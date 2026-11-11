"""Minimal Prefixwise proxy (stdlib only).

Drop-in OpenAI-compatible endpoint. Sees the exact bytes on the wire,
fingerprints the request, chains the provider's native diagnostics,
records a ledger entry, and forwards to the upstream when configured.

- ``PREFIXWISE_UPSTREAM_URL``: upstream base URL, e.g. https://api.openai.com.
  If unset, the proxy runs in mock mode (``PREFIXWISE_MOCK_UPSTREAM=1``
  or no upstream): it returns a synthetic cached_tokens usage so the
  fingerprint/classifier/ledger path can be exercised locally with no
  API key and no cost.
- ``PREFIXWISE_API_KEY``: upstream API key (never logged).
- ``PREFIXWISE_LEDGER_PATH``: SQLite ledger path (default in-memory).
- ``PREFIXWISE_HMAC_KEY``: per-install HMAC key for short blocks.

Endpoints:
- POST /v1/chat/completions  (OpenAI-compatible)
- POST /v1/responses         (OpenAI Responses API, comparison_response_id chained)
- POST /v1/messages          (Anthropic-compatible, diagnostics chained)
- POST /tasks               (record a task outcome)
- GET  /health, GET /ledger  (ledger roll-up as JSON)
- GET  /requests/<id>       (one ledger entry, by X-Prefixwise-Request-Id)

The client gets the upstream response and status unchanged. Prefixwise's
verdict rides in response headers: X-Prefixwise-Request-Id,
X-Prefixwise-Cause, X-Prefixwise-Legit, X-Prefixwise-Avoidable-Tokens.
Session/agent identity: X-Prefixwise-Session-Id / X-Prefixwise-Agent-Id
headers, or session_id / agent_id body fields (stripped before
forwarding), or the client's own x-session-id (forwarded unchanged: routers
route on it). Prefixwise sends no x-session-id of its own. Client auth
headers are passed through; PREFIXWISE_API_KEY, if set, overrides them. stream=true is relayed as SSE unchanged; its verdict is recorded when
the stream ends and served at GET /requests/<X-Prefixwise-Request-Id>.
Task outcomes: POST /tasks {"task_id": ..., "success": bool}; requests
join a task via X-Prefixwise-Task-Id or a task_id body field.

Engine join (v2): the proxy sets X-Request-Id and session_id on
forwarded requests. vLLM >= v0.30.0 (Sep 22, 2026) stamps session_id
onto every BlockStored KV event; LMCache --lookup-hash-log-dir logs
the same request id; SGLang accepts rid/session_id and
return_cached_tokens_details. Join is on request id, never hash
recomputation. Time-window fallback is flagged low confidence only.
The production-stack LookupMsg preflight is NOT claimed to work
against current LMCache (its import path breaks; pin unverified).
"""

from __future__ import annotations

import collections
import hashlib
import hmac
import json
import os
import re
import socket
import sys
import threading
from collections import OrderedDict
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .classifier import MIN_PROVIDER_TTL_SECONDS, _blocks_by_role, _role_unchanged, anthropic_min_prefix
from .details import Facts, body_facts, keeps_thinking, usage_facts
from .diagnostics import anthropic_diagnostics
from .fingerprint import Fingerprinter
from .fixes import apply_fixes, auto_breakpoints, parse_fixes
from . import api
from .pricing import PriceTable, breakdown, cache_reported, completion_tokens_of, normalize_usage  # noqa: F401 (re-exported)
from .ledger import Ledger, LedgerEntry
from .collector import (Collector, LANE_THREADS, MAX_SESSIONS, _CONVERSATION, _Lane,  # noqa: F401 (re-exported)
                        _SessionMap, _best_parent, _identity)
from .records import PARAM_KEYS, TurnRecord, param_fingerprint


def extract_blocks(body: dict, provider: str) -> list[tuple[str, object]]:
    """Split a rendered request into blocks in prefix order.

    OpenAI order is tools before developer/system (OpenAI Prompt
    Caching 201, verified Nov 2026); Anthropic is tools, system,
    messages. This is the v2 order, not the v1 assumption.
    """
    blocks = [(role, _strip_cache_markers(value)) for role, value in _extract_blocks_raw(body, provider)]
    if provider == "anthropic":
        blocks = [(role, _anthropic_text_blocks(role, value)) for role, value in blocks]
        if not keeps_thinking(str(body.get("model") or "")):
            # Older models drop earlier turns' thinking from the prompt themselves, so a client
            # that sends or omits it renders the same prompt: it must not look like an edit.
            blocks = [(role, _without_thinking(value)) for role, value in blocks]
    return blocks


def _without_thinking(value: object) -> object:
    if isinstance(value, dict) and value.get("role") == "assistant" and isinstance(value.get("content"), list):
        return {**value, "content": [b for b in value["content"]
                                     if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))]}
    return value


def _anthropic_text_blocks(role: str, value: object) -> object:
    """Anthropic treats string content as one text block: "x" and
    [{"type": "text", "text": "x"}] render the same prompt. Clients switch
    between the two (a message gets list form while it carries the moving
    cache breakpoint, then reverts to a string), so fingerprint one form."""
    if role == "system" and isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return {**value, "content": [{"type": "text", "text": value["content"]}]}
    return value


def _strip_cache_markers(value: object) -> object:
    """Drop cache_control markers before fingerprinting.

    A breakpoint says where the provider should cache, not what the prompt
    is. The standard Anthropic pattern moves the breakpoint to the newest
    message every turn; fingerprinting the marker made every ordinary turn
    look like HISTORY_EDIT. Breakpoint presence is checked separately
    (has_cache_breakpoint) on the untouched body.
    """
    if isinstance(value, dict):
        return {k: _strip_cache_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_markers(v) for v in value]
    return value


_SYSTEM_ROLES = ("system", "developer")


def _extract_blocks_raw(body: dict, provider: str) -> list[tuple[str, object]]:
    if provider != "anthropic" and is_responses_body(body):
        return _extract_responses_blocks(body)
    blocks: list[tuple[str, object]] = []
    # One block per tool: the classifier compares the tool multiset, so
    # a reordered tool list is TOOL_RESERIALIZE, not a catalog change.
    tools = body.get("tools") or []
    if provider == "anthropic":
        for tool in tools:
            blocks.append(("tools", tool))
        if body.get("system"):
            blocks.append(("system", body["system"]))
    else:
        for tool in tools:
            blocks.append(("tools", tool))
        if body.get("system") is not None:
            blocks.append(("system", body["system"]))
        # System and developer messages are the system prompt (OpenAI: tools, then developer /
        # system, then the conversation), each its own block, in order.
        for m in body.get("messages", []) or []:
            if m.get("role") in _SYSTEM_ROLES:
                blocks.append(("system", m.get("content")))
    for m in body.get("messages", []) or []:
        if m.get("role") in _SYSTEM_ROLES and provider != "anthropic":
            continue
        blocks.append(("tool_results" if _is_tool_result(m, provider) else "messages", m))
    return blocks


def _is_tool_result(m: dict, provider: str) -> bool:
    """Tool output is its own block group, so a mutated or cleared result is
    TOOL_RESULT_MUTATION (or a compaction), not a generic history edit.
    OpenAI chat: role "tool". Anthropic: a user turn made only of
    tool_result blocks. Responses: function_call_output items."""
    if m.get("type") == "function_call_output":
        return True
    if provider == "anthropic":
        content = m.get("content")
        return (m.get("role") == "user" and isinstance(content, list) and bool(content)
                and all(isinstance(c, dict) and c.get("type") == "tool_result" for c in content))
    return m.get("role") == "tool"


def cache_ttl_seconds(provider: str, body: dict, headers: dict | None = None) -> float | None:
    """The cache lifetime that applies to this request, or None if unknown.

    An X-Prefixwise-Cache-Ttl header wins. Anthropic: 5 minutes by default,
    1 hour when a cache_control breakpoint asks for ttl "1h". OpenAI: 24 h
    with prompt_cache_retention "24h"; the default in-memory retention is
    not a fixed number (5-10 min, up to 1 h), so it stays unknown. vLLM and
    other engines evict by memory pressure, not time.
    """
    raw = (headers or {}).get("x-prefixwise-cache-ttl")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    if provider == "anthropic":
        return 3600.0 if '"ttl": "1h"' in json.dumps(body) else 300.0
    if provider == "openai" and body.get("prompt_cache_retention") == "24h":
        return 86400.0
    return None


def is_responses_body(body: dict) -> bool:
    """OpenAI Responses API bodies carry input / instructions, not messages."""
    return "messages" not in body and ("input" in body or "instructions" in body)


def _extract_responses_blocks(body: dict) -> list[tuple[str, object]]:
    """Responses API: tools, then instructions, then input items, in prefix order.

    A string input is one user message. With previous_response_id the input
    holds only the new items; the proxy stitches them onto the stored prior
    turn (see PrefixwiseProxy.finalize).
    """
    blocks: list[tuple[str, object]] = [("tools", t) for t in body.get("tools") or []]
    if body.get("instructions") is not None:
        blocks.append(("system", body["instructions"]))
    items = body.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    for item in items or []:
        role = "tool_results" if isinstance(item, dict) and _is_tool_result(item, "openai") else "messages"
        blocks.append((role, item))
    return blocks


_MISSING = object()  # a field the client did not send
# The keys Prefixwise puts inside the fields it adds: an upstream may name either in its error.
_INNER_KEYS = {"prompt_cache_options": ("comparison_response_id",), "diagnostics": ("previous_message_id",),
               "stream_options": ("include_usage",), "vllm_xargs": ("kv_cache_report_mode",)}

# Body fields Prefixwise reads but upstream APIs do not accept. Stripped
# before forwarding (OpenAI rejects unknown fields with a 400). vLLM, and
# SGLang with --kv-events, get Prefixwise's session id: they stamp it on KV events.
PREFIXWISE_FIELDS = ("session_id", "agent_id", "task_id", "prefixwise_compaction_event",
                     "prefixwise_upstream_session_id")

# Client headers passed through to the upstream when present.
PASSTHROUGH_HEADERS = ("authorization", "x-api-key", "anthropic-version", "anthropic-beta",
                       "openai-organization", "openai-project", "openai-beta",
                       # Codex: session and thread identity, sticky routing (x-codex-turn-state),
                       # subagent and client ids. Dropping them could cost cache hits.
                       "session-id", "thread-id", "originator", "version", "x-openai-subagent",
                       "x-client-request-id", "chatgpt-account-id",
                       # W3C trace context: the upstream (a traced vLLM or gateway) joins the trace.
                       "traceparent", "tracestate",
                       # The client's own session key: routers with session affinity (sgl-router,
                       # llm-d, AIBrix) route on it. Prefixwise never sends one of its own.
                       "x-session-id")
PASSTHROUGH_PREFIXES = ("x-codex-",)
ANTHROPIC_VERSION = "2023-06-01"
UPSTREAM_TIMEOUT = 600  # seconds; long generations stream for minutes
# A client socket that sends nothing (or stops reading) for this long is
# dropped, so idle or half-sent connections cannot pin threads forever.
CLIENT_TIMEOUT = float(os.environ.get("PREFIXWISE_CLIENT_TIMEOUT", "120"))

# Upstream response headers relayed to the client: SDK retry hints, the
# provider's request id (support tickets), and rate-limit state.
RELAY_HEADERS = ("retry-after", "retry-after-ms", "x-should-retry", "x-request-id", "request-id",
                 "x-models-etag")
RELAY_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-", "x-codex-")


def relay_headers(headers) -> dict[str, str]:
    if not headers:
        return {}
    return {k: v for k, v in headers.items()
            if k.lower() in RELAY_HEADERS or k.lower().startswith(RELAY_PREFIXES)}
# Sessions remembered for classification; least recently used are dropped,
# so a long-running proxy does not grow without bound. A dropped session's
# next turn classifies as FIRST_WRITE (honest: no prior turn is known).
def has_cache_breakpoint(body: dict) -> bool:
    """Anthropic: cache_control anywhere (top level, tools, system, messages)."""
    if body.get("cache_control"):
        return True
    return any("cache_control" in json.dumps(body.get(k) or [])
               for k in ("tools", "system", "messages"))


def _mock_tool_call(body: dict, provider: str, api: str) -> tuple[str, dict] | None:
    """prefixwise check: a call to a function tool when the latest message is a user turn that
    carries no tool result, so an agent's tool loop runs a second turn (the result, then the
    answer) and the second turn is what the gate judges. `tool_choice` is honoured: none calls
    nothing, a forced tool is the one called, else the first function tool. Arguments are
    placeholders, by type, for the schema's required properties. None: no tools, the turn already
    answers a tool call, or tool_choice forbids it."""
    tools = body.get("tools")
    msgs = body.get("input") if api == "responses" else body.get("messages")
    if isinstance(msgs, str):
        msgs = [{"role": "user", "content": msgs}]
    if not isinstance(tools, list) or not tools or not isinstance(msgs, list) or not msgs:
        return None
    last = msgs[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None  # a tool result (OpenAI: role tool / function_call_output), or an assistant turn
    if provider == "anthropic" and isinstance(last.get("content"), list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in last["content"]):
        return None
    choice = body.get("tool_choice")
    forced = None
    if choice == "none" or (isinstance(choice, dict) and choice.get("type") == "none"):
        return None
    if isinstance(choice, dict):
        forced = (choice.get("name") or (choice.get("function") or {}).get("name")
                  if choice.get("type") in ("tool", "function") else None)
    candidates = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if provider == "anthropic":  # built-in server tools (web_search_2025...) have no schema: skipped
            if t.get("type") not in (None, "custom"):
                continue
            name, schema = t.get("name"), t.get("input_schema")
        elif api == "responses":
            if t.get("type") != "function":
                continue
            name, schema = t.get("name"), t.get("parameters")
        else:
            fn = t.get("function") if t.get("type") == "function" else None
            if not isinstance(fn, dict):
                continue
            name, schema = fn.get("name"), fn.get("parameters")
        if isinstance(name, str) and name:
            candidates.append((name, schema))
    for name, schema in candidates:
        if forced is None or name == forced:
            return name, _placeholder_args(schema)
    return None


def _placeholder_args(schema) -> dict:
    """Values of the right JSON type for a schema's required properties."""
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    out = {}
    for key in schema.get("required") if isinstance(schema.get("required"), list) else []:
        p = props.get(key) if isinstance(props.get(key), dict) else {}
        typ = p.get("type")
        if isinstance(typ, list):
            typ = next((t for t in typ if t != "null"), None)
        if isinstance(p.get("enum"), list) and p["enum"]:
            out[key] = p["enum"][0]
        elif typ == "integer":
            out[key] = 1
        elif typ == "number":
            out[key] = 1.0
        elif typ == "boolean":
            out[key] = False
        elif typ == "array":
            out[key] = []
        elif typ == "object":
            out[key] = _placeholder_args(p)
        else:
            out[key] = "prefixwise"
    return out


def _diagnostic_objects(payload: dict):
    usage = payload.get("usage") or {}
    for diag in (payload.get("prompt_cache_diagnostics"), payload.get("diagnostics"),
                 usage.get("prompt_cache_diagnostics")):
        if isinstance(diag, dict):
            yield diag


def native_reason_from(payload: dict) -> str | None:
    """The provider's own cache-miss reason, or None (a hit has none).

    Anthropic (observed live 2026-11-06): {"diagnostics": {"cache_miss_reason":
    {"type": "tools_changed", "cache_missed_input_tokens": N}}}, and
    "diagnostics": null on a hit. OpenAI (documented): {"prompt_cache_diagnostics":
    {"type": "cache_miss", "reason": "..."}}. Both shapes are read.
    """
    for diag in _diagnostic_objects(payload):
        miss = diag.get("cache_miss_reason")
        if isinstance(miss, dict) and miss.get("type"):
            return str(miss["type"])
        if isinstance(miss, str) and miss:
            return miss
        if diag.get("reason"):
            return str(diag["reason"])
    return None


def native_type_from(payload: dict) -> str | None:
    """The provider's verdict type (OpenAI: cache_hit, cache_miss,
    comparison_response_not_found, unavailable). Anthropic reports no type;
    a cache_miss_reason means cache_miss. Distinct from the reason: a hit
    has a type but no reason, and "unavailable" means no verdict at all."""
    for diag in _diagnostic_objects(payload):
        if diag.get("cache_miss_reason"):
            return "cache_miss"
        if diag.get("type"):
            return str(diag["type"])
    return None


def _error(status: int, message: str) -> dict:
    return {"error": {"type": "prefixwise_proxy_error", "message": message, "status": status}}


class StreamAccumulator:
    """Reads an SSE stream line by line as it is relayed, and rebuilds the
    parts of a non-streaming payload the classifier needs: id, usage, and
    any diagnostics. Never alters a line it forwards.

    OpenAI / vLLM: usage arrives only when stream_options.include_usage is
    set, in a final chunk with empty choices. If the proxy injected that
    option, the usage-only chunk is consumed here and not forwarded, so
    the client sees exactly the stream it asked for.

    Anthropic: message_start carries id, input and cache usage;
    message_delta carries output_tokens (and may repeat input fields).
    """

    def __init__(self, provider: str, suppress_usage_chunk: bool = False, api: str = "chat"):
        self.provider = provider
        self.api = api
        self.suppress_usage_chunk = suppress_usage_chunk
        self.payload: dict = {"usage": {}}
        self.saw_usage = False
        self.done = False  # the stream's terminal event was seen: the client has the whole answer

    def feed(self, line: bytes) -> bool:
        """Consume one raw line. Returns False if it must not be forwarded."""
        if not line.startswith(b"data:"):
            return True
        data = line[5:].strip()
        if data == b"[DONE]":
            self.done = True
            return True
        if not data:
            return True
        try:
            event = json.loads(data)
        except ValueError:
            return True
        if not isinstance(event, dict):
            return True
        if self.provider == "anthropic":
            self._feed_anthropic(event)
            return True
        if self.api == "responses":
            self._feed_responses(event)
            return True
        if event.get("id") and "id" not in self.payload:
            self.payload["id"] = event["id"]
        for key in ("prompt_cache_diagnostics", "diagnostics"):
            if isinstance(event.get(key), dict):
                self.payload[key] = event[key]
        if isinstance(event.get("usage"), dict):
            self.payload["usage"] = event["usage"]
            self.saw_usage = True
            if self.suppress_usage_chunk and not event.get("choices"):
                return False
        return True

    def _feed_responses(self, event: dict) -> None:
        """Responses API: every response.* lifecycle event carries the response
        object; response.completed (or .incomplete) carries usage and any
        prompt_cache_diagnostics. No stream_options needed."""
        if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
            self.done = True
        resp = event.get("response")
        if not isinstance(resp, dict):
            return
        if resp.get("id"):
            self.payload["id"] = resp["id"]
        if isinstance(resp.get("prompt_cache_diagnostics"), dict):
            self.payload["prompt_cache_diagnostics"] = resp["prompt_cache_diagnostics"]
        if isinstance(resp.get("usage"), dict):
            self.payload["usage"] = resp["usage"]
            self.saw_usage = True

    def _feed_anthropic(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "message_stop":
            self.done = True
        if kind == "message_start":
            msg = event.get("message") or {}
            if msg.get("id"):
                self.payload["id"] = msg["id"]
            for key in ("prompt_cache_diagnostics", "diagnostics"):
                if isinstance(msg.get(key), dict):
                    self.payload[key] = msg[key]
            if isinstance(msg.get("usage"), dict):
                self.payload["usage"].update(msg["usage"])
                self.saw_usage = True
        elif kind == "message_delta" and isinstance(event.get("usage"), dict):
            # message_delta usage is cumulative; only non-null values overwrite.
            self.payload["usage"].update({k: v for k, v in event["usage"].items() if v is not None})
            self.saw_usage = True


def _sources_of(events) -> list:
    """The per-cache event readers: a fleet's sources, or the one reader."""
    return list(getattr(events, "sources", None) or [events])


def _endpoint_label(endpoint: str) -> str:
    """tcp://10.0.0.5:5557 -> 10.0.0.5:5557: how a cache is named in verdicts."""
    return endpoint.split("://", 1)[-1]


def _reachable(endpoint: str, host: str) -> str:
    """An endpoint an engine bound on all interfaces (tcp://*:5557), as reached from here."""
    scheme, _, rest = endpoint.partition("://")
    if scheme == "tcp":
        h, _, port = rest.rpartition(":")
        if h in ("*", "0.0.0.0", "::", "[::]", ""):
            return f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"
    return endpoint


def _spans(setting):
    """OpenTelemetry export (prefixwise.otel), or None when off or unavailable."""
    from .otel import mode_from

    try:
        mode = mode_from(setting)
    except ValueError as exc:  # telemetry settings never stop the proxy or an app from working
        print(f"prefixwise: {exc}; OpenTelemetry export is off", file=sys.stderr)
        return None
    if mode is None:
        return None
    try:
        from .otel import Spans
        return Spans(mode)
    except ImportError:
        print("prefixwise: PREFIXWISE_OTEL is set but OpenTelemetry is not installed; "
              "pip install 'prefixwise[otel]'", file=sys.stderr)
        return None


class PrefixwiseProxy:
    def __init__(self, ledger_path: str = ":memory:", hmac_key: bytes | None = None,
                 upstream_url: str | None = None, api_key: str | None = None,
                 mock: bool | None = None, fixes: str | list[str] | tuple[str, ...] | None = None,
                 prices: str | None = None, engine: str | None = None, kv_events: str | None = None,
                 upstreams: dict[str, str] | None = None, full_attention: bool | None = None,
                 otel: bool | str | None = None, collector_url: str | None = None,
                 collector_token: str | None = None):
        # The control plane (prefixwise.collector): classification, pricing, the ledger and all
        # conversation state. In this process the edge below calls it directly; with a collector
        # elsewhere (PREFIXWISE_COLLECTOR), records are sent to it and this process keeps no ledger.
        url = collector_url if collector_url is not None else os.environ.get("PREFIXWISE_COLLECTOR")
        if url:
            from .transport import RemoteCollector, Sender
            token = collector_token if collector_token is not None else os.environ.get("PREFIXWISE_COLLECTOR_TOKEN")
            if url.startswith(("redis://", "rediss://", "unix://")):  # the record log (prefixwise.log)
                from .log import RedisBackend
                backend = RedisBackend(url)
            else:
                backend = url  # an HTTP collector: the Sender builds the backend
            self.collector = RemoteCollector(Sender(backend, token or None), PriceTable.load(prices))
        else:
            self.collector = Collector(Ledger(ledger_path), PriceTable.load(prices))
        self.collect_only = False  # `prefixwise collect`: a collector with no upstream (serves /api/v1/ingest)
        self.log_consumer = None  # `prefixwise collect --log`: reads the record log into this collector
        self.spans = _spans(otel if otel is not None else os.environ.get("PREFIXWISE_OTEL"))
        # Self-hosted engine behind the proxy (vllm, sglang), and its KV-event
        # stream for the engine join (prefixwise.engines).
        # None: from the environment; "" pins it off (prefixwise.sdk.Recorder).
        self.engine = (engine if engine is not None else os.environ.get("PREFIXWISE_ENGINE")) or None
        # Providers the engine serves (None: all). `prefixwise run` sets it when only one of the agent's
        # base URLs is the engine, so hosted traffic through the same proxy is not judged as an engine's.
        self.engine_providers: set[str] | None = None
        if self.engine not in (None, "vllm", "sglang"):
            raise ValueError(f"unknown engine {self.engine!r}: expected vllm or sglang")
        kv_events = (kv_events if kv_events is not None else os.environ.get("PREFIXWISE_KV_EVENTS")) or None
        if kv_events and self.engine not in ("vllm", "sglang"):
            raise ValueError("--kv-events needs --engine vllm or --engine sglang")
        if full_attention is None:
            full_attention = os.environ.get("PREFIXWISE_FULL_ATTENTION", "").lower() in ("1", "true", "yes")
        self.engine_events = None
        # SGLang (>= 0.5.20) reads a body session_id as the attribution label on its KV
        # events. Sent only once /server_info shows that is all it does (_check_sglang).
        self.sglang_join = False
        # /server_info showed --enable-session-radix-cache: a body session_id then decides eviction order.
        self.sglang_session_radix = False
        self._unprotected_warned = False
        self.engine_caches = 1  # KV caches behind the upstream, as far as known (data-parallel ranks, replicas)
        self._kv_auto = kv_events == "auto"
        endpoints = [e.strip() for e in kv_events.split(",") if e.strip()] if kv_events and not self._kv_auto else []
        if len(endpoints) > 1 and self.engine == "sglang":
            raise ValueError("several SGLang --kv-events endpoints: for one server's data-parallel ranks use "
                             "--kv-events auto; SGLang replicas behind a router are not supported yet")
        self._full_attention = full_attention
        if endpoints:
            self._start_events([(_endpoint_label(e), e) for e in endpoints], full_attention)
        # Opt-in request rewrites (see fixes.py). Off unless named here or
        # in PREFIXWISE_FIXES; turning them off restores pass-through.
        self.fixes = parse_fixes(fixes if fixes is not None else os.environ.get("PREFIXWISE_FIXES"))
        # Set by serve() from --auth-token / PREFIXWISE_AUTH_TOKEN; None = open.
        self.auth_token: str | None = None
        self.fingerprinter = Fingerprinter(hmac_key=hmac_key)
        self.upstream_url = (upstream_url or os.environ.get("PREFIXWISE_UPSTREAM_URL") or "").rstrip("/")
        # Per-provider upstreams ({"anthropic": url, "openai": url}) override upstream_url, so
        # one proxy can serve an agent that talks to both (prefixwise run).
        self.upstreams = {k: v.rstrip("/") for k, v in (upstreams or {}).items() if v}
        # Providers whose upstream base already ends in its API version, like an
        # OPENAI_BASE_URL (https://api.openai.com/v1, or a gateway's .../openai):
        # requests go to <base>/chat/completions, exactly as the SDK would send them.
        self.versioned = set()
        self.api_key = api_key or os.environ.get("PREFIXWISE_API_KEY")
        self.mock = mock if mock is not None else (
            not (self.upstream_url or self.upstreams) or os.environ.get("PREFIXWISE_MOCK_UPSTREAM") == "1")
        self.default_session = "default"  # for requests that carry no session id
        # prefixwise check (prefixwise.check): sees every request's blocks (never stored), makes the
        # mock answer a user turn that offers tools with a tool call so an agent's loop runs a second
        # turn, and holds a conversation's first mock response so per-second timestamps move.
        self.request_observer = None  # callable(provider, session_id, raw_blocks)
        self.check_mode = False  # POST /api/v1/check/scenario may rename default_session
        self.mock_tool_calls = False
        self._mock_calls: OrderedDict = OrderedDict()  # (session, model) -> was the last mock answer a tool call
        self.mock_first_turn_delay = 0.0
        self.engine_notes: list[str] = []  # what the engine check found; printed by serve()
        # Fields Prefixwise adds to requests that an upstream rejected, by provider: they are
        # not sent again (a gateway or compatible server that rejects unknown fields).
        self.rejected_fields: set[tuple[str, str]] = set()
        self._rejections: dict[tuple[str, str], int] = {}  # a field is dropped for good on its second rejection
        # Sub-reasons (prefixwise.details): request facts per lane and session, requests in
        # flight by tools+system prefix, and when another session last finished one.
        self.prefix_inflight: dict[str, dict[str, float]] = _SessionMap(MAX_SESSIONS)
        self.prefix_done: dict[str, tuple[float, str]] = _SessionMap(MAX_SESSIONS)
        # Completed requests by masked prefix key: (time, session, canonical key, dates-only key, ids-only
        # key, wrote inside tools+system). A first request whose masked key is here under another
        # canonical key differs from that session only in volatile values of the system prompt.
        self.prefix_seen: dict[str, tuple] = _SessionMap(MAX_SESSIONS)
        if self._kv_auto and self.engine == "vllm" and not self.mock:
            self._discover_vllm()
        if self.engine == "sglang" and (self.engine_events is not None or self._kv_auto) and not self.mock:
            self._check_sglang()
        # Per-session state, shared across the threaded server's handlers.
        # Every read and write holds self._lock: the classifier must compare
        # a turn with the turn recorded just before it, never a torn state.
        self._lock = threading.Lock()  # edge state: requests in flight, recordings pending
        self._upstream = threading.local()  # last upstream response headers, per thread
        self._recording: dict[tuple, threading.Event] = {}  # streams ended for the client, not yet recorded

    def close(self, timeout: float = 5.0) -> None:
        """Flush records still on their way to a remote collector; stop the log consumer and the
        KV-event reader."""
        if self.log_consumer is not None:
            self.log_consumer.stop(timeout)
        if getattr(self.collector, "remote", False):
            self.collector.close(timeout)
        elif self.ledger is not None:
            self.ledger.close_sinks(timeout)
        if self.engine_events is not None:
            self.engine_events.stop()

    # The collector's state and services, as the proxy's own names (the API, the CLI, tests).
    ledger = property(lambda self: self.collector.ledger)
    prices = property(lambda self: self.collector.prices)
    metrics = property(lambda self: self.collector.metrics)
    scorer = property(lambda self: self.collector.scorer)
    lane_threads = property(lambda self: self.collector.lane_threads)
    last_snapshot = property(lambda self: self.collector.last_snapshot)
    last_seen = property(lambda self: self.collector.last_seen)
    last_response_id = property(lambda self: self.collector.last_response_id)
    snapshot_by_response = property(lambda self: self.collector.snapshot_by_response)
    session_facts = property(lambda self: self.collector.session_facts)

    def _stitched_opening(self, previous_response_id: str):
        return self.collector.stitched_opening(previous_response_id)

    def _await_recording(self, lane, thread_key, stitched: bool) -> None:
        """Wait (at most 10 s) while this conversation's previous streamed turn has ended for the
        client but is still being recorded. A stitched turn whose parent is not recorded yet
        waits for every conversation of its lane being recorded."""
        with self._lock:
            if stitched and thread_key[1] is None:
                pending = [ev for key, ev in self._recording.items() if key[0] == lane]
            else:
                pending = [self._recording[thread_key]] if thread_key in self._recording else []
        for ev in pending:
            ev.wait(10)

    def begin_recording(self, ctx: dict) -> threading.Event:
        """The client has the whole streamed answer; until it is recorded, this conversation's
        next request waits (_await_recording)."""
        ev = threading.Event()
        with self._lock:
            self._recording[ctx["thread_key"]] = ev
        return ev

    def end_recording(self, ctx: dict, ev: threading.Event | None) -> None:
        if ev is None:
            return
        with self._lock:
            if self._recording.get(ctx["thread_key"]) is ev:
                del self._recording[ctx["thread_key"]]
        ev.set()

    def _engine_base(self) -> str:
        """The engine's own URL (no /v1): the upstream of a provider it serves, the OpenAI one by default."""
        for name in sorted(self.engine_providers or ("openai",), key=lambda n: n != "openai"):
            if self.upstreams.get(name):
                return self.upstreams[name].removesuffix("/v1")
        return (self.upstreams.get("openai") or self.upstream_url).removesuffix("/v1")

    def engine_for(self, provider: str) -> str | None:
        """The engine behind this provider's upstream, or None (a hosted API through the same proxy)."""
        if self.engine_providers is not None and provider not in self.engine_providers:
            return None
        return self.engine

    def _start_events(self, sources: list[tuple[str, str]], full_attention: bool = False,
                      ranks: list[int] | None = None) -> None:
        from .engines import EngineEvents, EngineFleet, SGLangEvents
        if self.engine == "sglang":
            make = lambda endpoint: SGLangEvents(endpoint, full_attention=full_attention)  # noqa: E731
        else:
            make = EngineEvents
        events = make(sources[0][1]) if len(sources) == 1 else EngineFleet(sources, make=make)
        for source, rank in zip(_sources_of(events), ranks or []):
            source.rank = rank  # each batch must carry this rank: a crossed endpoint is caught
        self.engine_events = events
        self.engine_caches = len(sources)
        events.start()

    def _discover_vllm(self) -> None:
        """--kv-events auto: every data-parallel rank's KV-event endpoint, from vLLM's
        GET /kv_event_sources ({rank: KVEventsConfig}, endpoints resolved per rank)."""
        base = self._engine_base()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            req = urllib.request.Request(base + "/kv_event_sources", headers=headers)
            with urllib.request.urlopen(req, timeout=5) as r:
                info = json.loads(r.read())
        except (OSError, ValueError) as exc:
            self.engine_notes.append(f"could not read vLLM's /kv_event_sources ({type(exc).__name__}; it needs "
                                     "a recent vLLM): give the endpoints with --kv-events; evictions stay inferred")
            return
        host = urllib.parse.urlsplit(base).hostname or "127.0.0.1"
        sources = []
        for rank, cfg in sorted((info or {}).items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
            cfg = cfg or {}
            if not cfg.get("enable_kv_cache_events") or cfg.get("publisher") != "zmq" or not cfg.get("endpoint"):
                continue
            sources.append((f"rank {rank}", _reachable(str(cfg["endpoint"]), host)))
        if not sources:
            self.engine_notes.append("vLLM publishes no KV events: start it with --kv-events-config "
                                     "'{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\"}'; "
                                     "evictions stay inferred")
            return
        self._start_events(sources, ranks=[int(label.split()[-1]) for label, _ in sources]
                           if all(label.split()[-1].isdigit() for label, _ in sources) else None)
        if len(sources) > 1:
            self.engine_notes.append(f"{len(sources)} data-parallel ranks: the KV events show which one served "
                                     "each request, so a request sent away from its cached prefix is a routing miss")

    def _check_sglang(self) -> None:
        """Read SGLang's /server_info once. Event verdicts stay off (session id not sent)
        unless it shows: version >= 0.5.20 (session on events), events published, no
        --enable-session-radix-cache (a session_id would pin the session's cache), and,
        with several data-parallel ranks (each with its own cache and event stream),
        --kv-events auto, no prefill/decode disaggregation and one node. With auto it
        subscribes to every rank. It also sizes the block index from the cache capacity,
        and a sliding-window model turns "not an eviction" off."""
        base = self._engine_base()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            req = urllib.request.Request(base + "/server_info", headers=headers)
            with urllib.request.urlopen(req, timeout=5) as r:
                info = json.loads(r.read())
        except (OSError, ValueError) as exc:
            self.engine_notes.append(f"could not read SGLang /server_info ({type(exc).__name__}"
                                     f"{', set PREFIXWISE_API_KEY' if '401' in str(exc) else ''}): "
                                     "evictions stay inferred")
            return
        problems = []
        version = str(info.get("version") or "")
        parts = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
        if not parts or tuple(int(x) for x in parts.groups()) < (0, 5, 20):
            problems.append(f"SGLang {version or '(version unknown)'} may put no session on its KV events "
                            "(needs >= 0.5.20)")
        if info.get("enable_session_radix_cache"):
            self.sglang_session_radix = True
            problems.append("SGLang runs with --enable-session-radix-cache, where a session_id pins the "
                            "session's cache")
            self.engine_notes.append("with --enable-session-radix-cache, send each conversation's id as a body "
                                     "session_id: SGLang does not read an x-session-id header as the session "
                                     "(sglang#47763). Prefixwise forwards a client's own session_id unchanged")
        events = info.get("kv_events") or {}
        if not events and not info.get("kv_events_config"):  # kv_events is None for ipc:// endpoints
            problems.append("SGLang is not publishing KV events: start it with --kv-events-config "
                            "'{\"publisher\": \"zmq\", \"endpoint\": \"tcp://*:5557\"}'")
        dp = max(int(info.get("dp_size") or 1), int(events.get("dp_size") or 1) if isinstance(events, dict) else 1)
        self.engine_caches = max(self.engine_caches, dp)  # several caches: never infer an eviction
        host = str((events or {}).get("endpoint_host") or "*") if isinstance(events, dict) else "*"
        if dp > 1 and not self._kv_auto:
            problems.append(f"SGLang runs {dp} data-parallel ranks, each with its own cache and events: "
                            "use --kv-events auto to follow every rank")
        if self._kv_auto and not (isinstance(events, dict) and events.get("endpoint_port_base")):
            problems.append("SGLang's /server_info describes no tcp KV-event endpoint (an ipc:// endpoint, or "
                            "an SGLang without the descriptor): start it with --kv-events-config "
                            "'{\"publisher\": \"zmq\", \"endpoint\": \"tcp://*:5557\"}'"
                            + ("" if dp > 1 else ", or give the endpoint"))
        elif self._kv_auto and host not in ("*", "0.0.0.0", "::", "[::]", ""):
            problems.append(f"SGLang's KV-event endpoint names a host ({host}), so SGLang connects out to it "
                            "instead of listening: bind it on all interfaces (tcp://*:PORT) for Prefixwise to "
                            "subscribe")
        if dp > 1 and str(info.get("disaggregation_mode") or "null") != "null":
            problems.append("SGLang runs prefill/decode disaggregation: a request's ranks are chosen per request")
        if dp > 1 and int(info.get("nnodes") or 1) > 1:
            problems.append("SGLang's ranks span several nodes, but /server_info advertises one event host")
        if problems:
            self.engine_notes.extend(problems)
            self.engine_notes.append("evictions stay inferred")
            if self._kv_auto:  # never subscribed: nothing to stop
                self.engine_events = None
            return
        if self._kv_auto:
            upstream = urllib.parse.urlsplit(base).hostname or "127.0.0.1"
            port = int(events["endpoint_port_base"])
            ranks = list(range(max(1, int(events.get("dp_size") or 1))))
            sources = [(f"rank {r}", _reachable(f"tcp://{host}:{port + r}", upstream)) for r in ranks]
            self._start_events(sources, self._full_attention, ranks=ranks if len(ranks) > 1 else None)
            if len(ranks) > 1:
                self.engine_notes.append(f"{len(ranks)} data-parallel ranks: the KV events show which one "
                                         "served each request, so a request sent away from its cached prefix "
                                         "is a routing miss")
        states = [s or {} for s in info.get("internal_states") or []]
        usage = [s.get("memory_usage") or {} for s in states]
        if any(u.get("token_capacity_swa") for u in usage):
            self.engine_events.hybrid = True  # the fleet sets every rank
            self.engine_notes.append("the model has sliding-window layers: evictions can be confirmed, "
                                     "never ruled out")
        capacity = max([int(u.get("token_capacity") or 0) for u in usage]
                       + [int(info.get("max_total_num_tokens") or 0)])
        blocks = capacity // max(1, int(info.get("page_size") or 1))
        if blocks:
            for source in _sources_of(self.engine_events):  # per rank: each has its own capacity
                source.max_alive = max(source.max_alive, blocks * 5 // 4 + 1024)
            if blocks > 2_000_000:
                self.engine_notes.append(f"SGLang caches up to {blocks:,} blocks: Prefixwise's index needs about "
                                         f"{blocks * 80 // 1_000_000:,} MB (--page-size 16 cuts it 16-fold)")
        self.sglang_join = True

    # -- shared request path ---------------------------------------------------------

    def prepare(self, provider: str, body: dict, headers: dict[str, str] | None = None,
                api: str = "chat") -> dict:
        """Read identity, strip Prefixwise fields, chain diagnostics.

        Returns a context dict for forward/finalize. body itself is not
        mutated; ctx["forward_body"] is what the upstream receives.
        """
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        for h in ("x-prefixwise-session-id", "x-prefixwise-agent-id", "x-prefixwise-task-id"):
            if headers.get(h) and len(headers[h]) > MAX_ID_CHARS:
                raise ClientError(f"{h} is longer than {MAX_ID_CHARS} characters")
        client_key = headers.get("x-session-id")  # the client's routing key, forwarded unchanged
        client_key = client_key if client_key and len(client_key) <= MAX_ID_CHARS else None
        # Known conversation ids first (Codex's own headers among them); the generic x-session-id after
        # them, since its meaning is up to each client; a user id last.
        given = (headers.get("x-prefixwise-session-id") or body.get("session_id")
                 or headers.get("thread-id") or headers.get("session-id")  # Codex
                 or client_key)
        session_is_user = not given and bool((body.get("metadata") or {}).get("user_id"))
        session_id = str(given or (body.get("metadata") or {}).get("user_id") or self.default_session)
        task_id = headers.get("x-prefixwise-task-id") or body.get("task_id")
        forward_body = {k: v for k, v in body.items() if k not in PREFIXWISE_FIELDS}
        # E1-open retry turns: a harness-only session id, used for the mock provider's cache lane.
        # The proxy's own session (snapshot chain, gap_seconds) stays session_id, and nothing
        # sent upstream changes (Prefixwise sends no session header). The retry turn also
        # carries a fresh cache_salt, which is what actually makes the
        # vLLM prefix cold (session_id alone is event metadata, not a
        # cache partition; cache_salt is mixed into the first block's
        # hash, docs.vllm.ai Automatic Prefix Caching).
        upstream_session_id = (body.get("prefixwise_upstream_session_id")
                               or headers.get("x-prefixwise-upstream-session-id"))
        upstream_session_id = str(upstream_session_id) if upstream_session_id else None
        # vLLM and SGLang put a body session_id on their KV events, so the engine
        # join can attribute blocks to sessions. Hosted APIs reject unknown fields.
        engine_token = None
        injected: dict = {}  # field -> the client's own value (_MISSING if none): undone if rejected

        def inject(field: str, value) -> bool:
            if (provider, field) in self.rejected_fields:
                return False
            injected.setdefault(field, forward_body.get(field, _MISSING))
            forward_body[field] = value
            return True
        engine = self.engine_for(provider)
        if (engine == "sglang" and api == "chat" and self.engine_events is not None and self.sglang_join
                and body.get("n") in (None, 1)):  # n > 1: the choices' stores would mix
            if inject("session_id", session_id):
                engine_token = self.engine_events.request_sent(session_id)
        if engine == "vllm" and api == "chat" and inject("session_id", session_id):
            xargs = body.get("vllm_xargs")
            if (self.engine_events is not None and (xargs is None or isinstance(xargs, dict))
                    and ("vllm", "vllm_xargs") not in self.rejected_fields):
                # Report every block a request uses, reused ones too, in order, so the
                # blocks of this session's previous request are known (docs/ENGINES.md).
                inject("vllm_xargs", {**(xargs or {}), "kv_cache_report_mode": "full"})
                engine_token = self.engine_events.request_sent(session_id)
        if engine == "sglang" and self.sglang_session_radix and "session_id" not in forward_body \
                and body.get("session_id"):
            # With --enable-session-radix-cache the body session_id is SGLang's own field: it decides what is
            # evicted first, so the client's value reaches the engine as sent. Elsewhere it changes no cache
            # hit (measured) and stays a Prefixwise field; inject() undoes it if the server refuses it. Any API:
            # SGLang reads it on chat, completions, responses and /generate alike.
            inject("session_id", body["session_id"])
        # The worker keeps this session's KV unprotected: it runs session radix cache, the client named the
        # session only in an x-session-id header, and SGLang reads the session from the body alone (per
        # sglang#47763 sgl-router does not forward the header either; #45620 would read it). A client that sent
        # a body session_id the server then refused is not told to send one.
        unprotected = (engine == "sglang" and self.sglang_session_radix and client_key is not None
                       and not body.get("session_id") and "session_id" not in forward_body)
        warn = False
        if unprotected:
            with self._lock:
                warn, self._unprotected_warned = not self._unprotected_warned, True
        if warn:
            print("prefixwise: requests name their session only in an x-session-id header, which SGLang does not "
                  "read: with --enable-session-radix-cache their KV is evicted first. Send the id as a body "
                  "session_id too (sglang#47763)", file=sys.stderr)
        # Native diagnostics chaining. Anthropic: diagnostics object (GA,
        # no beta header). OpenAI: prompt_cache_options.comparison_response_id,
        # Responses API only (it does not exist on /v1/chat/completions). A
        # client that sets its own comparison id keeps it.
        forward_body, fixes_applied = apply_fixes(forward_body, self.fixes)
        compaction = bool(body.get("prefixwise_compaction_event", False)) \
            or headers.get("x-prefixwise-compaction-event") in ("1", "true")
        # Only the system blocks and the opening message are fingerprinted here (cheap; a block's
        # hash does not depend on its position), to find this request's conversation.
        raw_blocks = extract_blocks(forward_body, provider)
        if self.request_observer is not None:
            self.request_observer(provider, session_id, raw_blocks)
        identity = _identity(self.fingerprinter.fingerprint_blocks(
            [blk for blk in raw_blocks if blk[0] in ("system", "system_dynamic")]
            + [blk for blk in raw_blocks if blk[0] in _CONVERSATION][:1]))
        lane = (session_id, str(body.get("model", "unknown")))
        stitched = api == "responses" and bool(body.get("previous_response_id"))
        # The conversation this request continues (a stitched Responses turn sends only its new
        # input: its conversation is its parent response's), and wait for that conversation's
        # previous turn if its stream ended but is still being recorded.
        thread_key = (lane, self._stitched_opening(body["previous_response_id"]) if stitched else identity[0])
        self._await_recording(lane, thread_key, stitched)
        if stitched:
            thread_key = (lane, self._stitched_opening(body["previous_response_id"]))
        # Chain the provider's diagnostics to the previous turn of this request's own
        # conversation, so its verdict compares the pair the classifier compares.
        previous_id = self.collector.chain_parent(session_id, lane, identity, compaction,
                                                  body["previous_response_id"] if stitched else None)
        # Chain only to an id from the same API: a session can mix Chat Completions
        # (chatcmpl-...) and Responses (resp_...) on one model, and OpenAI rejects a
        # foreign id with a 400 (found by the compatibility check, 2026-11-07).
        if provider == "anthropic" and "diagnostics" not in body:
            same_api = previous_id if previous_id and previous_id.startswith("msg") else None
            inject("diagnostics", anthropic_diagnostics(same_api)["diagnostics"])
        elif api == "responses" and previous_id and previous_id.startswith("resp"):
            # Merge into the client's own prompt_cache_options; a comparison
            # id the client chose wins.
            opts = dict(body.get("prompt_cache_options") or {})
            if "comparison_response_id" not in opts:
                opts["comparison_response_id"] = previous_id
                inject("prompt_cache_options", opts)
        # Caches are per model and per workspace (credentials), so both are in the prefix key.
        credential = hashlib.sha256((headers.get("x-api-key") or headers.get("authorization")
                                     or self.api_key or "").encode()).hexdigest()[:12]
        facts = body_facts(provider, forward_body, model=str(body.get("model", "")), credential=credential)
        facts.session_unprotected = unprotected
        sent = time.time()
        unmarked = None
        # Anthropic's own cache only: a self-hosted engine caches prefixes by itself and may refuse the
        # field (and the in-process middleware has no resend to fall back on).
        if ("auto_breakpoints" in self.fixes and provider == "anthropic" and facts.prefix_key_masked
                and self.engine_for(provider) is None
                and (provider, "auto_breakpoints") not in self.rejected_fields):
            # A write pays only when read: place markers only when this edge saw the same tools and
            # system prompt (this session's previous turn, or another session's) within the cache TTL.
            with self._lock:
                seen = self.prefix_seen.get(facts.prefix_key_masked)
            ttl = cache_ttl_seconds(provider, body, headers) or MIN_PROVIDER_TTL_SECONDS
            # the same bytes, not a prefix that matches only with its timestamp masked: that one is
            # rewritten every turn, and a marker on it would buy a write nobody reads
            if seen is not None and sent - seen[0] <= ttl and seen[2] == facts.prefix_key_canonical:
                marked = auto_breakpoints(forward_body, anthropic_min_prefix(facts.model))
                if marked is not None:
                    unmarked, forward_body = forward_body, marked
                    fixes_applied = [*fixes_applied, "auto_breakpoints"]
                    facts = body_facts(provider, forward_body, model=facts.model, credential=credential)
        request_id = str(uuid.uuid4())
        facts.sent_at = sent
        if facts.prefix_key:
            with self._lock:
                flying = self.prefix_inflight.get(facts.prefix_key)
                if flying is None:
                    flying = self.prefix_inflight[facts.prefix_key] = {}
                for rid, t in list(flying.items()):
                    if sent - t > 60:  # only the last second matters; drop what never completed
                        flying.pop(rid, None)
                if flying:
                    facts.fanout_age = sent - max(flying.values())
                flying[request_id] = sent
                done = self.prefix_done.get(facts.prefix_key)
                if done is not None and done[1] != session_id:
                    facts.shared_age, facts.shared_write_pos = sent - done[0], done[2]
                seen = self.prefix_seen.get(facts.prefix_key_masked) if facts.prefix_key_masked else None
                if seen is not None and seen[1] != session_id and seen[2] != facts.prefix_key_canonical:
                    facts.volatile_shared_age, facts.volatile_shared_wrote = sent - seen[0], seen[5]
                    facts.volatile_shared_kind = ("date" if seen[3] == facts.prefix_key_masked_time else
                                                  "id" if seen[4] == facts.prefix_key_masked_id else "mixed")
        return {
            "provider": provider, "api": api, "body": body, "headers": headers,
            "facts": facts, "sent_at": sent,
            "forward_body": forward_body, "session_id": session_id,
            "upstream_session_id": upstream_session_id, "engine_token": engine_token,
            "injected": injected,
            "unmarked_body": unmarked,  # the client's request before auto_breakpoints: resent on a 400
            "credential": credential,
            "agent_id": str(headers.get("x-prefixwise-agent-id") or body.get("agent_id") or "proxy"),
            "task_id": str(task_id) if task_id else None,
            "request_id": request_id, "model": str(body.get("model", "unknown")),
            # E1-open / hooks: a PreCompact/compaction event fired before
            # this turn, so the classifier can tell COMPACTION_REWRITE from
            # HISTORY_EDIT. Never forwarded.
            "compaction_event": bool(body.get("prefixwise_compaction_event", False))
            or headers.get("x-prefixwise-compaction-event") in ("1", "true"),
            "stream": bool(body.get("stream")),
            "t0": time.perf_counter(),  # for latency and Prefixwise's own overhead
            "fixes_applied": fixes_applied,
            "fork_of": headers.get("x-prefixwise-fork-of"),
            "thread_key": thread_key,  # (lane, opening): this conversation, for the recording wait
            "session_is_user": session_is_user,  # a user id, not a conversation id (OpenTelemetry)
            "ttl_seconds": cache_ttl_seconds(provider, body, headers),
        }

    def finalize(self, ctx: dict, status: int, payload: dict) -> tuple[dict, LedgerEntry]:
        """Classify and record one completed request from its (rebuilt) payload: build the turn
        record (the edge's half) and hand it to the collector."""
        provider = ctx["provider"]
        usage = payload.get("usage") or {}
        prompt_tokens, cached = normalize_usage(provider, usage)
        # Fingerprint what the provider actually received: with a fix on,
        # that is the rewritten request, not the client's original.
        fps = self.fingerprinter.fingerprint_blocks(extract_blocks(ctx["forward_body"], provider))
        verdict = None
        if ctx.get("engine_token") is not None:  # may wait a few ms for events
            reported = cache_reported(provider, usage)  # vLLM needs --enable-prompt-tokens-details
            if not reported and not getattr(self, "_warned_details", False):
                self._warned_details = True
                flag = "--enable-cache-report" if self.engine == "sglang" else "--enable-prompt-tokens-details"
                print(f"prefixwise: {self.engine} is not reporting cached tokens; start it with "
                      f"{flag}, or every request looks like a miss", file=sys.stderr)
            verdict = self.engine_events.request_done(ctx["engine_token"], cached if reported else None,
                                                      prompt_tokens)
        rec = self._record(ctx, status, fps, usage=usage, payload=payload, engine_verdict=verdict)
        result, entry = self.collector.ingest(rec)
        with self._lock:
            self._release_prefix(ctx, done_session=ctx["session_id"])
        result["response"] = payload
        if self.spans is not None:  # never raises
            self.spans.export(ctx, entry, payload, default_session=self.default_session,
                              upstream=self._span_upstream(ctx))
        return result, entry

    def _record(self, ctx: dict, status: int, fps: list, *, usage: dict | None = None,
                payload: dict | None = None, engine_verdict=None, error: str | None = None) -> TurnRecord:
        """The turn record for this request: everything the collector needs, no prompt text."""
        body, provider = ctx["body"], ctx["provider"]
        usage = usage or {}
        parts = breakdown(provider, usage)
        facts = ctx.get("facts") or Facts(provider=provider)
        if usage:
            facts = usage_facts(facts, usage, parts.cache_write_5m + parts.cache_write_1h + parts.cache_write)
        facts.done_at = ctx.get("span_end") or time.time()  # when the response completed
        payload = payload if isinstance(payload, dict) else {}
        verdict = None  # (evicted, resident[, routed]) from the engine join, as a list on the wire
        if engine_verdict is not None:
            routed = engine_verdict[2] if len(engine_verdict) > 2 and engine_verdict[2] is not None else None
            verdict = [engine_verdict[0], engine_verdict[1], list(routed) if routed is not None else None]
        return TurnRecord(
            provider=provider, api=ctx.get("api") or "chat", model=ctx["model"], session_id=ctx["session_id"],
            agent_id=ctx["agent_id"], request_id=ctx["request_id"], task_id=ctx["task_id"],
            session_is_user=bool(ctx.get("session_is_user")), fork_of=ctx.get("fork_of"),
            compaction_event=bool(ctx["compaction_event"]), stream=bool(ctx.get("stream")),
            previous_response_id=(str(body["previous_response_id"]) if ctx.get("api") == "responses"
                                  and body.get("previous_response_id") else None),
            blocks=fps, params={k: param_fingerprint(body.get(k)) for k in PARAM_KEYS if body.get(k) is not None},
            # the request the provider received: with auto_breakpoints on, the markers Prefixwise placed
            has_breakpoint=has_cache_breakpoint(ctx.get("forward_body") or body), ttl_seconds=ctx.get("ttl_seconds"), facts=facts,
            fixes_applied=list(ctx.get("fixes_applied") or []),
            sent_at=ctx.get("span_start") or ctx.get("sent_at") or facts.done_at,
            engine=self.engine_for(provider), engine_caches=self.engine_caches, engine_verdict=verdict,
            engine_block=getattr(self.engine_events, "block_size", None) if self.engine_for(provider) else None,
            status=status, error=error, usage=usage, response_model=(str(payload["model"]) if payload.get("model")
                                                                      else None),
            response_id=payload.get("id"), native_reason=native_reason_from(payload) if payload else None,
            latency_ms=ctx.get("upstream_ms"), overhead_ms=self._overhead_ms(ctx), done_at=facts.done_at)

    def finalize_safely(self, ctx: dict, status: int, payload: dict) -> tuple[dict, LedgerEntry | None]:
        """finalize(), failing open: the client already paid for this response,
        so a ledger or classifier failure (disk full, locked database, a bug)
        must never cost them it. The response passes through unrecorded."""
        try:
            return self.finalize(ctx, status, payload)
        except Exception as exc:
            print(f"prefixwise: request {ctx['request_id']} passed through unrecorded: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self.record_failure(ctx, status, error="not_recorded")  # at least a row, if the ledger works
            return {"request_id": ctx["request_id"], "status": status, "cause": None,
                    "recorded": False, "response": payload}, None

    @staticmethod
    def _overhead_ms(ctx: dict) -> float | None:
        """Prefixwise's own time: everything since the request arrived, minus upstream."""
        if "t0" not in ctx or ctx.get("upstream_ms") is None:
            return None
        return max(0.0, (time.perf_counter() - ctx["t0"]) * 1000 - ctx["upstream_ms"])

    def _release_prefix(self, ctx: dict, done_session: str | None = None) -> None:
        """The request is no longer in flight; if it succeeded, its prefix is now cached
        (as far as this proxy knows) for other sessions to reuse. Caller holds the lock
        by finalize and record_failure."""
        facts = ctx.get("facts")
        if facts is None or not facts.prefix_key:
            return
        flying = self.prefix_inflight.get(facts.prefix_key)
        if flying is not None:
            flying.pop(ctx.get("request_id"), None)
        # Writes happen only at breakpoints: other sessions can reuse this request's entry for
        # tools+system only if it wrote one ending there.
        inside = [b for b in facts.breakpoints if b < facts.sys_end]
        if done_session is not None and facts.cache_write > 0 and inside:
            self.prefix_done[facts.prefix_key] = (time.time(), done_session, max(inside))
        if done_session is not None and facts.prefix_key_masked:
            self.prefix_seen[facts.prefix_key_masked] = (
                time.time(), done_session, facts.prefix_key_canonical, facts.prefix_key_masked_time,
                facts.prefix_key_masked_id, bool(facts.cache_write > 0 and inside))

    def response_started(self, ctx: dict) -> None:
        """The upstream began answering: the request's prefill is done and its cache entry
        exists, so it no longer counts as a request other requests are waiting on."""
        facts = ctx.get("facts")
        if facts is None or not facts.prefix_key:
            return
        with self._lock:
            flying = self.prefix_inflight.get(facts.prefix_key)
            if flying is not None:
                flying.pop(ctx.get("request_id"), None)

    def record_failure(self, ctx: dict, status: int, payload: dict | None = None,
                       error: str | None = None) -> None:
        """A request that produced no verdict (upstream error, cut-off stream) is a
        row too, so error rates exist and every other rate has an honest
        denominator (docs/DATA.md). Never fails the request."""
        err = (payload or {}).get("error") if isinstance(payload, dict) else None
        kind = error or (err.get("type") if isinstance(err, dict) else None) or f"http_{status}"
        rec = self._record(ctx, status, [], error=str(kind)[:80])
        with self._lock:
            self._release_prefix(ctx)
        entry = None
        try:
            entry = self.collector.ingest_failure(rec)
        except Exception as exc:
            print(f"prefixwise: failure for {ctx['request_id']} not recorded: {type(exc).__name__}",
                  file=sys.stderr, flush=True)
        if self.spans is not None:
            if entry is None:  # the row never made it: still a span, from the same facts
                entry = LedgerEntry(session_id=rec.session_id, agent_id=rec.agent_id, provider=rec.provider,
                                    model=rec.model, cause=None, prompt_tokens=0, cached_tokens=0,
                                    avoidable_tokens=0, avoidable_cost=None, task_id=rec.task_id,
                                    request_id=rec.request_id, status=status, error=rec.error, api=rec.api,
                                    stream=rec.stream, latency_ms=rec.latency_ms, overhead_ms=rec.overhead_ms)
            self.spans.export(ctx, entry, payload if isinstance(payload, dict) else None,
                              default_session=self.default_session, upstream=self._span_upstream(ctx))

    # -- non-streaming ---------------------------------------------------------------

    def handle(self, provider: str, body: dict, headers: dict[str, str] | None = None,
               api: str = "chat") -> tuple[dict, LedgerEntry | None]:
        """Classify, forward, and record one non-streaming request.

        Returns (result, entry). result["status"] and result["response"]
        are the upstream status and payload, returned to the client
        unchanged. On an upstream error nothing is classified or
        recorded (no usage to score) and entry is None.
        """
        ctx = self.prepare(provider, body, headers, api=api)
        if ctx["stream"]:
            self.response_started(ctx)  # not sent at all: nothing in flight
            return {"request_id": ctx["request_id"], "status": 400, "cause": None,
                    "response": _error(400, "use handle_stream for stream=true")}, None
        self._upstream.headers = {}
        sent = time.perf_counter()
        status, payload = self._forward(provider, ctx["forward_body"], ctx["request_id"],
                                        ctx["upstream_session_id"] or ctx["session_id"],
                                        ctx["headers"], api=api)
        if status >= 400 and self._undo_rejected(ctx, status, payload):
            status, payload = self._forward(provider, ctx["forward_body"], ctx["request_id"],
                                            ctx["upstream_session_id"] or ctx["session_id"],
                                            ctx["headers"], api=api)
            if status < 400:
                self._markers_were_refused(ctx)
        self.response_started(ctx)
        ctx["upstream_ms"] = (time.perf_counter() - sent) * 1000
        upstream_headers = getattr(self._upstream, "headers", {}) or {}
        if status >= 400:
            if self.engine_events is not None:
                self.engine_events.discard(ctx.get("engine_token"))
            self.record_failure(ctx, status, payload)
            return {"request_id": ctx["request_id"], "status": status, "cause": None,
                    "response": payload, "upstream_headers": upstream_headers}, None
        result, entry = self.finalize_safely(ctx, status, payload)
        result["upstream_headers"] = upstream_headers
        return result, entry

    def _mock_payload(self, provider: str, model: str, request_id: str, session_id: str,
                      api: str = "chat", markers: bool = True, body: dict | None = None) -> dict:
        # Synthetic usage, shaped like a real cache with a breakpoint on the last block: 2,000
        # prompt tokens (so the OpenAI 1,024 minimum does not fire), all read when the session's
        # previous request on this model had the same tools and system prompt and its messages are
        # a prefix of these (a provider's cache serves everything the previous turn left in it),
        # else nothing read and, on Anthropic with cache_control, the whole prompt written.
        with self._lock:  # the session's latest turn on this model (each model has its own cache)
            lanes = getattr(self.collector, "lane_threads", None)  # none on an edge: the mock never hits there
            lane = lanes.get((session_id, model)) if lanes is not None else None
            prev = lane.threads[lane.latest][0] if lane is not None and lane.latest is not None else None
        cached = 0
        if prev is not None and body is not None:
            fps = self.fingerprinter.fingerprint_blocks(extract_blocks(body, provider))
            before, now = _blocks_by_role(prev.blocks), _blocks_by_role(fps)
            params = {k: param_fingerprint(body.get(k)) for k in PARAM_KEYS if body.get(k) is not None}
            if params == prev.params and all(_role_unchanged(r, before.get(r, []), now.get(r, []))
                                             for r in ("tools", "system", "system_dynamic", "messages", "tool_results")):
                cached = 2000  # a changed prompt-affecting parameter (a cache salt, tool_choice) renders anew
        write = 0
        if provider == "anthropic":  # as Anthropic bills: nothing without cache_control; a first write
            if not markers:
                cached = 0
            elif not cached or not getattr(prev, "has_breakpoint", True):  # the previous turn wrote nothing
                cached, write = 0, 2000
        call = None
        if self.mock_tool_calls and body is not None:
            # One tool call at a time: a turn that follows a tool call gets text, however the agent
            # rendered the result (a plain user message included), so the mock never drives a loop.
            with self._lock:
                last_was_call = self._mock_calls.get((session_id, model), False)
                call = None if last_was_call else _mock_tool_call(body, provider, api)
                self._mock_calls[(session_id, model)] = call is not None
                self._mock_calls.move_to_end((session_id, model))
                while len(self._mock_calls) > 10_000:
                    self._mock_calls.popitem(last=False)
        if self.mock_first_turn_delay and prev is None:
            time.sleep(self.mock_first_turn_delay)
        # Ids shaped like the real APIs' (msg_, resp_, chatcmpl-): diagnostics chaining checks them.
        rid = {"anthropic": "msg_", "responses": "resp_"}.get(
            provider if provider == "anthropic" else api, "chatcmpl-") + request_id[:8]
        call_id = "call_mock" + request_id[:8]
        if provider == "anthropic":
            content = ([{"type": "tool_use", "id": "toolu_mock" + request_id[:8], "name": call[0], "input": call[1]}]
                       if call else [{"type": "text", "text": "[prefixwise mock]"}])
            return {"id": rid, "type": "message", "role": "assistant", "model": model,
                    "content": content, "stop_reason": "tool_use" if call else "end_turn",
                    "usage": {"input_tokens": 2000 - cached - write, "cache_read_input_tokens": cached,
                              "cache_creation_input_tokens": write, "output_tokens": 1}}
        if api == "responses":
            item = ({"type": "function_call", "id": "fc_mock" + request_id[:8], "call_id": call_id,
                     "name": call[0], "arguments": json.dumps(call[1]), "status": "completed"} if call else
                    {"type": "message", "role": "assistant", "status": "completed",
                     "content": [{"type": "output_text", "text": "[prefixwise mock]", "annotations": []}]})
            return {"id": rid, "object": "response", "status": "completed", "model": model,
                    "output": [item],
                    "usage": {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": cached},
                              "output_tokens": 1, "total_tokens": 2001}}
        message = ({"role": "assistant", "content": None,
                    "tool_calls": [{"id": call_id, "type": "function",
                                    "function": {"name": call[0], "arguments": json.dumps(call[1])}}]} if call else
                   {"role": "assistant", "content": "[prefixwise mock]"})
        return {"id": rid, "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "tool_calls" if call else "stop", "message": message}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 1, "total_tokens": 2001,
                          "prompt_tokens_details": {"cached_tokens": cached}}}

    def _upstream_request(self, provider: str, body: dict, request_id: str, session_id: str,
                          client_headers: dict[str, str], api: str = "chat") -> urllib.request.Request:
        headers = {k: v for k, v in client_headers.items() if k in PASSTHROUGH_HEADERS or k.startswith(PASSTHROUGH_PREFIXES)}
        # No x-session-id of Prefixwise's own: a router with session affinity routes on it, and a
        # value Prefixwise invented ("default" for traffic with no session) would put every session
        # on one key. The client's own, if it sent one, is in PASSTHROUGH_HEADERS.
        headers.update({"content-type": "application/json", "x-request-id": request_id})
        if provider == "anthropic":
            if self.api_key:
                headers["x-api-key"] = self.api_key
                headers.pop("authorization", None)
            headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
        elif self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        path = ("/v1/messages" if provider == "anthropic"
                else "/v1/responses" if api == "responses" else "/v1/chat/completions")
        query = client_headers.get(QUERY_KEY)
        if query:  # e.g. Anthropic SDK's ?beta=true: forwarded as the client sent it
            path += "?" + query
        return urllib.request.Request(self.target(provider, path), method="POST",
                                      data=json.dumps(body).encode(), headers=headers)

    def _span_upstream(self, ctx: dict) -> str | None:
        """The upstream base URL a request went to, for span attributes; None in mock mode."""
        if self.mock:
            return None
        key = "anthropic" if ctx.get("provider") == "anthropic" else "openai"
        return self.upstreams.get(key, self.upstream_url) or None

    def target(self, provider: str, path: str) -> str:
        """The upstream URL for a request path like /v1/chat/completions."""
        key = "anthropic" if provider == "anthropic" else "openai"
        base = self.upstreams.get(key, self.upstream_url)
        # A base that already ends in /v1 (DeepSeek, Tencent TokenHub, Together and most OpenAI-compatible
        # APIs document theirs that way) takes the path without its own /v1, never /v1/v1/...
        if (key in self.versioned or base.rstrip("/").endswith("/v1")) and path.startswith("/v1/"):
            path = path[3:]
        return base.rstrip("/") + path

    def passthrough(self, method: str, raw_path: str, body: bytes | None,
                    client_headers: dict[str, str]) -> tuple[int, str, bytes]:
        """See _passthrough; relayed headers land in self._upstream.headers."""
        self._upstream.headers = {}
        return self._passthrough(method, raw_path, body, client_headers)

    def _passthrough(self, method: str, raw_path: str, body: bytes | None,
                     client_headers: dict[str, str]) -> tuple[int, str, bytes]:
        """Forward a request Prefixwise does not analyse (GET /v1/models,
        count_tokens, ...) to the upstream unchanged. Nothing is recorded.
        Returns (status, content type, body)."""
        if self.mock:
            return 404, "application/json", json.dumps(_error(
                404, f"{raw_path} is not available in mock mode (no upstream)")).encode()
        headers = {k: v for k, v in client_headers.items() if k in PASSTHROUGH_HEADERS or k.startswith(PASSTHROUGH_PREFIXES)}
        if body:
            headers["content-type"] = client_headers.get("content-type", "application/json")
        anthropic = raw_path.startswith("/v1/messages") or "anthropic-version" in client_headers \
            or "x-api-key" in client_headers
        if self.api_key:
            if anthropic:
                headers["x-api-key"] = self.api_key
                headers.pop("authorization", None)
            else:
                headers["authorization"] = f"Bearer {self.api_key}"
        if anthropic:
            headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
        req = urllib.request.Request(self.target("anthropic" if anthropic else "openai", raw_path),
                                     method=method, data=body or None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                self._upstream.headers = relay_headers(resp.headers)
                return resp.status, resp.headers.get("Content-Type", "application/json"), resp.read()
        except urllib.error.HTTPError as exc:
            self._upstream.headers = relay_headers(exc.headers)
            return exc.code, exc.headers.get("Content-Type", "application/json"), exc.read() or b""
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 502, "application/json", json.dumps(_error(502, f"upstream unreachable: {exc}")).encode()

    def _undo_rejected(self, ctx: dict, status: int, payload) -> bool:
        """The upstream rejected a field Prefixwise added (a gateway or compatible server
        that refuses unknown fields): put the client's request back as it was, and stop
        adding that field for this provider once it is rejected twice. True if the request
        should be sent again. Prefixwise must never turn a working request into an error."""
        injected = ctx.get("injected") or {}
        if status not in (400, 422) or not (injected or ctx.get("unmarked_body") is not None):
            return False
        text = (payload if isinstance(payload, str) else json.dumps(payload)).lower()
        undone = False
        if ctx.get("unmarked_body") is not None:
            # Any rejection of a request that carries placed markers: resend it as the client sent it,
            # whether or not the error names cache_control (a gateway may say only "extra inputs are not
            # permitted"). The markers sit inside blocks, so they are undone as a whole. Whether they
            # were the reason is known only when the resend succeeds (_markers_were_refused).
            undone = ctx["marker_retry"] = True
            ctx["forward_body"], ctx["unmarked_body"] = ctx["unmarked_body"], None
            ctx["fixes_applied"] = [f for f in ctx.get("fixes_applied") or [] if f != "auto_breakpoints"]
            old = ctx["facts"]
            facts = body_facts(ctx["provider"], ctx["forward_body"], model=old.model, credential=ctx.get("credential", ""))
            for name in ("sent_at", "fanout_age", "shared_age", "shared_write_pos", "volatile_shared_age",
                         "volatile_shared_wrote", "volatile_shared_kind", "session_unprotected"):
                setattr(facts, name, getattr(old, name))
            ctx["facts"] = facts
        # fields Prefixwise added that the error names go in the same pass: there is one resend
        rejected = [f for f in injected if any(name in text for name in (f, *_INNER_KEYS.get(f, ())))]
        for field in rejected:
            original = injected.pop(field)
            if original is _MISSING:
                ctx["forward_body"].pop(field, None)
            else:
                ctx["forward_body"][field] = original
            key = (ctx["provider"], field)
            with self._lock:
                self._rejections[key] = n = self._rejections.get(key, 0) + 1
                if n >= 2:  # twice: the upstream does not take it (once could be a bad id)
                    self.rejected_fields.add(key)
            if n == 2:
                print(f"prefixwise: the {ctx['provider']} upstream rejects the `{field}` field Prefixwise adds "
                      "or passes on; requests are resent without it, and it is no longer sent", file=sys.stderr)
            if field in ("session_id", "vllm_xargs") and self.engine_events is not None:
                self.engine_events.discard(ctx.get("engine_token"))
                ctx["engine_token"] = None
        return undone or bool(rejected)

    def _markers_were_refused(self, ctx: dict) -> None:
        """The resend without the placed markers went through, so the markers were what the upstream
        refused: count it, and after the second stop placing them for this provider."""
        if not ctx.pop("marker_retry", False):
            return
        key = (ctx["provider"], "auto_breakpoints")
        with self._lock:
            self._rejections[key] = n = self._rejections.get(key, 0) + 1
            if n >= 2:
                self.rejected_fields.add(key)
        if n == 2:
            print(f"prefixwise: the {ctx['provider']} upstream rejects the cache_control markers "
                  "auto_breakpoints adds; requests are resent without them, and they are no longer added",
                  file=sys.stderr)

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> tuple[int, dict]:
        raw = exc.read() or b""
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, _error(exc.code, raw.decode("utf-8", "replace")[:2000])

    def _forward(self, provider: str, body: dict, request_id: str, session_id: str,
                 client_headers: dict[str, str], api: str = "chat") -> tuple[int, dict]:
        """Send to the upstream. Returns (status, payload); never raises
        for upstream errors, so the client gets the upstream's own error."""
        if self.mock:
            return 200, self._mock_payload(provider, str(body.get("model", "unknown")),
                                           request_id, session_id, api, has_cache_breakpoint(body), body)
        req = self._upstream_request(provider, body, request_id, session_id, client_headers, api)
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                self._upstream.headers = relay_headers(resp.headers)
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            self._upstream.headers = relay_headers(exc.headers)
            return self._http_error(exc)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 502, _error(502, f"upstream unreachable: {exc}")
        except ValueError:
            return 502, _error(502, "upstream returned a non-JSON body")

    # -- streaming -------------------------------------------------------------------

    def open_stream(self, ctx: dict):
        """Open a streaming upstream request.

        Returns (status, lines, accumulator, error_payload). On success
        lines is an iterator of raw SSE lines to relay; on failure lines
        is None and error_payload is the upstream (or proxy) error.
        """
        provider, fwd, api = ctx["provider"], ctx["forward_body"], ctx.get("api", "chat")
        injected = False
        if (provider != "anthropic" and api != "responses"  # Responses always report usage
                and (provider, "stream_options") not in self.rejected_fields):
            opts = dict(fwd.get("stream_options") or {})
            if not opts.get("include_usage"):
                ctx.setdefault("injected", {}).setdefault("stream_options", fwd.get("stream_options", _MISSING))
                opts["include_usage"] = True
                fwd["stream_options"] = opts
                injected = True
        acc = StreamAccumulator(provider, suppress_usage_chunk=injected, api=api)
        if self.mock:
            return 200, iter(self._mock_stream(ctx)), acc, None
        for attempt in (1, 2):
            req = self._upstream_request(provider, ctx["forward_body"], ctx["request_id"],
                                         ctx["upstream_session_id"] or ctx["session_id"],
                                         ctx["headers"], api)
            try:
                resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
                ctx["upstream_headers"] = relay_headers(resp.headers)
                self.response_started(ctx)
                if attempt == 2:
                    self._markers_were_refused(ctx)
                break
            except urllib.error.HTTPError as exc:
                ctx["upstream_headers"] = relay_headers(exc.headers)
                status, payload = self._http_error(exc)
                if attempt == 1 and self._undo_rejected(ctx, status, payload):
                    if "stream_options" not in (ctx.get("injected") or {}):
                        acc = StreamAccumulator(provider, suppress_usage_chunk=False, api=api)
                    continue
                return status, None, acc, payload
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                return 502, None, acc, _error(502, f"upstream unreachable: {exc}")

        def lines():
            with resp:
                while True:
                    line = resp.readline()
                    if not line:
                        return
                    yield line
        return resp.status, lines(), acc, None

    def _mock_stream(self, ctx: dict) -> list[bytes]:
        payload = self._mock_payload(ctx["provider"], ctx["model"], ctx["request_id"],
                                     ctx["session_id"], ctx.get("api", "chat"),
                                     has_cache_breakpoint(ctx.get("forward_body") or ctx["body"]),
                                     ctx.get("forward_body") or ctx["body"])
        def ev(obj, name=None):  # one list item per line, as readline() yields them
            head = [f"event: {name}\n".encode()] if name else []
            return head + [b"data: " + json.dumps(obj).encode() + b"\n", b"\n"]
        if ctx["provider"] == "anthropic":
            u = payload["usage"]
            block = payload["content"][0]
            start = {"type": "message_start", "message": {
                "id": payload["id"], "type": "message", "role": "assistant", "model": ctx["model"],
                "content": [], "usage": {**u, "output_tokens": 0}}}
            out = ev(start, "message_start")
            if block["type"] == "tool_use":
                out += ev({"type": "content_block_start", "index": 0,
                           "content_block": {**block, "input": {}}}, "content_block_start")
                out += ev({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}},
                          "content_block_delta")
            else:
                out += ev({"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}}, "content_block_start")
                out += ev({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": "[prefixwise mock]"}}, "content_block_delta")
            out += ev({"type": "content_block_stop", "index": 0}, "content_block_stop")
            out += ev({"type": "message_delta", "delta": {"stop_reason": payload["stop_reason"]},
                       "usage": {"output_tokens": u["output_tokens"]}}, "message_delta")
            return out + ev({"type": "message_stop"}, "message_stop")
        if ctx.get("api") == "responses":
            # The full event sequence SDKs expect: item and part opened, text
            # delta, then every done event, then response.completed.
            created = {**payload, "status": "in_progress", "output": [], "usage": None}
            item = payload["output"][0]
            if item["type"] == "function_call":
                ids = {"item_id": item["id"], "output_index": 0}
                seq = [("response.created", {"response": created}),
                       ("response.in_progress", {"response": created}),
                       ("response.output_item.added", {"output_index": 0,
                                                       "item": {**item, "status": "in_progress", "arguments": ""}}),
                       ("response.function_call_arguments.delta", {**ids, "delta": item["arguments"]}),
                       ("response.function_call_arguments.done", {**ids, "arguments": item["arguments"]}),
                       ("response.output_item.done", {"output_index": 0, "item": item}),
                       ("response.completed", {"response": payload})]
            else:
                item = {**item, "id": "msg_mock"}
                part = item["content"][0]
                text = part["text"]
                seq = [("response.created", {"response": created}),
                       ("response.in_progress", {"response": created}),
                       ("response.output_item.added", {"output_index": 0,
                                                       "item": {**item, "status": "in_progress", "content": []}}),
                       ("response.content_part.added", {"item_id": "msg_mock", "output_index": 0,
                                                        "content_index": 0, "part": {**part, "text": ""}}),
                       ("response.output_text.delta", {"item_id": "msg_mock", "output_index": 0,
                                                       "content_index": 0, "delta": text}),
                       ("response.output_text.done", {"item_id": "msg_mock", "output_index": 0,
                                                      "content_index": 0, "text": text}),
                       ("response.content_part.done", {"item_id": "msg_mock", "output_index": 0,
                                                       "content_index": 0, "part": part}),
                       ("response.output_item.done", {"output_index": 0, "item": item}),
                       ("response.completed", {"response": {**payload, "output": [item]}})]
            out = []
            for i, (name, data) in enumerate(seq):
                out += ev({"type": name, "sequence_number": i, **data}, name)
            return out
        base = {"id": payload["id"], "object": "chat.completion.chunk", "model": ctx["model"]}
        choice = payload["choices"][0]
        if choice["finish_reason"] == "tool_calls":
            call = choice["message"]["tool_calls"][0]
            out = ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None,
                                                                 "tool_calls": [{"index": 0, **call}]}}]})
        else:
            out = ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant",
                                                                 "content": "[prefixwise mock]"}}]})
        out += ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]})
        if (ctx["forward_body"].get("stream_options") or {}).get("include_usage"):
            out += ev({**base, "choices": [], "usage": payload["usage"]})
        return out + [b"data: [DONE]\n", b"\n"]


MAX_BODY_BYTES = int(os.environ.get("PREFIXWISE_MAX_BODY_BYTES", str(32 * 1024 * 1024)))
MAX_ID_CHARS = 256  # session / agent / task ids: labels, not payloads


class ClientError(ValueError):
    """A request the proxy refuses: answered 400, never a dropped connection."""


def validate_body(body: dict) -> None:
    """Shape checks for the fields Prefixwise reads. A malformed request
    gets a 400 naming the field, instead of crashing a handler thread."""
    for key in ("messages", "tools"):
        if key in body and not isinstance(body[key], list):
            raise ClientError(f"'{key}' must be a list")
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            raise ClientError("every item in 'messages' must be an object")
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            raise ClientError("every item in 'tools' must be an object")
    inp = body.get("input")
    if inp is not None and not isinstance(inp, (str, list)):
        raise ClientError("'input' must be a string or a list")
    if isinstance(inp, list) and not all(isinstance(i, dict) for i in inp):
        raise ClientError("every item in 'input' must be an object")
    for key in ("session_id", "agent_id", "task_id"):
        if key in body and body[key] is not None and len(str(body[key])) > MAX_ID_CHARS:
            raise ClientError(f"'{key}' is longer than {MAX_ID_CHARS} characters")


# Endpoints Prefixwise classifies. Clients disagree on whether the base URL
# includes /v1 (dogfood 2026-11-06: opencode posted to /messages), so the
# bare forms are accepted too.
CLASSIFIED_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
_ALIASES = {p[len("/v1"):]: p for p in CLASSIFIED_PATHS}
LOCAL_PATHS = ("/health", "/ledger", "/tasks")
QUERY_KEY = "x-prefixwise-internal-query"  # carries the client's query string to the forwarder


def route(raw_path: str) -> tuple[str, str]:
    """(path, query): the query string split off, and /messages-style
    paths mapped to their /v1 form."""
    path, _, query = raw_path.partition("?")
    return _ALIASES.get(path, path), query


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")


class PrefixwiseHTTPServer(ThreadingHTTPServer):
    """Threaded server sized for agent fleets. The stdlib default listen
    backlog is 5, so a burst of parallel requests got connection resets."""

    request_queue_size = 256
    daemon_threads = True

    def handle_error(self, request, client_address):
        """A client hanging up (an agent cancelling a turn with Ctrl-C) is normal: say
        nothing. Anything else gets one line, never a traceback (socketserver's default
        prints one, which lands in the agent's terminal under `prefixwise run`)."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
            return
        print(f"prefixwise: error serving {client_address[0]}: {type(exc).__name__}: {exc}", file=sys.stderr)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.active = 0  # requests in flight, for a graceful drain on shutdown
        self._active_lock = threading.Lock()

    def process_request_thread(self, request, client_address):
        with self._active_lock:
            self.active += 1
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._active_lock:
                self.active -= 1

    def drain(self, seconds: float) -> int:
        """Wait up to `seconds` for in-flight requests; returns how many were left."""
        end = time.time() + seconds
        while self.active and time.time() < end:
            time.sleep(0.05)
        return self.active


def _role(proxy) -> str:
    if proxy.collect_only:
        return "collector"
    return "edge" if getattr(proxy.collector, "remote", False) else "proxy"


def _transport_gauges(proxy) -> dict[str, tuple[str, float]]:
    """The record transport's counters, as Prometheus gauges: an edge's sender, a collector's ingest."""
    if getattr(proxy.collector, "remote", False):
        s = proxy.collector.stats()
        return {"prefixwise_records_queued_total": ("Records handed to the sender.", s["queued"]),
                "prefixwise_records_sent_total": ("Records the collector accepted.", s["sent"]),
                "prefixwise_records_dropped_total": ("Records not delivered (queue full, or a batch failed).",
                                                     s["dropped"]),
                "prefixwise_records_rejected_total": ("Records the collector refused as malformed.", s["rejected"]),
                "prefixwise_records_pending": ("Records waiting to be sent.", s["pending"])}
    s = proxy.collector.stats
    log = proxy.log_consumer.snapshot() if proxy.log_consumer is not None else None
    extra = {} if log is None else {
        "prefixwise_log_records_consumed_total": ("Records read from the record log.", log["consumed"]),
        "prefixwise_log_records_acked_total": ("Records acknowledged after their row was written.", log["acked"]),
        "prefixwise_log_records_claimed_total": ("Records taken over from a consumer that died.", log["claimed"]),
        "prefixwise_log_records_replayed_total": ("Records replayed at start to rebuild state.", log["replayed"]),
        "prefixwise_log_errors_total": ("Log read errors.", log["errors"]),
        "prefixwise_log_replaying": ("Partitions still rebuilding state at start (0 = ready).", log["replaying"]),
        "prefixwise_log_conflicts": ("Partitions another collector was active on at start.", len(log["conflicts"]))}
    return {**extra, "prefixwise_ingest_records_total": ("Records classified.", s["records"]),
            "prefixwise_ingest_failures_total": ("Failure records (no verdict).", s["failures"]),
            "prefixwise_ingest_duplicates_total": ("Records seen twice (retried batches), skipped.", s["duplicates"]),
            "prefixwise_ingest_rejected_total": ("Records refused as malformed.", s["rejected"]),
            "prefixwise_ingest_concurrent_total": ("Turns sent before the compared turn ended (unknown).",
                                                   s["concurrent"])}


def make_handler(proxy: PrefixwiseProxy):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0: a streamed body ends when the connection closes, so no
        # chunked encoding is needed for SSE.
        timeout = CLIENT_TIMEOUT  # socket timeout on the client connection

        def log_message(self, *args):  # keep test output clean
            pass

        def _json(self, code: int, payload: dict, extra_headers: dict | None = None):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self, bearer: bool = False) -> bool:
            """With an auth token configured, every request but /health must
            carry it in X-Prefixwise-Token (constant-time compare). /metrics also
            takes it as Authorization: Bearer, which is how Prometheus sends it."""
            token = getattr(proxy, "auth_token", None)
            if not token:
                return True
            given = self.headers.get("X-Prefixwise-Token") or ""
            if not given and bearer:
                auth = self.headers.get("Authorization") or ""
                given = auth[7:] if auth[:7].lower() == "bearer " else ""
            if hmac.compare_digest(given.encode(), token.encode()):
                return True
            self._json(401, _error(401, "missing or wrong X-Prefixwise-Token"))
            return False

        def _client_headers(self) -> dict[str, str]:
            # Our internal query carrier can never be set by the client.
            headers = {k.lower(): v for k, v in self.headers.items() if k.lower() != QUERY_KEY}
            _, query = route(self.path)
            if query:
                headers[QUERY_KEY] = query
            return headers

        def _passthrough(self, method: str):
            if not self._authorized():
                return
            body = None
            if method in ("POST", "PUT", "PATCH"):
                try:
                    length = int(self.headers.get("Content-Length", 0) or 0)
                except ValueError:
                    return self._json(400, _error(400, "Content-Length is not a number"))
                if length < 0 or length > MAX_BODY_BYTES:
                    return self._json(413 if length > 0 else 400, _error(400, "bad Content-Length"))
                try:
                    body = self.rfile.read(length)
                except OSError:  # client stalled past CLIENT_TIMEOUT or hung up mid-body
                    self.close_connection = True
                    return
            status, ctype, data = proxy.passthrough(method, self.path, body, self._client_headers())
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (getattr(proxy._upstream, "headers", None) or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(data)

        def do_GET(self):
            path, _ = route(self.path)
            if path == "/health":
                health = {"ok": True, "mock": proxy.mock, "role": _role(proxy)}
                if getattr(proxy.collector, "remote", False):  # counts only: no hostnames on an open endpoint
                    health["collector"] = {k: v for k, v in proxy.collector.stats().items()
                                           if k not in ("url", "last_error")}
                else:
                    health["ingest"] = dict(proxy.collector.stats)
                    if proxy.log_consumer is not None:
                        health["log"] = {k: v for k, v in proxy.log_consumer.snapshot().items() if k != "last_error"}
                return self._json(200, health)
            if path == "/metrics":
                if not self._authorized(bearer=True):
                    return
                ev = proxy.engine_events
                data = proxy.metrics.render(proxy.engine, ev.snapshot() if ev else None,
                                            gauges=_transport_gauges(proxy)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if not self._authorized():
                return
            if path == "/engine":
                ev = proxy.engine_events
                return self._json(200, {"engine": proxy.engine,
                                        "kv_events": ev.snapshot() if ev else None})
            if path.startswith(api.PREFIX + "/") or path == api.PREFIX or path == "/ledger" \
                    or path.startswith("/requests/"):
                if self._no_ledger_here():
                    return
            if path.startswith(api.PREFIX + "/") or path == api.PREFIX:
                try:
                    return self._json(*api.get(proxy, path, route(self.path)[1]))
                except Exception as exc:  # never drop the connection; never echo internals
                    print(f"prefixwise: internal error on {path}: {type(exc).__name__}", file=sys.stderr)
                    return self._json(500, _error(500, "internal error in the Prefixwise API"))
            if path == "/ledger":
                return self._json(200, json.loads(proxy.ledger.to_json()))
            if path.startswith("/requests/"):
                row = proxy.ledger.get_request(path[len("/requests/"):])
                if row is None:
                    return self._json(404, _error(404, "unknown request id (or not recorded yet)"))
                return self._json(200, row)
            if proxy.collect_only:
                return self._json(404, _error(404, "this is a collector (prefixwise collect): it has no upstream; "
                                                  "point clients at an edge (prefixwise serve)"))
            return self._passthrough("GET")

        def _no_ledger_here(self) -> bool:
            """An edge with a remote collector has no ledger: say where it is (409), once per call."""
            if getattr(proxy.collector, "remote", False):
                self._json(409, _error(409, "this proxy is an edge: its records go to the collector at "
                                            f"{proxy.collector.url}; query the ledger there"))
                return True
            return False

        def do_HEAD(self):
            path, _ = route(self.path)
            if path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                return
            return self._passthrough("HEAD")

        def do_DELETE(self):
            return self._passthrough("DELETE")

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                raise ClientError("Content-Length is not a number")
            if length < 0:
                raise ClientError("Content-Length is negative")
            if length > MAX_BODY_BYTES:
                raise ClientError(f"body larger than {MAX_BODY_BYTES} bytes")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ClientError("body must be a JSON object")
            validate_body(body)
            return body

        def do_POST(self):
            path, _ = route(self.path)
            if path.startswith(api.PREFIX + "/"):
                if not self._authorized():
                    return
                if path == api.PREFIX + "/ingest" and getattr(proxy.collector, "remote", False):
                    return self._json(404, _error(404, "this is an edge, not a collector: records go to "
                                                      f"{proxy.collector.url}; POST there"))  # before reading the body
                if path != api.PREFIX + "/ingest" and path != api.PREFIX + "/explain" and self._no_ledger_here():
                    return
                try:
                    return self._json(*api.post(proxy, path, self._read_json()))
                except ClientError as exc:
                    code = 413 if "larger than" in str(exc) else 400
                    return self._json(code, _error(code, str(exc)))
                except ValueError as exc:
                    return self._json(400, _error(400, f"invalid JSON body: {exc}"))
                except OSError:
                    self.close_connection = True
                    return
                except Exception as exc:
                    print(f"prefixwise: internal error on {path}: {type(exc).__name__}", file=sys.stderr)
                    return self._json(500, _error(500, "internal error in the Prefixwise API"))
            if proxy.collect_only and path != "/tasks":
                return self._json(404, _error(404, "this is a collector (prefixwise collect): it has no upstream; "
                                                  "point clients at an edge (prefixwise serve)"))
            if path not in CLASSIFIED_PATHS and path != "/tasks":
                # Not analysed (count_tokens, files, batches, ...): forwarded unchanged.
                return self._passthrough("POST")
            if not self._authorized():
                return
            if path == "/tasks" and self._no_ledger_here():
                return
            try:
                body = self._read_json()
            except ClientError as exc:
                code = 413 if "larger than" in str(exc) else 400
                return self._json(code, _error(code, str(exc)))
            except ValueError as exc:
                return self._json(400, _error(400, f"invalid JSON body: {exc}"))
            except OSError:  # client stalled past CLIENT_TIMEOUT or hung up mid-body
                self.close_connection = True
                return
            try:
                return self._dispatch(body)
            except ClientError as exc:
                return self._json(400, _error(400, str(exc)))
            except Exception as exc:  # never drop the connection; never echo internals
                print(f"prefixwise: internal error on {self.path}: {type(exc).__name__}", file=sys.stderr)
                return self._json(500, _error(500, "internal error in the Prefixwise proxy"))

        def _dispatch(self, body: dict):
            path, _ = route(self.path)
            if path == "/tasks":
                return self._task(body)
            provider = "anthropic" if path == "/v1/messages" else "openai"
            api = "responses" if path == "/v1/responses" else "chat"
            if body.get("stream"):
                return self._stream(provider, body, api)
            result, _ = proxy.handle(provider, body, self._client_headers(), api=api)
            # The client gets the upstream response unchanged; Prefixwise's
            # verdict rides in headers so drop-in clients are unaffected.
            extra = {**result.get("upstream_headers", {}), "X-Prefixwise-Request-Id": result["request_id"]}
            if result.get("recorded") is False:
                extra["X-Prefixwise-Error"] = "not recorded (see proxy log)"
            if "queued" in result:  # an edge: the verdict is made at the collector
                extra["X-Prefixwise-Queued"] = "1" if result["queued"] else "0"
            if result.get("cause"):
                extra.update({"X-Prefixwise-Cause": result["cause"],
                              "X-Prefixwise-Legit": str(result["legit"]).lower(),
                              "X-Prefixwise-Avoidable-Tokens": str(result["avoidable_tokens"])})
                if result.get("fixes_applied"):
                    extra["X-Prefixwise-Fixes"] = ",".join(result["fixes_applied"])
            return self._json(result["status"], result["response"], extra)

        def _task(self, body: dict):
            task_id, success = body.get("task_id"), body.get("success")
            if not isinstance(task_id, str) or not task_id or not isinstance(success, bool):
                return self._json(400, _error(400, 'expected {"task_id": "<id>", "success": true|false}'))
            proxy.ledger.set_task_outcome(task_id, success)
            return self._json(200, {"ok": True, "task_id": task_id, "success": success})

        def _stream(self, provider: str, body: dict, api: str = "chat"):
            ctx = proxy.prepare(provider, body, self._client_headers(), api=api)
            try:
                return self._stream_ctx(ctx, provider, api)
            finally:  # whatever happened (cut-off upstream, client gone), it is not in flight now
                proxy.response_started(ctx)

        def _stream_ctx(self, ctx: dict, provider: str, api: str):
            sent = time.perf_counter()
            status, lines, acc, err = proxy.open_stream(ctx)
            relayed = ctx.get("upstream_headers") or {}
            if lines is None:
                if proxy.engine_events is not None:
                    proxy.engine_events.discard(ctx.get("engine_token"))
                ctx["upstream_ms"] = (time.perf_counter() - sent) * 1000
                proxy.record_failure(ctx, status, err)
                return self._json(status, err, {**relayed, "X-Prefixwise-Request-Id": ctx["request_id"]})
            # Headers go out before usage is known, so the verdict cannot
            # ride in them: fetch it afterwards from GET /requests/<id>.
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Prefixwise-Request-Id", ctx["request_id"])
            for k, v in relayed.items():
                self.send_header(k, v)
            self.end_headers()
            failure = None
            recording = None  # set once the client has the whole answer, before it is recorded
            try:
                try:
                    for line in lines:
                        keep = acc.feed(line)
                        if acc.done and recording is None:
                            # The terminal event: the client can act on it at once (the OpenAI SDK
                            # stops at [DONE]), so this conversation's next request must wait for
                            # this record from here on, before the line goes out.
                            recording = proxy.begin_recording(ctx)
                        if keep:
                            self.wfile.write(line)
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    failure = "client_disconnected"  # client went away mid-stream
                    status = 499  # client closed request (not an upstream error)
                except (TimeoutError, OSError):
                    failure = "upstream_dropped"  # upstream stalled or dropped
                ctx["upstream_ms"] = (time.perf_counter() - sent) * 1000
                # A cut-off stream has partial usage at best, and a guess is worse than
                # none: it is recorded as a failure. A stream that completed is a
                # response like any other, usage or not (no usage: verdict unknown).
                if failure is None:
                    if recording is None:
                        recording = proxy.begin_recording(ctx)
                    # HTTP/1.0: the stream ends when the connection does. End it for the client
                    # now, so classifying, the ledger write and any KV-event wait never delay it.
                    try:
                        self.wfile.flush()
                        self.connection.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    proxy.finalize_safely(ctx, status, acc.payload)
                else:
                    if proxy.engine_events is not None:
                        proxy.engine_events.discard(ctx.get("engine_token"))
                    proxy.record_failure(ctx, status, error=failure)
            finally:
                proxy.end_recording(ctx, recording)
            self.close_connection = True
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080, *, ledger_path: str | None = None,
          upstream_url: str | None = None, mock: bool | None = None, quiet: bool = False,
          fixes: str | list[str] | None = None, auth_token: str | None = None,
          allow_unauthenticated: bool = False, prices: str | None = None,
          engine: str | None = None, kv_events: str | None = None, full_attention: bool | None = None,
          otel: bool | str | None = None, collector_url: str | None = None, collect_only: bool = False,
          ready=None, log_url: str | None = None, log_partitions: str | None = None,
          sinks: list[str] | None = None):
    """Run the proxy until interrupted. Settings fall back to PREFIXWISE_* env vars.
    collect_only: `prefixwise collect`, a collector with no upstream; with log_url it also reads the
    record log's partitions (log_partitions: '0-7'; default all). ready: called with the bound port
    once the server listens (tests). sinks: URLs (prefixwise.sinks) that receive a copy of every row."""
    hmac_key = os.environ.get("PREFIXWISE_HMAC_KEY")
    ledger_path = ledger_path or os.environ.get("PREFIXWISE_LEDGER_PATH", ":memory:")
    proxy = PrefixwiseProxy(ledger_path=ledger_path, upstream_url=upstream_url, mock=mock or collect_only,
                            hmac_key=hmac_key.encode() if hmac_key else None, fixes=fixes,
                            prices=prices, engine=engine, kv_events=kv_events, full_attention=full_attention,
                            otel=otel, collector_url="" if collect_only else collector_url)
    proxy.collect_only = collect_only
    log_url = log_url if log_url is not None else (os.environ.get("PREFIXWISE_LOG") if collect_only else None)
    if log_url:
        from .log import LogConsumer, parse_partitions
        proxy.log_consumer = LogConsumer(proxy.collector, log_url, parse_partitions(log_partitions))
    for sink_url in sinks or (os.environ.get("PREFIXWISE_SINKS") or "").split(","):
        if sink_url.strip():
            if getattr(proxy.collector, "remote", False):
                print(f"prefixwise: sink {sink_url.strip()} ignored: this is an edge with no ledger; give the sink "
                      "to the collector", file=sys.stderr)
                continue
            from .sinks import sink_from_url
            proxy.ledger.add_sink(sink_from_url(sink_url.strip()))
    otel_where = None
    if proxy.spans is not None:  # Prefixwise is the application here: it sets up the exporter
        from .otel import install_provider
        otel_where = install_provider()
    proxy.auth_token = auth_token or os.environ.get("PREFIXWISE_AUTH_TOKEN") or None
    exposed = not _is_loopback(host)
    if exposed and collect_only and not proxy.auth_token and not allow_unauthenticated:
        # Anyone who can reach the port could write rows into the fleet's ledger.
        raise SystemExit(
            f"prefixwise: refusing to collect on {host} with no auth token: anyone who can reach the port could "
            "write records. Set PREFIXWISE_AUTH_TOKEN (edges send it as PREFIXWISE_COLLECTOR_TOKEN), bind to "
            "127.0.0.1, or pass --allow-unauthenticated.")
    if exposed and not proxy.auth_token and proxy.api_key and not proxy.mock and not allow_unauthenticated:
        # Anyone who can reach the port would spend this key.
        raise SystemExit(
            f"prefixwise: refusing to listen on {host} with PREFIXWISE_API_KEY set and no auth token: "
            "anyone who can reach the port could spend your key. Set PREFIXWISE_AUTH_TOKEN (clients send it "
            "as X-Prefixwise-Token), bind to 127.0.0.1, or pass --allow-unauthenticated.")
    server = PrefixwiseHTTPServer((host, port), make_handler(proxy))
    if ready is not None:
        ready(server.server_address[1])
    if not quiet and collect_only:
        print(f"Prefixwise collector on http://{host}:{server.server_address[1]}")
        print(f"  ledger:   {ledger_path}")
        for sink in proxy.ledger.sinks:
            print(f"  sink:     {sink.name}")
        print(f"  prices:   {len(proxy.prices.models)} models, table of {proxy.prices.as_of}")
        print(f"  auth:     {'X-Prefixwise-Token required' if proxy.auth_token else 'none'}"
              + ("  (WARNING: listening beyond localhost without a token)" if exposed and not proxy.auth_token else ""))
        if proxy.log_consumer is not None:
            lc = proxy.log_consumer
            print(f"  log:      {lc.name}, partitions {lc.partitions[0]}-{lc.partitions[-1]} of "
                  f"{len(lc.partitions)} (group {lc.group}, consumer {lc.consumer})")
            print(f"  edges:    PREFIXWISE_COLLECTOR={lc.name} prefixwise serve ... (or any edge: wrap, LiteLLM)")
        else:
            print("  edges:    PREFIXWISE_COLLECTOR=http://<this host>:<port> prefixwise serve ... "
                  "(PREFIXWISE_COLLECTOR_TOKEN for the token)")
        print("  POST /api/v1/ingest   GET /api/v1/*  GET /metrics  GET /health", flush=True)
    elif not quiet:
        target = "MOCK upstream (synthetic usage, no API calls)" if proxy.mock else proxy.upstream_url
        print(f"Prefixwise proxy on http://{host}:{server.server_address[1]}")
        print(f"  upstream: {target}")
        if getattr(proxy.collector, "remote", False):
            print(f"  collector: {proxy.collector.url} (records are sent there; no ledger here)")
        else:
            print(f"  ledger:   {ledger_path}")
        print(f"  fixes:    {', '.join(proxy.fixes) or 'none (pass-through)'}")
        print(f"  prices:   {len(proxy.prices.models)} models, table of {proxy.prices.as_of}"
              f" ({' + '.join(proxy.prices.sources)})")
        if proxy.engine:
            if proxy.engine_events is None:
                ev = "none (evictions are inferred)"
            elif proxy.engine_caches > 1:
                ev = ", ".join(f"{label} {s.endpoint}" for label, s in
                               zip(proxy.engine_events.labels, proxy.engine_events.sources))
            else:
                ev = proxy.engine_events.endpoint
            caches = (f"{proxy.engine_caches} KV caches" if proxy.engine_caches > 1 else "one self-hosted engine")
            print(f"  engine:   {proxy.engine} ({caches}); KV events: {ev}")
            if proxy.engine == "sglang" and proxy.engine_events is not None and proxy.engine_events.hybrid:
                print("  note:     evictions can be confirmed but not ruled out; pass --full-attention "
                      "if the model has no sliding-window or Mamba layers")
            for note in proxy.engine_notes:
                print(f"  note:     {note}")
            flag = "--enable-cache-report" if proxy.engine == "sglang" else "--enable-prompt-tokens-details"
            print(f"  note:     start {proxy.engine} with {flag}, or every turn looks like a miss")
        print(f"  auth:     {'X-Prefixwise-Token required' if proxy.auth_token else 'none'}"
              + ("  (WARNING: listening beyond localhost without a token)" if exposed and not proxy.auth_token else ""))
        if proxy.spans is not None:
            print(f"  otel:     a span per request ({proxy.spans.mode}) -> {otel_where}")
        if not os.environ.get("PREFIXWISE_HMAC_KEY"):
            print("  note:     set PREFIXWISE_HMAC_KEY so hashes of short prompt blocks cannot be guessed")
        print("  POST /v1/chat/completions  POST /v1/responses  POST /v1/messages  POST /tasks")
        print("  GET  /ledger  GET /requests/<id>  GET /metrics  GET /health", flush=True)
    # SIGTERM (docker stop, systemd, Kubernetes): stop accepting, let in-flight
    # requests and streams finish, then exit. As PID 1 in a container an
    # unhandled SIGTERM is ignored and the runtime SIGKILLs mid-stream.
    import signal

    def _stop(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, _stop)
    except ValueError:  # not the main thread (embedded use)
        pass
    if proxy.log_consumer is not None:
        proxy.log_consumer.start()  # replays its partitions, then reads them
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.socket.close()  # refuse new connections while draining
        left = server.drain(float(os.environ.get("PREFIXWISE_DRAIN_SECONDS", "8")))
        if left and not quiet:
            print(f"prefixwise: stopped with {left} request(s) still in flight", file=sys.stderr)
        server.server_close()
        proxy.close()  # records still on their way to a remote collector


if __name__ == "__main__":
    serve(port=int(os.environ.get("PORT", "8080")))
