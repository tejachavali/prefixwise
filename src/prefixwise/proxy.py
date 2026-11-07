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
forwarding). Client auth headers are passed through; PREFIXWISE_API_KEY,
if set, overrides them. stream=true is relayed as SSE unchanged; its verdict is recorded when
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

import hmac
import json
import os
import sys
import threading
from collections import OrderedDict
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .classifier import Classification, MissCause, RequestSnapshot, classify
from .diagnostics import AgreementScorer, anthropic_diagnostics, openai_comparison_options
from .fingerprint import Fingerprinter
from .fixes import apply_fixes, parse_fixes
from .pricing import PriceTable
from .ledger import Ledger, LedgerEntry, avoidable_for_turn


def extract_blocks(body: dict, provider: str) -> list[tuple[str, object]]:
    """Split a rendered request into blocks in prefix order.

    OpenAI order is tools before developer/system (OpenAI Prompt
    Caching 201, verified Nov 2026); Anthropic is tools, system,
    messages. This is the v2 order, not the v1 assumption.
    """
    blocks = [(role, _strip_cache_markers(value)) for role, value in _extract_blocks_raw(body, provider)]
    if provider == "anthropic":
        blocks = [(role, _anthropic_text_blocks(role, value)) for role, value in blocks]
    return blocks


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
        system = body.get("system") or next(
            (m.get("content") for m in body.get("messages", []) if m.get("role") == "system"), None)
        if system is not None:
            blocks.append(("system", system))
    for m in body.get("messages", []) or []:
        if m.get("role") == "system" and provider != "anthropic":
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


# Body fields Prefixwise reads but upstream APIs do not accept. Stripped
# before forwarding (OpenAI rejects unknown fields with a 400). vLLM
# keeps session_id: it stamps it onto KV events for the engine join.
PREFIXWISE_FIELDS = ("session_id", "agent_id", "task_id", "prefixwise_compaction_event",
                     "prefixwise_upstream_session_id")

# Client headers passed through to the upstream when present.
PASSTHROUGH_HEADERS = ("authorization", "x-api-key", "anthropic-version", "anthropic-beta",
                       "openai-organization", "openai-project")
ANTHROPIC_VERSION = "2023-06-01"
UPSTREAM_TIMEOUT = 600  # seconds; long generations stream for minutes
# A client socket that sends nothing (or stops reading) for this long is
# dropped, so idle or half-sent connections cannot pin threads forever.
CLIENT_TIMEOUT = float(os.environ.get("PREFIXWISE_CLIENT_TIMEOUT", "120"))

# Upstream response headers relayed to the client: SDK retry hints, the
# provider's request id (support tickets), and rate-limit state.
RELAY_HEADERS = ("retry-after", "retry-after-ms", "x-should-retry", "x-request-id", "request-id")
RELAY_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")


def relay_headers(headers) -> dict[str, str]:
    if not headers:
        return {}
    return {k: v for k, v in headers.items()
            if k.lower() in RELAY_HEADERS or k.lower().startswith(RELAY_PREFIXES)}
# Sessions remembered for classification; least recently used are dropped,
# so a long-running proxy does not grow without bound. A dropped session's
# next turn classifies as FIRST_WRITE (honest: no prior turn is known).
MAX_SESSIONS = int(os.environ.get("PREFIXWISE_MAX_SESSIONS", "10000"))


class _SessionMap(OrderedDict):
    """LRU dict: reads and writes refresh a key; oldest keys are evicted."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.limit:
            self.popitem(last=False)


def normalize_usage(provider: str, usage: dict) -> tuple[int, int]:
    """Return (prompt_tokens, cached_tokens) from a provider usage object.

    Anthropic reports input_tokens EXCLUDING cache reads and writes, so
    the full prompt is input + cache_read + cache_creation.
    OpenAI/vLLM report prompt_tokens with prompt_tokens_details.cached_tokens.
    """
    if provider == "anthropic":
        read = int(usage.get("cache_read_input_tokens", 0) or 0)
        write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        return int(usage.get("input_tokens", 0) or 0) + read + write, read
    if "prompt_tokens" not in usage and "input_tokens" in usage:  # Responses API
        details = usage.get("input_tokens_details") or {}
        return int(usage.get("input_tokens", 0) or 0), int(details.get("cached_tokens", 0) or 0)
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or usage.get("cached_tokens", 0) or 0)
    return prompt, cached


def completion_tokens_of(provider: str, usage: dict) -> int:
    if provider == "anthropic" or ("completion_tokens" not in usage and "output_tokens" in usage):
        return int(usage.get("output_tokens", 0) or 0)
    return int(usage.get("completion_tokens", 0) or 0)


def has_cache_breakpoint(body: dict) -> bool:
    """Anthropic: cache_control anywhere (top level, tools, system, messages)."""
    if body.get("cache_control"):
        return True
    return any("cache_control" in json.dumps(body.get(k) or [])
               for k in ("tools", "system", "messages"))


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

    def feed(self, line: bytes) -> bool:
        """Consume one raw line. Returns False if it must not be forwarded."""
        if not line.startswith(b"data:"):
            return True
        data = line[5:].strip()
        if not data or data == b"[DONE]":
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


class PrefixwiseProxy:
    def __init__(self, ledger_path: str = ":memory:", hmac_key: bytes | None = None,
                 upstream_url: str | None = None, api_key: str | None = None,
                 mock: bool | None = None, fixes: str | list[str] | tuple[str, ...] | None = None,
                 prices: str | None = None):
        self.ledger = Ledger(ledger_path)
        # Opt-in request rewrites (see fixes.py). Off unless named here or
        # in PREFIXWISE_FIXES; turning them off restores pass-through.
        self.fixes = parse_fixes(fixes if fixes is not None else os.environ.get("PREFIXWISE_FIXES"))
        # Set by serve() from --auth-token / PREFIXWISE_AUTH_TOKEN; None = open.
        self.auth_token: str | None = None
        # Dated price table: built-in verified rates + PREFIXWISE_PRICES / prices file.
        self.prices = PriceTable.load(prices)
        self.fingerprinter = Fingerprinter(hmac_key=hmac_key)
        self.scorer = AgreementScorer()
        self.upstream_url = (upstream_url or os.environ.get("PREFIXWISE_UPSTREAM_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("PREFIXWISE_API_KEY")
        self.mock = mock if mock is not None else (
            not self.upstream_url or os.environ.get("PREFIXWISE_MOCK_UPSTREAM") == "1")
        # Per-session state, shared across the threaded server's handlers.
        # Every read and write holds self._lock: the classifier must compare
        # a turn with the turn recorded just before it, never a torn state.
        self._lock = threading.Lock()
        self._upstream = threading.local()  # last upstream response headers, per thread
        # vLLM stamps a body session_id onto its KV events (the engine join).
        # SGLang reads session_id as its own stateful-session handle and
        # rejects ids it did not open, so engines like it set this False.
        self.forward_session_id = True
        self.last_snapshot: dict[str, RequestSnapshot] = _SessionMap(MAX_SESSIONS)
        self.last_response_id: dict[str, str | None] = _SessionMap(MAX_SESSIONS)
        self.last_seen: dict[str, float] = _SessionMap(MAX_SESSIONS)
        # Responses API: snapshot per response id, so a turn sent with
        # previous_response_id (input = new items only) can be stitched
        # onto the exact turn it continues, including branches.
        self.snapshot_by_response: dict[str, RequestSnapshot] = _SessionMap(MAX_SESSIONS)
        # Per-model lanes within a session, keyed (session, model). Agents
        # interleave small-model side calls (titles, summaries) with the main
        # thread; each turn is compared with the last turn to the SAME model,
        # so a side call never masks a cause on the main thread (dogfood,
        # 2026-11-06). The session-level maps above still say "has this
        # session been seen", which decides FIRST_WRITE vs MODEL_SWITCH.
        self.lane_snapshot: dict[tuple, RequestSnapshot] = _SessionMap(MAX_SESSIONS)
        self.lane_seen: dict[tuple, float] = _SessionMap(MAX_SESSIONS)
        self.lane_response_id: dict[tuple, str] = _SessionMap(MAX_SESSIONS)

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
        session_id = str(headers.get("x-prefixwise-session-id") or body.get("session_id")
                         or (body.get("metadata") or {}).get("user_id") or "default")
        task_id = headers.get("x-prefixwise-task-id") or body.get("task_id")
        forward_body = {k: v for k, v in body.items()
                        if k not in PREFIXWISE_FIELDS
                        or (provider == "vllm" and k == "session_id" and self.forward_session_id)}
        # E1-open retry turns: a harness-only upstream session id.
        # The proxy's own session (snapshot chain, gap_seconds) stays
        # session_id; only the identity sent upstream changes, so the
        # engine sees the retry as a new session. The retry turn also
        # carries a fresh cache_salt, which is what actually makes the
        # vLLM prefix cold (session_id alone is event metadata, not a
        # cache partition; cache_salt is mixed into the first block's
        # hash, docs.vllm.ai Automatic Prefix Caching).
        upstream_session_id = (body.get("prefixwise_upstream_session_id")
                               or headers.get("x-prefixwise-upstream-session-id"))
        upstream_session_id = str(upstream_session_id) if upstream_session_id else None
        if provider == "vllm" and self.forward_session_id:
            forward_body["session_id"] = upstream_session_id or session_id
        # Native diagnostics chaining. Anthropic: diagnostics object (GA,
        # no beta header). OpenAI: prompt_cache_options.comparison_response_id,
        # Responses API only (it does not exist on /v1/chat/completions). A
        # client that sets its own comparison id keeps it.
        forward_body, fixes_applied = apply_fixes(forward_body, self.fixes)
        with self._lock:
            # Chain the provider's diagnostics to this model's last turn when
            # there is one, so its verdict compares like with like.
            lane = (session_id, str(body.get("model", "unknown")))
            previous_id = self.lane_response_id.get(lane) or self.last_response_id.get(session_id)
        if provider == "anthropic":
            forward_body.update(anthropic_diagnostics(previous_id))
        elif api == "responses" and previous_id:
            # Merge into the client's own prompt_cache_options; a comparison
            # id the client chose wins.
            opts = dict(body.get("prompt_cache_options") or {})
            opts.setdefault("comparison_response_id", previous_id)
            forward_body["prompt_cache_options"] = opts
        return {
            "provider": provider, "api": api, "body": body, "headers": headers,
            "forward_body": forward_body, "session_id": session_id,
            "upstream_session_id": upstream_session_id,
            "agent_id": str(headers.get("x-prefixwise-agent-id") or body.get("agent_id") or "proxy"),
            "task_id": str(task_id) if task_id else None,
            "request_id": str(uuid.uuid4()), "model": str(body.get("model", "unknown")),
            # E1-open / hooks: a PreCompact/compaction event fired before
            # this turn, so the classifier can tell COMPACTION_REWRITE from
            # HISTORY_EDIT. Never forwarded.
            "compaction_event": bool(body.get("prefixwise_compaction_event", False))
            or headers.get("x-prefixwise-compaction-event") in ("1", "true"),
            "stream": bool(body.get("stream")),
            "fixes_applied": fixes_applied,
            "fork_of": headers.get("x-prefixwise-fork-of"),
            "ttl_seconds": cache_ttl_seconds(provider, body, headers),
        }

    def finalize(self, ctx: dict, status: int, payload: dict) -> tuple[dict, LedgerEntry]:
        """Classify and record one completed request from its (rebuilt) payload."""
        provider, body, session_id = ctx["provider"], ctx["body"], ctx["session_id"]
        usage = payload.get("usage") or {}
        prompt_tokens, cached = normalize_usage(provider, usage)
        native_reason = native_reason_from(payload)
        response_id = payload.get("id")

        # Fingerprint what the provider actually received: with a fix on,
        # that is the rewritten request, not the client's original.
        fps = self.fingerprinter.fingerprint_blocks(extract_blocks(ctx["forward_body"], provider))
        with self._lock:
            return self._finalize_locked(ctx, status, payload, usage, prompt_tokens, cached,
                                         native_reason, response_id, fps)

    def finalize_safely(self, ctx: dict, status: int, payload: dict) -> tuple[dict, LedgerEntry | None]:
        """finalize(), failing open: the client already paid for this response,
        so a ledger or classifier failure (disk full, locked database, a bug)
        must never cost them it. The response passes through unrecorded."""
        try:
            return self.finalize(ctx, status, payload)
        except Exception as exc:
            print(f"prefixwise: request {ctx['request_id']} passed through unrecorded: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return {"request_id": ctx["request_id"], "status": status, "cause": None,
                    "recorded": False, "response": payload}, None

    def _stitch(self, ctx: dict, fps: list) -> tuple[list | None, RequestSnapshot | None]:
        """Responses API with previous_response_id: the input holds only new
        items. The cached prefix is the prior turn's blocks plus these, so
        the comparable request is prior message blocks + new message blocks,
        under this turn's own tools and instructions (both are resent).
        Returns (blocks, prior snapshot); (None, None) if the prior turn was
        never seen by this proxy. Caller holds self._lock."""
        prev_id = ctx["body"].get("previous_response_id")
        prior = self.snapshot_by_response.get(prev_id)
        if prior is None:
            return None, None
        convo = ("messages", "tool_results")
        head = [b for b in fps if b.role not in convo]
        new_msgs = [b for b in fps if b.role in convo]
        old_msgs = [b for b in prior.blocks if b.role in convo]
        return head + old_msgs + new_msgs, prior

    def _finalize_locked(self, ctx, status, payload, usage, prompt_tokens, cached,
                         native_reason, response_id, fps) -> tuple[dict, LedgerEntry]:
        provider, body, session_id = ctx["provider"], ctx["body"], ctx["session_id"]
        now = time.time()
        lane = (session_id, ctx["model"])
        if lane in self.lane_snapshot:
            # Compare with this model's previous turn, and time the gap from it.
            prev = self.lane_snapshot.get(lane)
            gap_seconds = now - self.lane_seen[lane] if lane in self.lane_seen else None
        else:
            # First turn on this model: FIRST_WRITE for a new session, else
            # MODEL_SWITCH (the classifier compares models first).
            prev = self.last_snapshot.get(session_id)
            gap_seconds = (now - self.last_seen[session_id]) if session_id in self.last_seen else None
        self.last_seen[session_id] = now
        self.lane_seen[lane] = now
        unknown_parent = False
        if ctx.get("api") == "responses" and body.get("previous_response_id"):
            stitched, prior = self._stitch(ctx, fps)
            if stitched is None:
                unknown_parent = True  # continues a turn this proxy never saw
            else:
                fps, prev = stitched, prior
        curr = RequestSnapshot(model=ctx["model"], provider=provider, blocks=fps,
                               prompt_tokens=prompt_tokens, cached_tokens=cached,
                               gap_seconds=gap_seconds, compaction_event=ctx["compaction_event"],
                               params={k: body.get(k) for k in
                                       ("tool_choice", "thinking", "reasoning", "output_format",
                                        "text_format", "text", "verbosity") if body.get(k) is not None},
                               has_breakpoint=has_cache_breakpoint(body),
                               ttl_seconds=ctx.get("ttl_seconds"),
                               fork_parent_session_id=ctx.get("fork_of"),
                               session_id=session_id)
        if unknown_parent:
            result = Classification(MissCause.UNKNOWN, None,
                                    "previous_response_id points to a response this proxy never saw;"
                                    " the cached prefix cannot be reconstructed",
                                    native_reason=native_reason)
        else:
            result = classify(prev, curr, native_reason=native_reason)
        self.scorer.record(result.cause, native_reason, provider)
        self.last_snapshot[session_id] = curr
        self.lane_snapshot[lane] = curr
        if response_id:
            self.last_response_id[session_id] = response_id
            self.lane_response_id[lane] = response_id
            if ctx.get("api") == "responses":
                self.snapshot_by_response[response_id] = curr

        entry = LedgerEntry(
            session_id=session_id, agent_id=ctx["agent_id"],
            provider=provider, model=ctx["model"], cause=result.cause.value,
            prompt_tokens=prompt_tokens, cached_tokens=cached,
            avoidable_tokens=avoidable_for_turn(result.legit, prompt_tokens, cached,
                                                prev.prompt_tokens if prev else None),
            avoidable_cost=None, legit=result.legit, native_reason=native_reason,
            cost=self.prices.request_cost(str(payload.get("model") or ctx["model"]), provider, usage),
            task_id=ctx["task_id"], request_id=ctx["request_id"],
            completion_tokens=completion_tokens_of(provider, usage),
            fixes=",".join(ctx.get("fixes_applied") or []) or None,
        )
        priced_model = str(payload.get("model") or ctx["model"])
        entry.avoidable_cost = self.prices.avoidable_cost(priced_model, entry.avoidable_tokens, prompt_tokens)
        if entry.cost is not None:
            entry.price_as_of = self.prices.as_of
        self.ledger.record(entry)
        return {"request_id": ctx["request_id"], "status": status, "cause": result.cause.value,
                "legit": result.legit, "native_reason": native_reason, "usage": usage,
                "prompt_tokens": prompt_tokens, "cached_tokens": cached,
                "avoidable_tokens": entry.avoidable_tokens, "fixes_applied": ctx.get("fixes_applied") or [],
                "cost": entry.cost, "avoidable_cost": entry.avoidable_cost,
                "response": payload}, entry

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
            return {"request_id": ctx["request_id"], "status": 400, "cause": None,
                    "response": _error(400, "use handle_stream for stream=true")}, None
        self._upstream.headers = {}
        status, payload = self._forward(provider, ctx["forward_body"], ctx["request_id"],
                                        ctx["upstream_session_id"] or ctx["session_id"],
                                        ctx["headers"], api=api)
        upstream_headers = getattr(self._upstream, "headers", {}) or {}
        if status >= 400:
            return {"request_id": ctx["request_id"], "status": status, "cause": None,
                    "response": payload, "upstream_headers": upstream_headers}, None
        result, entry = self.finalize_safely(ctx, status, payload)
        result["upstream_headers"] = upstream_headers
        return result, entry

    def _mock_payload(self, provider: str, model: str, request_id: str, session_id: str,
                      api: str = "chat") -> dict:
        # Synthetic usage: later turns in a session reuse the prefix.
        # 2,000 prompt tokens so the OpenAI 1,024 minimum does not fire.
        with self._lock:
            cached = 1000 if session_id in self.last_snapshot else 0
        rid = f"resp-{request_id[:8]}"
        if provider == "anthropic":
            return {"id": rid, "type": "message", "role": "assistant", "model": model,
                    "content": [{"type": "text", "text": "[prefixwise mock]"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2000 - cached, "cache_read_input_tokens": cached,
                              "cache_creation_input_tokens": 0, "output_tokens": 1}}
        if api == "responses":
            return {"id": rid, "object": "response", "status": "completed", "model": model,
                    "output": [{"type": "message", "role": "assistant", "status": "completed",
                                "content": [{"type": "output_text", "text": "[prefixwise mock]",
                                             "annotations": []}]}],
                    "usage": {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": cached},
                              "output_tokens": 1, "total_tokens": 2001}}
        return {"id": rid, "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "[prefixwise mock]"}}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 1, "total_tokens": 2001,
                          "prompt_tokens_details": {"cached_tokens": cached}}}

    def _upstream_request(self, provider: str, body: dict, request_id: str, session_id: str,
                          client_headers: dict[str, str], api: str = "chat") -> urllib.request.Request:
        headers = {k: v for k, v in client_headers.items() if k in PASSTHROUGH_HEADERS}
        headers.update({"content-type": "application/json",
                        "x-request-id": request_id, "x-session-id": session_id})
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
        return urllib.request.Request(self.upstream_url + path, method="POST",
                                      data=json.dumps(body).encode(), headers=headers)

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
        headers = {k: v for k, v in client_headers.items() if k in PASSTHROUGH_HEADERS}
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
        req = urllib.request.Request(self.upstream_url + raw_path, method=method, data=body or None,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                self._upstream.headers = relay_headers(resp.headers)
                return resp.status, resp.headers.get("Content-Type", "application/json"), resp.read()
        except urllib.error.HTTPError as exc:
            self._upstream.headers = relay_headers(exc.headers)
            return exc.code, exc.headers.get("Content-Type", "application/json"), exc.read() or b""
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 502, "application/json", json.dumps(_error(502, f"upstream unreachable: {exc}")).encode()

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
                                           request_id, session_id, api)
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
        if provider != "anthropic" and api != "responses":  # Responses always report usage
            opts = dict(fwd.get("stream_options") or {})
            if not opts.get("include_usage"):
                opts["include_usage"] = True
                fwd["stream_options"] = opts
                injected = True
        acc = StreamAccumulator(provider, suppress_usage_chunk=injected, api=api)
        if self.mock:
            return 200, iter(self._mock_stream(ctx)), acc, None
        req = self._upstream_request(provider, fwd, ctx["request_id"],
                                     ctx["upstream_session_id"] or ctx["session_id"],
                                     ctx["headers"], api)
        try:
            resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
            ctx["upstream_headers"] = relay_headers(resp.headers)
        except urllib.error.HTTPError as exc:
            ctx["upstream_headers"] = relay_headers(exc.headers)
            status, payload = self._http_error(exc)
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
                                     ctx["session_id"], ctx.get("api", "chat"))
        def ev(obj, name=None):  # one list item per line, as readline() yields them
            head = [f"event: {name}\n".encode()] if name else []
            return head + [b"data: " + json.dumps(obj).encode() + b"\n", b"\n"]
        if ctx["provider"] == "anthropic":
            u = payload["usage"]
            start = {"type": "message_start", "message": {
                "id": payload["id"], "type": "message", "role": "assistant", "model": ctx["model"],
                "content": [], "usage": {**u, "output_tokens": 0}}}
            out = ev(start, "message_start")
            out += ev({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}}, "content_block_start")
            out += ev({"type": "content_block_delta", "index": 0,
                       "delta": {"type": "text_delta", "text": "[prefixwise mock]"}}, "content_block_delta")
            out += ev({"type": "content_block_stop", "index": 0}, "content_block_stop")
            out += ev({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                       "usage": {"output_tokens": u["output_tokens"]}}, "message_delta")
            return out + ev({"type": "message_stop"}, "message_stop")
        if ctx.get("api") == "responses":
            # The full event sequence SDKs expect: item and part opened, text
            # delta, then every done event, then response.completed.
            created = {**payload, "status": "in_progress", "output": [], "usage": None}
            item = {**payload["output"][0], "id": "msg_mock"}
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
        out = ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant",
                                                             "content": "[prefixwise mock]"}}]})
        out += ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
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

        def _authorized(self) -> bool:
            """With an auth token configured, every request but /health must
            carry it in X-Prefixwise-Token (constant-time compare)."""
            token = getattr(proxy, "auth_token", None)
            if not token:
                return True
            given = self.headers.get("X-Prefixwise-Token") or ""
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
                return self._json(200, {"ok": True, "mock": proxy.mock})
            if not self._authorized():
                return
            if path == "/ledger":
                return self._json(200, json.loads(proxy.ledger.to_json()))
            if path.startswith("/requests/"):
                row = proxy.ledger.get_request(path[len("/requests/"):])
                if row is None:
                    return self._json(404, _error(404, "unknown request id (or not recorded yet)"))
                return self._json(200, row)
            return self._passthrough("GET")

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
            if path not in CLASSIFIED_PATHS and path != "/tasks":
                # Not analysed (count_tokens, files, batches, ...): forwarded unchanged.
                return self._passthrough("POST")
            if not self._authorized():
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
            status, lines, acc, err = proxy.open_stream(ctx)
            relayed = ctx.get("upstream_headers") or {}
            if lines is None:
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
            completed = True
            try:
                for line in lines:
                    if acc.feed(line):
                        self.wfile.write(line)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                completed = False  # client went away mid-stream
            except (TimeoutError, OSError):
                completed = False  # upstream stalled or dropped
            # Record only a stream that finished with usage: a cut-off stream
            # has partial or no usage, and a guess is worse than no entry.
            if completed and acc.saw_usage:
                proxy.finalize_safely(ctx, status, acc.payload)
            self.close_connection = True
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080, *, ledger_path: str | None = None,
          upstream_url: str | None = None, mock: bool | None = None, quiet: bool = False,
          fixes: str | list[str] | None = None, auth_token: str | None = None,
          allow_unauthenticated: bool = False, prices: str | None = None):
    """Run the proxy until interrupted. Settings fall back to PREFIXWISE_* env vars."""
    hmac_key = os.environ.get("PREFIXWISE_HMAC_KEY")
    ledger_path = ledger_path or os.environ.get("PREFIXWISE_LEDGER_PATH", ":memory:")
    proxy = PrefixwiseProxy(ledger_path=ledger_path, upstream_url=upstream_url, mock=mock,
                            hmac_key=hmac_key.encode() if hmac_key else None, fixes=fixes,
                            prices=prices)
    proxy.auth_token = auth_token or os.environ.get("PREFIXWISE_AUTH_TOKEN") or None
    exposed = not _is_loopback(host)
    if exposed and not proxy.auth_token and proxy.api_key and not proxy.mock and not allow_unauthenticated:
        # Anyone who can reach the port would spend this key.
        raise SystemExit(
            f"prefixwise: refusing to listen on {host} with PREFIXWISE_API_KEY set and no auth token: "
            "anyone who can reach the port could spend your key. Set PREFIXWISE_AUTH_TOKEN (clients send it "
            "as X-Prefixwise-Token), bind to 127.0.0.1, or pass --allow-unauthenticated.")
    server = PrefixwiseHTTPServer((host, port), make_handler(proxy))
    if not quiet:
        target = "MOCK upstream (synthetic usage, no API calls)" if proxy.mock else proxy.upstream_url
        print(f"Prefixwise proxy on http://{host}:{server.server_address[1]}")
        print(f"  upstream: {target}")
        print(f"  ledger:   {ledger_path}")
        print(f"  fixes:    {', '.join(proxy.fixes) or 'none (pass-through)'}")
        print(f"  prices:   {len(proxy.prices.models)} models, table of {proxy.prices.as_of}"
              f" ({' + '.join(proxy.prices.sources)})")
        print(f"  auth:     {'X-Prefixwise-Token required' if proxy.auth_token else 'none'}"
              + ("  (WARNING: listening beyond localhost without a token)" if exposed and not proxy.auth_token else ""))
        if not os.environ.get("PREFIXWISE_HMAC_KEY"):
            print("  note:     set PREFIXWISE_HMAC_KEY so hashes of short prompt blocks cannot be guessed")
        print("  POST /v1/chat/completions  POST /v1/responses  POST /v1/messages  POST /tasks")
        print("  GET  /ledger  GET /requests/<id>  GET /health", flush=True)
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


if __name__ == "__main__":
    serve(port=int(os.environ.get("PORT", "8080")))
