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

import json
import os
import threading
from collections import OrderedDict
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .classifier import RequestSnapshot, classify
from .diagnostics import AgreementScorer, anthropic_diagnostics
from .fingerprint import Fingerprinter
from .ledger import Ledger, LedgerEntry, avoidable_for_turn


def extract_blocks(body: dict, provider: str) -> list[tuple[str, object]]:
    """Split a rendered request into blocks in prefix order.

    OpenAI order is tools before developer/system (OpenAI Prompt
    Caching 201, verified Nov 2026); Anthropic is tools, system,
    messages. This is the v2 order, not the v1 assumption.
    """
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
        blocks.append(("messages", m))
    return blocks


# Body fields Prefixwise reads but upstream APIs do not accept. Stripped
# before forwarding (OpenAI rejects unknown fields with a 400). vLLM
# keeps session_id: it stamps it onto KV events for the engine join.
PREFIXWISE_FIELDS = ("session_id", "agent_id", "task_id", "prefixwise_compaction_event")

# Client headers passed through to the upstream when present.
PASSTHROUGH_HEADERS = ("authorization", "x-api-key", "anthropic-version", "anthropic-beta",
                       "openai-organization", "openai-project")
ANTHROPIC_VERSION = "2023-06-01"
UPSTREAM_TIMEOUT = 600  # seconds; long generations stream for minutes
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
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or usage.get("cached_tokens", 0) or 0)
    return prompt, cached


def completion_tokens_of(provider: str, usage: dict) -> int:
    key = "output_tokens" if provider == "anthropic" else "completion_tokens"
    return int(usage.get(key, 0) or 0)


def has_cache_breakpoint(body: dict) -> bool:
    """Anthropic: cache_control anywhere (top level, tools, system, messages)."""
    if body.get("cache_control"):
        return True
    return any("cache_control" in json.dumps(body.get(k) or [])
               for k in ("tools", "system", "messages"))


def native_reason_from(payload: dict) -> str | None:
    """Provider miss reason. OpenAI puts prompt_cache_diagnostics on the
    response object; also checked: a top-level diagnostics object and,
    for older shapes, inside usage."""
    usage = payload.get("usage") or {}
    for diag in (payload.get("prompt_cache_diagnostics"), payload.get("diagnostics"),
                 usage.get("prompt_cache_diagnostics")):
        if isinstance(diag, dict) and diag.get("reason"):
            return str(diag["reason"])
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

    def __init__(self, provider: str, suppress_usage_chunk: bool = False):
        self.provider = provider
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
                 mock: bool | None = None):
        self.ledger = Ledger(ledger_path)
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
        self.last_snapshot: dict[str, RequestSnapshot] = _SessionMap(MAX_SESSIONS)
        self.last_response_id: dict[str, str | None] = _SessionMap(MAX_SESSIONS)
        self.last_seen: dict[str, float] = _SessionMap(MAX_SESSIONS)

    # -- shared request path ---------------------------------------------------------

    def prepare(self, provider: str, body: dict, headers: dict[str, str] | None = None) -> dict:
        """Read identity, strip Prefixwise fields, chain diagnostics.

        Returns a context dict for forward/finalize. body itself is not
        mutated; ctx["forward_body"] is what the upstream receives.
        """
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        session_id = str(headers.get("x-prefixwise-session-id") or body.get("session_id")
                         or (body.get("metadata") or {}).get("user_id") or "default")
        task_id = headers.get("x-prefixwise-task-id") or body.get("task_id")
        forward_body = {k: v for k, v in body.items()
                        if k not in PREFIXWISE_FIELDS or (provider == "vllm" and k == "session_id")}
        if provider == "vllm":
            forward_body["session_id"] = session_id
        # Native diagnostics chaining. Anthropic: diagnostics object (GA,
        # no beta header). OpenAI comparison_response_id exists only on
        # the Responses API, which this proxy does not serve yet, so it is
        # not sent to /v1/chat/completions.
        if provider == "anthropic":
            with self._lock:
                previous_id = self.last_response_id.get(session_id)
            forward_body.update(anthropic_diagnostics(previous_id))
        return {
            "provider": provider, "body": body, "headers": headers,
            "forward_body": forward_body, "session_id": session_id,
            "agent_id": str(headers.get("x-prefixwise-agent-id") or body.get("agent_id") or "proxy"),
            "task_id": str(task_id) if task_id else None,
            "request_id": str(uuid.uuid4()), "model": str(body.get("model", "unknown")),
            # E1-open / hooks: a PreCompact/compaction event fired before
            # this turn, so the classifier can tell COMPACTION_REWRITE from
            # HISTORY_EDIT. Never forwarded.
            "compaction_event": bool(body.get("prefixwise_compaction_event", False))
            or headers.get("x-prefixwise-compaction-event") in ("1", "true"),
            "stream": bool(body.get("stream")),
        }

    def finalize(self, ctx: dict, status: int, payload: dict) -> tuple[dict, LedgerEntry]:
        """Classify and record one completed request from its (rebuilt) payload."""
        provider, body, session_id = ctx["provider"], ctx["body"], ctx["session_id"]
        usage = payload.get("usage") or {}
        prompt_tokens, cached = normalize_usage(provider, usage)
        native_reason = native_reason_from(payload)
        response_id = payload.get("id")

        fps = self.fingerprinter.fingerprint_blocks(extract_blocks(body, provider))
        with self._lock:
            return self._finalize_locked(ctx, status, payload, usage, prompt_tokens, cached,
                                         native_reason, response_id, fps)

    def _finalize_locked(self, ctx, status, payload, usage, prompt_tokens, cached,
                         native_reason, response_id, fps) -> tuple[dict, LedgerEntry]:
        provider, body, session_id = ctx["provider"], ctx["body"], ctx["session_id"]
        now = time.time()
        gap_seconds = (now - self.last_seen[session_id]) if session_id in self.last_seen else None
        self.last_seen[session_id] = now
        prev = self.last_snapshot.get(session_id)
        curr = RequestSnapshot(model=ctx["model"], provider=provider, blocks=fps,
                               prompt_tokens=prompt_tokens, cached_tokens=cached,
                               gap_seconds=gap_seconds, compaction_event=ctx["compaction_event"],
                               params={k: body.get(k) for k in
                                       ("tool_choice", "thinking", "reasoning", "output_format",
                                        "text_format", "verbosity") if body.get(k) is not None},
                               has_breakpoint=has_cache_breakpoint(body))
        result = classify(prev, curr, native_reason=native_reason)
        self.scorer.record(result.cause, native_reason, provider)
        self.last_snapshot[session_id] = curr
        if response_id:
            self.last_response_id[session_id] = response_id

        entry = LedgerEntry(
            session_id=session_id, agent_id=ctx["agent_id"],
            provider=provider, model=ctx["model"], cause=result.cause.value,
            prompt_tokens=prompt_tokens, cached_tokens=cached,
            avoidable_tokens=avoidable_for_turn(result.legit, prompt_tokens, cached,
                                                prev.prompt_tokens if prev else None),
            avoidable_cost=None, legit=result.legit, native_reason=native_reason,
            task_id=ctx["task_id"], request_id=ctx["request_id"],
            completion_tokens=completion_tokens_of(provider, usage),
        )
        self.ledger.record(entry)
        return {"request_id": ctx["request_id"], "status": status, "cause": result.cause.value,
                "legit": result.legit, "native_reason": native_reason, "usage": usage,
                "prompt_tokens": prompt_tokens, "cached_tokens": cached,
                "avoidable_tokens": entry.avoidable_tokens, "response": payload}, entry

    # -- non-streaming ---------------------------------------------------------------

    def handle(self, provider: str, body: dict,
               headers: dict[str, str] | None = None) -> tuple[dict, LedgerEntry | None]:
        """Classify, forward, and record one non-streaming request.

        Returns (result, entry). result["status"] and result["response"]
        are the upstream status and payload, returned to the client
        unchanged. On an upstream error nothing is classified or
        recorded (no usage to score) and entry is None.
        """
        ctx = self.prepare(provider, body, headers)
        if ctx["stream"]:
            return {"request_id": ctx["request_id"], "status": 400, "cause": None,
                    "response": _error(400, "use handle_stream for stream=true")}, None
        status, payload = self._forward(provider, ctx["forward_body"], ctx["request_id"],
                                        ctx["session_id"], ctx["headers"])
        if status >= 400:
            return {"request_id": ctx["request_id"], "status": status, "cause": None,
                    "response": payload}, None
        return self.finalize(ctx, status, payload)

    def _mock_payload(self, provider: str, model: str, request_id: str, session_id: str) -> dict:
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
        return {"id": rid, "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "[prefixwise mock]"}}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 1, "total_tokens": 2001,
                          "prompt_tokens_details": {"cached_tokens": cached}}}

    def _upstream_request(self, provider: str, body: dict, request_id: str, session_id: str,
                          client_headers: dict[str, str]) -> urllib.request.Request:
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
        path = "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
        return urllib.request.Request(self.upstream_url + path, method="POST",
                                      data=json.dumps(body).encode(), headers=headers)

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> tuple[int, dict]:
        raw = exc.read() or b""
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, _error(exc.code, raw.decode("utf-8", "replace")[:2000])

    def _forward(self, provider: str, body: dict, request_id: str, session_id: str,
                 client_headers: dict[str, str]) -> tuple[int, dict]:
        """Send to the upstream. Returns (status, payload); never raises
        for upstream errors, so the client gets the upstream's own error."""
        if self.mock:
            return 200, self._mock_payload(provider, str(body.get("model", "unknown")),
                                           request_id, session_id)
        req = self._upstream_request(provider, body, request_id, session_id, client_headers)
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
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
        provider, fwd = ctx["provider"], ctx["forward_body"]
        injected = False
        if provider != "anthropic":
            opts = dict(fwd.get("stream_options") or {})
            if not opts.get("include_usage"):
                opts["include_usage"] = True
                fwd["stream_options"] = opts
                injected = True
        acc = StreamAccumulator(provider, suppress_usage_chunk=injected)
        if self.mock:
            return 200, iter(self._mock_stream(ctx)), acc, None
        req = self._upstream_request(provider, fwd, ctx["request_id"], ctx["session_id"],
                                     ctx["headers"])
        try:
            resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
        except urllib.error.HTTPError as exc:
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
                                     ctx["session_id"])
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
        base = {"id": payload["id"], "object": "chat.completion.chunk", "model": ctx["model"]}
        out = ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant",
                                                             "content": "[prefixwise mock]"}}]})
        out += ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        if (ctx["forward_body"].get("stream_options") or {}).get("include_usage"):
            out += ev({**base, "choices": [], "usage": payload["usage"]})
        return out + [b"data: [DONE]\n", b"\n"]


def make_handler(proxy: PrefixwiseProxy):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0: a streamed body ends when the connection closes, so no
        # chunked encoding is needed for SSE.
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

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"ok": True, "mock": proxy.mock})
            if self.path == "/ledger":
                return self._json(200, json.loads(proxy.ledger.to_json()))
            if self.path.startswith("/requests/"):
                row = proxy.ledger.get_request(self.path[len("/requests/"):])
                if row is None:
                    return self._json(404, _error(404, "unknown request id (or not recorded yet)"))
                return self._json(200, row)
            return self._json(404, _error(404, "not found"))

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            return body

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/v1/messages", "/tasks"):
                return self._json(404, _error(404, f"unsupported path {self.path}"))
            try:
                body = self._read_json()
            except ValueError as exc:
                return self._json(400, _error(400, f"invalid JSON body: {exc}"))
            if self.path == "/tasks":
                return self._task(body)
            provider = "anthropic" if self.path == "/v1/messages" else "openai"
            if body.get("stream"):
                return self._stream(provider, body)
            result, _ = proxy.handle(provider, body, dict(self.headers.items()))
            # The client gets the upstream response unchanged; Prefixwise's
            # verdict rides in headers so drop-in clients are unaffected.
            extra = {"X-Prefixwise-Request-Id": result["request_id"]}
            if result.get("cause"):
                extra.update({"X-Prefixwise-Cause": result["cause"],
                              "X-Prefixwise-Legit": str(result["legit"]).lower(),
                              "X-Prefixwise-Avoidable-Tokens": str(result["avoidable_tokens"])})
            return self._json(result["status"], result["response"], extra)

        def _task(self, body: dict):
            task_id, success = body.get("task_id"), body.get("success")
            if not isinstance(task_id, str) or not task_id or not isinstance(success, bool):
                return self._json(400, _error(400, 'expected {"task_id": "<id>", "success": true|false}'))
            proxy.ledger.set_task_outcome(task_id, success)
            return self._json(200, {"ok": True, "task_id": task_id, "success": success})

        def _stream(self, provider: str, body: dict):
            ctx = proxy.prepare(provider, body, dict(self.headers.items()))
            status, lines, acc, err = proxy.open_stream(ctx)
            if lines is None:
                return self._json(status, err, {"X-Prefixwise-Request-Id": ctx["request_id"]})
            # Headers go out before usage is known, so the verdict cannot
            # ride in them: fetch it afterwards from GET /requests/<id>.
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Prefixwise-Request-Id", ctx["request_id"])
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
                proxy.finalize(ctx, status, acc.payload)
            self.close_connection = True
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080, *, ledger_path: str | None = None,
          upstream_url: str | None = None, mock: bool | None = None, quiet: bool = False):
    """Run the proxy until interrupted. Settings fall back to PREFIXWISE_* env vars."""
    hmac_key = os.environ.get("PREFIXWISE_HMAC_KEY")
    ledger_path = ledger_path or os.environ.get("PREFIXWISE_LEDGER_PATH", ":memory:")
    proxy = PrefixwiseProxy(ledger_path=ledger_path, upstream_url=upstream_url, mock=mock,
                            hmac_key=hmac_key.encode() if hmac_key else None)
    server = ThreadingHTTPServer((host, port), make_handler(proxy))
    if not quiet:
        target = "MOCK upstream (synthetic usage, no API calls)" if proxy.mock else proxy.upstream_url
        print(f"Prefixwise proxy on http://{host}:{server.server_address[1]}")
        print(f"  upstream: {target}")
        print(f"  ledger:   {ledger_path}")
        print("  POST /v1/chat/completions  POST /v1/messages  POST /tasks")
        print("  GET  /ledger  GET /requests/<id>  GET /health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve(port=int(os.environ.get("PORT", "8080")))
