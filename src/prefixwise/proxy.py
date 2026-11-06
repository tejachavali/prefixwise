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
- GET  /health, GET /ledger  (ledger roll-up as JSON)

The client gets the upstream response and status unchanged. Prefixwise's
verdict rides in response headers: X-Prefixwise-Request-Id,
X-Prefixwise-Cause, X-Prefixwise-Legit, X-Prefixwise-Avoidable-Tokens.
Session/agent identity: X-Prefixwise-Session-Id / X-Prefixwise-Agent-Id
headers, or session_id / agent_id body fields (stripped before
forwarding). Client auth headers are passed through; PREFIXWISE_API_KEY,
if set, overrides them. stream=true is rejected with a 400 for now.

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
PREFIXWISE_FIELDS = ("session_id", "agent_id", "prefixwise_compaction_event")

# Client headers passed through to the upstream when present.
PASSTHROUGH_HEADERS = ("authorization", "x-api-key", "anthropic-version", "anthropic-beta",
                       "openai-organization", "openai-project")
ANTHROPIC_VERSION = "2023-06-01"


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


class PrefixwiseProxy:
    def __init__(self, ledger_path: str = ":memory:", hmac_key: bytes | None = None,
                 upstream_url: str | None = None, api_key: str | None = None):
        self.ledger = Ledger(ledger_path)
        self.fingerprinter = Fingerprinter(hmac_key=hmac_key)
        self.scorer = AgreementScorer()
        self.upstream_url = (upstream_url or os.environ.get("PREFIXWISE_UPSTREAM_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("PREFIXWISE_API_KEY")
        self.mock = not self.upstream_url or os.environ.get("PREFIXWISE_MOCK_UPSTREAM") == "1"
        self.last_snapshot: dict[str, RequestSnapshot] = {}
        self.last_response_id: dict[str, str | None] = {}
        self.last_seen: dict[str, float] = {}

    def handle(self, provider: str, body: dict,
               headers: dict[str, str] | None = None) -> tuple[dict, LedgerEntry | None]:
        """Classify, forward, and record one request.

        Returns (result, entry). result["status"] and result["response"]
        are the upstream status and payload, returned to the client
        unchanged. On an upstream error nothing is classified or
        recorded (no usage to score) and entry is None.
        """
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        session_id = str(headers.get("x-prefixwise-session-id") or body.get("session_id")
                         or (body.get("metadata") or {}).get("user_id") or "default")
        agent_id = str(headers.get("x-prefixwise-agent-id") or body.get("agent_id") or "proxy")
        request_id = str(uuid.uuid4())
        model = str(body.get("model", "unknown"))
        # E1-open / hooks: marks that a PreCompact/compaction event fired
        # before this turn, so the classifier can tell COMPACTION_REWRITE
        # from HISTORY_EDIT. Never forwarded.
        compaction_event = bool(body.get("prefixwise_compaction_event", False)) or \
            headers.get("x-prefixwise-compaction-event") in ("1", "true")

        if body.get("stream"):
            return {"request_id": request_id, "status": 400, "cause": None,
                    "response": _error(400, "Prefixwise proxy does not support stream=true yet; "
                                            "send a non-streaming request.")}, None

        forward_body = {k: v for k, v in body.items()
                        if k not in PREFIXWISE_FIELDS or (provider == "vllm" and k == "session_id")}
        if provider == "vllm":
            forward_body["session_id"] = session_id
        # Native diagnostics chaining. Anthropic: diagnostics object (GA,
        # no beta header). OpenAI comparison_response_id exists only on
        # the Responses API, which this proxy does not serve yet, so it is
        # not sent to /v1/chat/completions.
        if provider == "anthropic":
            forward_body.update(anthropic_diagnostics(self.last_response_id.get(session_id)))

        status, payload = self._forward(provider, forward_body, request_id, session_id, headers)
        if status >= 400:
            return {"request_id": request_id, "status": status, "cause": None,
                    "response": payload}, None

        now = time.time()
        gap_seconds = (now - self.last_seen[session_id]) if session_id in self.last_seen else None
        self.last_seen[session_id] = now
        usage = payload.get("usage") or {}
        prompt_tokens, cached = normalize_usage(provider, usage)
        native_reason = native_reason_from(payload)
        response_id = payload.get("id")

        fps = self.fingerprinter.fingerprint_blocks(extract_blocks(body, provider))
        prev = self.last_snapshot.get(session_id)
        curr = RequestSnapshot(model=model, provider=provider, blocks=fps,
                               prompt_tokens=prompt_tokens, cached_tokens=cached,
                               gap_seconds=gap_seconds, compaction_event=compaction_event,
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
            session_id=session_id, agent_id=agent_id,
            provider=provider, model=model, cause=result.cause.value,
            prompt_tokens=prompt_tokens, cached_tokens=cached,
            avoidable_tokens=avoidable_for_turn(result.legit, prompt_tokens, cached,
                                                prev.prompt_tokens if prev else None),
            avoidable_cost=None, legit=result.legit, native_reason=native_reason,
        )
        self.ledger.record(entry)
        return {"request_id": request_id, "status": status, "cause": result.cause.value,
                "legit": result.legit, "native_reason": native_reason, "usage": usage,
                "prompt_tokens": prompt_tokens, "cached_tokens": cached,
                "avoidable_tokens": entry.avoidable_tokens, "response": payload}, entry

    def _mock_payload(self, provider: str, model: str, request_id: str, session_id: str) -> dict:
        # Synthetic usage: later turns in a session reuse the prefix.
        # 2,000 prompt tokens so the OpenAI 1,024 minimum does not fire.
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

    def _forward(self, provider: str, body: dict, request_id: str, session_id: str,
                 client_headers: dict[str, str]) -> tuple[int, dict]:
        """Send to the upstream. Returns (status, payload); never raises
        for upstream errors, so the client gets the upstream's own error."""
        if self.mock:
            return 200, self._mock_payload(provider, str(body.get("model", "unknown")),
                                           request_id, session_id)
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
        req = urllib.request.Request(self.upstream_url + path, method="POST",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raw = exc.read() or b""
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                return exc.code, _error(exc.code, raw.decode("utf-8", "replace")[:2000])
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 502, _error(502, f"upstream unreachable: {exc}")
        except ValueError:
            return 502, _error(502, "upstream returned a non-JSON body")


def make_handler(proxy: PrefixwiseProxy):
    class Handler(BaseHTTPRequestHandler):
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
            return self._json(404, _error(404, "not found"))

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/v1/messages"):
                return self._json(404, _error(404, f"unsupported path {self.path}"))
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except ValueError as exc:
                return self._json(400, _error(400, f"invalid JSON body: {exc}"))
            provider = "anthropic" if self.path == "/v1/messages" else "openai"
            result, _ = proxy.handle(provider, body, dict(self.headers.items()))
            # The client gets the upstream response unchanged; Prefixwise's
            # verdict rides in headers so drop-in clients are unaffected.
            extra = {"X-Prefixwise-Request-Id": result["request_id"]}
            if result.get("cause"):
                extra.update({"X-Prefixwise-Cause": result["cause"],
                              "X-Prefixwise-Legit": str(result["legit"]).lower(),
                              "X-Prefixwise-Avoidable-Tokens": str(result["avoidable_tokens"])})
            return self._json(result["status"], result["response"], extra)
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080):
    hmac_key = os.environ.get("PREFIXWISE_HMAC_KEY")
    proxy = PrefixwiseProxy(ledger_path=os.environ.get("PREFIXWISE_LEDGER_PATH", ":memory:"),
                            hmac_key=hmac_key.encode() if hmac_key else None)
    server = ThreadingHTTPServer((host, port), make_handler(proxy))
    print(f"Prefixwise proxy on http://{host}:{port} (mock={proxy.mock})")
    print("POST /v1/chat/completions, POST /v1/messages, GET /ledger, GET /health")
    server.serve_forever()


if __name__ == "__main__":
    serve(port=int(os.environ.get("PORT", "8080")))
