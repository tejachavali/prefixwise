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
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .classifier import RequestSnapshot, classify
from .diagnostics import AgreementScorer, anthropic_diagnostics, openai_comparison_options
from .fingerprint import Fingerprinter
from .ledger import Ledger, LedgerEntry, avoidable_tokens


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

    def handle(self, provider: str, body: dict) -> tuple[dict, LedgerEntry]:
        session_id = body.get("session_id") or body.get("metadata", {}).get("user_id") or "default"
        session_id = str(session_id)
        request_id = str(uuid.uuid4())
        model = str(body.get("model", "unknown"))
        blocks = extract_blocks(body, provider)
        fps = self.fingerprinter.fingerprint_blocks(blocks)
        prev = self.last_snapshot.get(session_id)

        # Native diagnostics chaining (v2, verified mechanism).
        native_reason = None
        if provider == "anthropic":
            body.update(anthropic_diagnostics(self.last_response_id.get(session_id)))
        elif provider == "openai":
            body.update(openai_comparison_options(self.last_response_id.get(session_id)))

        usage, response_id = self._forward(provider, body, request_id, session_id)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        cached = int(usage.get("cached_tokens", 0) or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0)
        # Native reason, when the upstream returned one.
        diag = usage.get("prompt_cache_diagnostics") or {}
        native_reason = diag.get("reason")

        curr = RequestSnapshot(model=model, provider=provider, blocks=fps,
                               prompt_tokens=prompt_tokens, cached_tokens=cached,
                               params={k: body.get(k) for k in
                                       ("tool_choice", "thinking", "reasoning", "output_format",
                                        "text_format", "verbosity") if body.get(k) is not None},
                               has_breakpoint=bool(body.get("cache_control") or
                                                   any("cache_control" in str(m) for m in body.get("messages", []))))
        result = classify(prev, curr, native_reason=native_reason)
        self.scorer.record(result.cause, native_reason, provider)
        self.last_snapshot[session_id] = curr
        if response_id:
            self.last_response_id[session_id] = response_id

        # A first write is not waste: it is the baseline the prefix is built on.
        first_write = prompt_tokens if result.cause.value == "first_write" else 0
        entry = LedgerEntry(
            session_id=session_id, agent_id=str(body.get("agent_id", "proxy")),
            provider=provider, model=model, cause=result.cause.value,
            prompt_tokens=prompt_tokens, cached_tokens=cached,
            avoidable_tokens=avoidable_tokens(prompt_tokens, cached, first_write_tokens=first_write),
            avoidable_cost=None, legit=result.legit, native_reason=native_reason,
        )
        self.ledger.record(entry)
        return {"request_id": request_id, "cause": result.cause.value, "legit": result.legit,
                "native_reason": native_reason, "usage": usage}, entry

    def _forward(self, provider: str, body: dict, request_id: str, session_id: str) -> tuple[dict, str | None]:
        if self.mock:
            # Synthetic usage: second turn in a session reuses the prefix.
            # 2,000 prompt tokens so the OpenAI 1,024 minimum does not fire.
            cached = 1000 if session_id in self.last_snapshot else 0
            return {"prompt_tokens": 2000, "cached_tokens": cached}, f"resp-{request_id[:8]}"
        path = "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
        req = urllib.request.Request(self.upstream_url + path, method="POST",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key or ''}",
                                              "X-Request-Id": request_id,
                                              "X-Session-ID": session_id})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read() or b"{}")
        usage = payload.get("usage", {}) or {}
        return usage, payload.get("id")


def make_handler(proxy: PrefixwiseProxy):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output clean
            pass

        def _json(self, code: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"ok": True, "mock": proxy.mock})
            if self.path == "/ledger":
                return self._json(200, json.loads(proxy.ledger.to_json()))
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            provider = "anthropic" if self.path == "/v1/messages" else "openai"
            result, _ = proxy.handle(provider, body)
            return self._json(200, result)
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
