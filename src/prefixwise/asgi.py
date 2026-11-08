"""Prefixwise inside the engine's own server: an ASGI middleware, no second process or port.

    vllm serve <model> --enable-prefix-caching --enable-prompt-tokens-details \\
        --middleware prefixwise.PrefixwiseMiddleware

    # or in your own FastAPI / Starlette app
    app.add_middleware(PrefixwiseMiddleware, ledger="prefixwise.db")

The middleware is Prefixwise's **edge** (docs/ARCHITECTURE.md) running in-process: it fingerprints
each inference request as it comes in, lets the engine answer, reads the usage from the response
(or from the stream as it passes), and hands the collector a turn record. Everything the proxy does
to a request happens here too (`PrefixwiseProxy.prepare`: the session id and `vllm_xargs` the KV
join needs, `--fix` rewrites, `stream_options.include_usage` so a stream carries usage), and the
same verdict headers come back on a non-streaming response (`X-Prefixwise-Request-Id`,
`X-Prefixwise-Cause`, `X-Prefixwise-Legit`, `X-Prefixwise-Avoidable-Tokens`). A stream is
relayed chunk by chunk, never buffered; its row is written when it ends.

Raw ASGI 3 (`scope, receive, send`): no Starlette or FastAPI import. Requests that are not a POST
to an inference route pass straight through. Classification and the ledger write run off the event
loop (`run_in_executor`), so a slow disk or a KV-event wait never stalls the other streams.

Configuration, when vLLM instantiates the class with `app` only, comes from the environment:

- `PREFIXWISE_LEDGER` (or `PREFIXWISE_LEDGER_PATH`): the SQLite ledger (default `prefixwise.db`).
- `PREFIXWISE_ENGINE`: `vllm` or `sglang`; default from the process (`vllm` or `sglang` imported).
- `PREFIXWISE_KV_EVENTS`: a ZMQ endpoint (comma-separated for several), or `auto`. Inside vLLM the
  default is `auto`: on the first inference request the middleware reads the engine's own
  per-rank publisher configs (`app.state.engine_client.get_kv_event_sources()`, what
  `GET /kv_event_sources` serves) and subscribes; `sys.argv`'s `--kv-events-config` is the last
  resort. Needs `pip install "prefixwise[engines]"`.
- `PREFIXWISE_FIX`: a fix to apply to every request (`canonical_tools`).
- `PREFIXWISE_COLLECTOR` / `PREFIXWISE_COLLECTOR_TOKEN`: send records to a `prefixwise collect`
  (HTTP) or to the record log (`redis://`) instead of a ledger in this process. **Required with
  `--api-server-count` above 1:** each API-server process gets its own middleware, and a session's
  turns split across them read as misses unless one collector sees them all. The middleware says
  so once at startup when it sees that flag without a collector.
- `PREFIXWISE_FULL_ATTENTION=1` (SGLang): the model has no sliding-window or Mamba layers.

Limits: inside vLLM the middleware runs outside vLLM's own API-key check, so a request the key
rejects is recorded as a failed request (a row without a verdict), as the proxy records a 401.
The ledger, `/api/v1` and `prefixwise report` are not served by the engine's port: read the
ledger file with the CLI, or run a collector and point `PREFIXWISE_COLLECTOR` at it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import threading
import time
from urllib.parse import urlsplit

from .proxy import CLASSIFIED_PATHS, PrefixwiseProxy, StreamAccumulator, _reachable, route

_MISSING = object()


def _env(name: str, *alts: str, default: str | None = None) -> str | None:
    for key in (name, *alts):
        value = os.environ.get(key)
        if value:
            return value
    return default


def _default_engine() -> str | None:
    if "vllm" in sys.modules:
        return "vllm"
    if "sglang" in sys.modules:
        return "sglang"
    return None


def kv_events_from_argv(argv: list[str]) -> str | None:
    """The last resort for the KV-event endpoint: vLLM's own `--kv-events-config JSON` on the command
    line, its bind address (`tcp://*:5557`) turned into the local one. None when absent or not zmq."""
    for i, arg in enumerate(argv):
        raw = None
        if arg == "--kv-events-config" and i + 1 < len(argv):
            raw = argv[i + 1]
        elif arg.startswith("--kv-events-config="):
            raw = arg.split("=", 1)[1]
        if raw is None:
            continue
        try:
            cfg = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(cfg, dict) or cfg.get("publisher", "zmq") != "zmq" or not cfg.get("endpoint"):
            return None
        if cfg.get("enable_kv_cache_events") is False:
            return None
        return _reachable(str(cfg["endpoint"]), "127.0.0.1")
    return None


def api_server_count(argv: list[str]) -> int:
    """vLLM's --api-server-count / -asc from the command line; 1 when absent."""
    for i, arg in enumerate(argv):
        for flag in ("--api-server-count", "-asc"):
            value = None
            if arg == flag and i + 1 < len(argv):
                value = argv[i + 1]
            elif arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
            if value is not None:
                try:
                    return max(1, int(value))
                except ValueError:
                    return 1
    return 1


class PrefixwiseMiddleware:
    """ASGI 3 middleware: Prefixwise's edge inside the server it measures (module docstring)."""

    def __init__(self, app, *, ledger: str | None = None, engine: str | None = None,
                 kv_events: str | None = None, fixes: str | list[str] | None = None,
                 collector_url: str | None = None, full_attention: bool | None = None,
                 prices: str | None = None, argv: list[str] | None = None, env: dict | None = None):
        self.app = app
        env = os.environ if env is None else env
        argv = sys.argv if argv is None else argv

        def get(name, *alts, default=None):
            for key in (name, *alts):
                if env.get(key):
                    return env[key]
            return default
        ledger = ledger or get("PREFIXWISE_LEDGER", "PREFIXWISE_LEDGER_PATH", default="prefixwise.db")
        self.engine = engine or get("PREFIXWISE_ENGINE") or _default_engine()
        if fixes is None:
            fixes = [f for f in (get("PREFIXWISE_FIX", default="") or "").split(",") if f.strip()] or None
        if full_attention is None and get("PREFIXWISE_FULL_ATTENTION"):
            full_attention = get("PREFIXWISE_FULL_ATTENTION").lower() not in ("0", "false", "no", "off")
        collector_url = collector_url if collector_url is not None else get("PREFIXWISE_COLLECTOR", "PREFIXWISE_LOG")
        kv = kv_events if kv_events is not None else get("PREFIXWISE_KV_EVENTS")
        if kv is None and self.engine == "vllm":
            kv = "auto"  # from the engine itself, on the first request
        # `auto` is resolved lazily from the running engine: never passed to the proxy, which would
        # ask an upstream URL that does not exist here.
        self._lazy_kv = kv == "auto" and self.engine is not None
        self._pending_kv = kv if (kv and kv != "auto") else None
        self._full_attention = full_attention
        self._argv = list(argv)
        self.proxy = PrefixwiseProxy(ledger_path=ledger, upstream_url="http://in-process.invalid", api_key=None,
                                     mock=False, fixes=fixes, prices=prices, engine=self.engine,
                                     kv_events=None if self._lazy_kv else self._pending_kv,
                                     full_attention=full_attention, collector_url=collector_url or None)
        self.proxy.versioned = set()  # paths are the engine's own: no /v1 juggling
        self._kv_lock = threading.Lock()
        self._kv_done = not self._lazy_kv
        self.notes: list[str] = []
        self._warn_workers(argv, collector_url)

    # -- startup notes ------------------------------------------------------------------

    def _warn_workers(self, argv: list[str], collector_url: str | None) -> None:
        n = api_server_count(argv)
        if n > 1 and not collector_url:
            self._note(f"--api-server-count {n}: each API-server process runs its own Prefixwise with its own "
                       "ledger, so a session's turns split across them read as misses. Set PREFIXWISE_COLLECTOR to "
                       "a `prefixwise collect` URL (or PREFIXWISE_LOG to a redis:// log) so one collector sees "
                       "every turn")

    def _note(self, text: str) -> None:
        self.notes.append(text)
        print(f"prefixwise: {text}", file=sys.stderr, flush=True)

    # -- KV events from the engine itself -----------------------------------------------

    def _ensure_kv(self, scope: dict) -> None:
        """Once: subscribe to the engine's KV-event publishers, found from the app the request
        belongs to (vLLM keeps its engine client on app.state), else from the command line."""
        if self._kv_done:
            return
        with self._kv_lock:
            if self._kv_done:
                return
            self._kv_done = True
            sources = self._engine_sources(scope)
            if not sources:
                endpoint = kv_events_from_argv(self._argv)
                if endpoint:
                    sources = [("rank 0", endpoint)]
            if not sources:
                self._note("the engine publishes no KV events that this process can see (no --kv-events-config, or an "
                           "SGLang server: set PREFIXWISE_KV_EVENTS); evictions stay inferred")
                return
            try:
                ranks = [int(label.split()[-1]) for label, _ in sources] if len(sources) > 1 else None
                self.proxy._start_events(sources, bool(self._full_attention), ranks=ranks)
            except Exception as exc:  # noqa: BLE001 - pyzmq / msgspec missing, or a bad endpoint
                self.proxy.engine_events = None
                self._note(f"KV events not subscribed ({type(exc).__name__}: {exc}); pip install "
                           "'prefixwise[engines]'; evictions stay inferred")
                return
            self._note("subscribed to the engine's KV events: " + ", ".join(e for _, e in sources))

    @staticmethod
    def _engine_sources(scope: dict) -> list[tuple[str, str]]:
        app = scope.get("app")
        client = getattr(getattr(app, "state", None), "engine_client", None)
        getter = getattr(client, "get_kv_event_sources", None)
        if getter is None:
            return []
        try:
            info = getter()
            if inspect.isawaitable(info):
                return []  # an async client: nothing to await from here; the argv fallback follows
        except Exception:  # noqa: BLE001
            return []
        sources = []
        for rank, cfg in sorted((info or {}).items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
            cfg = cfg or {}
            if isinstance(cfg, dict) and cfg.get("enable_kv_cache_events") and cfg.get("publisher") == "zmq" \
                    and cfg.get("endpoint"):
                sources.append((f"rank {rank}", _reachable(str(cfg["endpoint"]), "127.0.0.1")))
        return sources

    # -- ASGI -----------------------------------------------------------------------------

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        path, _ = route(scope.get("path") or "")
        if path not in CLASSIFIED_PATHS:
            return await self.app(scope, receive, send)
        body = await _read_body(receive)
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if not isinstance(data, dict):  # let the engine answer the malformed request itself
            return await self.app(scope, _replay(body, receive), send)
        provider = "anthropic" if path == "/v1/messages" else "openai"
        api = "responses" if path == "/v1/responses" else "chat"
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        self._ensure_kv(scope)
        loop = asyncio.get_running_loop()
        try:
            ctx = await loop.run_in_executor(None, lambda: self.proxy.prepare(provider, data, headers, api))
        except Exception as exc:  # noqa: BLE001 - a Prefixwise error never fails the request
            print(f"prefixwise: request passed through unrecorded: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return await self.app(scope, _replay(body, receive), send)
        fwd = ctx["forward_body"]
        stream = bool(fwd.get("stream"))
        acc = None
        if stream:
            injected = False
            if provider != "anthropic" and api != "responses":
                opts = dict(fwd.get("stream_options") or {})
                if not opts.get("include_usage"):
                    opts["include_usage"] = True
                    fwd["stream_options"] = opts
                    injected = True
            acc = StreamAccumulator(provider, suppress_usage_chunk=injected, api=api)
        new_body = json.dumps(fwd).encode()
        new_scope = dict(scope)
        new_scope["headers"] = [(k, v) for k, v in (scope.get("headers") or []) if k.lower() != b"content-length"]
        new_scope["headers"].append((b"content-length", str(len(new_body)).encode()))
        sent_at = time.perf_counter()
        relay = _Relay(self, ctx, acc, send, loop, sent_at)
        try:
            await self.app(new_scope, _replay(new_body, receive), relay.send)
        except Exception:
            await relay.failed("app_error")
            raise
        await relay.finish()


async def _read_body(receive) -> bytes:
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.request":
            chunks.append(message.get("body") or b"")
            if not message.get("more_body"):
                break
        elif message["type"] == "http.disconnect":
            break
    return b"".join(chunks)


def _replay(body: bytes, receive):
    """A receive() that yields `body` once, then defers to the original (for http.disconnect)."""
    sent = False

    async def replayed():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()
    return replayed


class _Relay:
    """Wraps `send`: buffers a JSON response to add the verdict headers; relays a stream as it
    passes, reading usage from it; records the row when the response is complete."""

    def __init__(self, mw: PrefixwiseMiddleware, ctx: dict, acc: StreamAccumulator | None, send, loop, sent_at: float):
        self.mw, self.ctx, self.acc, self._send, self.loop, self.sent_at = mw, ctx, acc, send, loop, sent_at
        self.status = 200
        self.headers: list = []
        self.started = False  # http.response.start passed to the client
        self.body = bytearray()  # JSON responses, and error bodies on a stream request
        self.tail = b""  # a stream's incomplete last line
        self.recording = None
        self.done = False
        self.streaming = False

    async def send(self, message) -> None:
        kind = message.get("type")
        if kind == "http.response.start":
            self.status = int(message.get("status") or 200)
            self.headers = list(message.get("headers") or [])
            ctype = next((v for k, v in self.headers if k.lower() == b"content-type"), b"").decode("latin-1", "replace")
            self.streaming = self.acc is not None and self.status < 400 and "text/event-stream" in ctype
            self.mw.proxy.response_started(self.ctx)
            if self.streaming:  # headers go out before usage is known: the verdict rides no header here
                await self._start([(b"x-prefixwise-request-id", self.ctx["request_id"].encode())])
            return  # a JSON response: start is sent with the verdict headers, once the body is known
        if kind != "http.response.body":
            return await self._send(message)
        chunk = message.get("body") or b""
        more = bool(message.get("more_body"))
        if self.streaming:
            data = self.tail + chunk
            lines = data.split(b"\n")
            self.tail = lines.pop()  # incomplete, or b"" when the chunk ended on a newline
            out = bytearray()
            for line in lines:
                full = line + b"\n"
                keep = self.acc.feed(full)
                if self.acc.done and self.recording is None:
                    self.recording = self.mw.proxy.begin_recording(self.ctx)
                if keep:
                    out += full
            if not more and self.tail:
                if self.acc.feed(self.tail):
                    out += self.tail
                self.tail = b""
            try:
                await self._send({"type": "http.response.body", "body": bytes(out), "more_body": more})
            except Exception:
                await self.failed("client_disconnected", 499)
                raise
            if not more:
                await self._complete_stream()
            return
        self.body += chunk
        if more:
            return
        await self._complete_json()

    async def _start(self, extra: list) -> None:
        if self.started:
            return
        self.started = True
        await self._send({"type": "http.response.start", "status": self.status, "headers": self.headers + extra})

    async def _complete_json(self) -> None:
        if self.done:
            return
        self.done = True
        self.ctx["upstream_ms"] = (time.perf_counter() - self.sent_at) * 1000
        payload = None
        try:
            payload = json.loads(bytes(self.body)) if self.body else None
        except ValueError:
            payload = None
        extra = [(b"x-prefixwise-request-id", self.ctx["request_id"].encode())]
        if self.status >= 400 or not isinstance(payload, dict):
            err = payload if isinstance(payload, dict) else None
            await self.loop.run_in_executor(None, lambda: self.mw.proxy.record_failure(self.ctx, self.status, err))
        else:
            result, _ = await self.loop.run_in_executor(
                None, lambda: self.mw.proxy.finalize_safely(self.ctx, self.status, payload))
            if result.get("recorded") is False:
                extra.append((b"x-prefixwise-error", b"not recorded (see server log)"))
            if "queued" in result:
                extra.append((b"x-prefixwise-queued", b"1" if result["queued"] else b"0"))
            if result.get("cause"):
                extra += [(b"x-prefixwise-cause", str(result["cause"]).encode()),
                          (b"x-prefixwise-legit", str(result["legit"]).lower().encode()),
                          (b"x-prefixwise-avoidable-tokens", str(result["avoidable_tokens"]).encode())]
                if result.get("fixes_applied"):
                    extra.append((b"x-prefixwise-fixes", ",".join(result["fixes_applied"]).encode()))
        await self._start(extra)
        await self._send({"type": "http.response.body", "body": bytes(self.body), "more_body": False})

    async def _complete_stream(self) -> None:
        if self.done:
            return
        self.done = True
        self.ctx["upstream_ms"] = (time.perf_counter() - self.sent_at) * 1000
        if self.recording is None:
            self.recording = self.mw.proxy.begin_recording(self.ctx)
        rec = self.recording
        try:
            await self.loop.run_in_executor(
                None, lambda: self.mw.proxy.finalize_safely(self.ctx, self.status, self.acc.payload))
        finally:
            self.mw.proxy.end_recording(self.ctx, rec)

    async def failed(self, error: str, status: int | None = None) -> None:
        """The response did not complete: a row without a verdict, and the recording released."""
        if self.done:
            return
        self.done = True
        self.ctx["upstream_ms"] = (time.perf_counter() - self.sent_at) * 1000
        if self.mw.proxy.engine_events is not None:
            self.mw.proxy.engine_events.discard(self.ctx.get("engine_token"))
        code = status if status is not None else (self.status if self.status >= 400 else 500)
        try:
            await self.loop.run_in_executor(None, lambda: self.mw.proxy.record_failure(self.ctx, code, error=error))
        finally:
            self.mw.proxy.end_recording(self.ctx, self.recording)

    async def finish(self) -> None:
        """The app returned. A stream request answered with an error body, or an app that never
        sent a body, is completed here."""
        if self.done:
            return
        if self.acc is not None and not self.streaming:  # a stream request answered with JSON (an error)
            await self._complete_json()
        elif not self.started:
            await self.failed("no_response")
        else:
            await self._complete_json()


__all__ = ["PrefixwiseMiddleware", "kv_events_from_argv", "api_server_count"]
