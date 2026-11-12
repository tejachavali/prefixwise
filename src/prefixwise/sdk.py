"""Prefixwise inside your own code: no proxy, no base URL to change.

    from openai import OpenAI
    import prefixwise

    client = prefixwise.wrap(OpenAI())           # or Anthropic(), AsyncOpenAI(), AsyncAnthropic()
    with prefixwise.session("fix-bug-42"):       # optional: group turns, as X-Prefixwise-Session-Id does
        client.chat.completions.create(...)      # unchanged; Prefixwise records why the cache missed

Each completed call is classified and recorded exactly as the proxy would (same causes,
sub-reasons, prices, ledger), from the request the SDK sent and the response it got.
Nothing is sent anywhere and nothing is changed: the call and its return value are the
SDK's own. Recording can never break a call: a failure is reported once and ignored.

Streamed calls are recorded too: `create(..., stream=True)` and the SDKs' `.stream()`
helpers, sync and async. The stream the caller iterates is the SDK's own, item for item;
usage is read from its events as they pass, and the call is recorded once, when the
stream ends. A stream closed or cancelled before its end is a failure row
(`client_disconnected`), as the proxy records a client that hangs up: its usage is partial at
best; one the server or the network breaks is `upstream_dropped`. A stream dropped without
being closed or read to its end is not recorded. `with_raw_response` and
`with_streaming_response` streams are passed through untouched and unrecorded. On OpenAI Chat
Completions, usage is only streamed when `stream_options.include_usage` is set; when the
caller has not set it, Prefixwise asks for it and keeps the extra usage-only chunk out of
the caller's stream, so the caller sees exactly the chunks it asked for. The session is
the one current when the call was made, wherever the stream is consumed.

`Recorder` is the same machinery for any integration that has the request and the
response in hand (the LiteLLM callback uses it).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import os
import sys
import time

from .ledger import LedgerEntry

_session: contextvars.ContextVar[str | None] = contextvars.ContextVar("prefixwise_session", default=None)
_task: contextvars.ContextVar[str | None] = contextvars.ContextVar("prefixwise_task", default=None)


@contextlib.contextmanager
def session(session_id: str, task_id: str | None = None):
    """Calls inside this block belong to one session (and task, for tokens per successful task)."""
    s, t = _session.set(str(session_id)), _task.set(task_id)
    try:
        yield
    finally:
        _session.reset(s)
        _task.reset(t)


class Recorder:
    """Classify and record a request and its response, without forwarding anything."""

    def __init__(self, ledger_path: str | None = None, *, agent_id: str = "sdk"):
        from .proxy import PrefixwiseProxy

        path = ledger_path or os.environ.get("PREFIXWISE_LEDGER_PATH") or "prefixwise.db"
        # No fixes: the request recorded must be the request that was sent. No engine join
        # (PREFIXWISE_ENGINE / KV_EVENTS describe a proxy's upstream, not this process's calls).
        self.proxy = PrefixwiseProxy(ledger_path=path, mock=True, fixes=(), engine="", kv_events="")
        self.agent_id = agent_id
        self._warned = False
        if getattr(self.proxy.collector, "remote", False):
            import atexit
            atexit.register(self.close)  # a script's last records must reach the collector

    def close(self, timeout: float = 5.0) -> None:
        """Send records still queued for a remote collector (PREFIXWISE_COLLECTOR); a no-op otherwise."""
        self.proxy.close(timeout)

    @property
    def ledger(self):
        return self.proxy.ledger

    def record(self, provider: str, request: dict, response: dict, *, api: str = "chat",
               session_id: str | None = None, agent_id: str | None = None, task_id: str | None = None,
               latency_ms: float | None = None, session_is_user: bool = False,
               genai_provider: str | None = None, ended_at: float | None = None) -> LedgerEntry | None:
        """provider: "anthropic" or "openai" (any OpenAI-compatible server): whose caching rules
        apply; api: "chat", "responses" (OpenAI Responses) or "chat" for Anthropic Messages.
        session_is_user: session_id is an end-user id, not a conversation id (kept out of
        OpenTelemetry's gen_ai.conversation.id). genai_provider: the actual provider for
        OpenTelemetry (e.g. "aws.bedrock"). ended_at: unix time the response completed, when the
        call is recorded later than that (LiteLLM runs callbacks on a pool, after the fact); the
        send time is then ended_at - latency. Returns the row, or None if recording failed (it
        never raises)."""
        ctx = None
        try:
            ctx = self.proxy.prepare(provider, _plain(request), self._headers(session_id, agent_id, task_id), api=api)
            ctx["session_is_user"] = bool(session_is_user) or bool(ctx.get("session_is_user"))
            if genai_provider:
                ctx["genai_provider"] = genai_provider
            if ended_at is not None:
                ctx["span_end"] = ended_at
            if latency_ms is not None:  # the call is already over: it was sent latency_ms before it ended
                ctx["span_start"] = (ended_at if ended_at is not None else time.time()) - latency_ms / 1000
                if ctx.get("facts") is not None:  # sub-reasons time the gap from the send too
                    ctx["facts"].sent_at = ctx["span_start"]
            self.proxy.response_started(ctx)
            ctx["upstream_ms"] = latency_ms if latency_ms is not None else 0.0
            _, entry = self.proxy.finalize(ctx, 200, _plain(response))
            return entry
        except Exception as exc:  # never break the caller's request
            if not self._warned:
                self._warned = True
                print(f"prefixwise: could not record a call ({type(exc).__name__}: {exc}); "
                      "the call itself was unaffected", file=sys.stderr)
            if ctx is not None:
                try:  # still a row, so error rates and every other rate stay honest
                    self.proxy.record_failure(ctx, 200, error="not_recorded")
                except Exception:  # noqa: BLE001
                    pass
            return None

    def _headers(self, session_id, agent_id, task_id) -> dict:
        headers = {"x-prefixwise-agent-id": agent_id or self.agent_id}
        sid = session_id or _session.get()
        tid = task_id or _task.get()
        if sid:
            headers["x-prefixwise-session-id"] = str(sid)
        if tid:
            headers["x-prefixwise-task-id"] = str(tid)
        return headers

    def record_failure(self, provider: str, request: dict, *, status: int, error: str, api: str = "chat",
                       session_id: str | None = None, agent_id: str | None = None,
                       task_id: str | None = None, latency_ms: float | None = None) -> None:
        """A call that gave no verdict (a stream closed or broken before its end) is a row too,
        as in the proxy, so error rates exist and every other rate has an honest denominator.
        Never raises."""
        try:
            ctx = self.proxy.prepare(provider, _plain(request), self._headers(session_id, agent_id, task_id), api=api)
            if latency_ms is not None:
                ctx["span_start"] = time.time() - latency_ms / 1000
                ctx["upstream_ms"] = latency_ms
            self.proxy.record_failure(ctx, status, error=error)
        except Exception as exc:  # noqa: BLE001 - never break the caller's stream
            if not self._warned:
                self._warned = True
                print(f"prefixwise: could not record a call ({type(exc).__name__}: {exc}); "
                      "the call itself was unaffected", file=sys.stderr)


_DROP = object()


def _plain(value):
    """JSON-shaped copy of a request or response: SDK objects (messages from an earlier response,
    content blocks) as the dicts the SDK sent; NotGiven / Omit and other non-JSON values dropped."""
    if isinstance(value, dict):
        return {str(k): v for k, v in ((k, _plain(x)) for k, x in value.items()) if v is not _DROP}
    if isinstance(value, (list, tuple)):
        return [v for v in (_plain(x) for x in value) if v is not _DROP]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if type(value).__name__ in ("NotGiven", "Omit"):
        return _DROP
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _plain(dump(mode="json", exclude_none=True))
        except TypeError:
            return _plain(dump())
    return _DROP


_SDK_ONLY = ("extra_headers", "extra_query", "extra_body", "timeout")


def _request_body(kwargs: dict) -> dict:
    """What the SDK puts on the wire: its own options removed, extra_body merged in."""
    body = {k: v for k, v in kwargs.items() if k not in _SDK_ONLY}
    if isinstance(kwargs.get("extra_body"), dict):
        body.update(kwargs["extra_body"])
    return body


_default: Recorder | None = None


def default_recorder() -> Recorder:
    global _default
    if _default is None:
        _default = Recorder()
    return _default


def _as_dict(obj) -> dict | None:
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except TypeError:
            return dump()
    return None


class _Tap:
    """Reads one streamed call's events as they pass and records the call once, when it ends.
    Reading never raises: a Prefixwise bug leaves the caller's stream as it was, unrecorded."""

    _EVENTS = {"anthropic": ("message_start", "message_delta", "message_stop"),
               "responses": ("response.created", "response.completed", "response.incomplete", "response.failed")}

    def __init__(self, call: "_Call", injected: bool):
        from .proxy import StreamAccumulator

        self.call = call
        self.kind = "anthropic" if call.provider == "anthropic" else call.api
        self.acc = StreamAccumulator(call.provider, suppress_usage_chunk=injected, api=call.api)
        self.finished = False
        self.broken = False
        self.reading = False  # inside the stream's own next(): a close there is its natural end

    def hook_close(self, stream, asynchronous: bool) -> None:
        """The SDKs' helpers (ChatCompletionStream, ResponseStream) close a stream through its HTTP
        response, never through the stream: a close there before the end is a stream left early."""
        response = getattr(stream, "response", None)
        name = "aclose" if asynchronous else "close"
        original = getattr(response, name, None)
        if original is None:
            return
        tap = self
        if asynchronous:
            async def close(*args, **kwargs):
                if not tap.reading:
                    tap.end(False)
                return await original(*args, **kwargs)
        else:
            def close(*args, **kwargs):
                if not tap.reading:
                    tap.end(False)
                return original(*args, **kwargs)
        setattr(response, name, close)

    def see(self, item) -> bool:
        """True if the item goes on to the caller (only an usage chunk Prefixwise asked for does not)."""
        if self.broken or self.finished:
            return True
        try:
            if self.kind == "chat":
                event = {"id": getattr(item, "id", None), "model": getattr(item, "model", None),
                         "choices": getattr(item, "choices", None) or []}
                usage = getattr(item, "usage", None)
                if usage is not None:
                    event["usage"] = _as_dict(usage)
                for key in ("prompt_cache_diagnostics", "diagnostics"):
                    extra = (getattr(item, "model_extra", None) or {}).get(key)
                    if isinstance(extra, dict):
                        event[key] = extra
                return self.acc.feed_event(event)
            if getattr(item, "type", None) in self._EVENTS[self.kind]:
                self.acc.feed_event(_as_dict(item))
        except Exception:  # noqa: BLE001 - an event we cannot read: this call goes unrecorded
            self.broken = True
        return True

    def stopped(self, exc: BaseException) -> None:
        """The stream raised: the server or the network broke it (an Exception), or the caller
        stopped it (cancelled, interrupted, generator closed)."""
        self.end(False, "upstream_dropped" if isinstance(exc, Exception) else "client_disconnected")

    def end(self, completed: bool, error: str = "client_disconnected") -> None:
        """Record the call: completed, from what was read; otherwise as a failure row."""
        if self.finished:
            return
        self.finished = True
        if self.broken:
            return
        try:
            if completed:
                self.call.record(self.acc.payload)
            else:
                self.call.fail(499 if error == "client_disconnected" else 502, error)
        except Exception:  # noqa: BLE001 - recording never breaks the caller's stream
            pass


class _Call:
    """One call's request and context, taken when it was made (the session is the one current
    then, wherever its stream is consumed)."""

    def __init__(self, recorder: Recorder, provider: str, api: str, request: dict, agent_id: str | None):
        self.recorder, self.provider, self.api, self.agent_id = recorder, provider, api, agent_id
        self.request = request
        self.session_id, self.task_id = _session.get(), _task.get()
        self.started = time.perf_counter()

    def _ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def record(self, payload: dict) -> None:
        self.recorder.record(self.provider, self.request, payload, api=self.api, agent_id=self.agent_id,
                             session_id=self.session_id, task_id=self.task_id, latency_ms=self._ms())

    def fail(self, status: int, error: str) -> None:
        self.recorder.record_failure(self.provider, self.request, status=status, error=error, api=self.api,
                                     agent_id=self.agent_id, session_id=self.session_id,
                                     task_id=self.task_id, latency_ms=self._ms())


class _Proxy:
    """Stands in for an SDK object: unknown attributes go to it, and isinstance sees its class."""

    def __init__(self, inner):
        object.__setattr__(self, "_pw_inner", inner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_pw_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_pw_inner"), name, value)

    @property
    def __class__(self):  # isinstance(stream, openai.Stream) still holds
        return type(object.__getattribute__(self, "_pw_inner"))

    def __repr__(self):
        return repr(object.__getattribute__(self, "_pw_inner"))


class _StreamTap(_Proxy):
    """The SDK's Stream, item for item; the call is recorded when it ends."""

    def __init__(self, inner, tap: _Tap):
        super().__init__(inner)
        object.__setattr__(self, "_pw_tap", tap)
        tap.hook_close(inner, asynchronous=False)

    def __iter__(self):
        return self

    def __next__(self):
        inner, tap = object.__getattribute__(self, "_pw_inner"), object.__getattribute__(self, "_pw_tap")
        while True:
            tap.reading = True
            try:
                item = next(inner)
            except StopIteration:
                tap.end(True)
                raise
            except BaseException as exc:
                tap.stopped(exc)
                raise
            finally:
                tap.reading = False
            if tap.see(item):
                return item

    def __enter__(self):
        object.__getattribute__(self, "_pw_inner").__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return object.__getattribute__(self, "_pw_inner").__exit__(*exc)
        finally:
            object.__getattribute__(self, "_pw_tap").end(False)

    def close(self):
        try:
            return object.__getattribute__(self, "_pw_inner").close()
        finally:
            object.__getattribute__(self, "_pw_tap").end(False)


class _AsyncStreamTap(_Proxy):
    """The SDK's AsyncStream, item for item; the call is recorded when it ends."""

    def __init__(self, inner, tap: _Tap):
        super().__init__(inner)
        object.__setattr__(self, "_pw_tap", tap)
        tap.hook_close(inner, asynchronous=True)

    def __aiter__(self):
        return self

    async def __anext__(self):
        inner, tap = object.__getattribute__(self, "_pw_inner"), object.__getattribute__(self, "_pw_tap")
        while True:
            tap.reading = True
            try:
                item = await inner.__anext__()
            except StopAsyncIteration:
                tap.end(True)
                raise
            except BaseException as exc:
                tap.stopped(exc)
                raise
            finally:
                tap.reading = False
            if tap.see(item):
                return item

    async def __aenter__(self):
        await object.__getattribute__(self, "_pw_inner").__aenter__()
        return self

    async def __aexit__(self, *exc):
        try:
            return await object.__getattribute__(self, "_pw_inner").__aexit__(*exc)
        finally:
            object.__getattribute__(self, "_pw_tap").end(False)

    async def close(self):
        try:
            return await object.__getattribute__(self, "_pw_inner").close()
        finally:
            object.__getattribute__(self, "_pw_tap").end(False)

    async def aclose(self):
        await self.close()


def _is_stream(obj, name: str) -> bool:
    cls = type(obj)
    return cls.__name__ == name and cls.__module__.split(".")[0] in ("openai", "anthropic")


def _ask_for_usage(kwargs: dict) -> tuple[dict, bool]:
    """OpenAI Chat Completions streams usage only when asked: ask, unless the caller decided.
    A copy of stream_options is sent; the caller's dict is never changed."""
    opts = kwargs.get("stream_options")
    if isinstance(opts, dict) and "include_usage" in opts:
        return kwargs, False
    extra = kwargs.get("extra_body")
    if isinstance(extra, dict) and "stream_options" in extra:  # sent over ours: the caller's choice
        return kwargs, False
    base = opts if isinstance(opts, dict) else {}
    return {**kwargs, "stream_options": {**base, "include_usage": True}}, True


def _refused_usage(exc: Exception, kwargs: dict) -> bool:
    """A server that rejects stream_options (some OpenAI-compatible ones do): 400 naming it. Retried
    only when the caller sent none, so the retry is the caller's own request and can succeed."""
    return (getattr(exc, "status_code", None) in (400, 422) and "stream_options" in str(exc)
            and not isinstance(kwargs.get("stream_options"), dict))


def _raw(kwargs: dict) -> bool:
    """with_raw_response / with_streaming_response: the caller reads the HTTP response itself."""
    headers = kwargs.get("extra_headers")
    return isinstance(headers, dict) and any(str(k).lower() == "x-stainless-raw-response" for k in headers)


def _patch(resource, name: str, provider: str, api: str, recorder: Recorder, agent_id: str | None) -> None:
    original = getattr(resource, name, None)
    if original is None or getattr(original, "_prefixwise", False):
        return
    refused = []  # this server rejected stream_options once: never ask it again

    def note(kwargs, result, started):
        try:
            payload = _as_dict(result)
            if payload is None and type(result).__name__ == "LegacyAPIResponse":
                payload = _as_dict(result.parse())  # with_raw_response: parsed once, cached for the caller
        except Exception:  # noqa: BLE001 - a response we cannot read is simply not recorded
            payload = None
        if payload is not None:
            recorder.record(provider, _request_body(kwargs), payload, api=api, agent_id=agent_id,
                            latency_ms=(time.perf_counter() - started) * 1000)

    def tapped(result, call, injected):
        try:
            if _is_stream(result, "Stream"):
                return _StreamTap(result, _Tap(call, injected))
            if _is_stream(result, "AsyncStream"):
                return _AsyncStreamTap(result, _Tap(call, injected))
        except Exception:  # noqa: BLE001 - not wrapped: the caller gets the SDK's own stream
            pass
        return result  # with_raw_response and other shapes: passed through, unrecorded

    def sent(kwargs):
        """The call's arguments as sent, and whether Prefixwise asked for usage."""
        if kwargs.get("stream") is True and provider == "openai" and api == "chat" and not refused:
            return _ask_for_usage(kwargs)
        return kwargs, False

    async def finish(awaitable, kwargs, started):
        result = await awaitable
        note(kwargs, result, started)
        return result

    # Async SDK methods are often plain functions that return a coroutine (both SDKs
    # decorate `create`), so look at what the call returns, not at how it is declared.
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        if kwargs.get("stream") is True and _raw(kwargs):
            return original(*args, **kwargs)  # the caller reads the raw stream: untouched, unrecorded
        if kwargs.get("stream") is True:
            call = _Call(recorder, provider, api, _request_body(kwargs), agent_id)
            out, injected = sent(kwargs)
            try:
                result = original(*args, **out)
            except Exception as exc:
                if not injected or not _refused_usage(exc, kwargs):
                    raise
                refused.append(True)
                out, injected = kwargs, False
                result = original(*args, **out)
            if inspect.isawaitable(result):
                return _await_stream(result, kwargs, call, injected, original, args, refused, tapped)
            return tapped(result, call, injected)
        result = original(*args, **kwargs)
        if inspect.isawaitable(result):
            return finish(result, kwargs, started)
        note(kwargs, result, started)
        return result
    wrapper._prefixwise = True
    setattr(resource, name, wrapper)


async def _await_stream(awaitable, kwargs, call, injected, original, args, refused, tapped):
    try:
        result = await awaitable
    except Exception as exc:
        if not injected or not _refused_usage(exc, kwargs):
            raise
        refused.append(True)
        injected = False
        result = await original(*args, **kwargs)
    return tapped(result, call, injected)


class _ManagerTap(_Proxy):
    """Anthropic's messages.stream(): a context manager around a MessageStream. The call is
    recorded on exit from the final message the stream accumulated: as completed when the
    message got its stop reason, otherwise (left early) as a failure row."""

    def __init__(self, inner, call: _Call):
        super().__init__(inner)
        object.__setattr__(self, "_pw_call", call)
        object.__setattr__(self, "_pw_stream", None)

    def __enter__(self):
        stream = object.__getattribute__(self, "_pw_inner").__enter__()
        object.__setattr__(self, "_pw_stream", stream)
        return stream

    def __exit__(self, *exc):
        try:
            return object.__getattribute__(self, "_pw_inner").__exit__(*exc)
        finally:
            _record_snapshot(object.__getattribute__(self, "_pw_call"), object.__getattribute__(self, "_pw_stream"),
                             exc[1] if len(exc) > 1 else None)

    async def __aenter__(self):
        stream = await object.__getattribute__(self, "_pw_inner").__aenter__()
        object.__setattr__(self, "_pw_stream", stream)
        return stream

    async def __aexit__(self, *exc):
        try:
            return await object.__getattribute__(self, "_pw_inner").__aexit__(*exc)
        finally:
            _record_snapshot(object.__getattribute__(self, "_pw_call"), object.__getattribute__(self, "_pw_stream"),
                             exc[1] if len(exc) > 1 else None)


def _from_the_stream(exc) -> bool:
    """An error the SDK or its HTTP client raised (an error event mid-stream, a dropped connection),
    not one from the caller's own code in the `with` block, nor a cancellation."""
    return isinstance(exc, Exception) and type(exc).__module__.split(".")[0] in ("anthropic", "httpx", "httpx2", "httpcore")


def _record_snapshot(call: _Call, stream, exc=None) -> None:
    if stream is None:  # the request itself failed: the SDK raised it to the caller
        return
    try:
        try:
            snapshot = stream.current_message_snapshot
        except Exception:  # noqa: BLE001 - no message_start was read
            snapshot = None
        payload = _as_dict(snapshot) if snapshot is not None else None
        if payload and payload.get("stop_reason"):
            call.record(payload)
        elif _from_the_stream(exc):
            call.fail(502, "upstream_dropped")
        else:
            call.fail(499, "client_disconnected")
    except Exception:  # noqa: BLE001 - recording never breaks the caller's stream
        pass


def _patch_stream_helper(resource, provider: str, recorder: Recorder, agent_id: str | None) -> None:
    original = getattr(resource, "stream", None)
    if original is None or getattr(original, "_prefixwise", False):
        return

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        manager = original(*args, **kwargs)
        try:
            return _ManagerTap(manager, _Call(recorder, provider, "chat", _request_body({**kwargs, "stream": True}),
                                              agent_id))
        except Exception:  # noqa: BLE001 - not wrapped: the caller gets the SDK's own manager
            return manager
    wrapper._prefixwise = True
    setattr(resource, "stream", wrapper)


def wrap(client, *, recorder: Recorder | None = None, agent_id: str | None = None):
    """Record every call the client makes, streamed or not. Returns the same client.

    Anthropic / AsyncAnthropic: messages.create and beta.messages.create (stream=True too), and
    messages.stream / beta.messages.stream. OpenAI / AsyncOpenAI (and OpenAI-compatible servers
    through the openai SDK): chat.completions.create / .parse and responses.create / .parse
    (stream=True too), and chat.completions.stream and responses.stream, which call create;
    `with_raw_response` calls too. Not recorded: with_streaming_response, and a streamed call
    made through with_raw_response."""
    module = type(client).__module__.split(".")[0]
    if module not in ("anthropic", "openai"):
        raise TypeError(f"prefixwise.wrap: expected an Anthropic or OpenAI client, got {type(client).__name__}")
    recorder = recorder or default_recorder()
    if module == "anthropic":
        _patch(client.messages, "create", "anthropic", "chat", recorder, agent_id)
        _patch_stream_helper(client.messages, "anthropic", recorder, agent_id)
        beta = getattr(getattr(client, "beta", None), "messages", None)
        if beta is not None:
            _patch(beta, "create", "anthropic", "chat", recorder, agent_id)
            _patch_stream_helper(beta, "anthropic", recorder, agent_id)
    elif module == "openai":
        for name in ("create", "parse"):
            _patch(client.chat.completions, name, "openai", "chat", recorder, agent_id)
        if getattr(client, "responses", None) is not None:
            for name in ("create", "parse"):
                _patch(client.responses, name, "openai", "responses", recorder, agent_id)
    return client
