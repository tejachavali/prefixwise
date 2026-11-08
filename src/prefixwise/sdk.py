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

Streamed calls (stream=True) pass through unrecorded for now.

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

    @property
    def ledger(self):
        return self.proxy.ledger

    def record(self, provider: str, request: dict, response: dict, *, api: str = "chat",
               session_id: str | None = None, agent_id: str | None = None, task_id: str | None = None,
               latency_ms: float | None = None, session_is_user: bool = False,
               genai_provider: str | None = None) -> LedgerEntry | None:
        """provider: "anthropic" or "openai" (any OpenAI-compatible server): whose caching rules
        apply; api: "chat", "responses" (OpenAI Responses) or "chat" for Anthropic Messages.
        session_is_user: session_id is an end-user id, not a conversation id (kept out of
        OpenTelemetry's gen_ai.conversation.id). genai_provider: the actual provider for
        OpenTelemetry (e.g. "aws.bedrock"). Returns the row, or None if recording failed (it
        never raises)."""
        try:
            headers = {"x-prefixwise-agent-id": agent_id or self.agent_id}
            sid = session_id or _session.get()
            tid = task_id or _task.get()
            if sid:
                headers["x-prefixwise-session-id"] = str(sid)
            if tid:
                headers["x-prefixwise-task-id"] = str(tid)
            ctx = None
            ctx = self.proxy.prepare(provider, _plain(request), headers, api=api)
            ctx["session_is_user"] = bool(session_is_user)
            if genai_provider:
                ctx["genai_provider"] = genai_provider
            if latency_ms is not None:  # the call is already over: it was sent latency_ms ago
                ctx["span_start"] = time.time() - latency_ms / 1000
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


def _patch(resource, name: str, provider: str, api: str, recorder: Recorder, agent_id: str | None) -> None:
    original = getattr(resource, name, None)
    if original is None or getattr(original, "_prefixwise", False):
        return

    def note(kwargs, result, started):
        if kwargs.get("stream"):
            return  # streamed: not recorded yet
        try:
            payload = _as_dict(result)
            if payload is None and type(result).__name__ == "LegacyAPIResponse":
                payload = _as_dict(result.parse())  # with_raw_response: parsed once, cached for the caller
        except Exception:  # noqa: BLE001 - a response we cannot read is simply not recorded
            payload = None
        if payload is not None:
            recorder.record(provider, _request_body(kwargs), payload, api=api, agent_id=agent_id,
                            latency_ms=(time.perf_counter() - started) * 1000)

    async def finish(awaitable, kwargs, started):
        result = await awaitable
        note(kwargs, result, started)
        return result

    # Async SDK methods are often plain functions that return a coroutine (both SDKs
    # decorate `create`), so look at what the call returns, not at how it is declared.
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        if inspect.isawaitable(result):
            return finish(result, kwargs, started)
        note(kwargs, result, started)
        return result
    wrapper._prefixwise = True
    setattr(resource, name, wrapper)


def wrap(client, *, recorder: Recorder | None = None, agent_id: str | None = None):
    """Record every non-streamed call the client makes. Returns the same client.

    Anthropic / AsyncAnthropic: messages.create and beta.messages.create. OpenAI / AsyncOpenAI
    (and OpenAI-compatible servers through the openai SDK): chat.completions.create / .parse and
    responses.create / .parse; `with_raw_response` calls too. Not yet: streamed calls, including
    with_streaming_response and the SDKs' .stream() helpers."""
    module = type(client).__module__.split(".")[0]
    if module not in ("anthropic", "openai"):
        raise TypeError(f"prefixwise.wrap: expected an Anthropic or OpenAI client, got {type(client).__name__}")
    recorder = recorder or default_recorder()
    if module == "anthropic":
        _patch(client.messages, "create", "anthropic", "chat", recorder, agent_id)
        beta = getattr(getattr(client, "beta", None), "messages", None)
        if beta is not None:
            _patch(beta, "create", "anthropic", "chat", recorder, agent_id)
    elif module == "openai":
        for name in ("create", "parse"):
            _patch(client.chat.completions, name, "openai", "chat", recorder, agent_id)
        if getattr(client, "responses", None) is not None:
            for name in ("create", "parse"):
                _patch(client.responses, name, "openai", "responses", recorder, agent_id)
    return client
