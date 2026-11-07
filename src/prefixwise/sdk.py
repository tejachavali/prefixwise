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
        # No fixes: the request recorded must be the request that was sent.
        self.proxy = PrefixwiseProxy(ledger_path=path, mock=True, fixes=())
        self.agent_id = agent_id
        self._warned = False

    @property
    def ledger(self):
        return self.proxy.ledger

    def record(self, provider: str, request: dict, response: dict, *, api: str = "chat",
               session_id: str | None = None, agent_id: str | None = None, task_id: str | None = None,
               latency_ms: float | None = None) -> LedgerEntry | None:
        """provider: "anthropic" or "openai" (any OpenAI-compatible server); api: "chat",
        "responses" (OpenAI Responses) or "chat" for Anthropic Messages. Returns the row, or
        None if recording failed (it never raises)."""
        try:
            headers = {"x-prefixwise-agent-id": agent_id or self.agent_id}
            sid = session_id or _session.get()
            tid = task_id or _task.get()
            if sid:
                headers["x-prefixwise-session-id"] = str(sid)
            if tid:
                headers["x-prefixwise-task-id"] = str(tid)
            ctx = self.proxy.prepare(provider, dict(request), headers, api=api)
            self.proxy.response_started(ctx)
            ctx["upstream_ms"] = latency_ms if latency_ms is not None else 0.0
            _, entry = self.proxy.finalize_safely(ctx, 200, response)
            return entry
        except Exception as exc:  # never break the caller's request
            if not self._warned:
                self._warned = True
                print(f"prefixwise: could not record a call ({type(exc).__name__}: {exc}); "
                      "the call itself was unaffected", file=sys.stderr)
            return None


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
        payload = _as_dict(result)
        if payload is not None:
            recorder.record(provider, kwargs, payload, api=api, agent_id=agent_id,
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

    Anthropic / AsyncAnthropic: messages.create. OpenAI / AsyncOpenAI (and OpenAI-compatible
    servers through the openai SDK): chat.completions.create and responses.create."""
    recorder = recorder or default_recorder()
    module = type(client).__module__.split(".")[0]
    if module == "anthropic":
        _patch(client.messages, "create", "anthropic", "chat", recorder, agent_id)
    elif module == "openai":
        _patch(client.chat.completions, "create", "openai", "chat", recorder, agent_id)
        if getattr(client, "responses", None) is not None:
            _patch(client.responses, "create", "openai", "responses", recorder, agent_id)
    else:
        raise TypeError(f"prefixwise.wrap: expected an Anthropic or OpenAI client, got {type(client).__name__}")
    return client
