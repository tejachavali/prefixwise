"""Prefixwise as a LiteLLM callback: every call through LiteLLM (SDK or proxy server), recorded.

    import litellm
    from prefixwise.litellm_callback import PrefixwiseCallback
    litellm.callbacks = [PrefixwiseCallback()]          # ledger: ./prefixwise.db

LiteLLM proxy server (config.yaml):

    litellm_settings:
      callbacks: prefixwise.litellm_callback.prefixwise_callback

Sessions: LiteLLM's own `litellm_session_id`, else `metadata={"session_id": ..., "task_id": ...,
"agent_id": ...}`, else the `user` field. Completed, non-streamed chat completions are recorded
(streams and other call types pass through unrecorded for now); nothing is sent anywhere and
LiteLLM's behaviour is unchanged. Answers from LiteLLM's own response cache reached no provider
and are not recorded.

What is classified is the request body LiteLLM actually sent the provider (its
`complete_input_dict`): after its translation to the provider's format, the breakpoints it
injected (`cache_control_injection_points`) and every parameter. Claude, through Anthropic,
Vertex AI or Bedrock's Anthropic format, is classified by Anthropic's rules and prices, with
the 5-minute / 1-hour split of cache writes LiteLLM reports; everything else as
OpenAI-compatible. Like the proxy, only hashes of the prompt are stored, never its text.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time

from .otel import provider_name
from .sdk import Recorder

try:  # subclass LiteLLM's base when it is installed; plain object otherwise (tests, import checks)
    from litellm.integrations.custom_logger import CustomLogger as _Base
except Exception:  # noqa: BLE001 - litellm absent or broken: the callback is still importable
    _Base = object

# LiteLLM runs sync success callbacks on a pool of threads, so one session's calls can reach the
# callback out of order, and a turn compared with the turn after it reads as rewritten history.
# Calls are recorded in the order they were sent (the pre-call hook runs on the caller's thread,
# in send order); a call waits at most this long for the session's earlier calls.
ORDER_WAIT_SECONDS = 5.0
_REDACTED = "redacted-by-litellm"  # what LiteLLM logs in place of messages with logging turned off


def _dump(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else {}


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _session_of(kwargs) -> str | None:
    params = kwargs.get("litellm_params") or {}
    meta = params.get("metadata") or {}
    return params.get("litellm_session_id") or meta.get("session_id") or kwargs.get("user")


def _is_anthropic_body(body) -> bool:
    """Anthropic Messages format: typed content blocks (Bedrock's Converse format has none)."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return False
    for m in body["messages"]:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list) and any(not isinstance(b, dict) or "type" not in b for b in content):
            return False
    return True


def _anthropic_from_openai(messages: list, tools, model: str) -> dict:
    """No sent body in hand: rebuild Anthropic's format from LiteLLM's OpenAI-format messages."""
    system = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content")
            system += c if isinstance(c, list) else [{"type": "text", "text": c}]
    rest = [m for m in messages if m.get("role") != "system"]
    try:  # LiteLLM's own translation: tool calls and results become tool_use / tool_result blocks
        from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
        rest = anthropic_messages_pt(rest, model=model, llm_provider="anthropic")
    except Exception:  # noqa: BLE001 - keep the messages as they are
        pass
    body = {"model": model, "messages": rest}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    return body


class PrefixwiseCallback(_Base):
    def __init__(self, ledger_path: str | None = None, *, recorder: Recorder | None = None,
                 agent_id: str = "litellm"):
        if _Base is not object:
            super().__init__()
        self.recorder = recorder or Recorder(ledger_path, agent_id=agent_id)
        self._warned = False
        self._warned_redacted = False
        self._sent: dict = {}  # session -> {call id: send time}, in send order
        self._order = threading.Condition()

    # LiteLLM calls these with keyword arguments: the parameter names must be exactly these.
    def log_pre_api_call(self, model, messages, kwargs):
        session, call = _session_of(kwargs), kwargs.get("litellm_call_id")
        if session and call:
            with self._order:
                self._sent.setdefault(session, {})[call] = time.monotonic()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        with self._order:
            self._order.wait_for(lambda: self._first(kwargs), timeout=ORDER_WAIT_SECONDS)
        self._settled(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # On the event loop: wait without blocking it (the earlier call may be logged by it).
        deadline = time.monotonic() + ORDER_WAIT_SECONDS
        while time.monotonic() < deadline:
            with self._order:
                if self._first(kwargs):
                    break
            await asyncio.sleep(0.005)
        self._settled(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._done(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._done(kwargs)

    def _first(self, kwargs) -> bool:
        """No earlier call of this session is still waiting to be recorded. Caller holds _order."""
        sent = self._sent.get(_session_of(kwargs))
        call = kwargs.get("litellm_call_id")
        if not sent or call not in sent:
            return True
        stale = time.monotonic() - 2 * ORDER_WAIT_SECONDS
        for c in [c for c, t in sent.items() if t < stale and c != call]:
            sent.pop(c)  # never completed (a callback LiteLLM dropped): stop waiting for it
        return next(iter(sent)) == call

    def _done(self, kwargs) -> None:
        session, call = _session_of(kwargs), kwargs.get("litellm_call_id")
        with self._order:
            sent = self._sent.get(session)
            if sent is not None:
                sent.pop(call, None)
                if not sent:
                    self._sent.pop(session, None)
            self._order.notify_all()

    def _settled(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            self._record(kwargs, response_obj, start_time, end_time)
        finally:
            self._done(kwargs)

    def _record(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            if kwargs.get("stream") or kwargs.get("call_type") not in (None, "completion", "acompletion"):
                return
            if kwargs.get("cache_hit"):
                return  # answered from LiteLLM's response cache: nothing reached a provider
            params = kwargs.get("litellm_params") or {}
            meta = params.get("metadata") or {}
            provider = str(params.get("custom_llm_provider") or kwargs.get("custom_llm_provider") or "")
            response = _dump(response_obj)
            usage = response.get("usage")
            usage = usage if isinstance(usage, dict) else _dump(usage)
            if not usage:
                return
            model = str(kwargs.get("model") or response.get("model") or "unknown")
            if provider and model.startswith(provider + "/"):
                model = model[len(provider) + 1:]
            sent = (kwargs.get("additional_args") or {}).get("complete_input_dict")
            sent = dict(sent) if isinstance(sent, dict) else None
            messages = list(kwargs.get("messages") or [])
            tools = kwargs.get("tools") or (kwargs.get("optional_params") or {}).get("tools")
            latency = None
            try:
                latency = (end_time - start_time).total_seconds() * 1000
            except Exception:  # noqa: BLE001 - timing is optional
                pass
            session = params.get("litellm_session_id") or meta.get("session_id")  # (_session_of adds user)
            common = dict(session_id=session or kwargs.get("user"), session_is_user=not session,
                          task_id=meta.get("task_id"), agent_id=meta.get("agent_id"), latency_ms=latency,
                          genai_provider=provider_name(provider) if provider else None)
            returned = str(response.get("model") or model)
            prompt, completion = _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))
            details = usage.get("prompt_tokens_details") or {}
            details = details if isinstance(details, dict) else _dump(details)
            claude = provider == "anthropic" or "claude" in model.lower() \
                or "cache_creation_input_tokens" in usage
            if claude:
                if _is_anthropic_body(sent):
                    body = sent
                else:
                    if self._redacted(messages):
                        return
                    body = _anthropic_from_openai(messages, tools, model)
                body.setdefault("model", model)  # Bedrock and Vertex carry the model in the URL
                read = _int(usage.get("cache_read_input_tokens") or details.get("cached_tokens"))
                write = _int(usage.get("cache_creation_input_tokens") or details.get("cache_creation_tokens"))
                # LiteLLM's prompt_tokens for Claude includes cache reads and writes.
                shaped = {"input_tokens": max(0, prompt - read - write), "cache_read_input_tokens": read,
                          "cache_creation_input_tokens": write, "output_tokens": completion}
                split = details.get("cache_creation_token_details")
                split = split if isinstance(split, dict) else _dump(split)
                if split:
                    shaped["cache_creation"] = {k: _int(split.get(k)) for k in
                                                ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")}
                self.recorder.record("anthropic", body, {"id": response.get("id"), "model": returned, "usage": shaped},
                                     **common)
            else:
                if sent is not None and isinstance(sent.get("messages"), list):
                    body = sent
                else:
                    if self._redacted(messages):
                        return
                    body = {"model": model, "messages": messages}
                    if tools:
                        body["tools"] = tools
                body.setdefault("model", model)
                shaped = {"prompt_tokens": prompt, "completion_tokens": completion,
                          "prompt_tokens_details": {"cached_tokens": _int(details.get("cached_tokens"))}}
                self.recorder.record("openai", body, {"id": response.get("id"), "model": returned, "usage": shaped},
                                     **common)
        except Exception as exc:  # noqa: BLE001 - never break LiteLLM's logging
            if not self._warned:
                self._warned = True
                print(f"prefixwise: LiteLLM call not recorded ({type(exc).__name__}: {exc})", file=sys.stderr)

    def _redacted(self, messages: list) -> bool:
        """With turn_off_message_logging and no sent body, every request would look the same."""
        if not any(isinstance(m, dict) and m.get("content") == _REDACTED for m in messages):
            return False
        if not self._warned_redacted:
            self._warned_redacted = True
            print("prefixwise: LiteLLM redacted these messages (turn_off_message_logging) and passed on no "
                  "request body; such calls are not recorded", file=sys.stderr)
        return True


def __getattr__(name: str):
    """`prefixwise.litellm_callback.prefixwise_callback`: one shared instance, created on first use
    (for the LiteLLM proxy server's config.yaml)."""
    if name == "prefixwise_callback":
        global prefixwise_callback
        prefixwise_callback = PrefixwiseCallback()
        return prefixwise_callback
    raise AttributeError(name)
