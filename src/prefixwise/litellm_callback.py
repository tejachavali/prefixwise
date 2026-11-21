"""Prefixwise as a LiteLLM callback: every call through LiteLLM (SDK or proxy server), recorded.

    import litellm
    from prefixwise.litellm_callback import PrefixwiseCallback
    litellm.callbacks = [PrefixwiseCallback()]          # ledger: ./prefixwise.db

LiteLLM proxy server (config.yaml):

    litellm_settings:
      callbacks: prefixwise.litellm_callback.prefixwise_callback

Sessions: LiteLLM's own `litellm_session_id`, else `metadata={"session_id": ..., "task_id": ...,
"agent_id": ...}`, else the `user` field. Completed chat completions are recorded, streamed or not
(other call types pass through unrecorded); nothing is sent anywhere and LiteLLM's behaviour is
unchanged. Answers from LiteLLM's own response cache reached no provider and are not recorded.

A stream is recorded once, when LiteLLM has assembled it, with the usage the provider reported.
Claude streams always carry usage. An OpenAI-format stream carries it only when it was asked for: the
LiteLLM proxy server asks by default (`always_include_stream_usage`), and an SDK caller sets
`stream_options={"include_usage": True}`. Where no usage came back, LiteLLM fills in its own estimate
(its tokenizer's count of the messages, with no prompt-token details): such a stream is not
recorded, with one warning saying so, since an estimate with nothing cached would read as a miss
that never happened. Usage counts as the provider's when it carries prompt-token details (OpenAI
reports them even with nothing cached) or Claude's cache fields, or failing those, a prompt count
other than LiteLLM's estimate.

What is classified is the request body LiteLLM actually sent the provider (its
`complete_input_dict`): after its translation to the provider's format, the breakpoints it
injected (`cache_control_injection_points`) and every parameter. Claude, through Anthropic,
Vertex AI or Bedrock's Anthropic format, is classified by Anthropic's rules and prices, with
the 5-minute / 1-hour split of cache writes LiteLLM reports; everything else as
OpenAI-compatible. Like the proxy, only hashes of the prompt are stored, never its text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict

from .otel import provider_name
from .sdk import Recorder

try:  # subclass LiteLLM's base when it is installed; plain object otherwise (tests, import checks)
    from litellm.integrations.custom_logger import CustomLogger as _Base
except Exception:  # noqa: BLE001 - litellm absent or broken: the callback is still importable
    _Base = object

# LiteLLM runs sync success callbacks on a pool of threads, so a conversation's turns can reach
# the callback out of order, and a turn compared with the turn after it reads as rewritten history.
# Turns of one conversation are sequential (the next is sent after the previous answer), so the
# previous turn's callback is due when the next one's arrives: each turn waits at most this long
# for earlier turns of its own conversation (streams included), never for other conversations.
ORDER_WAIT_SECONDS = 2.0
# A call whose callback never comes (an abandoned call, a task LiteLLM dropped) is forgotten
# 2 x ORDER_WAIT_SECONDS after the next turn of its conversation was sent (its answer had come
# by then), or ORDER_FORGET_SECONDS after it was sent (longer than any generation), whichever
# comes first; at most ORDER_MAX_CALLS are tracked (past that, as on a LiteLLM proxy with more
# calls in flight, the oldest stop being ordered: recorded as they come, as before).
ORDER_FORGET_SECONDS = 900.0
ORDER_MAX_CALLS = 10_000
_REDACTED = "redacted-by-litellm"  # what LiteLLM logs in place of messages with logging turned off
# A sync stream's success callback carries no sent body; the one LiteLLM logged before the call is
# kept until the stream ends, for every stream, and let go when it is recorded or fails, or
# ORDER_FORGET_SECONDS after it was sent (a stream abandoned unread sends no callback). At most this
# many are kept; past that the oldest is let go, and its stream is classified from LiteLLM's messages.
STREAM_BODIES_MAX = 256


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


def _conversation(model, messages, kwargs) -> tuple | None:
    """(session, model, opening message): the turns that must be recorded in order. None when
    the call is not recorded (other call types) or has no session."""
    if kwargs.get("call_type") not in (None, "completion", "acompletion"):
        return None
    session = _session_of(kwargs)
    if not session:
        return None
    opening = next((m.get("content") for m in messages or [] if isinstance(m, dict)
                    and m.get("role") not in ("system", "developer")), None)
    digest = hashlib.sha256(_text(opening).encode()).hexdigest()
    return session, str(model), digest


def _sent_body(value) -> dict | None:
    """LiteLLM's `complete_input_dict`: a dict, or (Anthropic streams) the JSON it sent."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _fetches_image(messages) -> bool:
    """Would LiteLLM's token_counter fetch an image to size it? Only for a high-detail image given by
    an http(s) URL; base64 images and auto or low detail are counted locally."""
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        for b in content if isinstance(content, list) else []:
            image = b.get("image_url") if isinstance(b, dict) and b.get("type") == "image_url" else None
            if isinstance(image, dict) and image.get("detail") == "high" \
                    and str(image.get("url") or "").lower().startswith(("http://", "https://")):
                return True
    return False


def _text(content) -> str:
    """A message's content as the same string turn after turn: a string and text blocks with the
    same text agree, and markers (cache_control, which clients move to the last message) are
    ignored."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif isinstance(b, dict):
                parts.append(json.dumps({k: v for k, v in b.items() if k != "cache_control"},
                                        sort_keys=True, default=str))
            else:
                parts.append(str(b))
        return "".join(parts)
    return json.dumps(content, sort_keys=True, default=str)


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
        self._warned_stream_usage = False
        self._stream_bodies: OrderedDict = OrderedDict()  # call id -> the body logged before a stream
        # conversation -> OrderedDict(call id -> when it became due, or None), in send order
        self._turns: dict = {}
        self._key: OrderedDict = OrderedDict()  # call id -> (conversation, send time), oldest first
        self._order = threading.Condition()

    # LiteLLM calls these with keyword arguments: the parameter names must be exactly these.
    def log_pre_api_call(self, model, messages, kwargs):
        key, call = _conversation(model, messages, kwargs), kwargs.get("litellm_call_id")
        now = time.monotonic()
        if call and kwargs.get("stream"):
            body = _sent_body((kwargs.get("additional_args") or {}).get("complete_input_dict"))
            with self._order:
                self._sweep_bodies(now)
                if body is not None:
                    self._stream_bodies[call] = (body, now)
                    while len(self._stream_bodies) > STREAM_BODIES_MAX:
                        self._stream_bodies.popitem(last=False)
        if key is None or not call:
            return
        with self._order:
            turns = self._turns.setdefault(key, OrderedDict())
            for c, due in list(turns.items()):  # this conversation's earlier turns have been answered
                if due is None:
                    turns[c] = now
                elif due < now - 2 * ORDER_WAIT_SECONDS:
                    self._drop(c)
                    self._order.notify_all()
            self._sweep(now)
            self._turns.setdefault(key, OrderedDict())[call] = None
            self._key[call] = (key, now)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            with self._order:
                self._order.wait_for(lambda: self._first(kwargs), timeout=ORDER_WAIT_SECONDS)
            self._record(kwargs, response_obj, start_time, end_time)
        finally:
            self._done(kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:  # on the event loop: wait without blocking it (the earlier turn may be logged by it)
            deadline = time.monotonic() + ORDER_WAIT_SECONDS
            while time.monotonic() < deadline:
                with self._order:
                    if self._first(kwargs):
                        break
                await asyncio.sleep(0.01)
            self._record(kwargs, response_obj, start_time, end_time)
        finally:  # also when LiteLLM cancels the coroutine
            self._done(kwargs)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._done(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._done(kwargs)

    def _first(self, kwargs) -> bool:
        """No earlier turn of this call's conversation is still to be recorded. Caller holds _order."""
        call = kwargs.get("litellm_call_id")
        entry = self._key.get(call)
        turns = self._turns.get(entry[0]) if entry else None
        return not turns or next(iter(turns)) == call

    def _drop(self, call) -> None:
        """Caller holds _order."""
        self._stream_bodies.pop(call, None)
        entry = self._key.pop(call, None)
        if entry is not None:
            turns = self._turns.get(entry[0])
            if turns is not None:
                turns.pop(call, None)
                if not turns:
                    self._turns.pop(entry[0], None)

    def _sweep(self, now: float) -> None:
        """Forget calls sent ORDER_FORGET_SECONDS ago, and keep at most ORDER_MAX_CALLS. Caller
        holds _order."""
        stale = now - ORDER_FORGET_SECONDS
        dropped = False
        while self._key:
            call, (_, sent) = next(iter(self._key.items()))
            if sent >= stale and len(self._key) < ORDER_MAX_CALLS:
                break
            self._drop(call)
            dropped = True
        if dropped:
            self._order.notify_all()

    def _sweep_bodies(self, now: float) -> None:
        """Let go of kept stream bodies sent ORDER_FORGET_SECONDS ago (kept in send order). Caller
        holds _order."""
        stale = now - ORDER_FORGET_SECONDS
        while self._stream_bodies and next(iter(self._stream_bodies.values()))[1] < stale:
            self._stream_bodies.popitem(last=False)

    def _done(self, kwargs) -> None:
        with self._order:
            self._drop(kwargs.get("litellm_call_id"))
            self._order.notify_all()

    def _record(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            if kwargs.get("call_type") not in (None, "completion", "acompletion"):
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
            sent = _sent_body((kwargs.get("additional_args") or {}).get("complete_input_dict"))
            if sent is None and kwargs.get("stream"):
                with self._order:
                    kept = self._stream_bodies.get(kwargs.get("litellm_call_id"))
                sent = kept[0] if kept else None
            messages = list(kwargs.get("messages") or [])
            tools = kwargs.get("tools") or (kwargs.get("optional_params") or {}).get("tools")
            latency = ended = None
            try:
                latency = (end_time - start_time).total_seconds() * 1000
                ended = end_time.timestamp()  # the response ended then, not when this callback runs
            except Exception:  # noqa: BLE001 - timing is optional
                pass
            session = params.get("litellm_session_id") or meta.get("session_id")  # (_session_of adds user)
            common = dict(session_id=session or kwargs.get("user"), session_is_user=not session,
                          task_id=meta.get("task_id"), agent_id=meta.get("agent_id"), latency_ms=latency,
                          ended_at=ended, genai_provider=provider_name(provider) if provider else None)
            returned = str(response.get("model") or model)
            prompt, completion = _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))
            details = usage.get("prompt_tokens_details") or {}
            details = details if isinstance(details, dict) else _dump(details)
            claude = provider == "anthropic" or "claude" in model.lower() \
                or "cache_creation_input_tokens" in usage
            if kwargs.get("stream") and not self._stream_usage_reported(usage, model, messages, prompt, details):
                self._no_stream_usage()
                return
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
                shaped = {"prompt_tokens": prompt, "completion_tokens": completion}
                # only when the backend reported one: LiteLLM in front of Ollama sends no count
                if details or usage.get("cached_tokens") is not None:
                    shaped["prompt_tokens_details"] = {"cached_tokens": _int(details.get("cached_tokens")
                                                                             or usage.get("cached_tokens"))}
                self.recorder.record("openai", body, {"id": response.get("id"), "model": returned, "usage": shaped},
                                     **common)
        except Exception as exc:  # noqa: BLE001 - never break LiteLLM's logging
            if not self._warned:
                self._warned = True
                print(f"prefixwise: LiteLLM call not recorded ({type(exc).__name__}: {exc})", file=sys.stderr)

    @staticmethod
    def _stream_usage_reported(usage: dict, model, messages, prompt: int, details: dict) -> bool:
        """Did the provider report this stream's usage, or is it LiteLLM's estimate? With no usage
        in the stream, LiteLLM counts the messages with its tokenizer and leaves the prompt-token
        details out. The provider's usage has them (OpenAI's, even with nothing cached) or Claude's
        cache fields; a server that reports neither is believed when its count is not the estimate
        (OpenAI's own count can equal LiteLLM's tokenizer, so the count alone cannot decide)."""
        if prompt <= 0:
            return False
        if details or "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
            return True
        # The comparison counts exactly as LiteLLM's estimate does, and stays local: it runs inside
        # LiteLLM's logging, on the event loop for async calls, so nothing may block it. A model whose
        # tokenizer LiteLLM would download from Hugging Face is not counted (its count is taken as
        # reported); a prompt whose count would fetch an image (a high-detail image by URL) is not
        # either, and is not recorded: it cannot be told from an estimate without the fetch.
        if _fetches_image(messages):
            return False
        try:
            from litellm import token_counter
            from litellm.utils import huggingface_tokenizer_kind
            if huggingface_tokenizer_kind(str(model)) is not None:
                return True
            return prompt != token_counter(model=model, messages=messages)
        except Exception:  # noqa: BLE001 - no local tokenizer at hand: take the count as reported
            return True

    def _no_stream_usage(self) -> None:
        if not self._warned_stream_usage:
            self._warned_stream_usage = True
            print("prefixwise: a LiteLLM stream ended without usage from the provider (LiteLLM estimated it), "
                  'so it is not recorded; ask for it with stream_options={"include_usage": True} (the LiteLLM '
                  "proxy server asks by default)", file=sys.stderr)

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
