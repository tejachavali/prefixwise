"""Prefixwise as a LiteLLM callback: every call through LiteLLM (SDK or proxy server), recorded.

    import litellm
    from prefixwise.litellm_callback import PrefixwiseCallback
    litellm.callbacks = [PrefixwiseCallback()]          # ledger: ./prefixwise.db

LiteLLM proxy server (config.yaml):

    litellm_settings:
      callbacks: prefixwise.litellm_callback.prefixwise_callback

Sessions: LiteLLM's own `litellm_session_id`, else `metadata={"session_id": ..., "task_id": ...,
"agent_id": ...}`, else the run of a LiteLLM agent_eval (one attempt at one task by one target: the
run is the session, the task the task, the target the agent; the eval's judge is recorded apart, as
agent `agent_eval_judge` in session `<run>:judge`; read from the call's metadata, or through the
LiteLLM proxy server from `spend_logs_metadata`: agent_eval's own tool loop labels its calls only
through the gateway, its judge both ways), else the `user` field, else, on a Messages API call,
Anthropic's `metadata.user_id` (Claude Code's names its session). Completed chat completions and Anthropic
Messages API calls (`litellm.anthropic.messages`, the proxy server's /v1/messages, Claude Code's
route; the proxy server turns its `x-claude-code-session-id` header into `litellm_session_id`) are
recorded, streamed or not; other call types pass through unrecorded. Through the Messages API,
each call is classified from the body LiteLLM sent: Claude by Anthropic's rules, in Anthropic's
format, on Bedrock Converse (its cachePoints as breakpoints) or as a chat completion with its
cache_control markers (OpenRouter); OpenAI's models as the Responses API calls LiteLLM makes of them;
another model when LiteLLM sent it a chat completion (vLLM); DeepSeek's own Anthropic-compatible API by
the rules of an automatic prefix cache, as DeepSeek caches there (markers are ignored, no writes are
reported). Not another provider's own Anthropic-compatible API (MiniMax's): its cache rules are not
known to be Anthropic's or automatic. Nothing is sent anywhere
and LiteLLM's behaviour is unchanged. Answers from LiteLLM's own response cache reached no provider
and are not recorded.

A stream is recorded once, when LiteLLM has assembled it, with the usage the provider reported.
Claude streams always carry usage. An OpenAI-format stream carries it only when it was asked for: the
LiteLLM proxy server asks by default (`always_include_stream_usage`), and an SDK caller sets
`stream_options={"include_usage": True}`. Where no usage came back, LiteLLM fills in its own estimate
(its tokenizer's count of the messages, with no prompt-token details): such a stream is not
recorded, with one warning saying so, since an estimate with nothing cached would read as a miss
that never happened. Usage counts as the provider's when it carries prompt-token details (OpenAI
reports them even with nothing cached) or Claude's cache fields, or failing those, a prompt count
other than LiteLLM's estimate. Two cases cannot be counted locally: a prompt with a high-detail image
by URL (LiteLLM would fetch it to size it) is not recorded, and a model whose tokenizer LiteLLM
would download (Llama, Cohere) is believed only when usage was asked for; a server that ignores the
request would then have LiteLLM's estimate recorded, and one that sends usage unasked (vLLM with
`--enable-force-include-usage`) has it dropped unless the caller asks.

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
import re
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
# LiteLLM's Anthropic Messages API (`litellm.anthropic.messages`, the proxy server's /v1/messages, the
# route Claude Code uses): its sent body is logged before the call and cleared by the success callback,
# streamed or not, so it is kept like a stream's.
_MESSAGES_CALLS = ("anthropic_messages", "aanthropic_messages")
_RECORDED_CALLS = (None, "completion", "acompletion") + _MESSAGES_CALLS


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


def _eval_labels(kwargs) -> dict:
    """The labels LiteLLM's agent_eval puts on each model call of an evaluation run
    (agent_eval_run, agent_eval_task, agent_eval_target, agent_eval_trial): in the call's metadata,
    or, through the LiteLLM proxy server, under `spend_logs_metadata` (the x-litellm-spend-logs-metadata
    header). Empty when the call is not part of an evaluation."""
    params = kwargs.get("litellm_params") or {}
    for meta in (params.get("metadata"), params.get("litellm_metadata"), kwargs.get("metadata")):
        if not isinstance(meta, dict):
            continue
        for labels in (meta, meta.get("spend_logs_metadata")):
            if isinstance(labels, dict) and labels.get("agent_eval_run"):
                return labels
    return {}


def _eval_session(labels: dict) -> str | None:
    """The run is the session; its judge's calls (agent_eval_stage "judge", grading the attempt) are a
    conversation of their own, never part of the agent's."""
    run = labels.get("agent_eval_run")
    if not run:
        return None
    return f"{run}:judge" if labels.get("agent_eval_stage") == "judge" else run


def _eval_agent(labels: dict) -> str | None:
    if labels.get("agent_eval_stage") == "judge":
        return "agent_eval_judge"  # the grader, not the target: its tokens are not the agent's
    return labels.get("agent_eval_target")


def _user_of(kwargs) -> str | None:
    """The end user: OpenAI's `user`, else, on a Messages API call, Anthropic's `metadata.user_id`
    (Claude Code's names its session)."""
    if kwargs.get("user"):
        return str(kwargs["user"])
    if kwargs.get("call_type") not in _MESSAGES_CALLS:
        return None
    meta = (kwargs.get("litellm_params") or {}).get("metadata") or {}
    user_id = meta.get("user_id") if isinstance(meta, dict) else None
    return user_id if isinstance(user_id, str) and user_id else None


def _session_of(kwargs) -> str | None:
    params = kwargs.get("litellm_params") or {}
    meta = params.get("metadata") or {}
    return (params.get("litellm_session_id") or meta.get("session_id")
            or _eval_session(_eval_labels(kwargs)) or _user_of(kwargs))


def _conversation(model, messages, kwargs) -> tuple | None:
    """(session, model, opening message): the turns that must be recorded in order. None when
    the call is not recorded (other call types) or has no session."""
    if kwargs.get("call_type") not in _RECORDED_CALLS:
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
    an http(s) URL; base64 images and auto or low detail are counted locally. Blocks nest (a
    tool_result holds its own content list, which the counter walks too), so every level is looked at."""
    def walk(blocks, depth=0) -> bool:
        if depth > 16 or not isinstance(blocks, list):
            return False
        for b in blocks:
            if not isinstance(b, dict):
                continue
            image = b.get("image_url") if b.get("type") == "image_url" else None
            if isinstance(image, dict) and image.get("detail") == "high" \
                    and str(image.get("url") or "").lower().startswith(("http://", "https://")):
                return True
            if walk(b.get("content"), depth + 1):
                return True
        return False

    return any(isinstance(m, dict) and walk(m.get("content")) for m in messages or [])


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


def _marked(block: dict, point: dict) -> dict:
    """A Converse cachePoint as Anthropic's cache_control on the block before it."""
    marker = {"type": "ephemeral"}
    if isinstance(point, dict) and point.get("ttl"):
        marker["ttl"] = point["ttl"]
    return {**block, "cache_control": marker}


def _converse_blocks(blocks, before: list | None = None) -> list:
    """Bedrock Converse content blocks as Anthropic's; a cachePoint marks the block before it (in
    `before`, the blocks already converted, when it opens the list)."""
    out: list = []
    for b in blocks if isinstance(blocks, list) else []:
        if not isinstance(b, dict):
            continue
        if "cachePoint" in b:
            if out:
                out[-1] = _marked(out[-1], b["cachePoint"])
            elif before:
                before[-1] = _marked(before[-1], b["cachePoint"])
        elif "reasoningContent" in b:
            r = b["reasoningContent"] or {}
            if "redactedContent" in r:
                out.append({"type": "redacted_thinking", "data": r["redactedContent"]})
            else:
                t = r.get("reasoningText") or {}
                block = {"type": "thinking", "thinking": t.get("text")}
                if t.get("signature") is not None:
                    block["signature"] = t["signature"]
                out.append(block)
        elif "text" in b and len(b) == 1:
            out.append({"type": "text", "text": b["text"]})
        elif "toolUse" in b:
            u = b["toolUse"] or {}
            out.append({"type": "tool_use", "id": u.get("toolUseId"), "name": u.get("name"), "input": u.get("input")})
        elif "toolResult" in b:
            r = b["toolResult"] or {}
            block = {"type": "tool_result", "tool_use_id": r.get("toolUseId"),
                     "content": [{"type": "text", "text": c["text"]} if isinstance(c, dict) and "text" in c and len(c) == 1
                                 else c for c in r.get("content") or []]}
            if r.get("status"):
                block["is_error"] = r["status"] == "error"
            out.append(block)
        else:  # images, documents, guard content: kept as sent, each its own block
            out.append({"type": next(iter(b)), **b})
    return out


def _anthropic_from_converse(sent: dict, model: str) -> dict:
    """The body LiteLLM sent to Bedrock Converse, in Anthropic's format: the same blocks in the same
    order, with each cachePoint as cache_control on the block before it, as Bedrock caches them."""
    body: dict = {"model": model, "messages": []}
    for m in sent.get("messages") or []:
        if isinstance(m, dict):
            previous = body["messages"][-1]["content"] if body["messages"] else None
            body["messages"].append({"role": m.get("role"), "content": _converse_blocks(m.get("content"), previous)})
    system = _converse_blocks(sent.get("system"))
    if system:
        body["system"] = system
    tools: list = []
    for t in (sent.get("toolConfig") or {}).get("tools") or []:
        if not isinstance(t, dict):
            continue
        if "cachePoint" in t:
            if tools:
                tools[-1] = _marked(tools[-1], t["cachePoint"])
        elif "toolSpec" in t:
            spec = t["toolSpec"] or {}
            tools.append({"name": spec.get("name"), "description": spec.get("description"),
                          "input_schema": (spec.get("inputSchema") or {}).get("json")})
        else:  # a system tool: kept as sent
            tools.append({"type": next(iter(t)), **t})
    if tools:
        body["tools"] = tools
    choice = (sent.get("toolConfig") or {}).get("toolChoice")
    if choice:
        body["tool_choice"] = choice
    extra = sent.get("additionalModelRequestFields")  # Anthropic's own fields: thinking, betas
    if isinstance(extra, dict):
        for key, value in extra.items():
            body.setdefault(key, value)
    return body


def _chat_shaped(body: dict) -> bool:
    """A chat completion body: system, developer or tool messages, tool calls, or function tools.
    Anthropic's format has none of them (its system prompt is a top-level field)."""
    if "system" in body or "anthropic_version" in body:
        return False
    for m in body.get("messages") or []:
        if isinstance(m, dict) and (m.get("role") in ("system", "developer", "tool") or "tool_calls" in m):
            return True
    return any(isinstance(t, dict) and (t.get("type") == "function" or "function" in t)
               for t in body.get("tools") or [])


def _converse_shaped(body: dict) -> bool:
    return not _is_anthropic_body(body) or "toolConfig" in body or "inferenceConfig" in body


def _claude_name(model: str) -> str:
    """OpenRouter's model ids (anthropic/claude-sonnet-4.5) as Anthropic names them, so prices,
    minimums and per-model rules apply."""
    name = model.rsplit("/", 1)[-1]
    return re.sub(r"(?<=\d)\.(?=\d)", "-", name) if "claude" in name.lower() else model


def _marker_ttls(body) -> set:
    """The lifetimes the body's cache_control markers ask for (5m when none is given)."""
    found: set = set()
    def walk(x, depth=0):
        if depth > 32:
            return
        if isinstance(x, dict):
            marker = x.get("cache_control")
            if isinstance(marker, dict):
                found.add(marker.get("ttl") or "5m")
            for v in x.values():
                walk(v, depth + 1)
        elif isinstance(x, list):
            for v in x:
                walk(v, depth + 1)
    walk(body)
    return found


def _sent_natively(model: str, provider: str, sent) -> bool:
    """Did LiteLLM send a Messages API call on in Anthropic's format (the provider's own Messages
    API), rather than translate it to a chat completion? LiteLLM's own answer; failing that, the body."""
    try:
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager
        return ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model, provider=LlmProviders(provider)) is not None
    except Exception:  # noqa: BLE001 - another LiteLLM version, or an unknown provider
        return _anthropic_shaped(sent)


def _anthropic_shaped(body) -> bool:
    """Only Anthropic's format would have this: a top-level system prompt or anthropic_version, or
    tool_use / tool_result / thinking blocks. A chat completion body has none of them."""
    if not isinstance(body, dict):
        return False
    if "system" in body or "anthropic_version" in body:
        return True
    for m in body.get("messages") or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") in
                                             ("tool_use", "tool_result", "thinking", "redacted_thinking")
                                             for b in content):
            return True
    return False


def _plain_blocks(value):
    """A block or list of blocks without its cache_control markers (DeepSeek ignores them)."""
    if isinstance(value, dict):
        return {k: _plain_blocks(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_plain_blocks(v) for v in value]
    return value


def _openai_from_anthropic(body: dict, model: str) -> dict:
    """An Anthropic Messages body as a chat completion, block for block in the order sent, for a provider
    whose Messages API caches the way an OpenAI-compatible API does (DeepSeek's). The system prompt becomes
    the system message; tool_use blocks an assistant message's tool calls; tool_result blocks tool
    messages, ahead of the rest of their user message as Anthropic places them; any other block a content
    part (text and images as such, anything else, thinking included, as its JSON), so a change to any of
    them still shows. Markers are dropped."""
    def part(b):
        if not isinstance(b, dict):
            return {"type": "text", "text": json.dumps(b, ensure_ascii=False)}
        if b.get("type") == "text":
            return {"type": "text", "text": b.get("text", "")}
        if b.get("type") == "image" and isinstance(b.get("source"), dict):
            src = b["source"]
            url = src.get("url") or f"data:{src.get('media_type', '')};base64,{src.get('data', '')}"
            return {"type": "image_url", "image_url": {"url": url}}
        return {"type": "text", "text": json.dumps(_plain_blocks(b), ensure_ascii=False)}

    out = []
    system = body.get("system")
    if isinstance(system, str) and system:
        out.append({"role": "system", "content": system})
    elif isinstance(system, list) and system:
        out.append({"role": "system", "content": [part(b) for b in system]})
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, list):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            parts = [part(b) for b in content if not (isinstance(b, dict) and b.get("type") == "tool_use")]
            calls = [{"id": b.get("id"), "type": "function",
                      "function": {"name": b.get("name"),
                                   "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                     for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": parts or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        rest = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                result = b.get("content")
                text = result if isinstance(result, str) else json.dumps(_plain_blocks(result), ensure_ascii=False)
                if b.get("is_error"):
                    text = "[error] " + text
                out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": text})
            else:
                rest.append(part(b))
        if rest:
            out.append({"role": role, "content": rest})
    chat = {"model": model, "messages": out}
    tools = [t for t in body.get("tools") or [] if isinstance(t, dict)]
    if tools:
        chat["tools"] = [{"type": "function", "function": {
            "name": t.get("name"), "description": t.get("description", ""),
            "parameters": _plain_blocks(t.get("input_schema") or {"type": "object"})}}
            if "input_schema" in t or "type" not in t or t.get("type") == "custom"
            else _plain_blocks(t) for t in tools]
    for key in ("tool_choice", "thinking", "response_format"):
        if body.get(key) is not None:
            chat[key] = body[key]
    return chat


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
        if call and (kwargs.get("stream") or kwargs.get("call_type") in _MESSAGES_CALLS):
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
            if kwargs.get("call_type") not in _RECORDED_CALLS:
                return
            messages_api = kwargs.get("call_type") in _MESSAGES_CALLS
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
            if not sent and (kwargs.get("stream") or messages_api):
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
            labels = _eval_labels(kwargs)  # an agent_eval run: one attempt at one task, by one target
            session = (params.get("litellm_session_id") or meta.get("session_id")
                       or _eval_session(labels))  # (_session_of adds user)
            common = dict(session_id=session or _user_of(kwargs), session_is_user=not session,
                          task_id=meta.get("task_id") or labels.get("agent_eval_task"),
                          agent_id=meta.get("agent_id") or _eval_agent(labels), latency_ms=latency,
                          ended_at=ended, genai_provider=provider_name(provider) if provider else None)
            returned = str(response.get("model") or model)
            if "prompt_tokens" not in usage and "input_tokens" in usage:  # Anthropic's own shape
                usage = dict(usage, prompt_tokens=_int(usage.get("input_tokens"))
                             + _int(usage.get("cache_read_input_tokens")) + _int(usage.get("cache_creation_input_tokens")),
                             completion_tokens=_int(usage.get("output_tokens")))
            prompt, completion = _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))
            details = usage.get("prompt_tokens_details") or {}
            details = details if isinstance(details, dict) else _dump(details)
            responses = False
            if messages_api:
                # Classified only from the body LiteLLM sent: the caller's request has no cache
                # breakpoints LiteLLM injected, and on the chat-completion route its system prompt and
                # markers are gone. Claude (by name, or by Anthropic's body on Bedrock or Vertex AI) is
                # judged by Anthropic's rules in whatever format LiteLLM sent it: Anthropic's, Bedrock
                # Converse (translated back block for block, cachePoints as breakpoints), or a chat
                # completion with its cache_control markers (OpenRouter). Another model is recorded when
                # LiteLLM sent it a chat completion, or OpenAI's Responses API (OpenAI's models); not a
                # provider's own Messages API (DeepSeek's, MiniMax's), whose cache rules are not Anthropic's.
                if not isinstance(sent, dict):
                    return
                claude = provider == "anthropic" or "claude" in model.lower() or "anthropic_version" in sent
                listed = isinstance(sent.get("messages"), list)
                if claude:
                    if not listed:
                        return
                    if _converse_shaped(sent):
                        sent = _anthropic_from_converse(sent, model)
                    elif _chat_shaped(sent):  # a chat completion with its cache_control markers (OpenRouter)
                        chat = sent
                        model, returned = _claude_name(model), _claude_name(returned)
                        sent = _anthropic_from_openai(chat["messages"], chat.get("tools"), model)
                        for key in ("tool_choice", "thinking", "reasoning", "response_format"):
                            if chat.get(key) is not None:
                                sent[key] = chat[key]
                    # else Anthropic's own format, as sent
                elif listed:
                    # Recorded only as a chat completion: a Converse body (Nova, Llama: not Anthropic's
                    # cache) or an Anthropic one (a provider's own Messages API) is not, except DeepSeek's,
                    # which caches automatically, whatever the markers say, and reports no writes (probe,
                    # experiments/litellm/runs/2026-11-23-deepseek-probe.md): judged as the
                    # OpenAI-compatible API it caches like, from its reads.
                    if provider == "deepseek" and not _converse_shaped(sent) and (
                            _anthropic_shaped(sent) or _sent_natively(model, provider, sent)):
                        sent = _openai_from_anthropic(sent, model)
                        if not details and usage.get("cache_read_input_tokens") is not None:
                            details = {"cached_tokens": _int(usage.get("cache_read_input_tokens"))}
                    elif _converse_shaped(sent) or _anthropic_shaped(sent) or _sent_natively(model, provider, sent):
                        return
                elif isinstance(sent.get("input"), list):
                    responses = True
                else:
                    return
            else:
                claude = provider == "anthropic" or "claude" in model.lower() or "cache_creation_input_tokens" in usage
            if kwargs.get("stream") and not self._stream_usage_reported(usage, model, messages, prompt, details,
                                                                       kwargs):
                self._no_stream_usage()
                return
            if claude:
                if messages_api or _is_anthropic_body(sent):
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
                if not split and write and _marker_ttls(body) == {"1h"}:  # no split given (OpenRouter)
                    split = {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": write}
                if split:
                    shaped["cache_creation"] = {k: _int(split.get(k)) for k in
                                                ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")}
                self.recorder.record("anthropic", body, {"id": response.get("id"), "model": returned, "usage": shaped},
                                     **common)
            elif responses:  # OpenAI's Responses API, as LiteLLM sent it
                body = dict(sent)
                body.setdefault("model", model)
                shaped = {"input_tokens": prompt, "output_tokens": completion}
                if details or usage.get("cached_tokens") is not None:
                    shaped["input_tokens_details"] = {"cached_tokens": _int(details.get("cached_tokens")
                                                                            or usage.get("cached_tokens"))}
                self.recorder.record("openai", body, {"id": response.get("id"), "model": returned, "usage": shaped},
                                     api="responses", **common)
            else:
                if sent and isinstance(sent.get("messages"), list):
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
    def _stream_usage_reported(usage: dict, model, messages, prompt: int, details: dict, kwargs=None) -> bool:
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
        # LiteLLM's logging, on the event loop for async calls, so nothing may block it. A prompt whose
        # count would fetch an image (a high-detail image by URL) is not counted and not recorded: it
        # cannot be told from an estimate without the fetch. A model whose tokenizer LiteLLM would
        # download from Hugging Face (Llama, Cohere) is not counted either: its count is believed
        # only when usage was asked for (stream_options.include_usage, as the LiteLLM proxy server asks
        # by default), since an OpenAI-format server sends none unasked.
        if _fetches_image(messages):
            return False
        try:
            from litellm import token_counter
            from litellm.utils import huggingface_tokenizer_kind
            if huggingface_tokenizer_kind(str(model)) is not None:
                kwargs = kwargs or {}
                options = kwargs.get("stream_options") or (kwargs.get("optional_params") or {}).get("stream_options")
                return isinstance(options, dict) and options.get("include_usage") is True
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
