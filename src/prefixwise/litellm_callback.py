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
LiteLLM's behaviour is unchanged.

LiteLLM hands callbacks every call in OpenAI's format. For Anthropic-backed calls this rebuilds
Anthropic's shape (system prompt apart, usage as uncached + cache reads + cache writes), so the
Anthropic rules and prices apply; everything else is treated as OpenAI-compatible. LiteLLM does
not pass on the 5-minute / 1-hour split of cache writes, so `ttl_shortened` is never claimed
here. LiteLLM can add cache breakpoints itself (`cache_control_injection_points`); a call with
those, or with any cache activity, counts as carrying markers, so "no breakpoint" is never
claimed from a request LiteLLM changed.
"""

from __future__ import annotations

import sys

from .sdk import Recorder

try:  # subclass LiteLLM's base when it is installed; plain object otherwise (tests, import checks)
    from litellm.integrations.custom_logger import CustomLogger as _Base
except Exception:  # noqa: BLE001 - litellm absent or broken: the callback is still importable
    _Base = object


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


class PrefixwiseCallback(_Base):
    def __init__(self, ledger_path: str | None = None, *, recorder: Recorder | None = None,
                 agent_id: str = "litellm"):
        if _Base is not object:
            super().__init__()
        self.recorder = recorder or Recorder(ledger_path, agent_id=agent_id)
        self._warned = False

    # LiteLLM calls these with keyword arguments: the parameter names must be exactly these.
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time)

    def _record(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            if kwargs.get("stream") or kwargs.get("call_type") not in (None, "completion", "acompletion"):
                return
            params = kwargs.get("litellm_params") or {}
            meta = params.get("metadata") or {}
            provider = str(params.get("custom_llm_provider") or kwargs.get("custom_llm_provider") or "")
            response = _dump(response_obj)
            usage = _dump(response.get("usage")) if not isinstance(response.get("usage"), dict) \
                else response["usage"]
            if not usage:
                return
            messages = list(kwargs.get("messages") or [])
            tools = kwargs.get("tools") or (kwargs.get("optional_params") or {}).get("tools")
            model = str(kwargs.get("model") or response.get("model") or "unknown")
            latency = None
            try:
                latency = (end_time - start_time).total_seconds() * 1000
            except Exception:  # noqa: BLE001 - timing is optional
                pass
            session = params.get("litellm_session_id") or meta.get("session_id") or kwargs.get("user")
            common = dict(session_id=session, task_id=meta.get("task_id"), agent_id=meta.get("agent_id"),
                          latency_ms=latency)
            prompt, completion = _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))
            details = usage.get("prompt_tokens_details") or {}
            if provider == "anthropic":
                read = _int(usage.get("cache_read_input_tokens") or details.get("cached_tokens"))
                write = _int(usage.get("cache_creation_input_tokens") or details.get("cache_creation_tokens"))
                system = []
                for m in messages:
                    if m.get("role") == "system":
                        c = m.get("content")
                        system += c if isinstance(c, list) else [{"type": "text", "text": c}]
                body = {"model": model, "messages": [m for m in messages if m.get("role") != "system"]}
                if system:
                    body["system"] = system
                if tools:
                    body["tools"] = tools
                if read or write or kwargs.get("cache_control_injection_points") \
                        or (kwargs.get("optional_params") or {}).get("cache_control_injection_points"):
                    body["cache_control"] = {"type": "ephemeral"}  # markers reached Anthropic
                shaped = {"input_tokens": max(0, prompt - read - write), "cache_read_input_tokens": read,
                          "cache_creation_input_tokens": write, "output_tokens": completion}
                self.recorder.record("anthropic", body, {"id": response.get("id"), "model": model, "usage": shaped},
                                     **common)
            else:
                body = {"model": model, "messages": messages}
                if tools:
                    body["tools"] = tools
                shaped = {"prompt_tokens": prompt, "completion_tokens": completion,
                          "prompt_tokens_details": {"cached_tokens": _int(details.get("cached_tokens"))}}
                self.recorder.record("openai", body, {"id": response.get("id"), "model": model, "usage": shaped},
                                     **common)
        except Exception as exc:  # noqa: BLE001 - never break LiteLLM's logging
            if not self._warned:
                self._warned = True
                print(f"prefixwise: LiteLLM call not recorded ({type(exc).__name__}: {exc})", file=sys.stderr)


def __getattr__(name: str):
    """`prefixwise.litellm_callback.prefixwise_callback`: one shared instance, created on first use
    (for the LiteLLM proxy server's config.yaml)."""
    if name == "prefixwise_callback":
        global prefixwise_callback
        prefixwise_callback = PrefixwiseCallback()
        return prefixwise_callback
    raise AttributeError(name)
