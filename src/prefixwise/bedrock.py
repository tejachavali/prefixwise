"""Claude on Amazon Bedrock, recorded from the AWS SDK.

    import boto3
    import prefixwise

    client = prefixwise.wrap(boto3.client("bedrock-runtime"))
    with prefixwise.session("fix-bug-42"):
        client.converse(modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0", messages=[...])

`converse` and `invoke_model` are recorded when the model is Claude. Each call is judged by
Anthropic's caching rules, as on Anthropic's own API: Bedrock runs Anthropic's models with the same
prefix cache, and a Converse `cachePoint` is a breakpoint on the block before it. Nothing is sent
anywhere and the call and its return value are the SDK's own. Recording can never break a call: a
failure is reported once and ignored.

- `converse`: the request (messages, system, toolConfig, additionalModelRequestFields) is read in
  Anthropic's format, block for block. Usage: `inputTokens` is what was not read from or written
  to the cache, as on Anthropic's API; `cacheReadInputTokens` and `cacheWriteInputTokens` are read
  and written, and `cacheDetails` splits the write by lifetime when it accounts for all of it.
  When the usage carries no cache counts (no cachePoint), the write is "not reported", not zero.
- `invoke_model`: a body in Anthropic's format (`anthropic_version`). The response body is a
  stream the caller reads once; Prefixwise reads it and gives the caller a new one with the same
  bytes. A body passed as a file object is not read and the call is not recorded.

Claude is told by the model id: `anthropic.` or `claude` in it. An application inference profile
(an ARN with neither) is recorded on `invoke_model`, where the body says it is Anthropic's and the
response names the model, and not on `converse`, where nothing says which model ran.

Not recorded: `converse_stream`, `invoke_model_with_response_stream`, other models, and calls that
raise.
"""

from __future__ import annotations

import io
import json
import sys
import time

from .litellm_callback import _anthropic_from_converse, _converse_blocks

_STOP = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens",
         "stop_sequence": "stop_sequence"}


def is_bedrock_client(client) -> bool:
    """A boto3 / botocore client for the Bedrock runtime."""
    model = getattr(getattr(client, "meta", None), "service_model", None)
    return getattr(model, "service_name", None) == "bedrock-runtime"


def _claude(model_id) -> bool:
    m = str(model_id or "").lower()
    return "anthropic." in m or "claude" in m


def anthropic_usage(usage: dict) -> dict:
    """Bedrock Converse usage in Anthropic's shape. `inputTokens` excludes what was read from and
    written to the cache, as Anthropic's `input_tokens` does. The write count is carried only when
    Bedrock sent one, so a request without cache points is "not reported", not "wrote 0"."""
    out = {"input_tokens": int(usage.get("inputTokens") or 0), "output_tokens": int(usage.get("outputTokens") or 0)}
    read = usage.get("cacheReadInputTokens", usage.get("cacheReadInputTokenCount"))
    write = usage.get("cacheWriteInputTokens", usage.get("cacheWriteInputTokenCount"))
    if read is not None:
        out["cache_read_input_tokens"] = int(read)
    if write is not None:
        out["cache_creation_input_tokens"] = int(write)
        details = usage.get("cacheDetails")
        if isinstance(details, list) and details:
            by = {"5m": 0, "1h": 0}
            for d in details:
                if isinstance(d, dict) and d.get("ttl") in by:
                    by[d["ttl"]] += int(d.get("inputTokens") or 0)
            if by["5m"] + by["1h"] == int(write):  # a split that does not add up is no split
                out["cache_creation"] = {"ephemeral_5m_input_tokens": by["5m"], "ephemeral_1h_input_tokens": by["1h"]}
    return out


def converse_request(kwargs: dict) -> dict:
    """A Converse request in Anthropic's format, as Bedrock caches it."""
    model = str(kwargs.get("modelId") or "")
    body = _anthropic_from_converse(kwargs, model)
    config = kwargs.get("inferenceConfig") or {}
    if isinstance(config, dict):
        for theirs, ours in (("maxTokens", "max_tokens"), ("temperature", "temperature"), ("topP", "top_p"),
                             ("stopSequences", "stop_sequences")):
            if config.get(theirs) is not None:
                body.setdefault(ours, config[theirs])
    return body


def converse_response(response: dict, model: str) -> dict | None:
    """A Converse response in Anthropic's format; None when it carries no usage."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    message = ((response.get("output") or {}).get("message") or {})
    meta = response.get("ResponseMetadata") or {}
    return {"id": meta.get("RequestId") or f"bedrock-{time.time_ns()}", "type": "message", "role": "assistant",
            "model": model, "content": _converse_blocks(message.get("content")),
            "stop_reason": _STOP.get(response.get("stopReason"), response.get("stopReason")),
            "usage": anthropic_usage(usage)}


class _Body:
    """The response body handed back after Prefixwise read the original: the same bytes, with the
    methods callers use on botocore's StreamingBody."""

    def __init__(self, data: bytes, original):
        try:
            from botocore.response import StreamingBody
            self._b = StreamingBody(io.BytesIO(data), len(data))
        except Exception:  # noqa: BLE001 - botocore absent or changed: a plain reader still works
            self._b = io.BytesIO(data)
        self._original = original

    def __getattr__(self, name):
        return getattr(self._b, name)

    def __iter__(self):
        return iter(self._b)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self._b.close()
        finally:
            close = getattr(self._original, "close", None)
            if callable(close):
                close()


class _Failed:
    """A body whose read failed while Prefixwise read it: the caller meets the same error, when it
    reads, as it would have without Prefixwise."""

    def __init__(self, exc: BaseException, original):
        self._exc, self._original = exc, original

    def read(self, *args, **kwargs):
        raise self._exc

    iter_chunks = iter_lines = readlines = read

    def __iter__(self):
        raise self._exc

    def close(self):
        close = getattr(self._original, "close", None)
        if callable(close):
            close()


def _warn(recorder, exc) -> None:
    if not getattr(recorder, "_warned", False):
        recorder._warned = True
        print(f"prefixwise: could not record a Bedrock call ({type(exc).__name__}: {exc}); "
              "the call itself was unaffected", file=sys.stderr)


def wrap_bedrock(client, recorder, agent_id: str | None = None):
    """Record the client's `converse` and `invoke_model` calls to Claude. Returns the same client."""
    for name, patch in (("converse", _patch_converse), ("invoke_model", _patch_invoke)):
        original = getattr(client, name, None)
        if original is not None and not getattr(original, "_prefixwise", False):
            wrapper = patch(original, recorder, agent_id)
            wrapper._prefixwise = True
            setattr(client, name, wrapper)
    return client


def _patch_converse(original, recorder, agent_id):
    def converse(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        try:
            model = str(kwargs.get("modelId") or "")
            if _claude(model) and isinstance(result, dict):
                response = converse_response(result, model)
                if response is not None:
                    recorder.record("anthropic", converse_request(kwargs), response, api="chat", agent_id=agent_id,
                                    latency_ms=(time.perf_counter() - started) * 1000, genai_provider="aws.bedrock")
        except Exception as exc:  # noqa: BLE001 - recording never breaks the call
            _warn(recorder, exc)
        return result
    converse.__doc__ = getattr(original, "__doc__", None)
    return converse


def _patch_invoke(original, recorder, agent_id):
    def invoke_model(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        try:
            raw = kwargs.get("body")
            if isinstance(raw, (bytes, bytearray, str)) and isinstance(result, dict) and result.get("body") is not None:
                body = json.loads(raw)
                model = str(kwargs.get("modelId") or "")
                if isinstance(body, dict) and "anthropic_version" in body and isinstance(body.get("messages"), list):
                    stream = result["body"]
                    try:
                        data = stream.read()
                    except Exception as exc:  # the caller would have met this reading it: it still will
                        result["body"] = _Failed(exc, stream)
                        raise
                    result["body"] = _Body(data, stream)
                    latency = (time.perf_counter() - started) * 1000
                    response = json.loads(data)
                    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                        request = {**body, "model": model}
                        request.pop("anthropic_version", None)
                        response.setdefault("model", model)
                        recorder.record("anthropic", request, response, api="chat", agent_id=agent_id,
                                        latency_ms=latency, genai_provider="aws.bedrock")
        except Exception as exc:  # noqa: BLE001 - recording never breaks the call
            _warn(recorder, exc)
        return result
    invoke_model.__doc__ = getattr(original, "__doc__", None)
    return invoke_model
