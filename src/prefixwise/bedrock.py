"""Claude on Amazon Bedrock, recorded from the AWS SDK.

    import boto3
    import prefixwise

    client = prefixwise.wrap(boto3.client("bedrock-runtime"))
    with prefixwise.session("fix-bug-42"):
        client.converse(modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0", messages=[...])

`converse` and `invoke_model` are recorded when the model is Claude. Each call is judged by
Anthropic's caching rules, as on Anthropic's own API: Bedrock runs Anthropic's models with the same
prefix cache, and a Converse `cachePoint` is a breakpoint on the block before it. Nothing is sent
anywhere. The caller's arguments are not changed, and the return value is the SDK's own, except
that `invoke_model`'s response body is a new StreamingBody with the same bytes (below). Recording
never raises into the call: a failure is reported once and leaves a `not_recorded` row.

- `converse`: the request (messages, system, toolConfig, outputConfig, inferenceConfig,
  additionalModelRequestFields) is read in Anthropic's format, block for block. Bytes (images,
  documents, redacted thinking) are compared by their content. Usage: `inputTokens` is what was not
  read from or written to the cache, as on Anthropic's API; `cacheReadInputTokens` and
  `cacheWriteInputTokens` are read and written. The write is split by lifetime from
  `cacheDetails` when that accounts for all of it; otherwise from the request when every cache
  point asks for the same lifetime; otherwise it is priced as a 5-minute write. When the usage
  carries no cache counts, the write is "not reported", not zero.
- `invoke_model`: a body in Anthropic's format (`anthropic_version`). The response body is a
  stream the caller reads once; Prefixwise reads it to the end and gives the caller a new
  StreamingBody with the same bytes (its socket timeout, with nothing left to read, does nothing).
  If that read fails, every way of reading the body raises the same error for the caller. A body
  passed as a file object is not read and the call is not recorded.

Which model ran: a Claude model id, with or without a region prefix (`us.`, `eu.`, `global.`), or a
system inference profile's ARN, which names it. A prompt router or an application inference
profile names no single model: on `converse` the model a router chose is read from the response's
trace, on `invoke_model` from the response's `model`; without it the call is not recorded.

Not recorded: `converse_stream`, `invoke_model_with_response_stream`, other models, calls that
raise, `converse` under an application inference profile (nothing says which model ran), and
provisioned throughput (a `provisioned-model` ARN), which is billed by the hour, not per token.
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import time

from .converse import _anthropic_from_converse, _converse_blocks

_STOP = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens",
         "stop_sequence": "stop_sequence"}
# anthropic.claude-..., with a region prefix (us., eu., global., ...) or inside an ARN
# (foundation-model/..., inference-profile/..., default-prompt-router/...)
_CLAUDE = re.compile(r"(?:^|[./:])anthropic\.claude", re.I)


def is_bedrock_client(client) -> bool:
    """A boto3 / botocore client for the Bedrock runtime."""
    model = getattr(getattr(client, "meta", None), "service_model", None)
    return getattr(model, "service_name", None) == "bedrock-runtime"


def names_claude(model_id) -> bool:
    return bool(_CLAUDE.search(str(model_id or "")))


def _router(model_id: str) -> bool:
    return "prompt-router/" in model_id


def _provisioned(model_id: str) -> bool:
    return ":provisioned-model/" in model_id


def _jsonable(value):
    """The request with its bytes (images, documents, redacted thinking) as base64 text, as
    Anthropic's API carries them: a changed image or document reads as changed, and its size still
    tells the cache-size estimate that it is not text."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _lifetimes(kwargs: dict) -> set:
    """The lifetimes the request's cache points ask for ("5m" when none is given). Only blocks are
    looked at, where a cache point can be: the system blocks, each message's content blocks and
    the tools (not inside a tool's input or schema)."""
    blocks = list(kwargs.get("system") or [])
    for m in kwargs.get("messages") or []:
        if isinstance(m, dict):
            blocks.extend(m.get("content") or [])
    blocks.extend((kwargs.get("toolConfig") or {}).get("tools") or [])
    found = set()
    for b in blocks:
        point = b.get("cachePoint") if isinstance(b, dict) else None
        if isinstance(point, dict):
            found.add(point.get("ttl") or "5m")
    return found


def anthropic_usage(usage: dict, lifetimes: set | None = None) -> dict:
    """Bedrock usage in Anthropic's shape. `inputTokens` excludes what was read from and written
    to the cache, as Anthropic's `input_tokens` does (Converse's names, and InvokeModel's `*Count`
    names). The write count is carried only when Bedrock sent one, so a request without cache
    points is "not reported", not "wrote 0". The write is split by lifetime from `cacheDetails`
    when that adds up to it; else, when `lifetimes` (what the request's cache points asked for) is
    one lifetime, all of it is that one."""
    out = {"input_tokens": int(usage.get("inputTokens") or 0), "output_tokens": int(usage.get("outputTokens") or 0)}
    read = usage.get("cacheReadInputTokens", usage.get("cacheReadInputTokenCount"))
    write = usage.get("cacheWriteInputTokens", usage.get("cacheWriteInputTokenCount"))
    if read is not None:
        out["cache_read_input_tokens"] = int(read)
    if write is None:
        return out
    write = int(write)
    out["cache_creation_input_tokens"] = write
    by = None
    details = usage.get("cacheDetails")
    if isinstance(details, list) and details:
        split = {"5m": 0, "1h": 0}
        for d in details:
            if isinstance(d, dict) and d.get("ttl") in split:
                split[d["ttl"]] += int(d.get("inputTokens") or 0)
        if split["5m"] + split["1h"] == write:  # a split that does not add up is no split
            by = split
    if by is None and lifetimes and len(lifetimes) == 1 and next(iter(lifetimes)) in ("5m", "1h"):
        ttl = next(iter(lifetimes))
        by = {"5m": write if ttl == "5m" else 0, "1h": write if ttl == "1h" else 0}
    if by is not None:
        out["cache_creation"] = {"ephemeral_5m_input_tokens": by["5m"], "ephemeral_1h_input_tokens": by["1h"]}
    return out


def converse_request(kwargs: dict, model: str | None = None) -> dict:
    """A Converse request in Anthropic's format, as Bedrock caches it."""
    kwargs = _jsonable(kwargs)
    body = _anthropic_from_converse(kwargs, str(model or kwargs.get("modelId") or ""))
    config = kwargs.get("inferenceConfig") or {}
    if isinstance(config, dict):
        for theirs, ours in (("maxTokens", "max_tokens"), ("temperature", "temperature"), ("topP", "top_p"),
                             ("stopSequences", "stop_sequences")):
            if config.get(theirs) is not None:
                body.setdefault(ours, config[theirs])
    output = kwargs.get("outputConfig")
    if isinstance(output, dict) and output:  # effort and output format: Anthropic's output_config
        theirs = {}
        if output.get("effort") is not None:
            theirs["effort"] = output["effort"]
        if output.get("textFormat") is not None:
            theirs["format"] = output["textFormat"]
        if theirs:
            body["output_config"] = {**(body.get("output_config") or {}), **theirs}
    return body


def converse_response(response: dict, model: str, lifetimes: set | None = None) -> dict | None:
    """A Converse response in Anthropic's format; None when it carries no usage."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    message = ((response.get("output") or {}).get("message") or {})
    meta = response.get("ResponseMetadata") or {}
    return {"id": meta.get("RequestId") or f"bedrock-{time.time_ns()}", "type": "message", "role": "assistant",
            "model": model, "content": _converse_blocks(_jsonable(message.get("content"))),
            "stop_reason": _STOP.get(response.get("stopReason"), response.get("stopReason")),
            "usage": anthropic_usage(usage, lifetimes)}


def converse_model(model_id: str, response: dict) -> str | None:
    """The Claude model a converse call ran on, or None when it cannot be told or is not Claude.
    (A provisioned-model or application-inference-profile ARN names no model: None.)"""
    if _router(model_id):
        invoked = ((response.get("trace") or {}).get("promptRouter") or {}).get("invokedModelId")
        return str(invoked) if invoked and names_claude(invoked) else None
    return model_id if names_claude(model_id) else None


def _streaming_body(data: bytes):
    """A botocore StreamingBody of bytes already read. Its socket timeout has nothing left to
    bound: setting it does nothing."""
    from botocore.response import StreamingBody

    class ReadBody(StreamingBody):
        def set_socket_timeout(self, timeout):
            pass

    return ReadBody(io.BytesIO(data), len(data))


class _Raising(io.RawIOBase):
    """A raw stream every read of which raises one error."""

    def __init__(self, exc: BaseException):
        super().__init__()
        self._exc = exc

    def readable(self):
        return True

    def readinto(self, b):
        raise self._exc

    def read(self, size=-1):
        raise self._exc

    def readline(self, size=-1):
        raise self._exc

    def readlines(self, hint=-1):
        raise self._exc


def _failed_body(exc: BaseException, original):
    """A StreamingBody every read of which (read, readinto, readlines, iteration, a `with` block)
    raises what Prefixwise's read of the original raised: the caller meets the same error, when it
    reads, as it would have. Closing it closes the original."""
    from botocore.response import StreamingBody

    class FailedBody(StreamingBody):
        def set_socket_timeout(self, timeout):
            pass

        def close(self):
            try:
                original.close()
            finally:
                super().close()

    return FailedBody(_Raising(exc), 0)


def _warn(recorder, exc) -> None:
    if not getattr(recorder, "_bedrock_warned", False):
        recorder._bedrock_warned = True
        print(f"prefixwise: could not record a Bedrock call ({type(exc).__name__}: {exc})", file=sys.stderr)


def _not_recorded(recorder, model: str, agent_id, started) -> None:
    """A call that could not be read is still a row, so error rates stay honest. Never raises."""
    try:
        recorder.record_failure("anthropic", {"model": model, "messages": []}, status=200, error="not_recorded",
                                agent_id=agent_id, latency_ms=(time.perf_counter() - started) * 1000)
    except Exception:  # noqa: BLE001
        pass


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
        model = None
        try:
            if isinstance(result, dict):
                model = converse_model(str(kwargs.get("modelId") or ""), result)
            if model is not None:
                lifetimes = _lifetimes(kwargs)
                response = converse_response(result, model, lifetimes)
                if response is not None:
                    recorder.record("anthropic", converse_request(kwargs, model), response, api="chat",
                                    agent_id=agent_id, latency_ms=(time.perf_counter() - started) * 1000,
                                    genai_provider="aws.bedrock")
        except Exception as exc:  # noqa: BLE001 - recording never breaks the call
            _warn(recorder, exc)
            _not_recorded(recorder, model or str(kwargs.get("modelId") or ""), agent_id, started)
        return result
    converse.__doc__ = getattr(original, "__doc__", None)
    return converse


def _patch_invoke(original, recorder, agent_id):
    def invoke_model(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        model = str(kwargs.get("modelId") or "")
        try:
            raw = kwargs.get("body")
            if (isinstance(raw, (bytes, bytearray, str)) and isinstance(result, dict)
                    and result.get("body") is not None and not _provisioned(model)):
                body = json.loads(raw)
                if isinstance(body, dict) and "anthropic_version" in body and isinstance(body.get("messages"), list):
                    stream = result["body"]
                    try:
                        data = stream.read()
                    except Exception as exc:  # the caller would have met this reading it: it still will
                        result["body"] = _failed_body(exc, stream)
                        raise
                    result["body"] = _streaming_body(data)
                    try:
                        stream.close()
                    except Exception:  # noqa: BLE001 - read to its end already
                        pass
                    latency = (time.perf_counter() - started) * 1000
                    response = json.loads(data)
                    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                        # A model id that names no single model (an application inference profile,
                        # a prompt router) is judged and priced by the model the response names.
                        if _router(model) or not names_claude(model):
                            if not str(response.get("model") or "").lower().startswith("claude"):
                                return result
                            model = str(response["model"])
                        request = {**body, "model": model}
                        response.setdefault("model", model)
                        recorder.record("anthropic", request, response, api="chat", agent_id=agent_id,
                                        latency_ms=latency, genai_provider="aws.bedrock")
        except Exception as exc:  # noqa: BLE001 - recording never breaks the call
            _warn(recorder, exc)
            _not_recorded(recorder, model, agent_id, started)
        return result
    invoke_model.__doc__ = getattr(original, "__doc__", None)
    return invoke_model
