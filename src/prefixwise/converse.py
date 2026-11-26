"""Amazon Bedrock's Converse format read as Anthropic's Messages format, block for block, as Bedrock
caches it: a `cachePoint` is a breakpoint (cache_control) on the block before it. Used by the
LiteLLM callback and by prefixwise.bedrock. Imports nothing outside the standard library."""

from __future__ import annotations


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
