"""A test provider with a content-addressed prompt cache, for testing Prefixwise through LiteLLM.

Real providers cache by prompt content, not by session: a request reads the longest prefix of
itself that an earlier request left in the cache. This mock does the same, deterministically, and
updates its cache when the request arrives (no background recording, so no races):

- The prompt is rendered as blocks in the provider's order: tools, then system, then messages
  (each message, and each content block of a message, is a block). A block's "tokens" are its
  canonical JSON length / 4.
- OpenAI format (/v1/chat/completions): automatic caching. Every prefix ending at a block boundary
  is cached once the request is served; a read needs a cached prefix of at least 1,024 tokens.
- Anthropic format (/v1/messages): only prefixes ending at a block that carries cache_control are
  written (and billed as cache_creation_input_tokens); a read is the longest cached such prefix.
  Claude by any other route caches the same way: in chat format (OpenRouter: model name with
  "claude", usage with prompt_tokens_details.cache_write_tokens), and Bedrock Converse
  (/model/<id>/converse, a cachePoint marks the block before it; not streamed).
- OpenAI Responses API (/v1/responses): automatic, as chat completions; streamed or not.
- Streams: OpenAI sends a final usage chunk only when stream_options.include_usage is true;
  Anthropic sends usage in message_start and message_delta, as the real APIs do.
- Optional tool calls (--tool-turns N): answer the first N turns of a conversation that offers
  tools with a call to the first tool, so tool loops run.

Usage: python cachemock.py PORT [--tool-turns N]
"""

import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
CACHE: dict[str, int] = {}   # prefix hash -> tokens in that prefix
TOOL_TURNS = 0
OPENAI_MIN = 1024


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def strip_marker(x):
    """cache_control is a marker, not content: it does not change what is cached."""
    if isinstance(x, dict):
        return {k: strip_marker(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [strip_marker(v) for v in x]
    return x


def converse_blocks(body: dict) -> list[tuple[str, int, bool]]:
    """Bedrock Converse: tools, system, then message content; a cachePoint marks the block before it."""
    out = []
    def walk(items, tag):
        for b in items or []:
            if "cachePoint" in b:
                if out:
                    out[-1] = (out[-1][0], out[-1][1], True)
                continue
            text = tag + canon(b)
            out.append((text, max(1, len(text) // 4), False))
    walk((body.get("toolConfig") or {}).get("tools"), "tool:")
    walk(body.get("system"), "system:")
    for i, m in enumerate(body.get("messages") or []):
        walk(m.get("content"), f"{m.get('role')}:")
    return out


def responses_blocks(body: dict) -> list[tuple[str, int, bool]]:
    out = []
    for t in body.get("tools") or []:
        out.append(("tool:" + canon(t), 0, False))
    if body.get("instructions"):
        out.append(("instr:" + canon(body["instructions"]), 0, False))
    for it in body.get("input") or []:
        out.append(("item:" + canon(it), 0, False))
    return [(t, max(1, len(t) // 4), m) for t, _, m in out]


def blocks_of(body: dict, anthropic: bool) -> list[tuple[str, int, bool]]:
    """(canonical text, tokens, carries cache_control) per block, in render order."""
    if "toolConfig" in body or "inferenceConfig" in body:
        return converse_blocks(body)
    if isinstance(body.get("input"), list):
        return responses_blocks(body)
    out = []
    def add(x, role=""):
        text = role + canon(strip_marker(x))
        out.append((text, max(1, len(text) // 4), '"cache_control"' in canon(x)))
    for t in body.get("tools") or []:
        add(t, "tool:")
    system = body.get("system")
    if isinstance(system, list):
        for b in system:
            add(b, "system:")
    elif system:
        add(system, "system:")
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list) and content:
            for i, b in enumerate(content):
                add(b, f"{m.get('role')}[{i}]:")
            rest = {k: v for k, v in m.items() if k != "content"}
            if len(rest) > 1:
                add(rest, f"{m.get('role')}-meta:")
        else:
            add(m, f"{m.get('role')}:")
    return out


def serve_cache(body: dict, anthropic: bool) -> tuple[int, int, int]:
    """(prompt tokens, cached tokens read, tokens written)."""
    blocks = blocks_of(body, anthropic)
    h = hashlib.sha256()
    prefixes = []
    total = 0
    for text, tok, marked in blocks:
        h.update(text.encode())
        total += tok
        prefixes.append((h.hexdigest(), total, marked))
    with LOCK:
        read = 0
        for digest, upto, _ in prefixes:
            if CACHE.get(digest) == upto:
                read = upto
        if not anthropic and read < OPENAI_MIN:
            read = 0
        written = 0
        for digest, upto, marked in prefixes:
            if anthropic and not marked:
                continue
            if digest not in CACHE:
                CACHE[digest] = upto
                if anthropic:
                    written = max(written, upto - read)
    return total, read, written


def turns_so_far(body: dict) -> int:
    return sum(1 for m in body.get("messages") or [] if m.get("role") == "assistant")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        path = self.path.rstrip("/")
        converse = path.endswith("/converse")
        responses = path.endswith("/responses")
        # Claude by any route caches only at its breakpoints (cache_control, or a Converse cachePoint)
        anthropic = path.endswith("/messages") or converse or "claude" in str(body.get("model", "")).lower()
        prompt, read, written = serve_cache(body, anthropic)
        if converse:
            fresh = prompt - read - written
            return self.send(200, {"output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
                                   "stopReason": "end_turn", "metrics": {"latencyMs": 1},
                                   "usage": {"inputTokens": fresh, "outputTokens": 5, "totalTokens": prompt + 5,
                                             "cacheReadInputTokens": read, "cacheWriteInputTokens": written}})
        if responses:
            return self.oai_responses(body, prompt, read)
        tools = body.get("tools") or []
        call = None
        if TOOL_TURNS and tools and turns_so_far(body) < TOOL_TURNS:
            t = tools[0]
            name = t.get("name") or (t.get("function") or {}).get("name") or "tool"
            call = (name, {"name": "n0"})
        if anthropic:
            self.anthropic(body, prompt, read, written, call)
        else:
            self.openai(body, prompt, read, call)

    def send(self, status, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def oai_responses(self, body, prompt, read):
        usage = {"input_tokens": prompt, "input_tokens_details": {"cached_tokens": read}, "output_tokens": 5,
                 "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": prompt + 5}
        item = {"type": "message", "id": "msg_mock", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}]}
        resp = {"id": f"resp_{int(time.time()*1e6)}", "object": "response", "created_at": int(time.time()),
                "status": "completed", "model": body.get("model", "m"), "output": [item], "usage": usage,
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "error": None, "incomplete_details": None}
        if not body.get("stream"):
            return self.send(200, resp)
        start = {**resp, "status": "in_progress", "output": [], "usage": None}
        ev = [("response.created", {"type": "response.created", "sequence_number": 0, "response": start}),
              ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
                                              "item": {**item, "status": "in_progress", "content": []}}),
              ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 2, "item_id": "msg_mock",
                                              "output_index": 0, "content_index": 0, "delta": "done"}),
              ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": item}),
              ("response.completed", {"type": "response.completed", "sequence_number": 4, "response": resp})]
        self.send(200, b"".join(f"event: {n}\ndata: {json.dumps(d)}\n\n".encode() for n, d in ev), "text/event-stream")

    def openai(self, body, prompt, read, call, written=None):
        usage = {"prompt_tokens": prompt, "completion_tokens": 5, "total_tokens": prompt + 5,
                 "prompt_tokens_details": {"cached_tokens": read}}
        if written is not None:  # OpenRouter's Claude: writes reported as cache_write_tokens
            usage["prompt_tokens_details"]["cache_write_tokens"] = written
        msg = {"role": "assistant", "content": None if call else "done"}
        if call:
            msg["tool_calls"] = [{"id": f"call_{int(time.time()*1e6)}", "type": "function",
                                  "function": {"name": call[0], "arguments": json.dumps(call[1])}}]
        finish = "tool_calls" if call else "stop"
        base = {"id": "chatcmpl-mock", "created": int(time.time()), "model": body.get("model", "m")}
        if not body.get("stream"):
            return self.send(200, {**base, "object": "chat.completion",
                                   "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
        chunks = [{**base, "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {k: v for k, v in msg.items() if v is not None}, "finish_reason": None}]},
                  {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}]
        if (body.get("stream_options") or {}).get("include_usage"):
            chunks.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
        sse = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
        self.send(200, sse, "text/event-stream")

    def anthropic(self, body, prompt, read, written, call):
        if not self.path.rstrip("/").endswith("/messages"):  # Claude in chat format (OpenRouter)
            return self.openai(body, prompt, read, call, written)
        fresh = prompt - read - written
        content = ([{"type": "tool_use", "id": f"toolu_{int(time.time()*1e6)}", "name": call[0], "input": call[1]}]
                   if call else [{"type": "text", "text": "done"}])
        stop = "tool_use" if call else "end_turn"
        usage = {"input_tokens": fresh, "cache_read_input_tokens": read, "cache_creation_input_tokens": written,
                 "output_tokens": 5}
        msg = {"id": "msg_mock", "type": "message", "role": "assistant", "model": body.get("model", "m"),
               "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage}
        if not body.get("stream"):
            return self.send(200, msg)
        ev = [("message_start", {"type": "message_start", "message": {**msg, "content": [], "stop_reason": None,
                                                                     "usage": {**usage, "output_tokens": 1}}})]
        for i, b in enumerate(content):
            if b["type"] == "text":
                ev += [("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}),
                       ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}})]
            else:
                ev += [("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}),
                       ("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})]
            ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        ev += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                  "usage": {"output_tokens": 5}}),
               ("message_stop", {"type": "message_stop"})]
        sse = b"".join(f"event: {n}\ndata: {json.dumps(d)}\n\n".encode() for n, d in ev)
        self.send(200, sse, "text/event-stream")


if __name__ == "__main__":
    if "--tool-turns" in sys.argv:
        TOOL_TURNS = int(sys.argv[sys.argv.index("--tool-turns") + 1])
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
