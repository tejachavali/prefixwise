"""A fast fake model API for benchmarks: OpenAI chat completions and Anthropic messages.

Runs as its own process so its CPU does not compete with the proxy's GIL.
It reports realistic usage: prompt tokens ~ body bytes / 4, cached tokens =
the part of the prompt shared with the previous request in the same
conversation (by prefix bytes), rounded down to 128-token blocks.

Faults are chosen by model name, so no extra headers have to survive the proxy:
  bench-fault-500        HTTP 500 with a JSON error
  bench-fault-429        HTTP 429 with retry-after
  bench-fault-reset      closes the socket without a response
  bench-fault-midstream  streams a few chunks, then closes the socket
  bench-fault-badjson    200 with a body that is not JSON
  bench-fault-nousage    200 with no usage block
  bench-fault-slow       waits BENCH_SLOW_SECONDS (default 5) before answering

Usage: python benchmarks/upstream.py PORT [DELAY_MS] [CHUNKS] [CHUNK_DELAY_MS]
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DELAY = float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.0
CHUNKS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
CHUNK_DELAY = float(sys.argv[4]) / 1000 if len(sys.argv) > 4 else 0.0
SLOW = float(os.environ.get("BENCH_SLOW_SECONDS", "5"))

_last: dict[str, bytes] = {}
_lock = threading.Lock()


def _conversation_key(body: dict) -> str:
    return str(body.get("user") or body.get("metadata", {}).get("user_id") or "default")


def _usage_counts(raw: bytes, body: dict) -> tuple[int, int]:
    prompt = max(1, len(raw) // 4)
    key = _conversation_key(body)
    with _lock:
        prev = _last.get(key, b"")
        _last[key] = raw
    shared = 0
    n = min(len(prev), len(raw))
    # Shared prefix length, compared in 4 KiB steps (cheap, accurate enough).
    step = 4096
    while shared + step <= n and prev[shared:shared + step] == raw[shared:shared + step]:
        shared += step
    cached = (shared // 4) // 128 * 128
    return prompt, min(cached, prompt)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code: int, payload: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._send(200, b'{"object":"list","data":[]}')

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        body = json.loads(raw or b"{}")
        model = str(body.get("model", ""))
        anthropic = self.path.rstrip("/").endswith("/messages")
        if DELAY:
            time.sleep(DELAY)
        if model == "bench-fault-500":
            return self._send(500, b'{"error":{"type":"server_error","message":"upstream boom"}}')
        if model == "bench-fault-429":
            return self._send(429, b'{"error":{"type":"rate_limit","message":"slow down"}}',
                              extra={"retry-after": "1"})
        if model == "bench-fault-reset":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True
            return
        if model == "bench-fault-badjson":
            return self._send(200, b"<html>not json</html>", ctype="text/html")
        if model == "bench-fault-slow":
            time.sleep(SLOW)
        prompt, cached = _usage_counts(raw, body)
        if body.get("stream"):
            return self._stream(anthropic, body, model, prompt, cached)
        if model == "bench-fault-nousage":
            usage = None
        elif anthropic:
            usage = {"input_tokens": prompt - cached, "cache_read_input_tokens": cached,
                     "cache_creation_input_tokens": 0, "output_tokens": 50}
        else:
            usage = {"prompt_tokens": prompt, "completion_tokens": 50,
                     "prompt_tokens_details": {"cached_tokens": cached}}
        if anthropic:
            payload = {"id": f"msg_{time.time_ns()}", "type": "message", "role": "assistant", "model": model,
                       "content": [{"type": "text", "text": "ok " * 40}], "stop_reason": "end_turn"}
        else:
            payload = {"id": f"chatcmpl-{time.time_ns()}", "object": "chat.completion", "model": model,
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok " * 40},
                                    "finish_reason": "stop"}]}
        if usage is not None:
            payload["usage"] = usage
        self._send(200, json.dumps(payload).encode())

    def _stream(self, anthropic, body, model, prompt, cached):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        w = self.wfile

        def emit(chunk: bytes):
            w.write(chunk)
            w.flush()
            if CHUNK_DELAY:
                time.sleep(CHUNK_DELAY)

        chunks = 3 if model == "bench-fault-midstream" else CHUNKS
        if anthropic:
            start = {"type": "message_start", "message": {
                "id": f"msg_{time.time_ns()}", "type": "message", "role": "assistant", "model": model,
                "usage": {"input_tokens": prompt - cached, "cache_read_input_tokens": cached,
                          "cache_creation_input_tokens": 0, "output_tokens": 1}}}
            emit(b"event: message_start\ndata: " + json.dumps(start).encode() + b"\n\n")
            for _ in range(chunks):
                d = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok "}}
                emit(b"event: content_block_delta\ndata: " + json.dumps(d).encode() + b"\n\n")
            if model == "bench-fault-midstream":
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            emit(b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                 b'"usage":{"output_tokens":' + str(chunks).encode() + b"}}\n\n")
            emit(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
        else:
            cid = f"chatcmpl-{time.time_ns()}"
            for _ in range(chunks):
                c = {"id": cid, "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": {"content": "ok "}, "finish_reason": None}]}
                emit(b"data: " + json.dumps(c).encode() + b"\n\n")
            if model == "bench-fault-midstream":
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            if (body.get("stream_options") or {}).get("include_usage"):
                u = {"id": cid, "object": "chat.completion.chunk", "model": model, "choices": [],
                     "usage": {"prompt_tokens": prompt, "completion_tokens": chunks,
                               "prompt_tokens_details": {"cached_tokens": cached}}}
                emit(b"data: " + json.dumps(u).encode() + b"\n\n")
            emit(b"data: [DONE]\n\n")


if __name__ == "__main__":
    srv = Server(("127.0.0.1", int(sys.argv[1])), H)
    print("ready", flush=True)
    srv.serve_forever()
