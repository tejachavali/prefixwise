"""The proxy never costs a client its request or its answer (#184, items 10, 11, 13, 18).

- A request Prefixwise cannot read is forwarded as it was received and recorded as a failure row.
- A stream the upstream cuts is one failure row, `upstream_dropped`, however it was cut, and nothing
  is written into the stream after its last relayed line.
- A fix never moves the client's own cache marker.
"""
import json
import socket
import struct
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prefixwise.fixes import apply_fixes
from prefixwise.proxy import StreamAccumulator
from test_proxy import BODY, OPENAI_OK, FakeUpstream, post, run_proxy, start


def rows(proxy, columns="status, error, cause"):
    with proxy.ledger.reader() as conn:
        return conn.execute(f"SELECT {columns} FROM entries ORDER BY id").fetchall()


def raw_post(port, path, body: bytes) -> bytes:
    """The whole response as bytes, read to the end of the connection."""
    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.settimeout(10)
        sock.sendall(f"POST {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        out = b""
        while chunk := sock.recv(65536):
            out += chunk
        return out


def test_a_lone_surrogate_in_the_text_is_hashed_not_fatal():
    """A JavaScript client that cut a string inside an emoji sends a lone surrogate as an escape. It is
    valid JSON. Hashing it raised, the proxy answered 500, and the request never left."""
    up = FakeUpstream(payload=OPENAI_OK)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        body = b'{"model":"m","messages":[{"role":"user","content":"tool output, cut here: \\ud83d"}]}'
        status, resp, _ = post(port, "/v1/chat/completions", body)
        assert status == 200 and resp == OPENAI_OK
        assert up.requests[0]["body"]["messages"][0]["content"].endswith("\ud83d")   # forwarded as sent
        assert rows(proxy) == [(200, None, "first_write")]
    finally:
        server.shutdown(); up.server.shutdown()


def test_an_upstream_that_answers_with_a_number_as_its_id_does_not_break_the_next_request():
    up = FakeUpstream(payload={**OPENAI_OK, "id": 12345})
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        body = {"model": "claude-opus-5-5", "max_tokens": 9, "messages": [{"role": "user", "content": "hi"}]}
        for _ in range(2):
            assert post(port, "/v1/messages", body, {"X-Prefixwise-Session-Id": "s"})[0] == 200
    finally:
        server.shutdown(); up.server.shutdown()


def test_a_request_prefixwise_cannot_read_is_forwarded_as_received(monkeypatch):
    """Whatever goes wrong while reading a request (a bug here, an input no one thought of), the client
    still gets its answer. Nothing is added to the request, and the turn is a failure row."""
    up = FakeUpstream(payload=OPENAI_OK)
    proxy, server, port = run_proxy(upstream_url=up.url)

    def broken(*a, **k):
        raise RuntimeError("a bug in reading the request")

    monkeypatch.setattr(proxy, "prepare", broken)
    try:
        status, resp, headers = post(port, "/v1/chat/completions", BODY)
        assert status == 200 and resp == OPENAI_OK
        assert headers["X-Prefixwise-Error"].startswith("not recorded") and headers.get("X-Prefixwise-Cause") is None
        sent = up.requests[0]["body"]
        assert sent == {k: v for k, v in BODY.items() if k not in ("session_id", "agent_id")}  # Prefixwise's own fields only
        assert rows(proxy) == [(200, "not_recorded", None)]
        # Streamed: the client gets the stream, with nothing asked of the upstream that it did not ask for.
        up.stream_lines = [b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n',
                           b"data: [DONE]\n\n"]
        raw = raw_post(port, "/v1/chat/completions", json.dumps({**BODY, "stream": True}).encode())
        assert raw.rstrip().endswith(b"data: [DONE]") and b"internal error" not in raw
        assert "stream_options" not in up.requests[1]["body"]
        time.sleep(0.2)
        assert rows(proxy)[1] == (200, "not_recorded", None)
    finally:
        server.shutdown(); up.server.shutdown()


def test_a_request_that_is_the_clients_own_mistake_is_still_refused():
    proxy, server, port = run_proxy(mock=True)
    try:
        status, resp, _ = post(port, "/v1/chat/completions", BODY, {"X-Prefixwise-Session-Id": "x" * 5000})
        assert status == 400 and "longer than" in resp["error"]["message"]
    finally:
        server.shutdown()


FIRST = b'data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'


class CuttingUpstream:
    """Starts a chunked event stream, then ends it the way `mode` says, with no terminal event."""

    def __init__(self):
        self.mode = None
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(FIRST), FIRST))
                self.wfile.flush()
                time.sleep(0.2)
                if outer.mode == "dies inside a chunk":
                    self.wfile.write(b'400\r\ndata: {"id":"chatcmpl-1","mod')   # announces 1,024 bytes, sends 30
                    self.wfile.flush()
                elif outer.mode == "connection reset":
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                self.connection.close()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{start(self.server)}"


@pytest.mark.parametrize("mode", ["dies inside a chunk", "closes between chunks", "connection reset"])
def test_a_stream_the_upstream_cuts_is_one_failure_row(mode):
    """Three ways to cut a stream had three outcomes: an error written into the stream and no row, a
    success row, and a row blaming the client."""
    up = CuttingUpstream()
    up.mode = mode
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        body = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
        raw = raw_post(port, "/v1/chat/completions", body)
        head, _, stream = raw.partition(b"\r\n\r\n")
        assert b"200" in head.split(b"\r\n")[0]
        assert b"Hel" in stream and b"HTTP/1." not in stream and b"internal error" not in stream
        time.sleep(0.3)
        assert rows(proxy) == [(200, "upstream_dropped", None)]
    finally:
        server.shutdown(); up.server.shutdown()


def test_a_whole_answer_without_the_terminal_event_is_not_a_failure():
    """Some compatible servers close after the last chunk without `[DONE]`. The model finished: that
    is a response, usage or not."""
    up = FakeUpstream(stream_lines=[
        b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n',
        b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'])
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        raw_post(port, "/v1/chat/completions", json.dumps({**BODY, "stream": True}).encode())
        time.sleep(0.3)
        assert [r[:2] for r in rows(proxy)] == [(200, None)]
    finally:
        server.shutdown(); up.server.shutdown()


def test_the_accumulator_knows_a_finished_answer():
    chat = StreamAccumulator("openai")
    chat.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n')
    assert not chat.complete
    chat.feed(b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n')
    assert chat.complete and not chat.done
    usage_only = StreamAccumulator("openai")
    usage_only.feed(b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n')
    assert usage_only.complete
    claude = StreamAccumulator("anthropic")
    claude.feed(b'data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":5}}}\n')
    assert not claude.complete                      # usage at the start says nothing about the end
    claude.feed(b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n')
    assert claude.complete and not claude.done
    claude.feed(b'data: {"type":"message_stop"}\n')
    assert claude.done
    responses = StreamAccumulator("openai", api="responses")
    responses.feed(b'data: {"type":"response.output_text.delta","delta":"hi"}\n')
    assert not responses.complete
    responses.feed(b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n')
    assert responses.complete


def _tool(name, **extra):
    return {"name": name, "description": "d", "input_schema": {"type": "object"}, **extra}


def _fixed(tools):
    body = {"model": "claude-opus-5-5", "max_tokens": 9, "tools": tools, "messages": [{"role": "user", "content": "hi"}]}
    out, applied = apply_fixes(body, ("canonical_tools",))
    return [(t["name"], t.get("cache_control")) for t in out["tools"]], applied


MARK = {"type": "ephemeral"}


def test_canonical_tools_keeps_the_clients_marker_where_it_caches_the_whole_list():
    """A marker on the last tool caches every tool. Sorting moved it, with its tool, to the front, where
    it covered one tool."""
    tools = [_tool("web_search"), _tool("read_file"), _tool("bash", cache_control=MARK)]
    assert _fixed(tools) == ([("bash", None), ("read_file", None), ("web_search", MARK)], ["canonical_tools"])
    # the same list in another order gives the same bytes: that is what the fix is for
    again = [_tool("bash"), _tool("web_search"), _tool("read_file", cache_control=MARK)]
    assert _fixed(again)[0] == _fixed(tools)[0]
    assert tools[2]["cache_control"] == MARK and "cache_control" not in tools[0]   # the client's body is not touched


def test_canonical_tools_leaves_a_list_with_a_marker_in_the_middle_alone():
    """Such a marker covers the tools before it. Sorting would change which ones."""
    tools = [_tool("web_search"), _tool("read_file", cache_control=MARK), _tool("bash")]
    assert _fixed(tools) == ([("web_search", None), ("read_file", MARK), ("bash", None)], [])
    assert _fixed([_tool("b"), _tool("a")]) == ([("a", None), ("b", None)], ["canonical_tools"])   # no marker: as before


def test_overhead_counts_the_time_spent_reading_the_request():
    """`overhead_ms` was measured from after the request had been read and fingerprinted, the larger
    part of Prefixwise's own time on a big request."""
    up = FakeUpstream(payload=OPENAI_OK)
    proxy, server, port = run_proxy(upstream_url=up.url)
    proxy.request_observer = lambda *a: time.sleep(0.05)   # called once, while the request is read
    try:
        assert post(port, "/v1/chat/completions", BODY)[0] == 200
        assert rows(proxy, "overhead_ms")[0][0] >= 50
    finally:
        server.shutdown(); up.server.shutdown()
