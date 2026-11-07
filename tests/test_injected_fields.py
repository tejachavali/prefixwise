"""Prefixwise must never turn a working request into an error.

It adds fields to some requests (Anthropic `diagnostics`, OpenAI
`prompt_cache_options` and `stream_options.include_usage`, the engine join's
`session_id`). A gateway or compatible server that refuses unknown fields
answers 400: the request is resent as the client sent it, and after a second
rejection the field is not added again.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prefixwise.proxy import PrefixwiseProxy


def upstream(reject: set[str]):
    """A strict upstream: 400 naming the first unknown field, else a response with usage."""
    seen: list[dict] = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            text = json.dumps(body)
            bad = sorted(f for f in reject if f'"{f}"' in text)  # top-level or nested key
            if bad:
                data = json.dumps({"error": {"type": "invalid_request_error",
                                             "message": f"Unrecognized request argument supplied: {bad[0]}"}})
                self.send_response(400)
            elif body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk = {"id": "chatcmpl-1", "choices": [{"delta": {"content": "ok"}}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                return
            else:
                usage = ({"input_tokens": 2000, "output_tokens": 1, "cache_read_input_tokens": 0}
                         if "/messages" in self.path else
                         {"prompt_tokens": 2000, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}})
                prefix = "msg_" if "/messages" in self.path else ("resp_" if "/responses" in self.path else "chatcmpl-")
                data = json.dumps({"id": prefix + str(len(seen)), "usage": usage})
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data.encode())

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.fixture
def strict():
    servers = []

    def make(reject):
        srv, seen = upstream(reject)
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}", seen
    yield make
    for s in servers:
        s.shutdown()


MSG = {"model": "claude-haiku-5-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}


def test_anthropic_diagnostics_rejected_is_resent_without_it_then_dropped(strict, capsys):
    url, seen = strict({"diagnostics"})
    p = PrefixwiseProxy(upstream_url=url, api_key="k")
    h = {"x-prefixwise-session-id": "s"}
    res, entry = p.handle("anthropic", dict(MSG), h)
    assert res["status"] == 200 and entry is not None  # the client never sees the 400
    assert "diagnostics" in seen[0] and "diagnostics" not in seen[1]
    res, _ = p.handle("anthropic", dict(MSG), h)  # rejected again: dropped for good
    assert res["status"] == 200 and len(seen) == 4
    assert "rejects the `diagnostics` field" in capsys.readouterr().err
    res, _ = p.handle("anthropic", dict(MSG), h)
    assert res["status"] == 200 and len(seen) == 5 and "diagnostics" not in seen[4]


def test_a_client_error_that_is_not_about_an_added_field_is_returned_as_is(strict):
    url, seen = strict({"max_tokens"})  # the client's own field
    p = PrefixwiseProxy(upstream_url=url, api_key="k")
    res, entry = p.handle("anthropic", dict(MSG), {"x-prefixwise-session-id": "s"})
    assert res["status"] == 400 and entry is None and len(seen) == 1  # not retried


def test_client_values_are_restored_not_dropped(strict):
    # The client's own prompt_cache_options survive; only Prefixwise's addition goes.
    url, seen = strict({"comparison_response_id"})
    p = PrefixwiseProxy(upstream_url=url, api_key="k")
    h = {"x-prefixwise-session-id": "s"}
    body = {"model": "gpt-5.6", "input": "hi", "prompt_cache_options": {"mode": "implicit"}}
    assert p.handle("openai", dict(body), h, api="responses")[0]["status"] == 200  # first turn: nothing added
    res, _ = p.handle("openai", dict(body), h, api="responses")
    assert res["status"] == 200
    assert seen[1]["prompt_cache_options"]["comparison_response_id"].startswith("resp_")
    assert seen[2]["prompt_cache_options"] == {"mode": "implicit"}


def test_stream_options_rejected_on_a_stream(strict):
    url, seen = strict({"stream_options"})
    p = PrefixwiseProxy(upstream_url=url, api_key="k")
    ctx = p.prepare("openai", {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                    {"x-prefixwise-session-id": "s"})
    status, lines, _acc, err = p.open_stream(ctx)
    assert status == 200 and lines is not None and err is None
    out = b"".join(lines)
    assert b"ok" in out and "stream_options" in seen[0] and "stream_options" not in seen[1]


def test_engine_session_id_rejected_releases_the_engine_check(strict):
    url, seen = strict({"session_id"})
    p = PrefixwiseProxy(upstream_url=url, api_key="k", engine="vllm")
    res, _ = p.handle("vllm", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                      {"x-prefixwise-session-id": "s"})
    assert res["status"] == 200 and "session_id" in seen[0] and "session_id" not in seen[1]
