import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prefixwise.proxy import PrefixwiseProxy, has_cache_breakpoint, make_handler, normalize_usage


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def run_proxy(**kw):
    proxy = PrefixwiseProxy(**kw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    return proxy, server, start(server)


def post(port, path, body, headers=None):
    """Return (status, json body, response headers); never raises on 4xx/5xx."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                 data=json.dumps(body).encode() if not isinstance(body, bytes) else body,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), e.headers


class FakeUpstream:
    """Records what the proxy forwards; replies with a scripted status/payload."""

    def __init__(self, status=200, payload=None):
        self.status, self.payload, self.requests = status, payload, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                data = json.dumps(outer.payload).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{start(self.server)}"


OPENAI_OK = {"id": "chatcmpl-1", "object": "chat.completion",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
             "usage": {"prompt_tokens": 2000, "completion_tokens": 1,
                       "prompt_tokens_details": {"cached_tokens": 1800}}}
BODY = {"model": "gpt-5.6", "session_id": "s1", "agent_id": "a1",
        "messages": [{"role": "system", "content": "You are helpful."},
                     {"role": "user", "content": "hi"}]}


def test_mock_first_then_second_turn_verdict_in_headers():
    proxy, server, port = run_proxy()
    try:
        status, first, h1 = post(port, "/v1/chat/completions", BODY)
        assert status == 200 and first["object"] == "chat.completion"  # a real completion shape
        assert h1["X-Prefixwise-Cause"] == "first_write"
        _, _, h2 = post(port, "/v1/chat/completions", BODY)
        assert h2["X-Prefixwise-Cause"] == "none"  # mock caches 1,000 of 2,000 tokens
        assert h2["X-Prefixwise-Avoidable-Tokens"] == "0"
        assert proxy.ledger.tokens_per_successful_task() is None  # no task_success yet
    finally:
        server.shutdown()


def test_passthrough_strips_prefixwise_fields_and_sets_ids():
    up = FakeUpstream(payload=OPENAI_OK)
    proxy, server, port = run_proxy(upstream_url=up.url, api_key="sk-test")
    try:
        status, resp, headers = post(port, "/v1/chat/completions",
                                     {**BODY, "prefixwise_compaction_event": True})
        assert status == 200 and resp == OPENAI_OK  # upstream payload, unchanged
        fwd = up.requests[0]
        assert fwd["path"] == "/v1/chat/completions"
        for k in ("session_id", "agent_id", "prefixwise_compaction_event", "prompt_cache_options"):
            assert k not in fwd["body"]
        assert fwd["headers"]["authorization"] == "Bearer sk-test"
        assert fwd["headers"]["x-session-id"] == "s1"
        assert fwd["headers"]["x-request-id"] == headers["X-Prefixwise-Request-Id"]
        assert proxy.ledger.rollup_by_cause()[0]["cause"] == "first_write"
    finally:
        server.shutdown()
        up.server.shutdown()


def test_client_auth_passed_through_and_session_header():
    up = FakeUpstream(payload=OPENAI_OK)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        body = {k: v for k, v in BODY.items() if k not in ("session_id", "agent_id")}
        post(port, "/v1/chat/completions", body,
             {"Authorization": "Bearer client-key", "X-Prefixwise-Session-Id": "hdr-session"})
        fwd = up.requests[0]
        assert fwd["headers"]["authorization"] == "Bearer client-key"
        assert fwd["headers"]["x-session-id"] == "hdr-session"
        assert "x-prefixwise-session-id" not in fwd["headers"]
    finally:
        server.shutdown()
        up.server.shutdown()


def test_anthropic_auth_headers_and_usage():
    payload = {"id": "msg_1", "type": "message", "content": [],
               "usage": {"input_tokens": 50, "cache_read_input_tokens": 1900,
                         "cache_creation_input_tokens": 50, "output_tokens": 1}}
    up = FakeUpstream(payload=payload)
    proxy, server, port = run_proxy(upstream_url=up.url, api_key="sk-ant")
    try:
        body = {"model": "claude-sonnet-5-5", "session_id": "a",
                "system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "hi"}]}
        post(port, "/v1/messages", body)
        status, resp, h = post(port, "/v1/messages", body)
        fwd = up.requests[0]
        assert fwd["path"] == "/v1/messages"
        assert fwd["headers"]["x-api-key"] == "sk-ant"
        assert "authorization" not in fwd["headers"]
        assert fwd["headers"]["anthropic-version"]
        assert fwd["body"]["diagnostics"] == {"previous_message_id": None}
        assert up.requests[1]["body"]["diagnostics"] == {"previous_message_id": "msg_1"}
        # Breakpoint on the system block counts; 1,900 of 2,000 read from cache.
        assert h["X-Prefixwise-Cause"] == "none"
        entry = proxy.ledger.conn.execute(
            "SELECT prompt_tokens, cached_tokens FROM entries ORDER BY id DESC").fetchone()
        assert entry == (2000, 1900)
    finally:
        server.shutdown()
        up.server.shutdown()


def test_upstream_error_returned_unchanged_and_not_recorded():
    err = {"error": {"message": "Unrecognized request argument", "type": "invalid_request_error"}}
    up = FakeUpstream(status=400, payload=err)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        status, resp, headers = post(port, "/v1/chat/completions", BODY)
        assert status == 400 and resp == err
        assert "X-Prefixwise-Cause" not in headers
        assert proxy.ledger.rollup_by_cause() == []
        assert "s1" not in proxy.last_snapshot  # next good turn is still a first write
    finally:
        server.shutdown()
        up.server.shutdown()


def test_unreachable_upstream_is_502():
    proxy, server, port = run_proxy(upstream_url="http://127.0.0.1:1")
    try:
        status, resp, _ = post(port, "/v1/chat/completions", BODY)
        assert status == 502 and "unreachable" in resp["error"]["message"]
    finally:
        server.shutdown()


def test_bad_json_stream_and_unknown_path():
    proxy, server, port = run_proxy()
    try:
        assert post(port, "/v1/chat/completions", b"{not json")[0] == 400
        status, resp, _ = post(port, "/v1/chat/completions", {**BODY, "stream": True})
        assert status == 400 and "stream" in resp["error"]["message"]
        assert post(port, "/v1/embeddings", BODY)[0] == 404
    finally:
        server.shutdown()


def test_normalize_usage_and_breakpoints():
    assert normalize_usage("anthropic", {"input_tokens": 10, "cache_read_input_tokens": 80,
                                         "cache_creation_input_tokens": 10}) == (100, 80)
    assert normalize_usage("openai", {"prompt_tokens": 100,
                                      "prompt_tokens_details": {"cached_tokens": 64}}) == (100, 64)
    assert has_cache_breakpoint({"tools": [{"name": "t", "cache_control": {"type": "ephemeral"}}]})
    assert has_cache_breakpoint({"cache_control": {"type": "ephemeral"}})
    assert not has_cache_breakpoint({"system": "x", "messages": [{"role": "user", "content": "hi"}]})
