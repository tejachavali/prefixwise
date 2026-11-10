"""The client's x-session-id: routers with session affinity (sgl-router, llm-d, AIBrix) route on it,
so Prefixwise forwards it unchanged, uses it as the session, and never sends one of its own."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prefixwise.proxy import PrefixwiseProxy

OK = {"id": "chatcmpl-x", "choices": [{"message": {"role": "assistant", "content": "ok"}}],
      "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}}


class Upstream:
    def __init__(self):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                outer.seen.append({"x-session-id": self.headers.get("x-session-id"), "body": body})
                b = json.dumps(OK).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def _ask(p, headers=None, **body):
    return p.handle("openai", {"model": "m", "messages": [{"role": "user", "content": "hi"}], **body}, headers or {})


def test_each_clients_session_key_reaches_the_upstream_unchanged():
    up = Upstream()
    p = PrefixwiseProxy(upstream_url=up.url, mock=False)
    try:
        for conv in ("conv-1", "conv-2", "conv-3"):
            _ask(p, {"x-session-id": conv})
        assert [s["x-session-id"] for s in up.seen] == ["conv-1", "conv-2", "conv-3"]  # was "default" x3
    finally:
        up.server.shutdown()


def test_no_session_key_is_invented_and_the_prefixwise_id_is_not_turned_into_one():
    up = Upstream()
    p = PrefixwiseProxy(upstream_url=up.url, mock=False)
    try:
        _ask(p)
        _ask(p, {"x-prefixwise-session-id": "mine"})
        _ask(p, session_id="body-id")
        assert [s["x-session-id"] for s in up.seen] == [None, None, None]
    finally:
        up.server.shutdown()


def test_the_clients_key_groups_turns_into_sessions_and_explicit_ids_still_win():
    p = PrefixwiseProxy(mock=True)
    _ask(p, {"x-session-id": "conv-1"})
    _ask(p, {"x-session-id": "conv-2"})
    sessions = {r["session_id"] for r in p.ledger.recent(10)}
    assert sessions == {"conv-1", "conv-2"}  # not one "default" session
    _, entry = _ask(p, {"x-session-id": "conv-3", "x-prefixwise-session-id": "explicit"})
    assert entry.session_id == "explicit"
    _, entry = _ask(p, {"x-session-id": "conv-4"}, session_id="from-body")
    assert entry.session_id == "from-body"
    _, entry = _ask(p, {"x-session-id": "x" * 10_000})  # too long to use as an id: forwarded, not used
    assert entry.session_id == "default"


def test_sglang_body_session_carries_the_clients_key():
    """For SGLang's KV-event join the body session_id is the session Prefixwise registers, so with a
    client key it is that key: the router's key and the engine's label agree."""
    p = PrefixwiseProxy(mock=True)
    p.engine, p.sglang_join = "sglang", True

    class Events:
        def __init__(self):
            self.sent = []

        def request_sent(self, sid):
            self.sent.append(sid)
            return len(self.sent)

        def discard(self, token):
            pass

    p.engine_events = Events()
    ctx = p.prepare("openai", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, {"x-session-id": "conv-9"})
    assert ctx["forward_body"]["session_id"] == "conv-9" and p.engine_events.sent == ["conv-9"]
    p.response_started(ctx)


def test_codex_conversation_headers_rank_above_a_generic_session_key():
    """x-session-id means what each client makes it mean; Codex's thread-id is a known conversation id."""
    p = PrefixwiseProxy(mock=True)
    _, entry = _ask(p, {"x-session-id": "per-request-123", "thread-id": "codex-thread"})
    assert entry.session_id == "codex-thread"
    _, entry = _ask(p, {"x-session-id": "conv-5"}, metadata={"user_id": "u-1"})
    assert entry.session_id == "conv-5"  # a session key beats a user id
