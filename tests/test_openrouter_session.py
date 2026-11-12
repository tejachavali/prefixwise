"""OpenRouter routes a conversation to the provider that holds its cache by the body session_id
(falling back to prompt_cache_key; openrouter.ai/docs/guides/best-practices/prompt-caching).
Prefixwise removed a client's body session_id as one of its own fields on the way to any hosted API,
so through Prefixwise an agent lost OpenRouter's sticky routing, its cache missed, and the miss was
then blamed on routing. And with no session header, turns were grouped by neither field."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prefixwise.proxy import PrefixwiseProxy


class Upstream:
    def __init__(self):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.seen.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
                b = json.dumps({"id": "x", "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 1}}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/v1"


BODY = {"model": "moonshotai/kimi-k2", "messages": [{"role": "user", "content": "hi"}]}


def test_a_clients_session_id_reaches_openrouter(monkeypatch):
    monkeypatch.delenv("PREFIXWISE_FORWARD_SESSION_ID", raising=False)
    up = Upstream()
    try:
        p = PrefixwiseProxy(upstream_url=up.url, mock=False)
        p._takes_session_id = lambda provider: True  # what https://openrouter.ai/api/v1 resolves to
        _, entry = p.handle("openai", {**BODY, "session_id": "conv-42", "prompt_cache_key": "conv-42"}, {})
        assert up.seen[0]["session_id"] == "conv-42" and up.seen[0]["prompt_cache_key"] == "conv-42"
        assert entry.session_id == "conv-42"
    finally:
        up.server.shutdown()


def test_openrouter_is_recognised_by_its_host_and_other_hosted_apis_are_not(monkeypatch):
    monkeypatch.delenv("PREFIXWISE_FORWARD_SESSION_ID", raising=False)
    for url, takes in (("https://openrouter.ai/api/v1", True), ("https://api.openai.com/v1", False),
                       ("https://api.deepseek.com", False), ("https://evil-openrouter.ai.example.com", False)):
        assert PrefixwiseProxy(upstream_url=url, mock=False)._takes_session_id("openai") is takes, url
    monkeypatch.setenv("PREFIXWISE_FORWARD_SESSION_ID", "1")  # another gateway that reads it
    assert PrefixwiseProxy(upstream_url="https://gw.example.com/v1", mock=False)._takes_session_id("openai")


def test_a_hosted_api_still_gets_no_session_id(monkeypatch):
    monkeypatch.delenv("PREFIXWISE_FORWARD_SESSION_ID", raising=False)
    up = Upstream()
    try:
        p = PrefixwiseProxy(upstream_url=up.url, mock=False)  # 127.0.0.1: not OpenRouter
        p.handle("openai", {**BODY, "session_id": "conv-42"}, {})
        assert "session_id" not in up.seen[0]
    finally:
        up.server.shutdown()


def test_prompt_cache_key_then_user_group_turns_when_nothing_else_names_a_session():
    p = PrefixwiseProxy(mock=True)
    ctx = p.prepare("openai", {**BODY, "prompt_cache_key": "conv-7"}, {})
    assert ctx["session_id"] == "conv-7" and not ctx["session_is_user"]
    ctx = p.prepare("openai", {**BODY, "user": "u-123"}, {})
    assert ctx["session_id"] == "u-123" and ctx["session_is_user"]  # a user id, not a conversation id
    ctx = p.prepare("openai", {**BODY, "prompt_cache_key": "conv-7"}, {"x-prefixwise-session-id": "s-1"})
    assert ctx["session_id"] == "s-1"  # an explicit id still wins
    ctx = p.prepare("openai", {**BODY, "user": 12345}, {})
    assert ctx["session_id"] == p.default_session  # not a string: not an id
    long = "k" * 300
    assert p.prepare("openai", {**BODY, "prompt_cache_key": long}, {})["session_id"] == p.default_session
