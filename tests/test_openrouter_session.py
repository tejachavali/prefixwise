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


def test_user_groups_turns_when_nothing_else_names_a_session_but_never_a_cache_key():
    p = PrefixwiseProxy(mock=True)
    ctx = p.prepare("openai", {**BODY, "user": "u-123"}, {})
    assert ctx["session_id"] == "u-123" and ctx["session_is_user"]  # a user id, not a conversation id
    ctx = p.prepare("openai", {**BODY, "prompt_cache_key": "conv-7"}, {})
    assert ctx["session_id"] == p.default_session  # a changing key is the miss cache_key_changed names
    ctx = p.prepare("openai", {**BODY, "user": "u-123"}, {"x-prefixwise-session-id": "s-1"})
    assert ctx["session_id"] == "s-1" and not ctx["session_is_user"]  # an explicit id still wins
    assert p.prepare("openai", {**BODY, "user": 12345}, {})["session_id"] == p.default_session
    assert p.prepare("openai", {**BODY, "user": "u" * 300}, {})["session_id"] == p.default_session


def test_a_changed_cache_key_without_a_session_id_is_still_named():
    """The regression a prompt_cache_key session name would cause: turn 2 changes the key and
    misses; without any session id it must stay cache_key_changed, not a new conversation."""
    p = PrefixwiseProxy(mock=True)
    usages = [{"prompt_tokens": 3000, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}},
              {"prompt_tokens": 3020, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}]

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        return 200, {"id": request_id, "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                     "usage": usages.pop(0)}
    p.mock = False
    p._forward = fake
    msgs = [{"role": "system", "content": "You are careful. " * 300}, {"role": "user", "content": "hi"}]
    p.handle("openai", {"model": "gpt-5", "messages": msgs, "prompt_cache_key": "a"}, {})
    more = msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
    _, e = p.handle("openai", {"model": "gpt-5", "messages": more, "prompt_cache_key": "b"}, {})
    assert e.detail == "cache_key_changed", (e.cause, e.detail, e.evidence)


def test_check_scenarios_do_not_merge_on_a_shared_user():
    p = PrefixwiseProxy(mock=True)
    p.check_mode = True
    p.default_session = "test_a"
    assert p.prepare("openai", {**BODY, "user": "ci-bot"}, {})["session_id"] == "test_a"
