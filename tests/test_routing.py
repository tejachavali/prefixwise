"""Routing found by dogfooding (opencode, 2026-11-06): bare paths, query strings,
endpoints Prefixwise does not analyse, HEAD."""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler


class Upstream:
    def __init__(self):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, payload, status=200):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            def _record(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(n) if n else b""
                outer.seen.append({"method": self.command, "path": self.path, "body": body,
                                   "headers": {k.lower(): v for k, v in self.headers.items()}})

            def do_GET(self):
                self._record()
                self._reply({"data": [{"id": "claude-sonnet-5-5"}]})

            def do_HEAD(self):
                self._record()
                self._reply({})

            def do_POST(self):
                self._record()
                if self.path.startswith("/v1/messages/count_tokens"):
                    return self._reply({"input_tokens": 42})
                self._reply({"id": "msg_1", "type": "message", "content": [],
                             "usage": {"input_tokens": 10, "cache_read_input_tokens": 0,
                                       "cache_creation_input_tokens": 0, "output_tokens": 1}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


def start(proxy):
    server = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def call(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


MSG = {"model": "claude-sonnet-5-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}


def test_bare_messages_path_and_query_string_are_forwarded():
    up = Upstream()
    proxy = PrefixwiseProxy(upstream_url=up.url, api_key="k")
    server, base = start(proxy)
    try:
        status, _, headers = call(base + "/messages?beta=true", "POST", MSG)
        assert status == 200 and headers["X-Prefixwise-Cause"] == "first_write"
        assert up.seen[-1]["path"] == "/v1/messages?beta=true"
        assert "x-prefixwise-internal-query" not in up.seen[-1]["headers"]
    finally:
        server.shutdown()
        up.server.shutdown()


def test_unanalysed_endpoints_pass_through_and_are_not_recorded():
    up = Upstream()
    proxy = PrefixwiseProxy(upstream_url=up.url, api_key="k")
    server, base = start(proxy)
    try:
        status, body, _ = call(base + "/v1/models")
        assert status == 200 and json.loads(body)["data"][0]["id"] == "claude-sonnet-5-5"
        status, body, _ = call(base + "/v1/messages/count_tokens", "POST", MSG,
                               {"anthropic-version": "2023-06-01"})
        assert status == 200 and json.loads(body) == {"input_tokens": 42}
        assert up.seen[-1]["headers"]["x-api-key"] == "k"
        assert proxy.ledger.rollup_by_cause() == []  # nothing analysed, nothing recorded
    finally:
        server.shutdown()
        up.server.shutdown()


def test_head_is_answered():
    up = Upstream()
    server, base = start(PrefixwiseProxy(upstream_url=up.url))
    try:
        assert call(base + "/health", "HEAD")[0] == 200
        assert call(base + "/v1/models", "HEAD")[0] == 200
    finally:
        server.shutdown()
        up.server.shutdown()


def test_mock_mode_says_why_unanalysed_paths_fail():
    server, base = start(PrefixwiseProxy(mock=True))
    try:
        status, body, _ = call(base + "/v1/models")
        assert status == 404 and b"mock mode" in body
    finally:
        server.shutdown()


def test_client_cannot_inject_the_internal_query_header():
    up = Upstream()
    server, base = start(PrefixwiseProxy(upstream_url=up.url))
    try:
        call(base + "/v1/messages", "POST", MSG, {"X-Prefixwise-Internal-Query": "evil=1"})
        assert up.seen[-1]["path"] == "/v1/messages"
    finally:
        server.shutdown()
        up.server.shutdown()


def test_failed_upstream_requests_are_not_recorded():
    proxy = PrefixwiseProxy(upstream_url="http://127.0.0.1:1")
    server, base = start(proxy)
    try:
        assert call(base + "/v1/messages", "POST", MSG)[0] == 502
        assert proxy.ledger.rollup_by_cause() == []
    finally:
        server.shutdown()
