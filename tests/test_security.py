"""Security regressions: each test pins a weakness found in the pre-launch review."""

import json
import os
import socket
import stat
import threading

import pytest

from prefixwise.ledger import Ledger
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler, serve


def start(proxy):
    server = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def raw(port, request: bytes, timeout=3) -> bytes:
    s = socket.create_connection(("127.0.0.1", port))
    s.settimeout(timeout)
    s.sendall(request)
    out = b""
    try:
        while chunk := s.recv(4096):
            out += chunk
    except socket.timeout:
        pass
    return out


def post(port, path, body, headers=b""):
    data = json.dumps(body).encode() if not isinstance(body, bytes) else body
    return raw(port, b"POST " + path.encode() + b" HTTP/1.0\r\nContent-Type: application/json\r\n" + headers
               + b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)


@pytest.mark.parametrize("body,needle", [
    ({"model": "m", "messages": "x"}, b"'messages' must be a list"),
    ({"model": "m", "messages": [1, 2]}, b"must be an object"),
    ({"model": "m", "tools": "x", "messages": []}, b"'tools' must be a list"),
    ({"model": "m", "input": 5}, b"'input' must be a string or a list"),
    ({"model": "m", "messages": [], "session_id": "x" * 300}, b"longer than 256"),
])
def test_malformed_requests_get_a_400_not_a_dropped_connection(body, needle):
    server, port = start(PrefixwiseProxy(mock=True))
    try:
        reply = post(port, "/v1/chat/completions", body)
        assert reply.startswith(b"HTTP/1.0 400") and needle in reply
    finally:
        server.shutdown()


def test_bad_content_length_is_refused_immediately():
    server, port = start(PrefixwiseProxy(mock=True))
    try:
        neg = raw(port, b"POST /tasks HTTP/1.0\r\nContent-Length: -1\r\n\r\n{}")
        huge = raw(port, b"POST /tasks HTTP/1.0\r\nContent-Length: 999999999999\r\n\r\n{}")
        assert neg.startswith(b"HTTP/1.0 400") and huge.startswith(b"HTTP/1.0 413")
    finally:
        server.shutdown()


def test_unexpected_errors_are_a_500_without_internals():
    proxy = PrefixwiseProxy(mock=True)

    def boom(*a, **kw):
        raise RuntimeError("secret internal detail")
    proxy.handle = boom
    server, port = start(proxy)
    try:
        reply = post(port, "/v1/chat/completions", {"model": "m", "messages": []})
        assert reply.startswith(b"HTTP/1.0 500") and b"secret internal detail" not in reply
    finally:
        server.shutdown()


def test_auth_token_guards_everything_but_health():
    proxy = PrefixwiseProxy(mock=True)
    proxy.auth_token = "s3cret"
    server, port = start(proxy)
    try:
        assert post(port, "/v1/chat/completions", {"model": "m", "messages": []}).startswith(b"HTTP/1.0 401")
        assert raw(port, b"GET /ledger HTTP/1.0\r\nX-Prefixwise-Token: nope\r\n\r\n").startswith(b"HTTP/1.0 401")
        assert raw(port, b"GET /ledger HTTP/1.0\r\nX-Prefixwise-Token: s3cret\r\n\r\n").startswith(b"HTTP/1.0 200")
        assert raw(port, b"GET /health HTTP/1.0\r\n\r\n").startswith(b"HTTP/1.0 200")
        ok = post(port, "/v1/chat/completions", {"model": "m", "messages": []}, b"X-Prefixwise-Token: s3cret\r\n")
        assert ok.startswith(b"HTTP/1.0 200")
    finally:
        server.shutdown()


def test_token_is_not_forwarded_upstream():
    from prefixwise.proxy import PASSTHROUGH_HEADERS
    assert "x-prefixwise-token" not in PASSTHROUGH_HEADERS


def test_refuses_to_expose_a_key_without_auth(monkeypatch):
    monkeypatch.setenv("PREFIXWISE_API_KEY", "sk-test")
    monkeypatch.delenv("PREFIXWISE_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        serve(host="0.0.0.0", port=0, upstream_url="https://api.openai.com", quiet=True)
    assert "refusing to listen" in str(exc.value)


def test_ledger_file_is_owner_only(tmp_path):
    path = tmp_path / "l.db"
    Ledger(str(path))
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_ledger_never_stores_prompt_text(tmp_path):
    path = str(tmp_path / "l.db")
    proxy = PrefixwiseProxy(ledger_path=path, mock=True)
    secret = "my-very-private-prompt-text-12345"
    proxy.handle("openai", {"model": "m", "messages": [{"role": "user", "content": secret}]})
    files = list(tmp_path.glob("l.db*"))  # the database and its -wal / -shm side files
    assert any(f.name.endswith("-wal") for f in files) or len(files) >= 1
    for f in files:
        assert secret.encode() not in f.read_bytes(), f.name
    assert proxy.ledger.rollup_by_cause()  # the row really was written
