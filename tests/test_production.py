"""Production hardening found by benchmarks/bench.py: header relay, idle clients, mask cache."""

import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prefixwise import fingerprint
from prefixwise.fingerprint import Fingerprinter
from test_proxy import BODY, OPENAI_OK, post, run_proxy, start


class HeaderUpstream:
    """Replies with a fixed status, payload (or SSE lines) and response headers."""

    def __init__(self, status, payload=None, headers=None, stream_lines=None):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                if stream_lines is not None:
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for line in stream_lines:
                        self.wfile.write(line)
                    return
                data = json.dumps(payload).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{start(self.server)}"


RELAYED = {"retry-after": "7", "x-should-retry": "true", "x-request-id": "req_up_1",
           "x-ratelimit-remaining-requests": "0", "anthropic-ratelimit-tokens-remaining": "5"}


def test_429_keeps_retry_and_rate_limit_headers():
    up = HeaderUpstream(429, {"error": {"type": "rate_limit"}}, {**RELAYED, "set-cookie": "secret=1"})
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        status, _, headers = post(port, "/v1/chat/completions", BODY)
        assert status == 429
        for k, v in RELAYED.items():
            assert headers[k] == v  # SDKs back off on retry-after; request id for support
        assert headers.get("set-cookie") is None  # only the allowlist is relayed
    finally:
        server.shutdown(); up.server.shutdown()


def test_success_and_stream_relay_provider_request_id():
    up = HeaderUpstream(200, OPENAI_OK, {"x-request-id": "req_up_2"})
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        _, _, headers = post(port, "/v1/chat/completions", BODY)
        assert headers["x-request-id"] == "req_up_2"
        assert headers["X-Prefixwise-Request-Id"] != "req_up_2"  # ours stays separate
    finally:
        server.shutdown(); up.server.shutdown()
    lines = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    up = HeaderUpstream(200, headers={"request-id": "req_up_3"}, stream_lines=lines)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        c = socket.create_connection(("127.0.0.1", port))
        body = json.dumps({**BODY, "stream": True}).encode()
        c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        raw = b""
        while chunk := c.recv(65536):
            raw += chunk
        assert b"request-id: req_up_3" in raw.lower()
    finally:
        server.shutdown(); up.server.shutdown()


def test_idle_and_half_sent_clients_are_dropped(monkeypatch):
    import prefixwise.proxy as px
    monkeypatch.setattr(px, "CLIENT_TIMEOUT", 0.5)
    proxy, server, port = run_proxy()
    try:
        idle = socket.create_connection(("127.0.0.1", port))
        half = socket.create_connection(("127.0.0.1", port))
        half.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 1000\r\n\r\n{")
        time.sleep(1.5)
        idle.settimeout(2); half.settimeout(2)
        assert idle.recv(1024) == b""  # closed by the proxy
        assert half.recv(1024) == b""
        assert post(port, "/v1/chat/completions", BODY)[0] == 200  # others unaffected
    finally:
        server.shutdown()


def test_masked_hash_cache_is_exact_and_bounded(monkeypatch):
    monkeypatch.setattr(fingerprint, "MASK_CACHE_ENTRIES", 3)
    fp = Fingerprinter()
    blocks = [("system", f"Today is 2026-11-0{i}. Be brief.") for i in range(1, 6)]
    cached = [fp.fingerprint_block(r, 0, v) for r, v in blocks]
    again = [fp.fingerprint_block(r, 0, v) for r, v in blocks]
    fresh = [Fingerprinter().fingerprint_block(r, 0, v) for r, v in blocks]
    assert [f.masked for f in cached] == [f.masked for f in again] == [f.masked for f in fresh]
    assert len({f.masked for f in cached}) == 1  # only the date changed
    assert len(fp._masked_cache) == 3
    # HMAC mode never shares cache entries with plain hashing.
    keyed = Fingerprinter(hmac_key=b"k").fingerprint_block("system", 0, blocks[0][1])
    assert keyed.masked != cached[0].masked


def test_ledger_failure_fails_open():
    """Disk full / locked ledger: the client still gets the upstream response."""
    import sqlite3

    from test_proxy import FakeUpstream

    up = FakeUpstream(payload=OPENAI_OK)
    proxy, server, port = run_proxy(upstream_url=up.url)

    def broken(entry):
        raise sqlite3.OperationalError("database or disk is full")

    proxy.ledger.record = broken
    try:
        status, resp, headers = post(port, "/v1/chat/completions", BODY)
        assert status == 200 and resp == OPENAI_OK
        assert headers["X-Prefixwise-Error"].startswith("not recorded")
        assert headers.get("X-Prefixwise-Cause") is None
        # Streaming: the stream completes cleanly, with nothing appended.
        c = socket.create_connection(("127.0.0.1", port))
        body = json.dumps({**BODY, "stream": True}).encode()
        up.stream_lines = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                           b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
                           b"data: [DONE]\n\n"]
        c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        raw = b""
        while chunk := c.recv(65536):
            raw += chunk
        assert raw.rstrip().endswith(b"data: [DONE]") and b"internal error" not in raw
    finally:
        server.shutdown(); up.server.shutdown()


def test_sigterm_drains_in_flight_stream_then_exits(tmp_path):
    """docker stop: a stream already running finishes; then the process exits 0."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "benchmarks"))
    from bench import Upstream, free_port, wait_ready

    up = Upstream(chunks=20, chunk_delay_ms=50)  # a 1-second stream
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    env.pop("PREFIXWISE_API_KEY", None)
    proc = subprocess.Popen([sys.executable, "-m", "prefixwise", "serve", "--port", str(port), "--upstream",
                             f"http://127.0.0.1:{up.port}", "--ledger", str(tmp_path / "l.db")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    try:
        wait_ready(port)
        c = socket.create_connection(("127.0.0.1", port))
        body = json.dumps({**BODY, "stream": True}).encode()
        c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        first = c.recv(256)
        assert first.startswith(b"HTTP/1.0 200")
        proc.terminate()  # SIGTERM mid-stream
        raw = first
        while chunk := c.recv(65536):
            raw += chunk
        assert raw.rstrip().endswith(b"data: [DONE]")  # the stream finished
        assert proc.wait(10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        up.stop()


def test_prune_by_age_keeps_recent_and_undated_by_default(tmp_path):
    import sqlite3
    import subprocess
    import sys

    from prefixwise.ledger import Ledger, LedgerEntry

    path = str(tmp_path / "l.db")
    led = Ledger(path)
    for i in range(4):
        led.record(LedgerEntry(session_id="s", agent_id=None, provider="openai", model="m", cause="none",
                               legit=True, prompt_tokens=10, cached_tokens=10, avoidable_tokens=0, avoidable_cost=None,
                               task_id=f"t{i}", request_id=f"r{i}"))
    led.set_task_outcome("t0", True)
    db = sqlite3.connect(path)
    db.execute("UPDATE entries SET recorded_at = recorded_at - 40*86400 WHERE request_id IN ('r0','r1')")
    db.execute("UPDATE entries SET recorded_at = NULL WHERE request_id = 'r2'")  # a pre-0.3.1 row
    db.commit()
    import os
    from pathlib import Path

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}
    run = lambda *a: subprocess.run([sys.executable, "-m", "prefixwise", "prune", *a, "--ledger", path],
                                    capture_output=True, text=True, env=env)
    r = run("--older-than", "30d")
    assert r.returncode == 0 and "deleted 2" in r.stdout
    left = [x[0] for x in db.execute("SELECT request_id FROM entries ORDER BY request_id")]
    assert left == ["r2", "r3"]
    assert db.execute("SELECT count(*) FROM tasks WHERE task_id='t0'").fetchone()[0] == 0  # orphan gone
    assert "deleted 1" in run("--older-than", "30d", "--include-undated").stdout
    assert run("--older-than", "soon").returncode == 2


def test_live_overhead_script_runs_against_fake_upstream():
    """benchmarks/live_overhead.py end to end (fake upstream, dummy keys): one
    cache breakpoint per Anthropic request, every proxied request in the ledger.
    A live run once failed at pair 5: breakpoints piled up in the session history."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "benchmarks"))
    from bench import Upstream

    up = Upstream()
    try:
        for provider, var in (("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")):
            r = subprocess.run([sys.executable, str(root / "benchmarks" / "live_overhead.py"), "--provider", provider,
                                "--pairs", "8", "--url", f"http://127.0.0.1:{up.port}"],
                               capture_output=True, text=True, env={**os.environ, var: "dummy"})
            assert "ledger rows: 8/8" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
    finally:
        up.stop()
