"""Ledger sinks (prefixwise.sinks): a copy of every row, without the ledger ever waiting on it."""

import http.server
import json
import threading
import urllib.parse

import pytest

from prefixwise.ledger import Ledger, LedgerEntry
from prefixwise.sinks import ClickHouseSink, JSONLSink, sink_from_url
from test_transport import wait_for


def entry(rid, cause="none"):
    return LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model="claude-haiku-5-5", cause=cause,
                       prompt_tokens=2000, cached_tokens=1000, avoidable_tokens=0, avoidable_cost=0.0, request_id=rid)


def test_jsonl_sink_gets_every_row_after_the_ledger_commit(tmp_path):
    led = Ledger()
    path = tmp_path / "rows.jsonl"
    led.add_sink(sink_from_url(f"file://{path}"))
    led.record(entry("r1"))
    led.record(entry("r2", "volatile_system"), at=123.0)
    led.close_sinks()
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [l["request_id"] for l in lines] == ["r1", "r2"] and lines[1]["recorded_at"] == 123.0
    assert lines[1]["cause"] == "volatile_system" and "prompt_tokens" in lines[1] and "schema_version" in lines[1]
    assert led.summary()["requests"] == 2  # the ledger itself


class _ClickHouse(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.requests = []
        ch = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                ch.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"clickhouse://writer:pw@127.0.0.1:{self.srv.server_address[1]}/prefixwise.entries"
        self.start()

    def run(self):
        self.srv.serve_forever()


def test_clickhouse_sink_posts_json_each_row_in_batches():
    ch = _ClickHouse()
    led = Ledger()
    sink = sink_from_url(ch.url)
    assert isinstance(sink, ClickHouseSink)
    sink.sender.flush_seconds = 0.02
    led.add_sink(sink)
    try:
        for i in range(3):
            led.record(entry(f"r{i}"))
        wait_for(lambda: sum(len(b.splitlines()) for _, _, b in ch.requests) == 3)
        wait_for(lambda: sink.stats()["sent"] == 3)  # the sender counts a batch after it reads the reply
        path, headers, body = ch.requests[0]
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["query"][0]
        assert query == "INSERT INTO prefixwise.entries FORMAT JSONEachRow"
        assert headers["x-clickhouse-user"] == "writer" and headers["x-clickhouse-key"] == "pw"
        rows = [json.loads(l) for l in body.splitlines()]
        assert rows[0]["request_id"] == "r0" and rows[0]["model"] == "claude-haiku-5-5"
        assert rows[0]["legit"] is None and isinstance(rows[0]["prompt_tokens"], int)
        assert sink.stats()["sent"] == 3 and sink.stats()["dropped"] == 0
    finally:
        led.close_sinks(1)
        ch.srv.shutdown()


def test_a_failing_sink_never_fails_the_ledger(capsys):
    class Bad:
        name = "bad"

        def record(self, entry, at=None):
            raise RuntimeError("warehouse down")

        def close(self, timeout=5.0):
            pass
    led = Ledger()
    led.add_sink(Bad())
    for i in range(3):
        led.record(entry(f"r{i}"))  # must not raise
    assert led.summary()["requests"] == 3
    assert capsys.readouterr().err.count("prefixwise:") == 1  # said once


def test_sink_urls(tmp_path):
    assert isinstance(sink_from_url(str(tmp_path / "x.jsonl")), JSONLSink)
    s = sink_from_url("clickhouse://us%40er:p%40ss%25@h:8123/db.t")
    assert (s.sender.backend.user, s.sender.backend.password) == ("us@er", "p@ss%")
    s.close(0.1)
    with pytest.raises(ValueError):
        sink_from_url("clickhouse://host:8123/")  # no table
    with pytest.raises(ValueError):
        sink_from_url("postgres://host/db")
    with pytest.raises(ValueError):
        ClickHouseSink("http://h:8123", "entries; DROP TABLE x")
