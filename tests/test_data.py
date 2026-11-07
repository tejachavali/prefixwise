"""Data first (docs/DATA.md): every request leaves a row of facts, failures included."""

import sqlite3

from prefixwise.ledger import SCHEMA_VERSION, Ledger
from prefixwise.proxy import PrefixwiseProxy

SYSTEM = {"role": "system", "content": "You are a careful assistant. " * 40}


def proxy_with(payloads):
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    queue = list(payloads)

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        return queue.pop(0)

    p._forward = fake
    return p


def anthropic_ok(uncached, read, write, out=5):
    return 200, {"id": "msg", "model": "claude-sonnet-5-5",
                 "usage": {"input_tokens": uncached, "cache_read_input_tokens": read,
                           "cache_creation_input_tokens": write, "output_tokens": out}}


def body(*turns, system="Be brief."):
    msgs = []
    for i, t in enumerate(turns):
        if i:
            msgs.append({"role": "assistant", "content": "ok"})
        msgs.append({"role": "user", "content": t})
    return {"model": "claude-sonnet-5-5", "system": system * 200, "max_tokens": 10, "messages": msgs}


def test_a_verdict_row_carries_the_facts_behind_it():
    p = proxy_with([anthropic_ok(100, 0, 3000), anthropic_ok(40, 0, 3060)])
    h = {"x-prefixwise-session-id": "s"}
    p.handle("anthropic", body("first"), h)
    res = p.handle("anthropic", body("first", "second", system="Be terse."), h)[0]  # system changed
    row = p.ledger.get_request(res["request_id"])
    assert row["status"] == 200 and row["error"] is None and row["api"] == "chat" and row["stream"] == 0
    assert (row["uncached_tokens"], row["cache_write_tokens"]) == (40, 3060)
    assert row["latency_ms"] >= 0 and row["overhead_ms"] >= 0
    assert row["gap_seconds"] is not None and row["gap_seconds"] >= 0
    assert row["cause"] == "system_edit" and row["evidence"]
    assert (row["divergence_role"], row["divergence_index"]) == ("system", 0)
    assert row["schema_version"] == SCHEMA_VERSION


def test_failures_are_rows_without_a_verdict_and_do_not_skew_token_metrics():
    p = proxy_with([anthropic_ok(100, 0, 3000),
                    (529, {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})])
    p.handle("anthropic", body("one"), {"x-prefixwise-session-id": "s"})
    p.handle("anthropic", body("one", "two"), {"x-prefixwise-session-id": "s"})
    s = p.ledger.summary()
    assert (s["requests"], s["failed_requests"], s["prompt_tokens"]) == (1, 1, 3100)
    failed = p.ledger.recent(1)[0]
    assert (failed["status"], failed["error"], failed["cause"]) == (529, "overloaded_error", None)
    assert [r["cause"] for r in p.ledger.rollup("cause")] == ["first_write"]


def test_old_ledgers_gain_the_new_columns_and_old_rows_count_as_succeeded(tmp_path):
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)  # a schema-1 ledger with one row
    db.execute("""CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, agent_id TEXT,
                  provider TEXT, model TEXT, cause TEXT, legit INTEGER, prompt_tokens INTEGER,
                  cached_tokens INTEGER, avoidable_tokens INTEGER, avoidable_cost REAL, task_success INTEGER,
                  native_reason TEXT, classifier_version TEXT)""")
    db.execute("INSERT INTO entries (session_id, cause, prompt_tokens, cached_tokens, avoidable_tokens) "
               "VALUES ('s', 'none', 1000, 900, 0)")
    db.commit()
    s = Ledger(path).summary()
    assert (s["requests"], s["failed_requests"], s["cached_tokens"]) == (1, 0, 900)


def test_export_jsonl_and_csv(tmp_path):
    import csv
    import json
    import subprocess
    import sys
    from pathlib import Path

    path = str(tmp_path / "l.db")
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k", ledger_path=path)
    queue = [anthropic_ok(100, 0, 3000), anthropic_ok(40, 3000, 60)]
    p._forward = lambda *a, **k: queue.pop(0)
    p.handle("anthropic", body("a"), {"x-prefixwise-session-id": "s1"})
    p.handle("anthropic", body("a", "b"), {"x-prefixwise-session-id": "s2"})
    env = {"PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}
    run = lambda *a: subprocess.run([sys.executable, "-m", "prefixwise", "export", "--ledger", path, *a],
                                    capture_output=True, text=True, env={**__import__("os").environ, **env})
    rows = [json.loads(line) for line in run().stdout.splitlines()]
    assert [r["session_id"] for r in rows] == ["s1", "s2"] and rows[0]["cache_write_tokens"] == 3000
    only = [json.loads(line) for line in run("--where", "session=s2").stdout.splitlines()]
    assert len(only) == 1
    out = tmp_path / "x.csv"
    assert run("--format", "csv", "-o", str(out)).returncode == 0
    table = list(csv.DictReader(out.open()))
    assert len(table) == 2 and "evidence" in table[0]
    assert run("--where", "colour=red").returncode == 2


# -- review findings (docs/DATA.md principles) ---------------------------------------

def test_task_numbers_agree_everywhere_when_a_task_only_has_failed_requests():
    import json

    from prefixwise import metrics

    p = proxy_with([anthropic_ok(100, 0, 0, out=0), (500, {"error": {"type": "api_error"}})])
    p.handle("anthropic", body("a"), {"x-prefixwise-session-id": "s", "x-prefixwise-task-id": "t1"})
    p.handle("anthropic", body("b"), {"x-prefixwise-session-id": "u", "x-prefixwise-task-id": "t2"})
    p.ledger.set_task_outcome("t1", True)
    p.ledger.set_task_outcome("t2", True)
    s = p.ledger.summary()
    m = metrics.summary(p.ledger)
    legacy = json.loads(p.ledger.to_json())
    assert s["tasks"] == legacy["tasks"] == {"resolved": 2, "successful": 2} == \
        {"resolved": m["tasks_resolved"], "successful": m["tasks_successful"]}
    assert s["tokens_per_successful_task"] == legacy["tokens_per_successful_task"] \
        == m["tokens_per_successful_task"] == 50.0  # 100 tokens / 2 successful tasks


def test_percentiles_are_exact_and_cover_requests_with_a_verdict():
    from prefixwise import metrics
    from prefixwise.ledger import LedgerEntry

    led = Ledger()
    for i, ms in enumerate([10, 20, 30, 40, 1000]):
        led.record(LedgerEntry(session_id="s", agent_id="a", provider="p", model="m", cause="none",
                               prompt_tokens=1, cached_tokens=1, avoidable_tokens=0, avoidable_cost=None,
                               latency_ms=ms))
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="p", model="m", cause=None, prompt_tokens=0,
                           cached_tokens=0, avoidable_tokens=0, avoidable_cost=None, latency_ms=1, error="x",
                           status=500))  # a fast failure must not pull the percentiles down
    r = metrics.query(led, ["latency_ms_p50", "latency_ms_p95"])[0]
    assert r["latency_ms_p50"] == 30
    assert abs(r["latency_ms_p95"] - (40 + 0.8 * 960)) < 1e-9  # rank 3.8 of 0..4, interpolated
    g = metrics.query(led, ["latency_ms_p50"], by="cause")
    assert {x["cause"]: x["latency_ms_p50"] for x in g} == {"none": 30, None: None}


def test_mixed_schema_breakdown_sums_are_null_not_partial_and_old_rows_are_status_200(tmp_path):
    from prefixwise import metrics

    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, agent_id TEXT,
                  provider TEXT, model TEXT, cause TEXT, legit INTEGER, prompt_tokens INTEGER,
                  cached_tokens INTEGER, avoidable_tokens INTEGER, avoidable_cost REAL, task_success INTEGER,
                  native_reason TEXT, classifier_version TEXT)""")
    db.execute("INSERT INTO entries (session_id, cause, prompt_tokens, cached_tokens, avoidable_tokens) "
               "VALUES ('s', 'none', 1000, 900, 0)")
    db.commit()
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k", ledger_path=path)
    p._forward = lambda *a, **k: anthropic_ok(100, 0, 3000)
    p.handle("anthropic", body("a"), {"x-prefixwise-session-id": "s2"})
    m = metrics.summary(p.ledger)
    assert m["uncached_tokens"] is None and m["cache_write_tokens"] is None  # never a silently partial sum
    assert metrics.query(p.ledger, ["uncached_tokens"], filters={"session": "s2"})[0]["uncached_tokens"] == 100
    by_status = metrics.query(p.ledger, ["requests"], by="status")
    assert by_status == [{"status": 200, "requests": 2}]
    assert metrics.query(p.ledger, ["requests"], filters={"status": "200"})[0]["requests"] == 2


def test_a_response_without_usage_is_a_verdict_streamed_or_not():
    p = proxy_with([(200, {"id": "x", "content": []})])
    res = p.handle("anthropic", body("a"), {"x-prefixwise-session-id": "s"})[0]
    row = p.ledger.get_request(res["request_id"])
    assert row["error"] is None and row["cause"] is not None  # recorded with a verdict, not as a failure


def test_api_reads_do_not_hold_the_writers_lock(tmp_path):
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k", ledger_path=str(tmp_path / "l.db"))
    p._forward = lambda *a, **k: anthropic_ok(100, 0, 3000)
    p.handle("anthropic", body("a"), {"x-prefixwise-session-id": "s"})
    with p.ledger.reader() as conn:  # a long read is open...
        assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 1
        assert p.ledger._lock.acquire(timeout=0.1)  # ...and the writer's lock is free
        p.ledger._lock.release()
        p.handle("anthropic", body("a", "b"), {"x-prefixwise-session-id": "s"})  # recording proceeds
    assert p.ledger.summary()["requests"] == 2
