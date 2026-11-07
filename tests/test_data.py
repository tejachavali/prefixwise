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
