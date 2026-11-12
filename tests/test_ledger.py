from prefixwise.ledger import Ledger, LedgerEntry, avoidable_for_turn, avoidable_tokens, reuse_adjusted_reads


def entry(**kw):
    d = dict(session_id="s1", agent_id="a1", provider="openai", model="gpt-5.6",
             cause="volatile_system", prompt_tokens=1000, cached_tokens=200,
             avoidable_tokens=800, avoidable_cost=0.01, task_success=True)
    d.update(kw)
    return LedgerEntry(**d)


def test_avoidable_tokens():
    assert avoidable_tokens(1000, 200) == 800
    assert avoidable_tokens(1000, 200, first_write_tokens=800) == 0


def test_tokens_per_successful_task_counts_failures_and_tasks_not_requests():
    led = Ledger()
    # Task t1 succeeded in two requests; task t2 failed in one request.
    led.record(entry(task_id="t1", task_success=None, prompt_tokens=1000, completion_tokens=100))
    led.record(entry(task_id="t1", task_success=None, prompt_tokens=1500, completion_tokens=100))
    led.record(entry(task_id="t2", task_success=None, prompt_tokens=900, completion_tokens=0))
    assert led.tokens_per_successful_task() is None  # nothing resolved yet
    led.set_task_outcome("t1", True)
    led.set_task_outcome("t2", False)
    # (1100 + 1600 + 900) / 1 successful task; the failure's tokens count.
    assert led.tokens_per_successful_task() == 3600.0
    assert led.task_counts() == {"resolved": 2, "successful": 1}


def test_unresolved_and_untasked_requests_are_left_out():
    led = Ledger()
    led.record(entry(task_id="done", task_success=True, prompt_tokens=1000))
    led.record(entry(task_id="in-progress", task_success=None, prompt_tokens=5000))
    led.record(entry(task_id=None, task_success=None, prompt_tokens=7000))
    assert led.tokens_per_successful_task() == 1000.0


def test_task_outcome_overwrites_and_agent_filter():
    led = Ledger()
    led.record(entry(task_id="t", agent_id="a1", task_success=None, prompt_tokens=400))
    led.set_task_outcome("t", False)
    assert led.tokens_per_successful_task() is None
    led.set_task_outcome("t", True)  # retried and passed
    assert led.tokens_per_successful_task() == 400.0
    assert led.tokens_per_successful_task("a1") == 400.0
    assert led.tokens_per_successful_task("nope") is None


def test_old_ledger_file_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, agent_id TEXT, provider TEXT, model TEXT, cause TEXT, legit INTEGER,
        prompt_tokens INTEGER, cached_tokens INTEGER, avoidable_tokens INTEGER,
        avoidable_cost REAL, task_success INTEGER, native_reason TEXT, classifier_version TEXT)""")
    conn.execute("INSERT INTO entries (session_id, cause, prompt_tokens) VALUES ('s', 'none', 10)")
    conn.commit()
    conn.close()
    led = Ledger(str(path))
    led.record(entry(task_id="t", task_success=True, request_id="r1"))
    assert led.get_request("r1")["task_id"] == "t"
    assert sum(r["requests"] for r in led.rollup_by_cause()) == 2  # old row kept


def test_rollup_by_cause():
    led = Ledger()
    led.record(entry(cause="volatile_system", avoidable_tokens=800))
    led.record(entry(cause="tool_reserialize", avoidable_tokens=100))
    by_cause = {r["cause"]: r for r in led.rollup_by_cause()}
    assert by_cause["volatile_system"]["avoidable_tokens"] == 800


def test_reuse_adjusted_reads_is_a_model():
    # 1.25 / 0.9 ~= 1.4, 2.5 / 0.9 ~= 2.8 (model, not a verified fact).
    assert abs(reuse_adjusted_reads(1.25, 0.1) - 1.388) < 0.01
    assert abs(reuse_adjusted_reads(2.5, 0.1) - 2.777) < 0.01


def test_avoidable_for_turn_counts_waste_only():
    # Legitimate change (model switch, catalog change, compaction): not waste.
    assert avoidable_for_turn(True, 2000, 0, 2000) == 0
    # Undetermined legitimacy (NONE, routing, unknown): not counted, never a guess.
    assert avoidable_for_turn(None, 2400, 1900, 2000) == 0
    # Waste: the reusable prior prefix that was not read from cache.
    assert avoidable_for_turn(False, 2000, 400, 2000) == 1600
    # Tokens appended after the prior prefix are new content, not avoidable.
    assert avoidable_for_turn(False, 2500, 0, 2000) == 2000
    # No prior turn's usage to compare against: how much was reusable is not known, so nothing is
    # claimed (counting this whole prompt would count appended content as waste).
    assert avoidable_for_turn(False, 2000, 0, None) == 0
    assert avoidable_for_turn(False, 2000, 0, 0) == 0
