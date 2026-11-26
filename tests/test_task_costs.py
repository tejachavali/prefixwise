"""What a finished task cost: the report's unit when tasks have outcomes."""

import json
import time

import pytest

from prefixwise import metrics
from prefixwise.cli import main
from prefixwise.findings import _tasks, findings, render, task_costs
from prefixwise.ledger import Ledger, LedgerEntry

OPUS = "claude-opus-5-5"


def add(led, task, cost, cause="none", avoidable=0, avoidable_cost=None, error=None, at=None, prompt=1000):
    led.record(LedgerEntry(session_id=task or "s", agent_id="a", provider="anthropic", model=OPUS, cause=cause,
                           prompt_tokens=prompt, cached_tokens=0, avoidable_tokens=avoidable,
                           avoidable_cost=avoidable_cost, legit=None, task_id=task, completion_tokens=10,
                           cost=cost, error=error), at=at)


def three(path=":memory:"):
    """t1 and t3 succeeded, t2 failed, t4 has no outcome, and one request has no task."""
    led = Ledger(path)
    add(led, "t1", 0.10)
    add(led, "t1", 0.20, "tool_reserialize", 800, 0.05)
    add(led, "t2", 0.30, "volatile_system", 500, 0.02)
    add(led, "t3", 0.40)
    add(led, "t4", 0.50, "volatile_system", 900, 0.09)
    add(led, None, 0.90)
    for task, ok in (("t1", True), ("t2", False), ("t3", True)):
        led.set_task_outcome(task, ok)
    return led


def test_a_success_costs_what_all_resolved_tasks_cost_divided_by_the_successes():
    t = task_costs(three())
    assert (t["resolved"], t["successful"], t["failed"], t["unresolved"]) == (3, 2, 1, 1)
    assert t["cost_usd"] == pytest.approx(1.00)                # t1 + t2 + t3: not t4, not the request with no task
    assert t["cost_failed_usd"] == pytest.approx(0.30)
    assert t["cost_per_successful_usd"] == pytest.approx(0.50)  # the failed task was paid for too
    assert (t["tokens"], t["tokens_failed"], t["tokens_per_successful"]) == (4040, 1010, 2020.0)
    assert t["avoidable_tokens"] == 1300 and t["avoidable_cost_usd"] == pytest.approx(0.07)  # t4's 900 not counted
    assert t["top_waste"]["cause"] == "tool_reserialize" and t["top_waste"]["avoidable_tokens"] == 800


def test_it_is_the_same_number_as_the_metric():
    led = three()
    s = metrics.summary(led)
    t = task_costs(led)
    assert t["cost_per_successful_usd"] == pytest.approx(s["cost_per_successful_task_usd"])
    assert t["tokens_per_successful"] == s["tokens_per_successful_task"]
    assert (t["resolved"], t["successful"]) == (s["tasks_resolved"], s["tasks_successful"])


def test_no_outcome_no_lines():
    led = Ledger()
    add(led, "t1", 0.10)
    add(led, None, 0.10)
    assert task_costs(led) is None and findings(led)["tasks"] is None
    assert "Tasks:" not in render(findings(led))


def test_a_failed_request_has_no_tokens_or_cost_in_the_sums_and_its_task_still_counts():
    led = three()
    add(led, "t1", None, error="upstream_dropped", prompt=0)
    add(led, "t5", None, error="upstream_dropped", prompt=0)
    led.set_task_outcome("t5", False)
    t = task_costs(led)
    assert (t["resolved"], t["failed"]) == (4, 2)
    assert t["cost_usd"] == pytest.approx(1.00) and t["tokens"] == 4040
    assert t["cost_per_successful_usd"] == pytest.approx(0.50)
    # even one that carries a cost and avoidable tokens (a stream cut after its usage arrived)
    led.record(LedgerEntry(session_id="t1", agent_id="a", provider="anthropic", model=OPUS, cause="volatile_system",
                           prompt_tokens=9000, cached_tokens=0, avoidable_tokens=9000, avoidable_cost=0.60,
                           legit=None, task_id="t1", completion_tokens=10, cost=0.80, error="upstream_dropped"))
    t = task_costs(led)
    assert t["cost_usd"] == pytest.approx(1.00) and t["avoidable_cost_usd"] == pytest.approx(0.07)
    assert t["top_waste"]["cause"] == "tool_reserialize"


def test_no_dollars_unless_every_request_of_these_tasks_is_priced():
    led = three()
    add(led, "t3", None)  # answered, with a verdict, on a model with no rate
    t = task_costs(led)
    assert t["cost_usd"] is None and t["cost_per_successful_usd"] is None and t["avoidable_cost_usd"] is None
    assert t["top_waste"]["avoidable_cost_usd"] is None
    assert t["tokens_per_successful"] == 2525.0
    text = "\n".join(_tasks(t))
    assert "$" not in text and "Each success cost 2,525 tokens" in text
    assert "not every request of these tasks is priced" in text
    # a request outside these tasks that is not priced changes nothing
    led = three()
    add(led, None, None)
    add(led, "t4", None)
    assert task_costs(led)["cost_per_successful_usd"] == pytest.approx(0.50)


def test_a_window_counts_only_the_requests_in_it():
    led = Ledger()
    now = time.time()
    add(led, "old", 0.70, at=now - 7200)
    add(led, "new", 0.20, at=now - 60)
    add(led, "both", 0.30, at=now - 7200)
    add(led, "both", 0.10, at=now - 60)
    for task in ("old", "new", "both"):
        led.set_task_outcome(task, True)
    assert task_costs(led)["cost_usd"] == pytest.approx(1.30)
    t = task_costs(led, since=now - 3600)
    assert (t["resolved"], t["cost_usd"]) == (2, pytest.approx(0.30))   # `old` has no request in the window
    assert task_costs(led, until=now - 3600)["cost_usd"] == pytest.approx(1.00)
    assert task_costs(led, since=now + 60) is None


def test_the_lines_say_it_in_words():
    text = "\n".join(_tasks(task_costs(three())))
    assert text.splitlines()[0] == ("Tasks: 3 with an outcome, 2 succeeded. Each success cost $0.5000 (2,020 tokens), "
                                    "the failed task included: it was $0.3000 of the $1.00 spent (30%).")
    assert "Avoidable on these tasks: $0.0350 a success, 7% of what they cost." in text
    assert "Most of it: the same tools were sent in a different order or format." in text
    assert "1 more task has requests and no outcome yet, and is not counted here" in text
    # two failed, several without an outcome
    led = three()
    add(led, "t6", 0.20)
    add(led, "t7", 0.10)
    led.set_task_outcome("t6", False)
    text = "\n".join(_tasks(task_costs(led)))
    assert "the 2 failed tasks included: they were $0.5000 of the $1.20 spent (42%)" in text
    assert "2 more tasks have requests and no outcome yet, and are not counted here" in text
    # nothing avoidable, nothing unresolved, none failed: one line
    led = Ledger()
    add(led, "t1", 0.25)
    led.set_task_outcome("t1", True)
    assert _tasks(task_costs(led)) == ["Tasks: 1 with an outcome, 1 succeeded. Each success cost $0.2500 (1,010 tokens)."]


def test_none_succeeded():
    led = Ledger()
    add(led, "t1", 0.25, "volatile_system", 400, 0.10)
    add(led, "t2", 0.75)
    led.set_task_outcome("t1", False)
    led.set_task_outcome("t2", False)
    t = task_costs(led)
    assert t["cost_per_successful_usd"] is None and t["tokens_per_successful"] is None
    lines = _tasks(t)
    assert lines[0] == "Tasks: 2 with an outcome, none succeeded: $1.00 spent on them."
    assert lines[1].startswith("  Avoidable on these tasks: 10% of what they cost.")


def test_the_report_and_its_json_carry_it(tmp_path, capsys):
    path = str(tmp_path / "l.db")
    three(path)
    assert main(["report", "--ledger", path]) == 0
    out = capsys.readouterr().out
    assert "Tasks: 3 with an outcome, 2 succeeded. Each success cost $0.5000" in out
    assert main(["report", "--ledger", path, "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["findings"]["tasks"]["cost_per_successful_usd"] == pytest.approx(0.50)
