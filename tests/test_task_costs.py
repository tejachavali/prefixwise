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
    assert "Tokens per successful task" not in render(findings(three()))  # the line this replaces is gone


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
    assert t["top_waste"]["cause"] == "tool_reserialize" and t["avoidable_tokens"] == 1300


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


# -- what an independent review found ------------------------------------------------------------

def test_the_cause_named_is_the_one_that_cost_most_and_is_called_most_only_above_half():
    # more tokens on a cheap model, more dollars on a dear one: dollars decide
    led = Ledger()
    add(led, "t1", 0.10, "volatile_system", 600, 0.01)
    add(led, "t1", 0.10, "tool_reserialize", 500, 0.09)
    led.set_task_outcome("t1", True)
    t = task_costs(led)
    assert t["top_waste"]["cause"] == "tool_reserialize" and t["top_waste"]["share"] == pytest.approx(0.9)
    assert "Most of it: the same tools were sent in a different order or format." in _tasks(t)[1]
    # three causes, none above half: its share is given, and it is not called most
    led = Ledger()
    for cause, tokens in (("volatile_system", 400), ("tool_reserialize", 300), ("history_edit", 300)):
        add(led, "t1", 0.10, cause, tokens, tokens / 1e4)
    led.set_task_outcome("t1", True)
    t = task_costs(led)
    assert t["top_waste"]["cause"] == "volatile_system" and t["top_waste"]["share"] == pytest.approx(0.4)
    line = _tasks(t)[1]
    assert "The largest part (40%): the system prompt changed only in a date, time or id." in line
    assert "Most of it" not in line


def test_avoidable_dollars_are_not_given_when_a_priced_request_has_none():
    """A price file with input and output rates and no cache-read rate: the request has a cost, its
    avoidable part has none. The tokens are given, not "$0.00, 0%"."""
    led = Ledger()
    add(led, "t1", 0.0031, "volatile_system", 3000, None)
    add(led, "t1", 0.0020, "tool_reserialize", 500, 0.0004)
    led.set_task_outcome("t1", True)
    t = task_costs(led)
    assert t["cost_usd"] == pytest.approx(0.0051) and t["avoidable_cost_usd"] is None
    assert t["top_waste"]["cause"] == "volatile_system" and t["top_waste"]["avoidable_cost_usd"] is None  # by tokens
    line = _tasks(t)[1]
    assert line.startswith("  Avoidable on these tasks: 3,500 tokens (their dollars are not known: a rate is missing")
    assert "0%" not in line and "$" not in line


def test_with_a_window_the_lines_say_when_a_counted_task_has_requests_outside_it():
    led = Ledger()
    now = time.time()
    add(led, "t1", 1.00, at=now - 7200)
    add(led, "t1", 0.10, at=now - 60)
    add(led, "t2", 0.30, at=now - 60)
    add(led, "open", 0.50, at=now - 7200)  # no outcome: straddles the window too, and is not one of "these tasks"
    add(led, "open", 0.50, at=now - 60)
    for task in ("t1", "t2"):
        led.set_task_outcome(task, True)
    whole = task_costs(led)
    assert (whole["window_only"], whole["tasks_cut"]) == (False, 0)
    assert "window" not in "\n".join(_tasks(whole))
    t = task_costs(led, since=now - 3600)
    assert (t["window_only"], t["tasks_cut"]) == (True, 1) and t["cost_per_successful_usd"] == pytest.approx(0.20)
    lines = _tasks(t)
    assert lines[0] == "Tasks: 2 with an outcome, 2 succeeded. Each success cost $0.2000 (1,010 tokens) in this window."
    assert lines[1] == ("  1 of these tasks also has requests outside this window, which are not counted: "
                        "it cost more than this.")
    # a window that holds every request of its tasks says nothing about windows
    t = task_costs(led, since=now - 9000)
    assert (t["window_only"], t["tasks_cut"]) == (True, 0) and "window" not in "\n".join(_tasks(t))
    # the other end: requests after the window
    t = task_costs(led, until=now - 3600)
    assert (t["window_only"], t["resolved"], t["tasks_cut"]) == (True, 1, 1)
    assert "1 of these tasks also has requests outside this window" in "\n".join(_tasks(t))
    # `prefixwise run` prints the same findings for its own run, and says "run"
    text = render(findings(led, since=now - 3600), command="run")
    assert "Each success cost $0.2000 (1,010 tokens) in this run." in text
    assert "also has requests outside this run" in text and "window" not in text.split("Tasks:")[1]
    assert "this run" not in render(findings(led, since=now - 9000), command="run").split("Tasks:")[1]


def test_exactly_half_is_not_most_and_a_tie_goes_to_the_cause_with_more_tokens():
    led = Ledger()
    add(led, "t1", 0.10, "volatile_system", 300, 0.02)
    add(led, "t1", 0.10, "tool_reserialize", 500, 0.02)
    led.set_task_outcome("t1", True)
    t = task_costs(led)
    assert t["top_waste"]["cause"] == "tool_reserialize" and t["top_waste"]["share"] == pytest.approx(0.5)
    assert "The largest part (50%): the same tools" in _tasks(t)[1]


def test_a_model_priced_at_zero_and_an_odd_price_file():
    # everything costs $0.00: the avoidable tokens are given, with no claim that a rate is missing
    led = Ledger()
    add(led, "t1", 0.0, "volatile_system", 800, 0.0)
    led.set_task_outcome("t1", True)
    line = _tasks(task_costs(led))[1]
    assert line.startswith("  Avoidable on these tasks: 800 tokens.") and "not known" not in line
    # a read rate above the input rate gives a negative avoidable cost on a row: counted as zero
    led = Ledger()
    add(led, "t1", 0.20, "volatile_system", 800, -0.04)
    add(led, "t1", 0.20, "tool_reserialize", 100, 0.01)
    led.set_task_outcome("t1", True)
    t = task_costs(led)
    assert t["avoidable_cost_usd"] == pytest.approx(0.01) and t["top_waste"]["cause"] == "tool_reserialize"
    assert "-" not in _tasks(t)[1]


def test_a_task_whose_every_request_failed_cost_nothing_as_the_metric_says():
    led = Ledger()
    add(led, "t1", None, error="upstream_dropped", prompt=0)
    led.set_task_outcome("t1", True)
    t = task_costs(led)
    assert t["cost_per_successful_usd"] == 0.0 == metrics.summary(led)["cost_per_successful_task_usd"]
    assert "No dollars" not in "\n".join(_tasks(t))


def test_an_empty_task_id_is_no_task():
    led = three()
    add(led, "", 0.10)
    assert task_costs(led)["unresolved"] == 1  # t4 alone


def test_a_sum_too_small_for_four_decimals_is_said_as_such():
    led = Ledger()
    add(led, "t1", 0.00002, "volatile_system", 10, 0.00001)
    led.set_task_outcome("t1", True)
    text = "\n".join(_tasks(task_costs(led)))
    assert "$0.0000" not in text and "Each success cost under $0.0001" in text
