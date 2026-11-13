"""`prefixwise report --compare A B`: two tasks side by side, with the difference. A user comparing
one long session against several short ones for the same task asked for exactly this view; the
report summed the two tasks together."""

import json

import pytest

from prefixwise.cli import main
from prefixwise.ledger import Ledger, LedgerEntry


def _entry(task, session, prompt, cached, out, cost):
    return LedgerEntry(session_id=session, agent_id="a", provider="openai", model="m", cause="none", legit=None,
                       prompt_tokens=prompt, cached_tokens=cached, avoidable_tokens=0, avoidable_cost=0.0,
                       task_id=task, completion_tokens=out, cost=cost)


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "l.db")
    led = Ledger(path)
    for i in range(20):  # one long session: prompts grow, nearly all cached
        led.record(_entry("long", "long-0", 500 + 50 * i, 450 + 50 * i if i else 0, 60, 0.00005))
    for s in range(4):   # four short sessions, each with its own first write
        for i in range(6):
            led.record(_entry("short", f"short-{s}", 500 + 40 * i, 460 + 40 * i if i else 0, 50, 0.00003))
    led.set_task_outcome("long", True)
    led.set_task_outcome("short", True)
    return path


def test_two_tasks_side_by_side_with_the_difference(ledger, capsys):
    assert main(["report", "--ledger", ledger, "--compare", "long", "short"]) == 0
    out = capsys.readouterr().out
    lines = {l.strip().split("  ")[0]: l for l in out.splitlines() if l.strip()}
    assert "20" in lines["requests"] and "24" in lines["requests"] and "+20.0%" in lines["requests"]
    assert "$0.00100" in lines["cost"] and "$0.000720" in lines["cost"] and "-28.0%" in lines["cost"]
    assert "pts" in lines["from cache"] and lines["outcome"].count("succeeded") == 2


def test_json_carries_both_tasks_and_the_change(ledger, capsys):
    assert main(["report", "--ledger", ledger, "--compare", "long", "short", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["tasks"] == ["long", "short"] and d["long"]["requests"] == 20 and d["short"]["requests"] == 24
    assert d["change"]["requests"] == pytest.approx(0.2) and d["change"]["cost_usd"] == pytest.approx(-0.28)
    assert d["long"]["outcome"] is True


def test_an_unknown_task_is_an_error_not_a_zero_row(ledger, capsys):
    assert main(["report", "--ledger", ledger, "--compare", "long", "nope"]) == 1
    assert "no requests with task id 'nope'" in capsys.readouterr().err


def test_cost_only_when_every_request_is_priced(tmp_path, capsys):
    path = str(tmp_path / "p.db")
    led = Ledger(path)
    led.record(_entry("a", "s1", 1000, 0, 10, 0.001))
    led.record(_entry("a", "s1", 1100, 1000, 10, None))  # one unpriced request: no partial total
    led.record(_entry("b", "s2", 1000, 0, 10, 0.001))
    assert main(["report", "--ledger", path, "--compare", "a", "b"]) == 0
    out = capsys.readouterr().out
    cost = next(l for l in out.splitlines() if l.startswith("cost"))
    assert cost.split()[1] == "-" and "every request is priced" in out
    assert "Record outcomes" in out  # neither task has an outcome


def test_the_report_points_to_the_per_task_views_only_with_several_tasks(ledger, tmp_path, capsys):
    assert main(["report", "--ledger", ledger]) == 0
    assert "--compare TASK TASK" in capsys.readouterr().out
    one = str(tmp_path / "one.db")
    Ledger(one).record(_entry("only", "s", 1000, 0, 10, 0.001))
    assert main(["report", "--ledger", one]) == 0
    assert "--compare" not in capsys.readouterr().out
