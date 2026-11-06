import json

import pytest

from prefixwise.cli import main
from prefixwise.ledger import Ledger, LedgerEntry

TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}},
         {"type": "function", "function": {"name": "calc", "parameters": {}}}]
BASE = {"model": "gpt-5.6", "tools": TOOLS,
        "messages": [{"role": "system", "content": "You are an agent. Today is 2026-11-05."},
                     {"role": "user", "content": "fix the bug"}]}


def write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def variant(**changes):
    body = json.loads(json.dumps(BASE))
    for k, fn in changes.items():
        body[k] = fn(body[k])
    return body


def test_explain_pair_volatile_system(tmp_path, capsys):
    a = write(tmp_path / "a.json", BASE)
    b = write(tmp_path / "b.json", variant(messages=lambda m: [
        {**m[0], "content": m[0]["content"].replace("05", "06")}, m[1]]))
    assert main(["explain", a, b]) == 0
    out = capsys.readouterr().out
    assert "VOLATILE_SYSTEM" in out and "WASTE" in out and "only volatile values" in out


def test_explain_pair_tools_reordered_json(tmp_path, capsys):
    a = write(tmp_path / "a.json", BASE)
    b = write(tmp_path / "b.json", variant(tools=lambda t: t[::-1]))
    main(["explain", a, b, "--json"])
    [res] = json.loads(capsys.readouterr().out)
    assert res["cause"] == "tool_reserialize" and res["legit"] is False
    assert res["divergence"]["kind"].startswith("tools reordered")
    assert res["offline"] is True


def test_explain_session_jsonl_with_usage(tmp_path, capsys):
    t2 = variant(messages=lambda m: m + [{"role": "assistant", "content": "ok"},
                                         {"role": "user", "content": "next"}])
    lines = [
        {"request": BASE, "usage": {"prompt_tokens": 3000}},
        {"request": t2, "usage": {"prompt_tokens": 3200, "prompt_tokens_details": {"cached_tokens": 2900}}},
        {"request": t2, "usage": {"prompt_tokens": 3200, "prompt_tokens_details": {"cached_tokens": 0}},
         "gap_seconds": 5},
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    main(["explain", str(path), "--json"])
    causes = [r["cause"] for r in json.loads(capsys.readouterr().out)]
    assert causes == ["first_write", "none", "retry_replay"]


def test_explain_sessions_are_kept_apart(tmp_path, capsys):
    lines = [{"request": BASE, "session_id": "a"}, {"request": BASE, "session_id": "b"}]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(lines))
    main(["explain", str(path), "--json"])
    assert [r["cause"] for r in json.loads(capsys.readouterr().out)] == ["first_write", "first_write"]


def test_explain_anthropic_detected(tmp_path, capsys):
    body = {"model": "claude-sonnet-5-5", "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}]}
    a = write(tmp_path / "a.json", body)
    main(["explain", a, a, "--json"])
    [res] = json.loads(capsys.readouterr().out)
    # Identical, but no cache_control anywhere: Anthropic will not cache it.
    assert res["provider"] == "anthropic" and res["cause"] == "breakpoint_misplaced"


def test_explain_bad_input(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(SystemExit):
        main(["explain", str(bad)])


def _ledger(tmp_path):
    path = str(tmp_path / "l.db")
    led = Ledger(path)
    for i, (cause, legit, avoid) in enumerate([("first_write", True, 0), ("volatile_system", False, 900),
                                               ("none", None, 0)]):
        led.record(LedgerEntry(session_id="s", agent_id="a", provider="openai", model="m",
                               cause=cause, legit=legit, prompt_tokens=1000, cached_tokens=100 * i,
                               avoidable_tokens=avoid, avoidable_cost=None, task_id="t1",
                               request_id=f"r{i}", completion_tokens=10))
    return path


def test_task_report_request(tmp_path, capsys):
    path = _ledger(tmp_path)
    assert main(["task", "t1", "--success", "--ledger", path]) == 0
    capsys.readouterr()
    main(["report", "--ledger", path, "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rep["summary"]["tokens_per_successful_task"] == 3030.0
    assert rep["rows"][0]["cause"] == "volatile_system"  # most avoidable first
    main(["report", "--ledger", path, "--by", "task_id"])
    out = capsys.readouterr().out
    assert "1 succeeded / 1 resolved" in out and "t1" in out
    assert main(["request", "r1", "--ledger", path, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cause"] == "volatile_system"
    assert main(["request", "nope", "--ledger", path]) == 1


def test_missing_ledger_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["report", "--ledger", str(tmp_path / "none.db")])
    assert "no ledger" in str(exc.value)
