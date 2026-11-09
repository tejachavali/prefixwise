"""`prefixwise report --session ID|latest` and GET /api/v1/sessions/{id}/turns: one session turn by
turn, with the block where the prompt first changed. From r/opencodeCLI (Individual-Quote-958): "the
first point where two consecutive requests stop sharing a prefix, plus which part changed."""

import json

from prefixwise.cli import main
from prefixwise.proxy import PrefixwiseProxy

A = {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
B = {"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}
Q1 = [{"role": "user", "content": "q"}]
Q2 = Q1 + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}]
Q3 = Q2 + [{"role": "assistant", "content": "b"}, {"role": "user", "content": "q3"}]


def _session(path: str) -> PrefixwiseProxy:
    p = PrefixwiseProxy(ledger_path=path, mock=True)
    p.handle("openai", {"model": "m", "session_id": "s", "tools": [A, B], "messages": Q1})
    p.handle("openai", {"model": "m", "session_id": "s", "tools": [A, B], "messages": Q2})
    p.handle("openai", {"model": "m", "session_id": "s", "tools": [B, A], "messages": Q3})  # tools reordered
    p.handle("openai", {"model": "m", "session_id": "s", "tools": [B, A],
                        "messages": [{"role": "user", "content": "compacted"}]})  # history rewritten
    return p


def test_turns_number_the_session_and_name_the_first_changed_block(tmp_path):
    path = str(tmp_path / "l.db")
    led = _session(path).ledger
    turns = led.turns("s")
    assert [t["turn"] for t in turns] == [1, 2, 3, 4]
    assert [t["cause"] for t in turns] == ["first_write", "none", "tool_reserialize", "history_edit"]
    assert turns[0]["first_changed"] is None  # nothing to compare with
    assert turns[1]["first_changed"] == "messages[3]"  # the appended turn: the only change, and a hit
    assert turns[2]["first_changed"] == "tools[0]"  # the tool list broke the prefix at its first tool
    assert turns[3]["first_changed"] == "messages[2]"
    assert turns[1]["hit_rate"] == 1.0 and turns[2]["hit_rate"] == 0.0
    assert led.turns("nope") == [] and led.latest_session() == "s"
    assert all("content" not in json.dumps(t) and "q2" not in json.dumps(t) for t in turns)  # hashes and counts only


def test_report_session_renders_the_table(tmp_path, capsys):
    path = str(tmp_path / "l.db")
    _session(path)
    assert main(["report", "--ledger", path, "--session", "s"]) == 0
    out = capsys.readouterr().out
    head, table = out.split("\n\n", 1)
    assert head.startswith("Session s: 4 requests") and "cache hit rate 25.0%" in head and "4,000 avoidable" in head
    lines = [ln for ln in table.splitlines() if ln[:1].isdigit()]
    assert len(lines) == 4
    assert lines[0].split()[:5] == ["1", "2,000", "0", "0.0%", "first_write"] and lines[0].split()[5] == "-"
    assert lines[1].split()[:6] == ["2", "2,000", "2,000", "100.0%", "none", "messages[3]"]
    assert lines[2].split()[:6] == ["3", "2,000", "0", "0.0%", "tool_reserialize", "tools[0]"]
    assert lines[3].split()[4:6] == ["history_edit", "messages[2]"]
    assert "first changed: the block where" in out


def test_report_session_latest_and_json_and_unknown(tmp_path, capsys):
    path = str(tmp_path / "l.db")
    p = _session(path)
    p.handle("openai", {"model": "m", "session_id": "later", "tools": [A], "messages": Q1})
    assert main(["report", "--ledger", path, "--session", "latest", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["session_id"] == "later" and [t["turn"] for t in data["turns"]] == [1]
    assert main(["report", "--ledger", path, "--session", "nope"]) == 1
    assert "no requests recorded for session 'nope'" in capsys.readouterr().err
    empty = str(tmp_path / "e.db")
    PrefixwiseProxy(ledger_path=empty, mock=True)
    assert main(["report", "--ledger", empty, "--session", "latest"]) == 1
    assert "no sessions yet" in capsys.readouterr().err


def test_report_session_shows_a_failed_request_and_a_detail(tmp_path, capsys):
    from prefixwise.ledger import Ledger, LedgerEntry

    path = str(tmp_path / "l.db")
    led = Ledger(path)
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model="m", cause="first_write",
                           prompt_tokens=3000, cached_tokens=0, avoidable_tokens=0, avoidable_cost=None, request_id="r1"))
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model="m", cause="none",
                           prompt_tokens=0, cached_tokens=0, avoidable_tokens=0, avoidable_cost=None, request_id="r2",
                           status=529, error="overloaded"))
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model="m", cause="ttl_expired",
                           prompt_tokens=3000, cached_tokens=0, avoidable_tokens=3000, avoidable_cost=0.0045,
                           request_id="r3", detail="ttl_shortened", divergence_role="messages", divergence_index=4))
    assert main(["report", "--ledger", path, "--session", "s"]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln[:1].isdigit()]
    assert "failed: overloaded" in lines[1]
    assert "ttl_expired (ttl_shortened)" in lines[2] and "messages[4]" in lines[2] and "$0.0045" in lines[2]


def test_api_session_turns(tmp_path):
    from prefixwise.api import get

    path = str(tmp_path / "l.db")
    p = _session(path)
    status, body = get(p, "/api/v1/sessions/s/turns", "")
    assert status == 200 and body["session_id"] == "s" and [t["turn"] for t in body["turns"]] == [1, 2, 3, 4]
    assert body["turns"][2]["first_changed"] == "tools[0]"
    status, body = get(p, "/api/v1/sessions/latest/turns", "")
    assert status == 200 and body["session_id"] == "s"
    assert get(p, "/api/v1/sessions/nope/turns", "")[0] == 404


def test_turns_are_in_time_order_and_since_keeps_the_numbering(tmp_path, capsys):
    """Imported logs are recorded out of time order (a Claude Code main log, then its subagents'):
    the view orders by time and names the agent when a session has more than one."""
    from prefixwise.ledger import Ledger, LedgerEntry

    path = str(tmp_path / "l.db")
    led = Ledger(path)

    def row(rid, agent, at, cause="none"):
        return LedgerEntry(session_id="s", agent_id=agent, provider="anthropic", model="m", cause=cause,
                           prompt_tokens=1000, cached_tokens=500, avoidable_tokens=0, avoidable_cost=None, request_id=rid)

    led.record(row("m1", "main", 100.0, "first_write"), at=100.0)
    led.record(row("m2", "main", 300.0), at=300.0)
    led.record(row("s1", "sub", 200.0, "first_write"), at=200.0)  # recorded last, happened in between
    turns = led.turns("s")
    assert [(t["turn"], t["request_id"]) for t in turns] == [(1, "m1"), (2, "s1"), (3, "m2")]
    assert [(t["turn"], t["request_id"]) for t in led.turns("s", since=250.0)] == [(3, "m2")]
    assert led.latest_session() == "s"
    main(["report", "--ledger", path, "--session", "s"])
    out = capsys.readouterr().out
    header = [ln for ln in out.splitlines() if ln.startswith("turn")][0].split()
    assert header[:2] == ["turn", "agent"]
    assert main(["report", "--ledger", path, "--session", "s", "--since", "250"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln[:1].isdigit()]
    assert len(lines) == 1 and lines[0].split()[0] == "3"
    assert main(["report", "--ledger", path, "--session", "s", "--since", "1000"]) == 1
    assert "in that window" in capsys.readouterr().err
