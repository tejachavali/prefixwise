"""`--fix auto_breakpoints` (docs/specs/check-and-optimize.md section 2.2): cache_control markers
for an Anthropic request that carries none, placed only when a write can pay."""

import json

from prefixwise.fixes import FIX_ADVICE, FIX_STATUS, LOOKBACK_BLOCKS, auto_breakpoints, parse_fixes
from prefixwise.proxy import PrefixwiseProxy

BIG = "x" * 8000  # about 2,000 tokens: above every Anthropic minimum


def _marked(value) -> bool:
    return "cache_control" in json.dumps(value)


def _turn(n: int, system=BIG, **extra) -> dict:
    msgs = [{"role": "user", "content": "q"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f"a{i}"}, {"role": "user", "content": f"q{i + 1}"}]
    return {"model": "claude-sonnet-5", "session_id": "s", "system": system, "messages": msgs, **extra}


def test_markers_end_the_prefix_and_the_last_message_and_never_mutate():
    body = _turn(1)
    out = auto_breakpoints(body)
    assert not _marked(body)
    assert out["system"] == [{"type": "text", "text": BIG, "cache_control": {"type": "ephemeral"}}]
    assert out["messages"][-1]["content"] == [{"type": "text", "text": "q1", "cache_control": {"type": "ephemeral"}}]
    assert not _marked(out["messages"][:-1])
    # a system list keeps its blocks; only the last is marked (it covers the tools before it)
    tools = [{"name": "t", "input_schema": {"type": "object"}}]
    out = auto_breakpoints({**_turn(0), "tools": tools, "system": [{"type": "text", "text": BIG}, {"type": "text", "text": "b"}]})
    assert [_marked(b) for b in out["system"]] == [False, True] and not _marked(out["tools"])
    # no system: the last tool ends the prefix
    out = auto_breakpoints({"model": "m", "tools": [{"name": "a", "input_schema": {"d": BIG}}, {"name": "b", "input_schema": {}}],
                            "messages": [{"role": "user", "content": "q"}]})
    assert [_marked(t) for t in out["tools"]] == [False, True]


def test_nothing_placed_when_the_client_set_a_marker_or_there_is_nothing_to_cache():
    own = _turn(1)
    own["messages"][0] = {"role": "user", "content": [{"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}]}
    assert auto_breakpoints(own) is None  # a client's markers are never moved or added to
    assert auto_breakpoints({"model": "m", "messages": [{"role": "user", "content": "q"}]}) is None  # no prefix
    assert auto_breakpoints(_turn(1, system="short")) is None  # below the minimum: Anthropic would ignore it
    assert auto_breakpoints(_turn(1, system="short"), min_prefix_tokens=1) is not None


def test_third_marker_on_the_previous_user_turn_only_past_the_lookback():
    msgs = [{"role": "user", "content": "q"}]
    msgs += [{"role": "assistant", "content": [{"type": "text", "text": str(i)}]} for i in range(LOOKBACK_BLOCKS + 1)]
    msgs += [{"role": "user", "content": "last"}]
    out = auto_breakpoints({"system": BIG, "messages": msgs})
    assert [i for i, m in enumerate(out["messages"]) if _marked(m)] == [0, len(msgs) - 1]
    short = auto_breakpoints({"system": BIG, "messages": msgs[:3] + [msgs[-1]]})
    assert [i for i, m in enumerate(short["messages"]) if _marked(m)] == [3]
    # a tool_result user turn is a block like any other
    tr = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}]},
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]}]
    out = auto_breakpoints({"system": BIG, "messages": tr})
    assert out["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_status_and_advice():
    from pathlib import Path

    status = FIX_STATUS["auto_breakpoints"]
    assert status["status"] == "proven"  # E4 live 2026-11-09; the status must cite the committed run
    assert (Path(__file__).resolve().parents[1] / status["evidence"].split()[-1]).exists()
    assert FIX_ADVICE["breakpoint_misplaced"]["toggle"] == "auto_breakpoints"
    assert "auto_breakpoints" in parse_fixes("optimize")  # proven: --optimize turns it on
    assert set(parse_fixes("optimize,auto_breakpoints")) == {"canonical_tools", "auto_breakpoints"}


def test_proxy_places_markers_only_once_the_prefix_was_seen_within_the_ttl():
    """Off: an Anthropic session without markers is breakpoint_misplaced every turn. On: the first
    turn is sent as is (no write for a prefix nobody has reused), the second carries markers and
    from then on the prefix is read."""
    off, on = PrefixwiseProxy(mock=True), PrefixwiseProxy(mock=True, fixes=["auto_breakpoints"])
    for p in (off, on):
        r, e = p.handle("anthropic", _turn(0))
        assert r["cause"] == "first_write" and e.fixes is None
    r, e = off.handle("anthropic", _turn(1))
    assert r["cause"] == "breakpoint_misplaced" and e.fixes is None
    r, e = on.handle("anthropic", _turn(1))  # the first marked turn writes: the unmarked turn before it cached nothing
    assert r["cause"] == "first_write" and e.fixes == "auto_breakpoints" and r["fixes_applied"] == ["auto_breakpoints"]
    assert r["response"]["usage"]["cache_creation_input_tokens"] > 0 and e.cached_tokens == 0
    r, e = on.handle("anthropic", _turn(2))
    assert r["cause"] == "none" and e.cached_tokens > 0


def test_proxy_leaves_a_prefix_alone_that_matches_only_with_its_timestamp_masked():
    """A system prompt rewritten every turn (a clock in it) is seen every turn under its masked key,
    but a marker on it would buy a write nobody reads: no markers, and the cause stays volatile_system."""
    p = PrefixwiseProxy(mock=True, fixes=["auto_breakpoints"])
    for n in range(3):
        r, e = p.handle("anthropic", _turn(n, system=BIG + f" now 2026-11-09T19:0{n}:00Z"))
        assert e.fixes is None
    assert r["cause"] == "volatile_system"


def test_no_marker_on_a_thinking_block():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}]}]
    out = auto_breakpoints({"system": BIG, "messages": msgs})
    assert _marked(out["system"]) and not _marked(out["messages"])


def test_proxy_places_markers_for_a_new_session_whose_prefix_another_session_wrote():
    p = PrefixwiseProxy(mock=True, fixes=["auto_breakpoints"])
    p.handle("anthropic", _turn(0))
    other = {**_turn(0), "session_id": "s2"}
    r, e = p.handle("anthropic", other)
    assert e.fixes == "auto_breakpoints"


def test_proxy_sends_the_clients_body_as_it_was_and_leaves_no_fix_recorded_when_it_was_not_applied():
    p = PrefixwiseProxy(mock=True, fixes=["auto_breakpoints"])
    own = _turn(0)
    own["messages"][0] = {"role": "user", "content": [{"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}]}
    p.handle("anthropic", own)
    r, e = p.handle("anthropic", {**_turn(1), "messages": own["messages"] + _turn(1)["messages"][1:]})
    assert e.fixes is None and r["fixes_applied"] == []


def test_a_400_about_cache_control_resends_the_clients_request_and_stops_after_twice(capsys):
    p = PrefixwiseProxy(mock=True, fixes=["auto_breakpoints"])
    real, sent = p._forward, []

    def fake(provider, body, *a, **k):
        sent.append(body)
        if "cache_control" in json.dumps(body):
            return 400, {"type": "error", "error": {"type": "invalid_request_error",
                                                    "message": "cache_control is not supported here"}}
        return real(provider, body, *a, **k)

    p._forward = fake
    p.handle("anthropic", _turn(0))
    for n in (1, 2):
        r, e = p.handle("anthropic", _turn(n))
        assert r["status"] == 200 and e is not None and e.fixes is None and r["fixes_applied"] == []
        assert _marked(sent[-2]) and not _marked(sent[-1]) and sent[-1]["messages"] == _turn(n)["messages"]
        assert e.cause == "breakpoint_misplaced"  # the record says what the provider received: no marker
    assert ("anthropic", "auto_breakpoints") in p.rejected_fields
    r, e = p.handle("anthropic", _turn(3))
    assert not _marked(sent[-1]) and r["fixes_applied"] == []  # no third try
    assert "auto_breakpoints" in capsys.readouterr().err


def test_fixes_says_whether_the_placed_markers_paid(tmp_path, capsys):
    from prefixwise.cli import main
    from prefixwise.fixes import breakpoints_paid

    path = str(tmp_path / "l.db")
    p = PrefixwiseProxy(ledger_path=path, mock=True, fixes=["auto_breakpoints"])
    assert breakpoints_paid(p.ledger) is None  # nothing rewritten yet
    for n in range(4):
        p.handle("anthropic", _turn(n))
    paid = breakpoints_paid(p.ledger)
    # turn 2 wrote 2,000 (the first marked turn), turns 3 and 4 read 2,000 each: 3 rewritten rows
    assert paid["requests"] == 3 and paid["written_tokens"] == 2000 and paid["read_tokens"] == 4000
    assert paid["write_surcharge_token_equivalents"] == 500 and paid["read_saving_token_equivalents"] == 3600
    assert paid["paid"] is True and paid["net_token_equivalents"] == 3100
    assert main(["fixes", "--ledger", path]) == 0
    out = capsys.readouterr().out
    assert "auto_breakpoints rewrote 3 requests" in out and "paid for itself" in out and "read 4,000 back" in out
    assert main(["fixes", "--ledger", path, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["auto_breakpoints"]["paid"] is True
    # one write, nothing read back yet: not paid
    q = PrefixwiseProxy(ledger_path=str(tmp_path / "m.db"), mock=True, fixes=["auto_breakpoints"])
    q.handle("anthropic", _turn(0))
    q.handle("anthropic", _turn(1))
    assert breakpoints_paid(q.ledger)["paid"] is False
