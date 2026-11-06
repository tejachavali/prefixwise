import json

import pytest

from prefixwise.cli import main
from prefixwise.fixes import apply_fixes, canonical_tools, parse_fixes
from prefixwise.proxy import PrefixwiseProxy

A = {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}
B = {"type": "function", "function": {"parameters": {"properties": {}, "type": "object"}, "name": "calc"}}


def test_canonical_tools_is_order_and_key_order_independent():
    one = canonical_tools([A, B])
    two = canonical_tools([json.loads(json.dumps(B)), json.loads(json.dumps(A))])
    assert json.dumps(one) == json.dumps(two)
    assert [t["function"]["name"] for t in one] == ["calc", "search"]
    # Anthropic / Responses shape: name at the top level.
    flat = canonical_tools([{"name": "z", "input_schema": {}}, {"name": "a", "input_schema": {}}])
    assert [t["name"] for t in flat] == ["a", "z"]


def test_apply_fixes_never_mutates_and_reports_what_changed():
    body = {"model": "m", "tools": [A, B], "messages": []}
    out, applied = apply_fixes(body, ("canonical_tools",))
    assert applied == ["canonical_tools"] and body["tools"] == [A, B]
    assert apply_fixes(body, ())[0] is body  # off: pass-through
    assert apply_fixes({"model": "m"}, ("canonical_tools",))[1] == []


def test_parse_fixes_rejects_unknown():
    assert parse_fixes("canonical_tools") == ("canonical_tools",)
    with pytest.raises(ValueError):
        parse_fixes("make_it_faster")


def test_proxy_with_fix_turns_reordered_tools_into_a_hit():
    """Off: a reordered tool list is TOOL_RESERIALIZE. On: the forwarded
    request is identical, so the turn is a plain append."""
    msgs = [{"role": "user", "content": "q"}]
    turn1 = {"model": "m", "session_id": "s", "tools": [A, B], "messages": msgs}
    turn2 = {**turn1, "tools": [B, A], "messages": msgs + [{"role": "assistant", "content": "a"},
                                                            {"role": "user", "content": "q2"}]}
    off, on = PrefixwiseProxy(mock=True), PrefixwiseProxy(mock=True, fixes=["canonical_tools"])
    for p in (off, on):
        p.handle("openai", dict(turn1))
    assert off.handle("openai", dict(turn2))[0]["cause"] == "tool_reserialize"
    result, entry = on.handle("openai", dict(turn2))
    assert result["cause"] == "none" and entry.fixes == "canonical_tools"


def test_fixes_command_ranks_by_predicted_savings(tmp_path, capsys):
    path = str(tmp_path / "l.db")
    proxy = PrefixwiseProxy(ledger_path=path, mock=True)
    msgs = [{"role": "user", "content": "q"}]
    proxy.handle("openai", {"model": "m", "session_id": "s", "tools": [A, B], "messages": msgs})
    proxy.handle("openai", {"model": "m", "session_id": "s", "tools": [B, A], "messages": msgs})
    assert main(["fixes", "--ledger", path, "--json"]) == 0
    recs = json.loads(capsys.readouterr().out)["recommendations"]
    assert recs[0]["cause"] == "tool_reserialize" and recs[0]["proxy_toggle"] == "canonical_tools"
    assert recs[0]["predicted_savings_tokens"] > 0
    main(["fixes", "--ledger", path])
    assert "prefixwise serve --fix canonical_tools" in capsys.readouterr().out


def test_canonical_tools_is_proven_with_evidence(tmp_path, capsys):
    """E4 live 2026-11-06 passed; the status must cite the committed run."""
    from pathlib import Path

    from prefixwise.fixes import FIX_STATUS

    status = FIX_STATUS["canonical_tools"]
    assert status["status"] == "proven"
    run = status["evidence"].split()[-1]
    assert (Path(__file__).resolve().parents[1] / run).exists()
    path = str(tmp_path / "l.db")
    proxy = PrefixwiseProxy(ledger_path=path, mock=True)
    msgs = [{"role": "user", "content": "q"}]
    proxy.handle("openai", {"model": "m", "session_id": "s", "tools": [A, B], "messages": msgs})
    proxy.handle("openai", {"model": "m", "session_id": "s", "tools": [B, A], "messages": msgs})
    main(["fixes", "--ledger", path, "--json"])
    rec = json.loads(capsys.readouterr().out)["recommendations"][0]
    assert rec["status"] == "proven" and "experiments/e4/runs/" in rec["evidence"]
