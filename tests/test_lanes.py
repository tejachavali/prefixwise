"""Per-model lanes (dogfood 2026-11-06: opencode interleaves Haiku side calls with Sonnet turns)."""

import json

from prefixwise.cli import main
from prefixwise.proxy import PrefixwiseProxy

SYS = [{"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}]
TOOLS = [{"name": "read", "input_schema": {}}, {"name": "write", "input_schema": {}}]


def main_turn(n, tools=TOOLS):
    msgs = []
    for i in range(1, n + 1):
        msgs.append({"role": "user", "content": f"step {i}"})
        if i < n:
            msgs.append({"role": "assistant", "content": f"done {i}"})
    return {"model": "claude-sonnet-5-5", "session_id": "s", "system": SYS, "tools": tools, "messages": msgs}


def side_call(n):  # e.g. "generate a title for this conversation"
    return {"model": "claude-haiku-5-5", "session_id": "s", "system": [{"type": "text", "text": "Title it."}],
            "messages": [{"role": "user", "content": f"title for step {n}"}]}


def causes(proxy, bodies):
    return [proxy.handle("anthropic", dict(b))[0]["cause"] for b in bodies]


def test_side_calls_do_not_mask_the_main_thread():
    proxy = PrefixwiseProxy(mock=True)
    got = causes(proxy, [main_turn(1), side_call(1), main_turn(2), side_call(2), main_turn(3)])
    # Before lanes: first_write, model_switch, model_switch, model_switch, model_switch.
    assert got[0] == "first_write"
    assert got[1] == "model_switch"          # first call on the small model
    assert got[2] == "none"                  # main thread compared with its own last turn
    assert got[3] == "breakpoint_misplaced"  # the side call's own lane: no cache_control, honestly flagged
    assert got[4] == "none"


def test_a_real_cause_on_the_main_thread_is_still_found_after_a_side_call():
    proxy = PrefixwiseProxy(mock=True)
    got = causes(proxy, [main_turn(1), side_call(1), main_turn(2, tools=TOOLS[::-1])])
    assert got[2] == "tool_reserialize"  # before lanes this was hidden as model_switch


def test_diagnostics_chain_to_the_same_models_last_response():
    proxy = PrefixwiseProxy(mock=True)
    sent = []
    real = proxy._forward

    def spy(provider, body, *a, **kw):
        sent.append(body.get("diagnostics"))
        return real(provider, body, *a, **kw)
    proxy._forward = spy
    r1, _ = proxy.handle("anthropic", main_turn(1))
    proxy.handle("anthropic", side_call(1))
    proxy.handle("anthropic", main_turn(2))
    assert sent[2] == {"previous_message_id": r1["response"]["id"]}  # not the side call's id


def test_explain_offline_uses_the_same_lanes(tmp_path, capsys):
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(b) for b in [main_turn(1), side_call(1), main_turn(2)]) + "\n")
    main(["explain", str(path), "--json"])
    got = [r["cause"] for r in json.loads(capsys.readouterr().out)]
    assert got[:2] == ["first_write", "model_switch"] and got[2] != "model_switch"
