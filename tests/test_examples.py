"""The files in examples/ back the README and docs/USE_CASES.md; keep their verdicts pinned."""

import json
from pathlib import Path

from prefixwise.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_turn_pair_is_volatile_system(capsys):
    main(["explain", str(EXAMPLES / "turn1.json"), str(EXAMPLES / "turn2.json"), "--json"])
    [res] = json.loads(capsys.readouterr().out)
    assert res["cause"] == "volatile_system"
    assert res["divergence"]["role"] == "system"


def test_session_has_the_two_documented_regressions(capsys):
    assert main(["explain", str(EXAMPLES / "session.jsonl"), "--json", "--fail-on-waste"]) == 2
    causes = [r["cause"] for r in json.loads(capsys.readouterr().out)]
    assert causes == ["first_write", "none", "tool_result_mutation", "tool_reserialize"]
