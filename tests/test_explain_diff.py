"""`prefixwise explain --diff`: the hashes say where two requests diverge; --diff shows what, from
the files the user holds. Nothing new is stored: the proxy and ledger are untouched."""

import json
import subprocess
import sys
from pathlib import Path

from prefixwise.explain import Turn, explain_pair


def _body(stamp, extra=""):
    return {"model": "gpt-5.6-luna",
            "messages": [{"role": "system", "content": f"You are careful. Now: {stamp}.{extra}"},
                         {"role": "user", "content": "hi"}]}


def test_diff_shows_the_first_differing_block_only_when_asked():
    a, b = Turn(body=_body("2026-11-09 07:15:00")), Turn(body=_body("2026-11-09 07:15:30"))
    plain = explain_pair(a, b, "openai")
    assert plain.cause == "volatile_system" and plain.diff is None
    e = explain_pair(a, b, "openai", diff=True)
    assert e.cause == "volatile_system" and e.divergence["role"] == "system"
    assert "-" in e.diff and "07:15:00" in e.diff and "07:15:30" in e.diff
    assert "previous request" in e.diff and "this request" in e.diff
    # a hit has nothing to diff
    assert explain_pair(a, Turn(body=_body("2026-11-09 07:15:00")), "openai", diff=True).diff is None
    # growth after the previous prompt is not a divergence: no diff
    grown = _body("2026-11-09 07:15:00")
    grown["messages"] += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    assert explain_pair(a, Turn(body=grown), "openai", diff=True).diff is None


def test_cli_prints_the_diff(tmp_path):
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PATH": "/usr/bin:/bin"}
    (tmp_path / "a.json").write_text(json.dumps(_body("2026-11-09 07:15:00")))
    (tmp_path / "b.json").write_text(json.dumps(_body("2026-11-09 07:15:30")))
    out = subprocess.run([sys.executable, "-m", "prefixwise", "explain", str(tmp_path / "a.json"),
                          str(tmp_path / "b.json"), "--diff"], capture_output=True, text=True, cwd=tmp_path,
                         env=env)
    assert out.returncode == 0, out.stderr
    assert "VOLATILE_SYSTEM" in out.stdout and "what changed (from your files; nothing is stored)" in out.stdout
    assert "07:15:30" in out.stdout
    plain = subprocess.run([sys.executable, "-m", "prefixwise", "explain", str(tmp_path / "a.json"),
                            str(tmp_path / "b.json")], capture_output=True, text=True, cwd=tmp_path,
                           env=env)
    assert "07:15:30" not in plain.stdout  # without --diff no prompt text is printed
