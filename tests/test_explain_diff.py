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
    assert "VOLATILE_SYSTEM" in out.stdout and "what changed (as fingerprinted, from your files; nothing is stored)" in out.stdout
    assert "07:15:30" in out.stdout
    plain = subprocess.run([sys.executable, "-m", "prefixwise", "explain", str(tmp_path / "a.json"),
                            str(tmp_path / "b.json")], capture_output=True, text=True, cwd=tmp_path,
                           env=env)
    assert "07:15:30" not in plain.stdout  # without --diff no prompt text is printed


def test_diff_handles_whitespace_only_changes_dict_blocks_and_large_blocks():
    """Review scenarios: a trailing-newline change is named, not silent; an Anthropic system prompt (a
    dict block) diffs line by line, not as one multi-kilobyte JSON string; a 20,000-line block with
    scattered edits is diffed in well under a second because the common head and tail are dropped."""
    import time
    # whitespace only
    a = Turn(body={"model": "m", "messages": [{"role": "system", "content": "a\nb"}, {"role": "user", "content": "hi"}]})
    b = Turn(body={"model": "m", "messages": [{"role": "system", "content": "a\nb\n"}, {"role": "user", "content": "hi"}]})
    e = explain_pair(a, b, "openai", diff=True)
    assert e.cause == "system_edit" and e.diff is not None
    # an Anthropic system prompt: many rules, the timestamp at the end changes
    rules = "\n".join(f"Rule {i}: be careful about thing {i}." for i in range(60))
    sa = Turn(body={"model": "m", "system": rules + "\nNow: 2026-11-09 07:15:00", "messages": [{"role": "user", "content": "hi"}]})
    sb = Turn(body={"model": "m", "system": rules + "\nNow: 2026-11-09 07:15:30", "messages": [{"role": "user", "content": "hi"}]})
    e = explain_pair(sa, sb, "anthropic", diff=True)
    changed = [l for l in e.diff.splitlines() if l.startswith(("-Now", "+Now"))]
    assert len(changed) == 2 and all(len(l) < 60 for l in changed), e.diff
    # a huge block with an edit every third line
    big = [f"line {i} of a very long system prompt" for i in range(20_000)]
    edited = [l + " x" if i % 3 == 0 else l for i, l in enumerate(big)]
    la = Turn(body={"model": "m", "messages": [{"role": "system", "content": "\n".join(big)}, {"role": "user", "content": "hi"}]})
    lb = Turn(body={"model": "m", "messages": [{"role": "system", "content": "\n".join(edited)}, {"role": "user", "content": "hi"}]})
    t0 = time.perf_counter()
    e = explain_pair(la, lb, "openai", diff=True)
    assert time.perf_counter() - t0 < 1.0 and e.diff and "more lines" in e.diff
