"""E4 harness: recording, mock replay end to end, scoring rules."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "e4"))

from e4 import DEFAULT_SESSIONS, MockCache, load, main, record, render_tools, score  # noqa: E402


def test_recording_is_deterministic(tmp_path):
    a = record(tmp_path / "a.jsonl", 3, 5, 0.5, seed=1)
    b = record(tmp_path / "b.jsonl", 3, 5, 0.5, seed=1)
    assert a == b and any(s["reloaded"] for s in a)


def test_committed_recording_matches_the_default_seed(tmp_path):
    fresh = record(tmp_path / "x.jsonl", 6, 10, 0.5, seed=20261106)
    assert load(DEFAULT_SESSIONS) == fresh


def test_reloaded_tools_differ_in_bytes_but_not_in_set():
    a = render_tools("vllm", ["calculator", "git_log"], None)
    b = render_tools("vllm", ["git_log", "calculator"], 7)
    assert json.dumps(a) != json.dumps(b)
    assert sorted(t["function"]["name"] for t in a) == sorted(t["function"]["name"] for t in b)


def test_mock_cache_reuses_only_the_shared_prefix():
    c = MockCache()
    p1, c1 = c.lookup("x" * 4000)
    p2, c2 = c.lookup("x" * 4000 + "y" * 400)
    p3, c3 = c.lookup("z" + "x" * 4000)
    assert c1 == 0 and 0 < c2 <= 1000 and c3 == 0


def test_mock_replay_passes(tmp_path, capsys):
    assert main(["--mock", "--output-dir", str(tmp_path)]) == 0
    rep = json.loads(next(tmp_path.glob("*.json")).read_text())["report"]
    assert rep["paired_requests"] == 60 and rep["causes"]["on"].get("tool_reserialize", 0) == 0
    assert rep["realised_over_predicted"] > 0.9


def test_score_requires_flat_success_and_enough_requests():
    def rows(arm, n, cached, success, cause="none", avoid=0):
        return [{"arm": arm, "session": 0, "turn": i, "reloaded": False, "cause": cause, "cached_tokens": cached,
                 "prompt_tokens": 1000, "avoidable_tokens": avoid, "success": success} for i in range(n)]
    good = score(rows("off", 60, 0, True, "tool_reserialize", 900), rows("on", 60, 900, True))
    assert good["pass"]
    worse = score(rows("off", 60, 0, True, "tool_reserialize", 900), rows("on", 60, 900, False))
    assert not worse["pass"] and not worse["success_flat"]  # savings never excuse broken tasks
    small = score(rows("off", 20, 0, True, "tool_reserialize", 900), rows("on", 20, 900, True))
    assert not small["pass"]


def test_no_marker_recording_is_committed_and_has_no_drift(tmp_path):
    from e4 import NO_MARKER_SESSIONS
    fresh = record(tmp_path / "x.jsonl", 6, 10, 0.0, seed=20261106)
    assert load(NO_MARKER_SESSIONS) == fresh and not any(s["reloaded"] for s in fresh)


def test_mock_cache_bills_by_marker_as_anthropic_does():
    c = MockCache()
    assert c.lookup_marked("x" * 4000, False) == (1000, 0, 0)  # no marker: nothing read, nothing written
    p, read, written = c.lookup_marked("x" * 4000, True)
    assert (read, written) == (0, 1000)  # the first marked request writes
    p, read, written = c.lookup_marked("x" * 4000 + "y" * 400, True)
    assert read > 0 and read + written == p


def test_mock_replay_of_auto_breakpoints_passes_and_reports_dollars(tmp_path, capsys):
    assert main(["--mock", "--fix", "auto_breakpoints", "--output-dir", str(tmp_path)]) == 0
    rep = json.loads(next(tmp_path.glob("*.json")).read_text())["report"]
    assert rep["fix"] == "auto_breakpoints" and rep["paired_requests"] == 60
    assert rep["causes"]["off"]["breakpoint_misplaced"] == 54 and "breakpoint_misplaced" not in rep["causes"]["on"]
    assert rep["cached_tokens"]["off"] == 0 and rep["cache_write_tokens"]["off"] == 0  # no marker: nothing cached
    assert rep["requests_rewritten"] == 59  # every request but the very first: nothing had been seen yet
    assert rep["cost_usd"]["on"] < rep["cost_usd"]["off"] and rep["saved_usd"] > 0 and not rep["markers_rejected"]
    out = capsys.readouterr().out
    assert "write surcharge included" in out and "Pass: YES" in out


def test_auto_breakpoints_fails_when_it_costs_more_or_the_markers_are_rejected():
    def rows(n, cached, cost, cause="none", avoid=0, **extra):
        return [{"session": 0, "turn": i, "reloaded": False, "cause": cause, "cached_tokens": cached,
                 "prompt_tokens": 1000, "avoidable_tokens": avoid, "success": True, "cost": cost,
                 "cache_write_tokens": 0, "fixes_applied": [], **extra} for i in range(n)]
    off = rows(60, 0, 0.001, "breakpoint_misplaced", 900)
    assert score(off, rows(60, 900, 0.0002), "auto_breakpoints")["pass"]
    assert not score(off, rows(60, 900, 0.002), "auto_breakpoints")["pass"]  # tokens saved, dollars not
    assert not score(off, rows(60, 900, 0.0002, markers_rejected=True), "auto_breakpoints")["pass"]
    assert not score(off, rows(60, 900, None), "auto_breakpoints")["pass"]  # unpriced: no verdict on dollars
    assert score(rows(60, 0, None, "tool_reserialize", 900), rows(60, 900, None))["pass"]  # canonical_tools: as before
