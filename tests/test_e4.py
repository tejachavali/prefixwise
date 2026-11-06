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
