"""E1-hosted harness: mock mode end to end (no network), estimate, budget cap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "e1_hosted"))

from e1_hosted import estimate, main, run_provider  # noqa: E402


def test_mock_run_passes_both_providers(tmp_path):
    assert main(["--mock", "--repeats", "2", "--output-dir", str(tmp_path)]) == 0
    assert list(tmp_path.glob("*.md")) and list(tmp_path.glob("*.json"))


def test_every_injected_cause_is_named_in_mock():
    for provider in ("anthropic", "openai"):
        alt = "alt-model" if provider == "anthropic" else None
        rows, rep = run_provider(provider, model="m", alt_model=alt, repeats=1, mock=True,
                                 max_input_tokens=None)
        wrong = [(r["injected"], r["predicted"]) for r in rows if not r["agree_injected"]]
        assert not wrong, (provider, wrong)
        assert rep["agreement_native"] == 1.0


def test_budget_cap_stops_before_spending():
    rows, rep = run_provider("anthropic", model="m", alt_model=None, repeats=5, mock=True,
                             max_input_tokens=30_000)
    assert 0 < rep["input_tokens"] <= 30_000 and len(rows) < 100


def test_estimate_counts_the_full_plan():
    est = estimate(["anthropic", "openai"], 5, {"anthropic": ("m", "alt"), "openai": ("m", None)})
    assert est["anthropic"]["turns"] == 100 and est["openai"]["turns"] == 100
    assert est["anthropic"]["input_tokens"] > 500_000


def test_probe_exercises_real_misses_and_passes_in_mock(capsys):
    assert main(["--mock", "--probe", "--output-dir", "/tmp/claude-probe-test"]) == 0
    out = capsys.readouterr().out
    assert "PROBE anthropic: OK" in out and "PROBE openai: OK" in out


def test_probe_verdict_rules():
    from e1_hosted import probe_verdict

    hit = {"turn": 2, "injected": "none", "cached_tokens": 2900, "prompt_tokens": 3000,
           "native_reason": None, "native_type": "cache_hit", "real_miss": False}
    miss = {"turn": 3, "injected": "tool_reserialize", "cached_tokens": 0, "prompt_tokens": 3000,
            "native_type": "unavailable", "real_miss": True, "cache_fields": {}, "predicted": "tool_reserialize",
            "agree_native": None}
    # Null reason on a hit is fine; null reason on a real miss stops the run.
    ok, notes = probe_verdict([hit, {**miss, "native_reason": None}])
    assert not ok and "REAL miss but no native reason" in " ".join(notes)
    ok, _ = probe_verdict([hit, {**miss, "native_reason": "tools_changed", "agree_native": True}])
    assert ok
    # Without any real miss the probe proves nothing.
    ok, notes = probe_verdict([hit])
    assert not ok and "not exercised" in notes[-1]


def test_error_ends_the_session_so_labels_cannot_drift():
    import e1_hosted

    calls = {"n": 0}
    orig = e1_hosted.mock_forward_factory

    def failing_factory(provider, state):
        fwd = orig(provider, state)

        def forward(*a, **kw):
            calls["n"] += 1
            if state["injected"] == "model_switch":
                return 400, {"error": {"message": "tool_choice not supported"}}
            return fwd(*a, **kw)
        return forward

    e1_hosted.mock_forward_factory = failing_factory
    try:
        rows, rep = e1_hosted.run_provider("anthropic", model="m", alt_model="alt", repeats=1, mock=True,
                                           max_input_tokens=None)
    finally:
        e1_hosted.mock_forward_factory = orig
    session_b = [r for r in rows if r["session"] == 1]
    assert [r["turn"] for r in session_b][-1] == 7 and "error" in session_b[-1]  # nothing after the error
    assert all(r["agree_injected"] for r in rows if "error" not in r)


def test_anthropic_param_change_uses_a_tool_choice_every_model_accepts():
    from datetime import date

    from e1_hosted import SessionState

    s = SessionState("anthropic", "m", "x", date(2026, 11, 6))
    s.apply("param_rendered", 3, None, "m")
    s.next_user_turn()
    assert s.body()["tool_choice"] == {"type": "none"}
