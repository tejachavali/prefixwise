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
