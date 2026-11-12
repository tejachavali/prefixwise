"""The share served from the cache had every avoidable miss been fixed: (cached + avoidable) / prompt.
Whether there is room to raise a hit rate is what the aggregate hit rate cannot say; this can. Checked against a real run: 74.6% served from cache with 12.5% of prompt
tokens avoidable predicts 87.1%, and the same agent with --optimize read 87.1%
(experiments/demo/2026-11-10-optimize-real-run.md)."""

import random

from prefixwise.findings import ceiling_line, hit_rate_if_fixed


def test_the_real_run_is_predicted():
    s = {"prompt_tokens": 26_932, "cached_tokens": 20_089, "avoidable_tokens": 3_375,
         "cache_hit_rate": 20_089 / 26_932}
    assert round(hit_rate_if_fixed(s), 3) == 0.871 and round(s["cache_hit_rate"], 3) == 0.746
    s["cache_hit_rate_if_fixed"] = hit_rate_if_fixed(s)
    assert ceiling_line(s).startswith("Fixing these would take the share served from cache from 74.6% to 87.1%")


def test_bounds():
    rng = random.Random(7)
    for _ in range(2000):
        prompt = rng.randint(1, 10**7)
        cached = rng.randint(0, prompt)
        avoidable = rng.randint(0, prompt - cached)  # avoidable is never more than what was not read
        s = {"prompt_tokens": prompt, "cached_tokens": cached, "avoidable_tokens": avoidable,
             "cache_hit_rate": cached / prompt}
        fixed = hit_rate_if_fixed(s)
        assert s["cache_hit_rate"] <= fixed <= 1.0
        assert (fixed == s["cache_hit_rate"]) == (avoidable == 0)


def test_nothing_to_say_without_usage_or_waste():
    assert hit_rate_if_fixed({"prompt_tokens": 0, "cached_tokens": 0, "avoidable_tokens": 0}) is None
    s = {"prompt_tokens": 100, "cached_tokens": 90, "avoidable_tokens": 0, "cache_hit_rate": 0.9,
         "cache_hit_rate_if_fixed": 0.9}
    assert ceiling_line(s) is None


def test_the_short_figure_never_hides_the_gain_or_promises_a_perfect_cache():
    """Found by review: whole percents showed 75% now and 75% if fixed, or 100% for 99.95%."""
    from prefixwise.findings import _pct_down
    assert _pct_down(0.7490, 0.7460) == "74.9%"   # 75% and 75% would hide the gain
    assert _pct_down(0.9995, 0.9955) == "99.9%"   # never rounded up into a perfect cache
    assert _pct_down(0.871, 0.746) == "87%"
    assert _pct_down(1.0, 0.9) == "100%"
    assert _pct_down(0.29, 0.1) == "29%"           # float noise does not round 29 down to 28
    rng = random.Random(9)
    for _ in range(5000):  # never below or equal to the current share as the block prints it
        now = rng.random() * 0.999
        fixed = now + rng.random() * (1 - now)
        if fixed - now > 2e-4:  # two decimals resolve it
            text = _pct_down(fixed, now)
            d = len(text[:-1].partition(".")[2])
            assert float(text[:-1]) > float(f"{now * 100:.{d}f}") and float(text[:-1]) < 100
            assert float(text[:-1]) <= fixed * 100 + 1e-9  # rounded down, never up


def test_the_report_names_the_causes_before_the_ceiling(tmp_path):
    from prefixwise.findings import findings, render
    from prefixwise.ledger import Ledger, LedgerEntry
    led = Ledger(str(tmp_path / "l.db"))
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="openai", model="gpt-5", cause="first_write",
                           legit=True, prompt_tokens=10_000, cached_tokens=0, avoidable_tokens=0, avoidable_cost=0.0, task_success=None))
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="openai", model="gpt-5", cause="tool_reserialize",
                           legit=False, prompt_tokens=10_000, cached_tokens=2_000, avoidable_tokens=8_000, avoidable_cost=0.01, task_success=None))
    text = render(findings(led))
    lines = text.splitlines()
    at = next(i for i, l in enumerate(lines) if l.startswith("Fixing these would take"))
    assert any(l.startswith("Avoidable:") for l in lines[:at]) and any(l.startswith("  1.") for l in lines[:at])
    assert "whose cause is not known" in lines[at]
