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


def test_the_two_shares_are_printed_so_the_gain_shows_and_nothing_is_overstated():
    """Found by review: 75% now and 75% if fixed; 74.9% if fixed under a headline of 75%; 99.995%
    printed as 100.0%; a gain two decimals cannot show printed as none, or as a loss."""
    from prefixwise.findings import from_to
    assert from_to(0.746, 0.749) == ("74.6%", "74.9%")       # whole percents would hide it
    assert from_to(20_089 / 26_932, (20_089 + 3_375) / 26_932, 1) == ("74.6%", "87.1%")  # the real run
    assert from_to(0.746, 0.871) == ("75%", "87%")
    assert from_to(0.99996, 0.99998) is None                  # 99.99% to 99.99% would show no gain
    assert from_to(0.99996, 1.0) == ("99%", "100%")
    assert from_to(0.746, 0.74601) is None                    # too small to show: left out
    assert from_to(0.10, 0.29) == ("10%", "29%")              # float noise does not make 29 a 28
    assert from_to(None, 0.5) is None
    rng = random.Random(9)
    for _ in range(20_000):
        now = rng.random()
        fixed = now + rng.random() * (1 - now)
        pair = from_to(now, fixed, rng.choice([0, 1]))
        if pair is None:
            assert fixed - now < 0.0002  # only a gain under two decimals' resolution is left out
            continue
        n, f = float(pair[0][:-1]), float(pair[1][:-1])
        assert f > n                                   # the gain always shows
        assert f <= fixed * 100 + 1e-9                 # never rounded up
        assert n < 100 or now >= 1                     # never a perfect cache that is not one
        assert f < 100 or fixed >= 1


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


def test_a_headline_share_is_never_rounded_up_to_a_perfect_cache():
    """Found by review: 99.6% read as "100% from cache" in the headlines."""
    from prefixwise.findings import from_to, share_pct
    assert share_pct(0.996) == "99%" and share_pct(1.0) == "100%" and share_pct(0.746) == "75%"
    assert share_pct(0.99996, 1) == "99.9%" and share_pct(0.0) == "0%"
    assert from_to(0.996, 1.0) == ("99%", "100%")  # and the pair agrees with the headline
