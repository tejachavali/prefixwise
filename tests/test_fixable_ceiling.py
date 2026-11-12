"""The share served from the cache had every avoidable miss been fixed: (cached + avoidable) / prompt.
An inference provider asked whether there was room from an 89% hit rate to 93%; the aggregate hit
rate cannot say, this can. Checked against a real run: 74.6% served from cache with 12.5% of prompt
tokens avoidable predicts 87.1%, and the same agent with --optimize read 87.1%
(experiments/demo/2026-11-10-optimize-real-run.md)."""

import random

from prefixwise.findings import ceiling_line, hit_rate_if_fixed


def test_the_real_run_is_predicted():
    s = {"prompt_tokens": 26_932, "cached_tokens": 20_089, "avoidable_tokens": 3_375,
         "cache_hit_rate": 20_089 / 26_932}
    assert round(hit_rate_if_fixed(s), 3) == 0.871 and round(s["cache_hit_rate"], 3) == 0.746
    s["cache_hit_rate_if_fixed"] = hit_rate_if_fixed(s)
    assert ceiling_line(s).startswith("Fixing them would take the share served from cache from 74.6% to 87.1%")


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
