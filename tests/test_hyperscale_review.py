"""Findings of the hyperscale review (2026-11-08), each pinned by a test.

1. A request that is itself below the provider's minimum cacheable prefix cannot be read from
   the cache, so a prompt change in it is not waste (below_min_prefix wins, the change is kept
   as secondary).
2. A miss's avoidable cost is priced at what the provider billed for the tokens it did not read
   (on Anthropic, the cache-write surcharge), not at the plain input rate.
3. Anthropic ledgers with ttl_expired misses inside the hour get a 5-minute vs 1-hour verdict.
"""

import pytest

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.findings import findings, render, ttl_advice
from prefixwise.fingerprint import Fingerprinter
from prefixwise.ledger import Ledger, LedgerEntry, avoidable_for_turn
from prefixwise.pricing import PriceTable

FP = Fingerprinter()
T = PriceTable.load()


def snap(system, prompt, cached, provider="openai", model="gpt-5.6-terra"):
    blocks = FP.fingerprint_blocks([("system", system), ("messages", {"role": "user", "content": "hi"})])
    return RequestSnapshot(model=model, provider=provider, blocks=blocks, prompt_tokens=prompt,
                           cached_tokens=cached, has_breakpoint=True)


# 1. below_min_prefix before prompt-change rules

def test_change_in_a_request_too_short_to_cache_is_not_waste():
    prev = snap("Today is 2026-11-07.", 1500, 1200)  # long enough to cache
    curr = snap("Today is 2026-11-08.", 900, 0)  # shrank below OpenAI's 1,024
    c = classify(prev, curr)
    assert c.cause is MissCause.BELOW_MIN_PREFIX and c.legit is None
    assert MissCause.VOLATILE_SYSTEM in c.secondary and c.score is False
    assert avoidable_for_turn(c.legit, 900, 0, 1500) == 0


def test_change_above_the_minimum_is_still_named():
    c = classify(snap("Today is 2026-11-07.", 1500, 1200), snap("Today is 2026-11-08.", 1600, 0))
    assert c.cause is MissCause.VOLATILE_SYSTEM and c.legit is False


def test_short_request_that_read_the_cache_is_not_below_minimum():
    # Something was read, so the minimum did not stop caching: the usual rules decide.
    c = classify(snap("Today is 2026-11-07.", 1500, 1200), snap("Today is 2026-11-08.", 1000, 512))
    assert c.cause is not MissCause.BELOW_MIN_PREFIX


# 2. avoidable cost at the billed rate

def anthropic_usage(write_5m=0, write_1h=0, uncached=0, read=0):
    return {"input_tokens": uncached, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write_5m + write_1h,
            "cache_creation": {"ephemeral_5m_input_tokens": write_5m, "ephemeral_1h_input_tokens": write_1h},
            "output_tokens": 10}


def test_miss_rewritten_to_the_cache_costs_the_write_rate():
    # claude-sonnet-5-5: input 2.0, cache_read 0.1, 5m write 2.5, 1h write 4.0 (per 1M).
    m = "claude-sonnet-5-5"
    five = T.avoidable_cost(m, 10_000, 10_000, "anthropic", anthropic_usage(write_5m=10_000))
    hour = T.avoidable_cost(m, 10_000, 10_000, "anthropic", anthropic_usage(write_1h=10_000))
    plain = T.avoidable_cost(m, 10_000, 10_000)
    assert five == pytest.approx(10_000 * (2.5 - 0.1) / 1e6)
    assert hour == pytest.approx(10_000 * (4.0 - 0.1) / 1e6)
    assert plain == pytest.approx(10_000 * (2.0 - 0.1) / 1e6)  # no usage: the input rate, as before


def test_mixed_writes_and_uncached_are_blended():
    u = anthropic_usage(write_5m=6_000, uncached=4_000)
    paid = (6_000 * 2.5 + 4_000 * 2.0) / 10_000
    assert T.avoidable_cost("claude-sonnet-5-5", 8_000, 10_000, "anthropic", u) == pytest.approx(8_000 * (paid - 0.1) / 1e6)


def test_all_read_falls_back_to_input_rate():
    # Nothing paid beyond reads: no billed rate to use, so the input rate (never a guess of zero).
    u = anthropic_usage(read=10_000)
    assert T.avoidable_cost("claude-sonnet-5-5", 1_000, 10_000, "anthropic", u) == pytest.approx(1_000 * 1.9 / 1e6)


def test_openai_write_rate_is_used():
    # gpt-5.6-terra: input 2.0, cache_read 0.2, cache_write 2.5.
    u = {"prompt_tokens": 10_000, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 10_000},
         "completion_tokens": 5}
    assert T.avoidable_cost("gpt-5.6-terra", 10_000, 10_000, "openai", u) == pytest.approx(10_000 * 2.3 / 1e6)


# 3. 5-minute vs 1-hour cache

def row(cause, gap, writes, model="claude-sonnet-5-5", detail=None, provider="anthropic"):
    return LedgerEntry(session_id="s", agent_id="a", provider=provider, model=model, cause=cause,
                       prompt_tokens=writes, cached_tokens=0, avoidable_tokens=0, avoidable_cost=None,
                       cache_write_tokens=writes, gap_seconds=gap, detail=detail)


def test_one_hour_cache_pays_when_misses_within_the_hour_dominate():
    led = Ledger()
    led.record(row("first_write", None, 20_000))
    for _ in range(5):
        led.record(row("ttl_expired", 900, 20_000))  # 15 minutes: the 1-hour cache keeps it
    [a] = ttl_advice(led, prices=T)
    assert a["misses"] == 5 and a["pays"]
    assert a["saved_at_most_usd"] == pytest.approx(100_000 * (2.5 - 0.1) / 1e6)
    assert a["extra_write_cost_usd"] == pytest.approx(120_000 * (4.0 - 2.5) / 1e6)
    assert "worth trying" in render(findings(led))


def test_one_hour_cache_does_not_pay_when_writes_dominate():
    led = Ledger()
    for _ in range(50):
        led.record(row("none", 10, 20_000))  # writes everywhere, read soon after
    led.record(row("ttl_expired", 900, 1_000))
    [a] = ttl_advice(led, prices=T)
    assert not a["pays"] and "keep the 5-minute cache" in render(findings(led))


def test_gaps_outside_the_hour_and_already_one_hour_are_not_counted():
    led = Ledger()
    led.record(row("ttl_expired", 200, 1_000))  # under 5 minutes: not a TTL question
    led.record(row("ttl_expired", 7200, 1_000))  # beyond an hour: 1 hour would not keep it
    led.record(row("ttl_expired", 900, 1_000, detail="ttl_shortened"))  # already asked for 1 hour
    led.record(row("ttl_expired", 900, 1_000, model="gpt-5.6-terra", provider="openai"))
    assert ttl_advice(led, prices=T) == []
