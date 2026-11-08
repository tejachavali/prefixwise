"""A full cache hit wins over every inference (2026-11-08 review), and the price of a cache write
that was never read is visible.

1. Blocks unchanged and the cache served the whole reusable prefix: `none`, whatever else differs
   (a param the provider did not render, no cache_control yet a hit, a prompt the minimum table
   says is too short). Before, those requests were PARAM_RENDERED / BREAKPOINT_MISPLACED /
   BELOW_MIN_PREFIX: a hit reported as a miss.
2. A miss that read nothing right after a cache write: the previous turn's write tokens were
   never read, and the surcharge paid to write them is shown, separately from avoidable cost.
"""

import pytest

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.findings import findings, render, unread_writes
from prefixwise.fingerprint import Fingerprinter
from prefixwise.ledger import Ledger, LedgerEntry, unread_write
from prefixwise.pricing import PriceTable

FP = Fingerprinter()
T = PriceTable.load()
BLOCKS = FP.fingerprint_blocks([("system", "Be careful."), ("messages", {"role": "user", "content": "hi"})])


def snap(prompt, cached, provider="openai", model="gpt-5.6-terra", **kw):
    kw.setdefault("has_breakpoint", True)
    return RequestSnapshot(model=model, provider=provider, blocks=BLOCKS, prompt_tokens=prompt,
                           cached_tokens=cached, **kw)


def test_param_change_the_cache_absorbed_is_a_hit():
    prev, curr = snap(4000, 0, params={"tool_choice": "auto"}), snap(4200, 4000, params={"tool_choice": "none"})
    c = classify(prev, curr)
    assert c.cause is MissCause.NONE and MissCause.PARAM_RENDERED in c.secondary


def test_param_change_with_nothing_read_is_still_param_rendered():
    c = classify(snap(4000, 0, params={"tool_choice": "auto"}), snap(4200, 0, params={"tool_choice": "none"}))
    assert c.cause is MissCause.PARAM_RENDERED


def test_anthropic_hit_without_markers_is_a_hit():
    prev = snap(4000, 0, provider="anthropic", model="claude-sonnet-5-5", has_breakpoint=False)
    curr = snap(4200, 4000, provider="anthropic", model="claude-sonnet-5-5", has_breakpoint=False)
    assert classify(prev, curr).cause is MissCause.NONE
    cold = snap(4200, 0, provider="anthropic", model="claude-sonnet-5-5", has_breakpoint=False)
    assert classify(prev, cold).cause is MissCause.BREAKPOINT_MISPLACED


def test_hit_below_the_minimum_table_is_a_hit():
    # The provider cached it, so the table's minimum did not apply to this model.
    assert classify(snap(900, 0), snap(950, 900)).cause is MissCause.NONE
    assert classify(snap(900, 0), snap(950, 0)).cause is MissCause.BELOW_MIN_PREFIX


# 2. cache writes never read

def test_write_surcharge_rates():
    # claude-sonnet-5-5: input 2.0, 5m write 2.5, 1h write 4.0 per 1M.
    assert T.write_surcharge("claude-sonnet-5-5", 10_000, write_5m=10_000) == pytest.approx(10_000 * 0.5 / 1e6)
    assert T.write_surcharge("claude-sonnet-5-5", 10_000, write_1h=10_000) == pytest.approx(10_000 * 2.0 / 1e6)
    assert T.write_surcharge("gpt-5.6-terra", 10_000, cache_write=10_000) == pytest.approx(10_000 * 0.5 / 1e6)
    assert T.write_surcharge("unknown-model", 10_000, write_5m=10) is None


def test_unread_write_only_after_a_cold_miss():
    assert unread_write("volatile_system", 0, 8_000, 0, 0) == (8_000, 0, 0)
    assert unread_write("ttl_expired", 0, 0, 8_000, 0) == (0, 8_000, 0)
    assert unread_write("volatile_system", 100, 8_000, 0, 0) == (0, 0, 0)  # something was read
    for kept in ("none", "first_write", "fork_boundary", "model_switch"):
        assert unread_write(kept, 0, 8_000, 0, 0) == (0, 0, 0)
    assert unread_write("routing_miss", 0, 0, 0, 8_000) == (0, 0, 8_000)  # single-rate provider


def test_report_shows_the_price_of_unread_writes():
    led = Ledger()
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model="claude-sonnet-5-5",
                           cause="first_write", prompt_tokens=8_000, cached_tokens=0, avoidable_tokens=0,
                           avoidable_cost=None, cache_write_tokens=8_000))
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model="claude-sonnet-5-5",
                           cause="volatile_system", prompt_tokens=8_100, cached_tokens=0, avoidable_tokens=8_000,
                           avoidable_cost=0.0192, legit=False, unread_write_tokens=8_000, unread_write_cost=0.004))
    [u] = unread_writes(led)
    assert (u["requests"], u["tokens"], u["surcharge_usd"]) == (1, 8_000, 0.004)
    out = render(findings(led))
    assert "Cache writes never read: 8,000 tokens" in out and "$0.0040" in out and "not counted twice" in out
