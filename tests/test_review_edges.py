"""Edge cases from the 2026-11-08 review of the write-rate pricing (#55), unread writes (#56) and
the held-prefix cap (#55):

1. A miss priced at the rate it paid: a request that wrote nothing to the cache paid the input
   rate, so its avoidable cost is (input - read), not (write - read); the report shows the
   surcharge on an unread write apart from avoidable cost, never added to it.
2. A breakpoint that stays early while the stable prompt grows past it is `breakpoint_misplaced`,
   not `none`: the cache serves all it holds, and the stable tail after the breakpoint is paid
   uncached on every turn. Its waste is that tail, not capped at what the cache held.
"""

import pytest
from test_details import MODEL, USER, a_usage, body, proxy, turn

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.findings import findings, render
from prefixwise.fingerprint import Fingerprinter
from prefixwise.ledger import Ledger, LedgerEntry
from prefixwise.pricing import PriceTable

T = PriceTable.load()
FP = Fingerprinter()
BLOCKS = FP.fingerprint_blocks([("system", "Be careful."), ("messages", {"role": "user", "content": "hi"})])
SONNET = "claude-sonnet-5-5"  # input 2.0, cache_read 0.1, 5m write 2.5 per 1M


def snap(prompt, cached, written=0, has_breakpoint=True, provider="anthropic", model=SONNET, blocks=None):
    return RequestSnapshot(model=model, provider=provider, blocks=blocks or BLOCKS, prompt_tokens=prompt,
                           cached_tokens=cached, cache_written=written, has_breakpoint=has_breakpoint)


# -- 1. the rate paid, and no double count ---------------------------------------------

def test_a_miss_that_wrote_nothing_is_priced_at_the_input_rate_not_the_write_rate():
    no_write = {"input_tokens": 10_000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert T.avoidable_cost(SONNET, 10_000, 10_000, "anthropic", no_write) == pytest.approx(10_000 * (2.0 - 0.1) / 1e6)
    wrote = {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 10_000}
    assert T.avoidable_cost(SONNET, 10_000, 10_000, "anthropic", wrote) == pytest.approx(10_000 * (2.5 - 0.1) / 1e6)
    # part written (an early breakpoint), the rest plain input: the blend of what was paid
    part = {"input_tokens": 8_000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2_000}
    paid = (8_000 * 2.0 + 2_000 * 2.5) / 10_000
    assert T.avoidable_cost(SONNET, 10_000, 10_000, "anthropic", part) == pytest.approx(10_000 * (paid - 0.1) / 1e6)


def row(cause, avoidable_cost, unread_cost=None, unread_tokens=None):
    e = LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model=SONNET, cause=cause,
                    prompt_tokens=10_000, cached_tokens=0, avoidable_tokens=10_000 if avoidable_cost else 0,
                    avoidable_cost=avoidable_cost, legit=False if avoidable_cost else None)
    e.unread_write_tokens, e.unread_write_cost = unread_tokens, unread_cost
    return e


def test_the_report_keeps_the_unread_write_surcharge_apart_from_avoidable_cost():
    led = Ledger()
    led.record(row("first_write", 0.0))
    led.record(row("history_edit", 0.024, unread_cost=0.005, unread_tokens=10_000))  # the same turn carries both
    led.record(row("volatile_system", 0.024))
    f = findings(led, prices=T)
    assert f["summary"]["avoidable_cost_usd"] == pytest.approx(0.048)  # 0.024 + 0.024, no 0.005
    assert sum(u["surcharge_usd"] for u in f["unread_writes"]) == pytest.approx(0.005)
    text = render(f)
    assert "Cache writes never read" in text and "not counted twice" in text


# -- 2. a breakpoint that stays early ------------------------------------------------------

def test_an_early_static_breakpoint_is_misplaced_not_a_hit():
    # Turn 1: 20,000 tokens, breakpoint after the first 2,000 (the write). Turn 2: 22,000 tokens,
    # reads those 2,000: the cache served all it held, yet 18,000 stable tokens were paid uncached.
    c = classify(snap(20_000, 0, written=2_000), snap(22_000, 2_000))
    assert c.cause is MissCause.BREAKPOINT_MISPLACED and c.legit is False
    assert "breakpoint too early" in c.evidence and "18,000 stable tokens" in c.evidence
    # steady state: turn 3 again reads 2,000 of a 22,000-token shared prefix
    c = classify(snap(22_000, 2_000, written=0), snap(24_000, 2_000))
    assert c.cause is MissCause.BREAKPOINT_MISPLACED and "20,000 stable tokens" in c.evidence


def test_a_moving_breakpoint_or_a_short_tail_stays_a_hit():
    # The breakpoint moves forward each turn: everything shared was held, nothing is left out.
    assert classify(snap(20_000, 0, written=20_000), snap(22_000, 20_000)).cause is MissCause.NONE
    assert classify(snap(20_000, 18_000, written=2_000), snap(22_000, 20_000)).cause is MissCause.NONE
    # The tail after the breakpoint is below the model's minimum cacheable size: no breakpoint could cache it.
    assert classify(snap(2_400, 0, written=2_000), snap(2_600, 2_000)).cause is MissCause.NONE
    # Not everything held was served: another cause (eviction, routing), never this rule.
    assert classify(snap(20_000, 0, written=2_000), snap(22_000, 1_000)).cause is not MissCause.BREAKPOINT_MISPLACED
    # OpenAI reports no breakpoints: the rule is Anthropic's.
    assert classify(snap(20_000, 0, written=2_000, provider="openai", model="gpt-5.6-terra"),
                    snap(22_000, 2_000, provider="openai", model="gpt-5.6-terra")).cause is MissCause.NONE
    # No breakpoint at all is the existing rule, unchanged.
    assert classify(snap(20_000, 0, has_breakpoint=False), snap(22_000, 0, has_breakpoint=False)).cause \
        is MissCause.BREAKPOINT_MISPLACED


def test_the_proxy_charges_the_stable_tail_not_the_held_cap():
    # haiku: breakpoint on the system block only. Turn 1 writes 2,000 (the system) and sends
    # 18,000 uncached; turn 2 reads the 2,000 and sends 20,000 uncached.
    p = proxy([a_usage(write5=2_000, fresh=18_000), a_usage(read=2_000, fresh=20_000)])
    turn(p, body([USER]))
    e = turn(p, body([USER, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]))
    assert e.cause == "breakpoint_misplaced" and e.legit == 0
    assert e.avoidable_tokens == 18_000  # the shared prefix (20,000) minus what was read, not capped at 2,000
    r = T.rates_for(MODEL)
    assert e.avoidable_cost == pytest.approx(18_000 * (r["input"] - r["cache_read"]) / 1e6)  # turn 2 wrote nothing
    assert e.unread_write_tokens is None  # the write was read: nothing unread


def test_a_replaced_one_shot_or_a_shorter_turn_is_not_an_early_breakpoint():
    # Side calls: the same 2,000-token system block (breakpoint on it) and a different 18,000-token
    # document each time. The breakpoint is right where it should be; the documents are not stable.
    doc_a = FP.fingerprint_blocks([("system", "Be careful."), ("messages", {"role": "user", "content": "doc A " * 50})])
    doc_b = FP.fingerprint_blocks([("system", "Be careful."), ("messages", {"role": "user", "content": "doc B " * 50})])
    c = classify(snap(20_000, 0, written=2_000, blocks=doc_a), snap(20_000, 2_000, blocks=doc_b))
    assert c.cause is MissCause.NONE and MissCause.HISTORY_EDIT in c.secondary
    # A rewritten, shorter last turn: the previous prompt is not all repeated, so no tail is claimed.
    assert classify(snap(20_000, 0, written=2_000), snap(15_000, 2_000)).cause is MissCause.NONE


def test_the_held_cap_still_applies_to_other_misplaced_breakpoint_verdicts():
    # No cache_control at all (nothing held, nothing read): the existing verdict and its waste are unchanged.
    p = proxy([a_usage(fresh=8_000), a_usage(fresh=8_300)])
    turn(p, body([USER], system_marker=False))
    e = turn(p, body([USER, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}], system_marker=False))
    assert e.cause == "breakpoint_misplaced" and e.avoidable_tokens == 8_000
