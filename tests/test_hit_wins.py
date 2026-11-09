"""A full cache hit wins over every inference (2026-11-08 review), and the price of a cache write
that was never read is visible.

1. Blocks unchanged and the cache served the whole reusable prefix: `none`, whatever else differs
   (a param the provider did not render, no cache_control yet a hit, a prompt the minimum table
   says is too short). Before, those requests were PARAM_RENDERED / BREAKPOINT_MISPLACED /
   BELOW_MIN_PREFIX: a hit reported as a miss.
2. A miss that read nothing right after a cache write: the previous turn's write tokens were
   never read, and the surcharge paid to write them is shown, separately from avoidable cost.
"""

import copy

import pytest
from test_details import MODEL, USER, a_usage, body, proxy, turn

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


def test_unread_write_only_after_a_cold_miss_that_replaced_the_prefix():
    assert unread_write("volatile_system", 0, 8_000, 0, 0) == (8_000, 0, 0)
    assert unread_write("ttl_expired", 0, 0, 8_000, 0) == (0, 8_000, 0)
    assert unread_write("history_edit", 0, 0, 0, 8_000) == (0, 0, 8_000)  # single-rate provider
    # a read that stops short of the write leaves the rest unread: the previous turn wrote after what
    # it read, so only the part of this read beyond that counts as read back
    assert unread_write("volatile_system", 100, 8_000, 0, 0) == (7_900, 0, 0)
    assert unread_write("history_edit", 10_000, 4_000, 0, 0, prev_cached_tokens=10_000) == (4_000, 0, 0)
    assert unread_write("history_edit", 12_000, 4_000, 0, 0, prev_cached_tokens=10_000) == (2_000, 0, 0)
    assert unread_write("history_edit", 14_000, 4_000, 0, 0, prev_cached_tokens=10_000) == (0, 0, 0)
    assert unread_write("history_edit", 9_000, 4_000, 0, 0, prev_cached_tokens=10_000) == (4_000, 0, 0)
    assert unread_write("history_edit", 10_000, 3_000, 1_000, 0, prev_cached_tokens=10_000) == (3_000, 1_000, 0)
    assert unread_write("history_edit", 12_000, 3_000, 1_000, 0, prev_cached_tokens=10_000) == (1_500, 500, 0)
    assert unread_write("history_edit", 10_000, 0, 0, 4_000, prev_cached_tokens=10_000) == (0, 0, 4_000)
    # The entry was not replaced: a later turn may still read it (review: a retry, then a hit).
    for kept in ("none", "first_write", "fork_boundary", "model_switch", "retry_replay", "routing_miss",
                 "evicted", "unknown", "breakpoint_misplaced"):
        assert unread_write(kept, 0, 8_000, 0, 0) == (0, 0, 0)


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


# End to end through the proxy (review: both production paths were untested).

DATED = lambda d: [{"type": "text", "text": "You are a careful agent. " * 200 + f" today {d}",  # noqa: E731
                    "cache_control": {"type": "ephemeral"}}]
MORE = [USER, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]


def dated(messages, day):
    b = body(messages)
    b["system"] = DATED(day)
    return b


def test_proxy_prices_an_unread_write_without_the_ttl_split():
    # Anthropic usage without cache_creation: the write was 5 minutes (pricing.breakdown reads it so).
    def nosplit(**k):
        u = a_usage(**k)
        u.pop("cache_creation")
        return u
    p = proxy([nosplit(write5=8_000), nosplit(write5=8_100, fresh=100)])
    turn(p, dated([USER], "2026-11-07"))
    e = turn(p, dated(MORE, "2026-11-08"))
    assert e.cause == "volatile_system" and e.unread_write_tokens == 8_000
    assert e.unread_write_cost == pytest.approx(T.write_surcharge(MODEL, 8_200, write_5m=8_000))


def test_side_call_on_the_same_model_does_not_claim_the_main_threads_write():
    # A one-message side call falls back to the main thread's turn for comparison; the main
    # thread's next turn then reads that write, so it was not left unread.
    side = dated([{"role": "user", "content": "title this"}], "2026-11-08")
    p = proxy([a_usage(write5=8_000), a_usage(write5=8_100, fresh=100), a_usage(write5=7_000, fresh=0),
               a_usage(read=8_100, fresh=50)])
    turn(p, dated([USER], "2026-11-07"))
    turn(p, dated(MORE, "2026-11-08"))
    e_side = turn(p, side)
    e_main = turn(p, dated(MORE + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "x"}], "2026-11-08"))
    assert e_side.unread_write_tokens is None
    assert e_main.cause == "none" and e_main.unread_write_tokens is None


def test_retry_does_not_claim_the_write_the_next_turn_reads():
    b = dated([USER], "2026-11-07")
    p = proxy([a_usage(write5=8_000), a_usage(write5=8_000, fresh=0), a_usage(read=8_000)])
    turn(p, b)
    e2 = turn(p, copy.deepcopy(b))
    e3 = turn(p, copy.deepcopy(b))
    assert e2.cause == "retry_replay" and e2.unread_write_tokens is None
    assert e3.cause == "none"


def test_static_breakpoint_hit_with_a_param_change_is_none():
    # The commonest Anthropic layout: the breakpoint stays on the system block, each turn's tail
    # is uncached. Turn 2 reads everything turn 1 left in the cache (8,000 of its 8,200), so a
    # changed param that the provider did not render is no miss.
    p = proxy([a_usage(write5=8_000), a_usage(read=8_000, write5=100, fresh=0)])
    turn(p, dated([USER], "2026-11-07"))
    b2 = dated(MORE, "2026-11-07")
    b2["tool_choice"] = {"type": "any"}
    e = turn(p, b2)
    assert e.cause == "none" and "param_rendered" in (e.secondary_causes or "")


def test_steady_state_hit_with_a_param_change_is_none():
    # Turn 2 read the system block and wrote nothing (the breakpoint stays on it); turn 3 reads
    # the same 8,000 with a changed param. The held prefix is what turn 2 read: a full hit.
    p = proxy([a_usage(write5=8_000), a_usage(read=8_000, fresh=300), a_usage(read=8_000, fresh=500)])
    turn(p, dated([USER], "2026-11-07"))
    turn(p, dated(MORE, "2026-11-07"))
    b3 = dated(MORE + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "x"}], "2026-11-07")
    b3["tool_choice"] = {"type": "any"}
    e = turn(p, b3)
    assert e.cause == "none" and "8000 of 8000" in e.evidence


def test_compaction_counts_the_write_it_replaced():
    # A compaction rewrites the opening message, so identity alone would call it another
    # conversation; the compaction header says it is the same one, and its write was replaced.
    p = proxy([a_usage(write5=8_000), a_usage(write5=8_100, fresh=100), a_usage(write5=3_000, fresh=0)])
    turn(p, dated([USER], "2026-11-07"))
    turn(p, dated(MORE, "2026-11-08"))
    summary = dated([{"role": "user", "content": "Summary of the conversation so far: ..."}], "2026-11-08")
    e = p.handle("anthropic", summary, {"x-prefixwise-session-id": "s", "x-prefixwise-compaction-event": "1"})[1]
    assert e.cause == "compaction_rewrite" and e.unread_write_tokens == 8_100


def test_compaction_while_another_conversation_is_interleaved_claims_nothing():
    # Main thread M and subagent S share the lane (same model and system prompt). S's turn is the
    # lane's latest when M compacts: M's rewritten opening matches neither, so it is compared with
    # S's turn; S's write is not M's to call unread (S's next turn reads it).
    p = proxy([a_usage(write5=8_000), a_usage(write5=6_000), a_usage(write5=8_100, fresh=100),
               a_usage(write5=6_100, fresh=100), a_usage(write5=3_000, fresh=0)])
    sub = [{"role": "user", "content": "subagent: explore"}]
    turn(p, dated([USER], "2026-11-07"))
    turn(p, dated(sub, "2026-11-07"))
    turn(p, dated(MORE, "2026-11-07"))
    turn(p, dated(sub + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}], "2026-11-07"))
    summary = dated([{"role": "user", "content": "Summary of the conversation so far: ..."}], "2026-11-07")
    e = p.handle("anthropic", summary, {"x-prefixwise-session-id": "s", "x-prefixwise-compaction-event": "1"})[1]
    assert e.cause == "compaction_rewrite" and e.unread_write_tokens is None


def test_a_trimmed_history_with_a_hit_on_the_system_prompt_leaves_the_write_unread():
    """Reddit (Specialist-Can2889): an agent that trims older messages every turn while its system
    prompt and tools still hit. Turn 1 reads 10,000 and writes 4,000 of messages; turn 2 rewrites
    history, reads the 10,000 again and nothing of the write: 4,000 unread tokens, priced."""
    from test_details import a_usage, body, proxy
    p = proxy([a_usage(read=10_000, write5=4_000, fresh=0), a_usage(read=10_000, write5=4_200, fresh=0)])
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a long answer " * 50},
            {"role": "user", "content": "q2"}]
    p.handle("anthropic", body(msgs, last_marker=True), {"x-prefixwise-session-id": "s"})
    trimmed = [msgs[0], {"role": "assistant", "content": "trimmed to the question"}, msgs[2],
               {"role": "assistant", "content": "a2"}, {"role": "user", "content": "q3"}]
    res, entry = p.handle("anthropic", body(trimmed, last_marker=True), {"x-prefixwise-session-id": "s"})
    assert res["cause"] == "history_edit"
    assert entry.unread_write_tokens == 4_000 and entry.unread_write_cost is not None and entry.unread_write_cost > 0
