"""Wrong verdicts an independent review of the core found and reproduced (#184, items 1, 2, 4, 5, 6
and 7). Each test is the review's reproduction, through a real proxy whose upstream is scripted."""
import pytest

import prefixwise.proxy as px
from prefixwise.collector import LANE_THREADS, Collector
from prefixwise.ledger import Ledger
from prefixwise.pricing import PriceTable
from prefixwise.proxy import PrefixwiseProxy, extract_blocks

SYS = "You are a careful agent. " * 400  # about 2,000 tokens
U1 = {"role": "user", "content": "start"}
A1 = {"role": "assistant", "content": "ok"}
GO = {"role": "user", "content": "go"}
S = {"x-prefixwise-session-id": "s"}
OPUS = "claude-opus-5-5"


def scripted(usages, **kw):
    proxy = PrefixwiseProxy(upstream_url="http://unused.invalid", api_key="k", **kw)
    queue = list(usages)

    def forward(provider, body, request_id, session_id, client_headers, api="chat"):
        prefix = "msg_" if provider == "anthropic" else "chatcmpl-"
        return 200, {"id": prefix + request_id[:8], "usage": queue.pop(0)}

    proxy._forward = forward
    return proxy


def claude(read=0, write=0, fresh=200):
    return {"input_tokens": fresh, "output_tokens": 5, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write, "cache_creation": {"ephemeral_5m_input_tokens": write}}


def gpt(prompt, cached):
    return {"prompt_tokens": prompt, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": cached}}


def marked(text=SYS):
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def idle(proxy, seconds, session="s"):
    """Pretend the session's previous request was sent, and answered, `seconds` earlier."""
    for (sid, _), lane in proxy.lane_threads.items():
        if sid == session:
            snap, done, facts, rid = lane.threads[lane.latest]
            lane.threads[lane.latest] = (snap, done - seconds, facts, rid)
            snap.sent_at -= seconds
    if session in proxy.last_seen:
        proxy.last_seen[session] -= seconds


def lifetime_rows(proxy):
    with proxy.ledger.reader() as conn:
        return [r[0] for r in conn.execute("SELECT lifetime_evidence FROM entries ORDER BY id")]


# -- 1. another conversation is not an unchanged prompt that missed ------------------------------

def test_independent_one_shot_requests_are_not_routing_misses():
    """Four classification calls with no session id: one short system prompt, a different document
    each time, nothing read. They were `routing_miss` ("blocks and params equal") and were kept as
    evidence of how long the cache lasts."""
    proxy = scripted([gpt(2500, 0)] * 4)
    got = []
    for i in range(4):
        body = {"model": "gpt-5.6-terra", "messages": [
            {"role": "system", "content": "Classify the document as spam or not spam."},
            {"role": "user", "content": f"Document {i}: " + f"lorem {i} ipsum " * 800}]}
        got.append(proxy.handle("openai", body, {})[1])
        if i == 2:
            idle(proxy, 900, session="default")
    assert [e.cause for e in got] == ["first_write"] * 4 and all(e.legit for e in got)
    assert "a new conversation in this session" in got[1].evidence
    assert lifetime_rows(proxy) == [None] * 4


def test_a_shared_system_prompt_read_in_full_is_not_a_miss_of_the_document():
    doc = lambda i: [{"role": "user", "content": [{"type": "text", "text": f"Document {i}: " + f"alpha {i} " * 900,  # noqa: E731
                                                   "cache_control": {"type": "ephemeral"}}]}]
    proxy = scripted([claude(write=5000, fresh=0), claude(read=2000, write=3000, fresh=0),
                      claude(read=2000, write=3000, fresh=0)])
    got = [proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": doc(i)}, {})[1]
           for i in range(3)]
    assert [e.cause for e in got] == ["first_write"] * 3
    assert "2,000 of its 5,000 prompt tokens were read" in got[1].evidence


def conversation_pair(headers):
    system = {"role": "system", "content": "You are a support agent. " * 300}
    a1 = [system, {"role": "user", "content": "My printer is broken, model X100."}]
    a2 = a1 + [{"role": "assistant", "content": "Try turning it off."}, {"role": "user", "content": "Did not help."}]
    b1 = [system, {"role": "user", "content": "How do I reset my password?"}]
    proxy = scripted([gpt(2000, 0), gpt(2100, 1920), gpt(2050, 1920)])
    return [proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": m}, headers)[1] for m in (a1, a2, b1)]


def test_a_new_conversation_with_no_session_id_is_not_a_history_edit():
    """Traffic that names no conversation all lands in one session. A second user's first message
    after a first user's two turns was `history_edit`, avoidable waste."""
    b1 = conversation_pair({})[2]
    assert b1.cause == "none" and b1.avoidable_tokens == 0 and "a new conversation in this session" in b1.evidence


def test_in_a_named_conversation_a_rewritten_first_message_is_still_an_edit():
    b1 = conversation_pair(S)[2]
    assert b1.cause == "history_edit" and b1.legit is False


# -- 2. a cache entry's age counts from when the request that wrote it was sent ------------------

@pytest.mark.parametrize("generation, idle_for", [(5, 335), (280, 60)])
@pytest.mark.parametrize("edited", [False, True])
def test_generation_time_counts_toward_the_cache_lifetime(monkeypatch, generation, idle_for, edited):
    """340 s after the request that wrote the entry, a 5-minute entry is gone, whether those seconds
    were idle time or the model writing its answer. Measured on real logs: two entries read nothing
    318 s after the request that wrote them, 153 s and 292 s after its response ended. With 280 s of
    generation this was `routing_miss`, or `history_edit` waste when the history had changed."""
    clock = [1_800_000_000.0]
    monkeypatch.setattr(px.time, "time", lambda: clock[0])
    proxy = PrefixwiseProxy(upstream_url="http://unused.invalid", api_key="k")
    usages, takes = [claude(write=3000), claude(write=3100)], [generation, 1]

    def forward(provider, body, request_id, session_id, client_headers, api="chat"):
        clock[0] += takes.pop(0)  # the model generates for this long
        return 200, {"id": "msg_" + request_id[:8], "usage": usages.pop(0)}

    proxy._forward = forward
    hist = [{"role": "user", "content": "plan the report"}, {"role": "assistant", "content": "plan"},
            {"role": "user", "content": "write it"}]
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 64000, "system": marked(), "messages": hist}, S)
    clock[0] += idle_for
    if edited:
        hist = [{"role": "user", "content": "plan the report (edited)"}] + hist[1:]
    entry = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(),
                                       "messages": hist + [{"role": "assistant", "content": "report"}, GO]}, S)[1]
    assert entry.cause == "ttl_expired" and entry.legit is True and entry.avoidable_tokens == 0
    assert entry.gap_seconds == pytest.approx(idle_for)  # the stored gap stays the idle time
    if generation > 100:
        assert "340s since the previous turn was sent (its response ended 60s ago)" in entry.evidence
    else:
        assert "340s since the previous turn was sent (its response ended 335s ago)" in entry.evidence


def test_a_retry_is_still_told_by_idle_time():
    """A request repeated right after a slow answer is a retry: that rule reads the idle gap."""
    proxy = scripted([claude(write=3000), claude(write=3000)])
    body = {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": [U1]}
    proxy.handle("anthropic", body, S)
    lane = proxy.lane_threads[("s", OPUS)]
    lane.threads[lane.latest][0].sent_at -= 200   # it took 200 s to answer; the retry follows at once
    assert proxy.handle("anthropic", body, S)[1].cause == "retry_replay"


# -- 4, 5. no change is claimed against a turn that is not this turn's predecessor ----------------

@pytest.mark.parametrize("agents", [LANE_THREADS, LANE_THREADS + 1, LANE_THREADS + 4])
def test_more_conversations_than_the_lane_remembers_do_not_turn_hits_into_waste(agents):
    """Each conversation reads everything its own previous turn left. With nine interleaved, 26 of 27
    such hits were `history_edit`, compared with another conversation's turn."""
    usages, order, size = [], [], {i: 3000 + 137 * i for i in range(agents)}
    for t in range(4):
        for i in range(agents):
            grow = 400 + 50 * ((i * 7 + t * 3) % 5)
            usages.append(claude(read=0 if t == 0 else size[i], write=size[i] if t == 0 else grow, fresh=0))
            size[i] += grow if t else 0
            order.append((i, t))
    proxy = scripted(usages)
    convs = {i: [{"role": "user", "content": f"research topic {i}"}] for i in range(agents)}
    later = []
    for i, t in order:
        if t:
            convs[i] += [{"role": "assistant", "content": f"notes {i}.{t} " * 50}, {"role": "user", "content": f"continue {i}.{t}"}]
        last = {**convs[i][-1], "content": [{"type": "text", "text": convs[i][-1]["content"],
                                             "cache_control": {"type": "ephemeral"}}]}
        entry = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(),
                                           "messages": convs[i][:-1] + [last]}, {"x-prefixwise-session-id": "swarm"})[1]
        if t:
            later.append(entry)
    assert {e.cause for e in later} == {"none"} and sum(e.avoidable_tokens for e in later) == 0
    if agents > LANE_THREADS:
        assert any("no longer remembered" in e.evidence for e in later)


def test_a_forgotten_conversation_that_missed_is_unknown_not_blamed():
    proxy = scripted([claude(write=3000, fresh=0)] * (LANE_THREADS + 1) + [claude(write=3400, fresh=0)])
    turn = lambda i, more=(): {"model": OPUS, "max_tokens": 9, "system": marked(),  # noqa: E731
                               "messages": [{"role": "user", "content": f"topic {i}"}, *more]}
    for i in range(LANE_THREADS + 1):
        proxy.handle("anthropic", turn(i), S)
    entry = proxy.handle("anthropic", turn(0, [A1, GO]), S)[1]      # conversation 0 was dropped to make room
    assert entry.cause == "unknown" and entry.avoidable_tokens == 0 and "no longer remembered" in entry.evidence


def test_a_record_delivered_late_is_judged_by_its_own_read():
    """Turn 2's record reaches the collector after turn 3's. It read all 3,000 tokens turn 1 left; it
    was compared with turn 3 and called `history_edit`. Nor may it become the conversation's latest."""
    edge = PrefixwiseProxy(upstream_url="http://unused.invalid", api_key="k")

    def body(n):
        msgs = [{"role": "user", "content": "start"}]
        for i in range(1, n):
            msgs += [{"role": "assistant", "content": f"answer {i} " * 200}, {"role": "user", "content": f"next {i}"}]
        msgs[-1] = {**msgs[-1], "content": [{"type": "text", "text": msgs[-1]["content"],
                                             "cache_control": {"type": "ephemeral"}}]}
        return {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": msgs}

    t0 = 1_800_000_000.0

    def record(n, usage):  # turn n: sent at t0 + 10n, answered 5 s later
        ctx = edge.prepare("anthropic", body(n), S)
        ctx["sent_at"], ctx["span_end"] = t0 + 10 * n, t0 + 10 * n + 5
        fps = edge.fingerprinter.fingerprint_blocks(extract_blocks(ctx["forward_body"], "anthropic"))
        return edge._record(ctx, 200, fps, usage=usage, payload={"id": f"msg_{n}", "usage": usage})

    recs = {1: record(1, claude(write=3000, fresh=0)), 2: record(2, claude(read=3000, write=500, fresh=0)),
            3: record(3, claude(read=3500, write=500, fresh=0)), 4: record(4, claude(read=4000, write=500, fresh=0))}
    collector = Collector(Ledger(":memory:"), PriceTable.load())
    got = {n: collector.ingest(recs[n])[1] for n in (1, 3, 2, 4)}
    assert got[2].cause == "none" and got[2].avoidable_tokens == 0 and "arrived late" in got[2].evidence
    assert got[4].cause == "none" and "4000 of 4000" in got[4].evidence   # still compared with turn 3
    missed = Collector(Ledger(":memory:"), PriceTable.load())
    late = record(2, claude(read=0, write=3500, fresh=0))
    assert [missed.ingest(r)[1].cause for r in (recs[1], recs[3], late)][2] == "unknown"


# -- 6, 7. rules that must look at what was, or could still be, in the cache ----------------------

def test_no_markers_after_the_longest_lifetime_is_an_expiry_not_waste():
    """No cache_control anywhere: 3,000 avoidable tokens, even two hours later, when no marker could
    have kept the entry alive."""
    def run(gap):
        proxy = scripted([claude(fresh=3000), claude(fresh=3100)])
        proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": [U1]}, S)
        idle(proxy, gap)
        return proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": [U1, A1, GO]}, S)[1]
    soon, late = run(10), run(7200)
    assert (soon.cause, soon.legit, soon.avoidable_tokens) == ("breakpoint_misplaced", False, 3000)
    assert (late.cause, late.legit, late.avoidable_tokens) == ("ttl_expired", True, 0)
    assert "breakpoint_misplaced" in (late.secondary_causes or "")


def test_a_change_in_a_prompt_with_no_markers_says_both_need_fixing():
    proxy = scripted([claude(fresh=3000), claude(fresh=3100)])
    stamp = lambda s: {"model": OPUS, "max_tokens": 9, "system": SYS + f" Now: 2026-11-26T09:00:{s}Z",  # noqa: E731
                       "messages": [U1]}
    proxy.handle("anthropic", stamp("01"), S)
    entry = proxy.handle("anthropic", {**stamp("09"), "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "volatile_system"   # still the first thing to fix: a marker on it would buy unread writes
    assert "also carries no cache_control" in entry.evidence and "breakpoint_misplaced" in entry.secondary_causes


def test_with_markers_and_nothing_held_a_change_had_nothing_to_lose():
    """The previous turn's marked prefix was too small to cache: it read 0 and wrote 0. A timestamp
    that then changed cost nothing, and was charged 2,500 avoidable tokens."""
    proxy = scripted([claude(fresh=2500), claude(fresh=2600)])
    stamp = lambda s, msgs: {"model": "claude-sonnet-4-6", "max_tokens": 9,  # noqa: E731
                             "system": marked("You are a careful agent. " * 150 + f" Now: 2026-11-26T09:00:{s}Z"),
                             "messages": msgs}
    proxy.handle("anthropic", stamp("01", [{"role": "user", "content": "context " * 1500}]), S)
    entry = proxy.handle("anthropic", stamp("09", [{"role": "user", "content": "context " * 1500}, A1, GO]), S)[1]
    assert entry.avoidable_tokens == 0


def test_a_turn_that_left_nothing_in_the_cache_is_not_a_routing_miss():
    """Sonnet 4.6 (minimum 1,024): the only marker is on a 900-token system prompt. Anthropic reports
    nothing written and nothing read on every turn. There was nothing for a router to miss."""
    proxy = scripted([claude(fresh=2500), claude(fresh=2600), claude(fresh=2700)])
    msgs = [{"role": "user", "content": "context " * 1500}]
    got = []
    for i in range(3):
        got.append(proxy.handle("anthropic", {"model": "claude-sonnet-4-6", "max_tokens": 9,
                                              "system": marked("You are a careful agent. " * 150), "messages": msgs}, S)[1])
        msgs = msgs + [A1, {"role": "user", "content": f"go {i}"}]
    assert [e.cause for e in got] == ["first_write", "below_min_prefix", "below_min_prefix"]
    assert "left nothing in the cache" in got[1].evidence and "1024" in got[1].evidence
