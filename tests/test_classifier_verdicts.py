"""Wrong verdicts an independent review of the core found and reproduced (#184, items 1, 2, 4, 5, 6
and 7), and the rules a second review asked of the fix (#187). Each test is a reproduction, through
a real proxy whose upstream is scripted, or through a collector fed the records an edge builds."""
import collections

import pytest

import prefixwise.proxy as px
from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.collector import LANE_THREADS, Collector
from prefixwise.findings import ttl_advice
from prefixwise.fingerprint import Fingerprinter
from prefixwise.ledger import Ledger, LedgerEntry
from prefixwise.pricing import PriceTable
from prefixwise.proxy import PrefixwiseProxy, extract_blocks
from prefixwise.records import TurnRecord

SYS = "You are a careful agent. " * 400  # about 2,000 tokens
U1 = {"role": "user", "content": "start"}
A1 = {"role": "assistant", "content": "ok"}
GO = {"role": "user", "content": "go"}
S = {"x-prefixwise-session-id": "s"}
OPUS = "claude-opus-5-5"
T0 = 1_800_000_000.0
AGENT = {"role": "system", "content": "You are a coding agent. " * 300}


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


def marked(text=SYS, ttl=None):
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}}]


def last_marked(msgs):
    """The messages, with cache_control on the last one (the usual moving breakpoint)."""
    last = msgs[-1]
    return msgs[:-1] + [{**last, "content": [{"type": "text", "text": last["content"],
                                              "cache_control": {"type": "ephemeral"}}]}]


def shift(proxy, session="s", sent=0.0, done=0.0):
    """Move the session's latest turn back in time: sent `sent` seconds earlier than it was, and
    answered `done` seconds earlier."""
    for (sid, _), lane in proxy.lane_threads.items():
        if sid == session:
            snap, when, facts, rid = lane.threads[lane.latest]
            lane.threads[lane.latest] = (snap, when - done, facts, rid)
            if getattr(snap, "sent_at", None):
                snap.sent_at -= sent
            if facts.sent_at:
                facts.sent_at -= sent
            if facts.done_at:
                facts.done_at -= done
    if session in proxy.last_seen:
        proxy.last_seen[session] -= done


def idle(proxy, seconds, session="s"):
    """Pretend the session's previous request was sent, and answered, `seconds` earlier."""
    shift(proxy, session, sent=seconds, done=seconds)


def lifetime_rows(proxy):
    with proxy.ledger.reader() as conn:
        return [r[0] for r in conn.execute("SELECT lifetime_evidence FROM entries ORDER BY id")]


class Edge:
    """An edge that only builds records: each is stamped with the times given and handed to a
    collector by the test, in whatever order the test delivers them."""

    def __init__(self, **kw):
        self.proxy = PrefixwiseProxy(upstream_url="http://unused.invalid", api_key="k", **kw)
        self.count = 0

    def record(self, provider, body, headers, sent, done, usage, api="chat", response_id=None):
        ctx = self.proxy.prepare(provider, body, headers, api=api)
        ctx["sent_at"], ctx["span_end"] = sent, done
        ctx["facts"].sent_at = sent
        fps = self.proxy.fingerprinter.fingerprint_blocks(extract_blocks(ctx["forward_body"], provider))
        self.count += 1
        rid = response_id or f"{'msg_' if provider == 'anthropic' else 'chatcmpl-'}{self.count}"
        return self.proxy._record(ctx, 200, fps, usage=usage, payload={"id": rid, "usage": usage})


def collector():
    return Collector(Ledger(":memory:"), PriceTable.load())


def chat(user, turns, mark=True):
    """One user's conversation after `turns` user messages."""
    msgs = [{"role": "user", "content": f"hello from {user}"}]
    for i in range(1, turns):
        msgs += [{"role": "assistant", "content": f"answer {user}.{i} " * 100}, {"role": "user", "content": f"next {user}.{i}"}]
    return last_marked(msgs) if mark else msgs


def snap(blocks, prompt, cached, provider="anthropic", model=OPUS, **kw):
    fields = dict(gap_seconds=5.0, ttl_seconds=300.0, has_breakpoint=True)
    fields.update(kw)
    return RequestSnapshot(model=model, provider=provider, blocks=blocks, prompt_tokens=prompt,
                           cached_tokens=cached, **fields)


FP = Fingerprinter()


def blocks_of(body, provider="anthropic"):
    return FP.fingerprint_blocks(extract_blocks(body, provider))


# -- 1. another conversation is not an unchanged prompt that missed ------------------------------

def test_independent_one_shot_requests_are_not_routing_misses():
    """Four classification calls with no session id: one short system prompt, a different document
    each time, nothing read. They were `routing_miss` ("blocks and params equal") and were kept as
    evidence of how long the cache lasts. Whether the short shared part was missed is not known, so
    the verdict is the conversation's first write with its legitimacy left open."""
    proxy = scripted([gpt(2500, 0)] * 4)
    got = []
    for i in range(4):
        body = {"model": "gpt-5.6-terra", "messages": [
            {"role": "system", "content": "Classify the document as spam or not spam."},
            {"role": "user", "content": f"Document {i}: " + f"lorem {i} ipsum " * 800}]}
        got.append(proxy.handle("openai", body, {})[1])
        if i == 2:
            idle(proxy, 900, session="default")
    assert [e.cause for e in got] == ["first_write"] * 4
    assert [e.legit for e in got] == [True, None, None, None] and sum(e.avoidable_tokens for e in got) == 0
    assert "a new conversation in this session" in got[1].evidence
    assert "the size of that shared part is not known" in got[1].evidence
    assert lifetime_rows(proxy) == [None] * 4


def test_a_shared_system_prompt_read_in_full_is_not_a_miss_of_the_document():
    """The 2,000-token system prompt is read on every call; the 3,000-token document is new each
    time. Under half of the prompt was read, and nothing was missed: these were `routing_miss`."""
    doc = lambda i: [{"role": "user", "content": [{"type": "text", "text": f"Document {i}: " + f"alpha {i} " * 900,  # noqa: E731
                                                   "cache_control": {"type": "ephemeral"}}]}]
    proxy = scripted([claude(write=5000, fresh=0), claude(read=2000, write=3000, fresh=0),
                      claude(read=2000, write=3000, fresh=0)])
    got = [proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": doc(i)}, {})[1]
           for i in range(3)]
    assert [e.cause for e in got] == ["first_write"] * 3 and [e.legit for e in got] == [True, None, None]
    assert "2,000 of its 5,000 prompt tokens were read" in got[1].evidence


def conversation_pair(headers, proxy=None):
    system = {"role": "system", "content": "You are a support agent. " * 300}
    a1 = [system, {"role": "user", "content": "My printer is broken, model X100."}]
    a2 = a1 + [{"role": "assistant", "content": "Try turning it off."}, {"role": "user", "content": "Did not help."}]
    b1 = [system, {"role": "user", "content": "How do I reset my password?"}]
    proxy = proxy or scripted([gpt(2000, 0), gpt(2100, 1920), gpt(2050, 1920)])
    return [proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": m}, headers)[1] for m in (a1, a2, b1)]


def test_a_new_conversation_with_no_session_id_is_not_a_history_edit():
    """Traffic that names no conversation all lands in one session. A second user's first message
    after a first user's two turns was `history_edit`, avoidable waste."""
    b1 = conversation_pair({})[2]
    assert b1.cause == "none" and b1.avoidable_tokens == 0 and "a new conversation in this session" in b1.evidence


def test_in_a_named_conversation_a_rewritten_first_message_is_still_an_edit():
    b1 = conversation_pair(S)[2]
    assert b1.cause == "history_edit" and b1.legit is False


def test_an_end_user_id_names_a_person_not_a_conversation():
    """With only an end-user id (OpenAI's `user`) to go by, the session is that person's: their
    next conversation is a new one, not an edit of the last."""
    system = {"role": "system", "content": "You are a support agent. " * 300}
    proxy = scripted([gpt(2000, 0), gpt(2100, 1920), gpt(2050, 1920)])
    send = lambda *msgs: proxy.handle("openai", {"model": "gpt-5.6-terra", "user": "user-4711",  # noqa: E731
                                                 "messages": [system, *msgs]}, {})[1]
    send({"role": "user", "content": "My printer is broken."})
    send({"role": "user", "content": "My printer is broken."}, {"role": "assistant", "content": "Try again."},
         {"role": "user", "content": "Did not help."})
    entry = send({"role": "user", "content": "How do I reset my password?"})
    assert entry.session_id == "user-4711" and entry.cause == "none" and "a new conversation in this session" in entry.evidence


def test_an_edge_that_says_the_session_is_a_user_id_is_believed(tmp_path):
    """The SDK recorder and the LiteLLM callback pass the user id as the session and say so
    (`session_is_user`); such a record names no conversation either, whatever else it carries."""
    from prefixwise.sdk import Recorder
    rec = Recorder(str(tmp_path / "sdk.db"))
    system = {"role": "system", "content": "You are a support agent. " * 300}
    reply = lambda usage: {"id": "chatcmpl-1", "choices": [], "usage": usage}  # noqa: E731
    send = lambda usage, *msgs: rec.record("openai", {"model": "gpt-5.6-terra", "messages": [system, *msgs]}, reply(usage),  # noqa: E731
                                           session_id="user-4711", session_is_user=True)
    first = {"role": "user", "content": "My printer is broken."}
    send(gpt(2000, 0), first)
    send(gpt(2100, 1920), first, {"role": "assistant", "content": "Try again."}, {"role": "user", "content": "Did not help."})
    entry = send(gpt(2050, 1920), {"role": "user", "content": "How do I reset my password?"})
    assert entry.cause == "none" and "a new conversation in this session" in entry.evidence
    rec.close()


def test_a_new_conversation_does_not_inherit_the_previous_ones_tool_results_or_tail():
    """After a conversation with a tool result, a new single message with no session id: the tool
    result that is "gone" was never this conversation's (it was `tool_result_mutation`, waste). And
    when the cache serves all the previous turn held, the stable tail of the previous conversation
    is not this one's to have cached (it was `breakpoint_misplaced`, "both turns share a
    6,000-token prefix")."""
    tools = [{"name": "search", "description": "search things", "input_schema": {"type": "object", "properties": {}}}]
    first = [{"role": "user", "content": "find the report " + "and its appendix " * 900},
             {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "report"}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "found it"}]}]
    # (Read nothing: the marked system prompt the previous request cached was missed, a routing miss.)
    for usage, cause in ((claude(read=2000, fresh=4500), "none"), (claude(fresh=6500), "routing_miss")):
        proxy = scripted([claude(write=2000, fresh=4000), usage])
        body = lambda msgs: {"model": OPUS, "max_tokens": 9, "tools": tools, "system": marked(), "messages": msgs}  # noqa: E731
        proxy.handle("anthropic", body(first), {})
        entry = proxy.handle("anthropic", body([{"role": "user", "content": "another task " + "with a long brief " * 1100}]), {})[1]
        assert (entry.cause, entry.avoidable_tokens) == (cause, 0)
        assert "tool_result_mutation" not in (entry.secondary_causes or "")


def check_proxy():
    """The proxy as `prefixwise check` sets it up (cli.cmd_check): scenarios are named through
    default_session, which the pytest plugin sets per test."""
    proxy = scripted([gpt(2000, 0), gpt(2100, 1920), gpt(2050, 1920)])
    proxy.check_mode = True
    proxy.default_session = "tests/test_agent.py::test_refactor"
    return proxy


def run_proxy():
    """The proxy as `prefixwise run` sets it up (cli.cmd_run): one default session per run."""
    proxy = scripted([gpt(2000, 0), gpt(2100, 1920), gpt(2050, 1920)])
    proxy.default_session = "run-20261126-090000"
    proxy.default_session_shared = False
    return proxy


@pytest.mark.parametrize("setup", [check_proxy, run_proxy])
def test_a_check_scenario_and_a_run_are_one_conversation_not_a_shared_session(setup):
    """`prefixwise check --strict` failed a scenario whose agent rewrote its history into one
    message; counting the scenario as a shared session made that a new conversation, and the gate
    passed. The same for the one session `prefixwise run` gives a run."""
    proxy = setup()
    b1 = conversation_pair({}, proxy)[2]
    assert b1.session_id == proxy.default_session
    assert (b1.cause, b1.legit) == ("history_edit", False) and b1.avoidable_tokens == 2050 - 1920


def test_run_turns_the_shared_default_session_off(monkeypatch):
    """`prefixwise run` itself sets the flag the test above relies on."""
    import prefixwise.cli as cli
    seen = {}

    class Stop(Exception):
        pass

    def listen(proxy, port):
        seen["proxy"] = proxy
        raise Stop

    monkeypatch.setattr(cli, "_listen_local", listen)
    with pytest.raises(Stop):
        cli.main(["run", "--mock", "--", "true"])
    assert seen["proxy"].default_session.startswith("run-") and seen["proxy"].default_session_shared is False
    assert PrefixwiseProxy(mock=True).default_session_shared is True  # `serve`: the default session is shared


# -- a new conversation's first write does not hide a miss of a prefix known to be cached ---------

BIG = "You classify support tickets. " * 2000  # about 10,000 tokens


def one_shots(spacing, usages, system=None, doc_marked=False, session=None):
    """One-shot classification calls in one named session, `spacing` seconds apart."""
    proxy = scripted(usages)
    got = []
    for i in range(len(usages)):
        text = f"Ticket {i}: the printer in room {i} is out of toner"
        content = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}] if doc_marked else text
        body = {"model": OPUS, "max_tokens": 9, "system": marked(BIG) if system is None else system,
                "messages": [{"role": "user", "content": content}]}
        got.append(proxy.handle("anthropic", body, session if session is not None else {"x-prefixwise-session-id": "classifier"})[1])
        idle(proxy, spacing, session="classifier" if session is None else "default")
    return proxy, got


@pytest.mark.parametrize("spacing, cause, legit", [(360, "ttl_expired", True), (10, "routing_miss", None)])
def test_a_new_conversation_that_missed_a_prefix_known_to_be_cached_is_judged_as_a_miss(spacing, cause, legit):
    """A 10,000-token marked system prompt and a 60-token document per call; every call writes the
    10,000 tokens again and reads nothing. The previous call left that prefix in the cache at a
    marker this call repeats, so reading nothing is a miss of it: an expiry 6 minutes later, a
    routing miss 10 seconds later. Calling each the new conversation's `first_write` hid both, and
    the 1-hour-TTL advice with the expiries."""
    proxy, got = one_shots(spacing, [claude(write=10000, fresh=60)] * 3)
    assert [e.cause for e in got] == ["first_write", cause, cause]
    assert [e.legit for e in got] == [True, legit, legit] and sum(e.avoidable_tokens for e in got) == 0
    assert "a new conversation on the tools and system prompt the previous request cached" in got[1].evidence
    assert "blocks and params equal" not in got[1].evidence
    advice = ttl_advice(proxy.ledger)
    assert (sum(a["misses"] for a in advice) == 2) if cause == "ttl_expired" else not advice


def test_a_new_conversation_that_read_the_shared_prefix_is_a_hit():
    _, got = one_shots(10, [claude(write=10000, fresh=60), claude(read=10000, fresh=60)])
    assert (got[1].cause, got[1].legit, got[1].avoidable_tokens) == ("none", None, 0)


@pytest.mark.parametrize("case", ["part read", "no marker inside tools and system", "marker not repeated",
                                  "previous turn left nothing"])
def test_a_new_conversation_whose_shared_part_is_not_known_to_be_cached_is_a_first_write(case):
    """Where the previous turn is not known to have left the shared part in the cache at a marker
    this request repeats, no miss of it is claimed, and none is ruled out: `first_write`, with its
    legitimacy left open."""
    first, second, system2 = claude(write=10000, fresh=60), claude(write=10000, fresh=60), None
    kw = {}
    if case == "part read":  # something of the shared part was read: nothing says how much of it there is
        second = claude(read=4000, write=6000, fresh=60)
    elif case == "no marker inside tools and system":
        kw = {"system": BIG, "doc_marked": True}
        first = second = claude(write=10060, fresh=0)
    elif case == "previous turn left nothing":
        first = claude(fresh=10060)
    proxy = scripted([first, second])
    bodies = []
    for i in range(2):
        text = f"Ticket {i}: the printer in room {i} is out of toner"
        content = ([{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
                   if kw.get("doc_marked") or (case == "marker not repeated" and i == 1) else text)
        system = kw.get("system", marked(BIG))
        if case == "marker not repeated" and i == 1:
            system = BIG  # the same system prompt, its marker gone; the only marker is on the message
        bodies.append({"model": OPUS, "max_tokens": 9, "system": system, "messages": [{"role": "user", "content": content}]})
    got = [proxy.handle("anthropic", b, {"x-prefixwise-session-id": "classifier"})[1] for b in bodies]
    assert (got[1].cause, got[1].legit, got[1].avoidable_tokens) == ("first_write", None, 0)
    assert "the size of that shared part is not known" in got[1].evidence


def test_new_conversations_on_an_automatic_cache_are_a_hit_or_a_first_write():
    """OpenAI caches without markers, so nothing says how much of a new conversation's prompt the
    previous one left cached: a read of at least half is a hit; anything less is a first write with
    its legitimacy left open, never a routing miss or an expiry, and never lifetime evidence."""
    system = {"role": "system", "content": "You classify support tickets. " * 2000}
    proxy = scripted([gpt(10060, 0), gpt(10060, 9984), gpt(10060, 0), gpt(10060, 0)])
    got = []
    for i in range(4):
        got.append(proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [
            system, {"role": "user", "content": f"Ticket {i}: out of toner"}]}, {"x-prefixwise-session-id": "c"})[1])
        idle(proxy, 10 if i < 2 else 900, session="c")
    assert [(e.cause, e.legit) for e in got] == [("first_write", True), ("none", None), ("first_write", None),
                                                 ("first_write", None)]
    assert lifetime_rows(proxy)[2:] == [None, None]


def test_behind_an_engine_a_new_conversation_is_never_judged_by_anthropic_markers():
    one = lambda text: blocks_of({"model": OPUS, "system": SYS, "messages": [{"role": "user", "content": text}]})  # noqa: E731
    prev = snap(one("ticket 1"), 10060, 0, cache_written=10000)
    curr = lambda **kw: snap(one("ticket 2"), 10060, 0, shared_breakpoint=True, **kw)  # noqa: E731
    assert classify(prev, curr()).cause is MissCause.ROUTING_MISS
    untimed = classify(prev, curr(gap_seconds=None))   # no timing: the shared prefix was missed, the cause is not known
    assert untimed.cause is MissCause.UNKNOWN and not untimed.cold_unchanged and "a new conversation on the tools" in untimed.evidence
    assert classify(prev, curr(engine=True)).cause is MissCause.FIRST_WRITE
    assert classify(prev, snap(one("ticket 2"), 10060, 0)).cause is MissCause.FIRST_WRITE  # no shared marker
    # Nor on another provider, whatever fields a request in its shape carries.
    other = classify(snap(one("ticket 1"), 10060, 0, provider="openai", model="gpt-5.6-terra", cache_written=10000),
                     snap(one("ticket 2"), 10060, 0, provider="openai", model="gpt-5.6-terra", shared_breakpoint=True))
    assert other.cause is MissCause.FIRST_WRITE and other.legit is None


def test_explain_follows_the_same_rules_offline():
    """`prefixwise explain` on saved requests with their usage: a turn after one that left nothing,
    and a new conversation that missed the prefix the previous request cached."""
    from prefixwise.explain import Turn, explain_session
    small = lambda msgs: {"model": "claude-sonnet-4-6", "max_tokens": 9, "system": marked("You are a careful agent. " * 150),  # noqa: E731
                          "messages": msgs}
    context = [{"role": "user", "content": "context " * 1500}]
    out = explain_session([Turn(small(context), claude(fresh=2500)), Turn(small(context + [A1, GO]), claude(fresh=2600), gap_seconds=5)])
    assert out[1].cause == "below_min_prefix" and "left nothing in the cache" in out[1].evidence
    ticket = lambda i, system: {"model": OPUS, "max_tokens": 9, "system": system,  # noqa: E731
                                "messages": [{"role": "user", "content": f"Ticket {i}: out of toner"}]}
    turns = lambda system: [Turn(ticket(0, system), claude(write=10000, fresh=60)),  # noqa: E731
                            Turn(ticket(1, system), claude(write=10000, fresh=60), gap_seconds=10)]
    assert explain_session(turns(marked(BIG)))[1].cause == "routing_miss"
    unmarked_system = explain_session([Turn(ticket(0, marked(BIG)), claude(write=10000, fresh=60)),
                                       Turn({**ticket(1, BIG), "messages": last_marked([{"role": "user", "content": "Ticket 1"}])},
                                            claude(write=10000, fresh=60), gap_seconds=10)])
    assert (unmarked_system[1].cause, unmarked_system[1].legit) == ("first_write", None)
    # A marked part far above the minimum that wrote nothing: the markers were not honoured.
    large = lambda msgs: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": msgs}  # noqa: E731
    out = explain_session([Turn(large(context), claude(fresh=6000)), Turn(large(context + [A1, GO]), claude(fresh=6100), gap_seconds=5)])
    assert out[1].cause == "unknown" and "Its markers were not honoured" in out[1].evidence
    # A marker on the last block of a prompt that wrote nothing was ignored: not a prefix below the minimum.
    whole = lambda msgs: {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": last_marked(msgs)}  # noqa: E731
    out = explain_session([Turn(whole([U1]), claude(fresh=3000)), Turn(whole([U1, A1, GO]), claude(fresh=3100), gap_seconds=5)])
    assert out[1].cause == "routing_miss"


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
    assert entry.cache_age_seconds == pytest.approx(340)  # and the age is stored beside it
    if generation > 100:
        assert "340s since the previous turn was sent (its response ended 60s ago)" in entry.evidence
    else:
        assert "340s since the previous turn was sent (its response ended 335s ago)" in entry.evidence


def test_a_retry_is_still_told_by_idle_time():
    """A request repeated right after a slow answer is a retry: that rule reads the idle gap."""
    proxy = scripted([claude(write=3000), claude(write=3000)])
    body = {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": [U1]}
    proxy.handle("anthropic", body, S)
    shift(proxy, sent=200)   # it took 200 s to answer; the retry follows at once
    assert proxy.handle("anthropic", body, S)[1].cause == "retry_replay"


def test_an_age_is_never_printed_as_a_gap_when_the_idle_gap_is_another_number():
    """Asked for 1 hour, answered after 280 s, the next turn 60 s later reads nothing: a routing
    miss whose evidence said "gap 340s" on a row whose gap_seconds is 60."""
    def second(sent, done, read=0):
        proxy = scripted([claude(write=3000), claude(read=read, write=3100 - read)])
        proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(ttl="1h"), "messages": last_marked([U1])}, S)
        shift(proxy, sent=sent, done=done)
        return proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(ttl="1h"),
                                          "messages": last_marked([U1, A1, GO])}, S)[1]
    entry = second(340, 60)
    assert entry.cause == "routing_miss" and entry.gap_seconds == pytest.approx(60, abs=1)
    assert "340s since the previous turn was sent (its response ended 60s ago) within TTL 3600s" in entry.evidence
    assert "gap 340s" not in entry.evidence
    assert "gap 60s within TTL 3600s" in second(60, 60).evidence  # the two agree: one number, called the gap
    part = second(700, 400, read=1000)  # a part read after more than 5 minutes: expiry or routing, not told apart
    assert part.cause == "unknown" and "gap 700s" not in part.evidence
    assert "700s since the previous turn was sent (its response ended 400s ago)" in part.evidence


class SeenLifetime:
    """An observed lifetime (prefixwise.lifetime.Observed): the cache has held an unchanged prefix
    across idle gaps of up to `alive_up_to` seconds."""
    alive_up_to, cold_from = 1000.0, None

    def describe(self):
        return "still read after 1000s idle (5 turns)"


def test_a_stated_lifetime_is_judged_by_age_and_an_observed_one_by_the_idle_gap():
    body = {"model": "gpt-5.6-terra", "messages": [{"role": "system", "content": SYS}, U1]}
    grown = {**body, "messages": body["messages"] + [A1, GO]}
    prev = snap(blocks_of(body, "openai"), 3000, 0, provider="openai", model="gpt-5.6-terra", ttl_seconds=None)
    turn = lambda age, gap, **kw: classify(prev, snap(blocks_of(grown, "openai"), 3100, 0, provider="openai",  # noqa: E731
                                                      model="gpt-5.6-terra", has_breakpoint=False, ttl_seconds=None,
                                                      gap_seconds=gap, cache_age_seconds=age, **kw))
    # No lifetime stated or observed: within 5 minutes of the entry being written it cannot have
    # expired. 280 s of generation and 60 s idle is 340 s: no longer certain.
    assert turn(200, 60).cause is MissCause.ROUTING_MISS
    late = turn(340, 60)
    assert late.cause is MissCause.UNKNOWN and "expiry or routing" in late.evidence
    assert "340s since the previous turn was sent (its response ended 60s ago), with unknown TTL" in late.evidence
    # An observed lifetime was learned from idle gaps, so it is compared with the idle gap.
    seen = turn(1100, 900, observed_lifetime=SeenLifetime())
    assert seen.cause is MissCause.ROUTING_MISS and "gap 900s within the observed lifetime" in seen.evidence
    assert turn(1300, 1100, observed_lifetime=SeenLifetime()).cause is MissCause.UNKNOWN


def test_the_evidence_names_the_lifetime_that_decided():
    """Observed lifetime 1,000 s. Idle 290 s but 500 s since the previous turn was sent: past the 5
    minutes any provider keeps, so the observed lifetime is what says "too soon to have expired",
    and the evidence said "within shortest provider TTL 300s". Within 5 minutes by age, that TTL
    does decide."""
    body = {"model": "gpt-5.6-terra", "messages": [{"role": "system", "content": SYS}, U1]}
    grown = blocks_of({**body, "messages": body["messages"] + [A1, GO]}, "openai")
    prev = snap(blocks_of(body, "openai"), 3000, 0, provider="openai", model="gpt-5.6-terra", ttl_seconds=None)
    turn = lambda age, gap: classify(prev, snap(grown, 3100, 0, provider="openai", model="gpt-5.6-terra",  # noqa: E731
                                                has_breakpoint=False, ttl_seconds=None, gap_seconds=gap,
                                                cache_age_seconds=age, observed_lifetime=SeenLifetime()))
    by_observation = turn(500, 290)
    assert by_observation.cause is MissCause.ROUTING_MISS
    assert "gap 290s within the observed lifetime on this model" in by_observation.evidence
    assert "shortest provider TTL" not in by_observation.evidence
    by_default = turn(200, 60)
    assert "within shortest provider TTL 300s" in by_default.evidence and "observed lifetime" not in by_default.evidence


def test_an_unchanged_prefix_with_no_timing_is_still_marked_as_one_that_came_back_cold():
    """(The flag lifetimes are learned from; a reused variable had turned it into None.)"""
    body = {"model": "gpt-5.6-terra", "messages": [{"role": "system", "content": SYS}, U1]}
    prev = snap(blocks_of(body, "openai"), 3000, 0, provider="openai", model="gpt-5.6-terra", ttl_seconds=None)
    curr = snap(blocks_of({**body, "messages": body["messages"] + [A1, GO]}, "openai"), 3100, 0, provider="openai",
                model="gpt-5.6-terra", has_breakpoint=False, ttl_seconds=None, gap_seconds=None)
    verdict = classify(prev, curr)
    assert verdict.cause is MissCause.UNKNOWN and verdict.cold_unchanged is True


def test_a_model_switch_has_a_gap_and_no_cache_age():
    """Another model's turn left no entry of this model to age."""
    proxy = scripted([claude(write=3000), claude(write=3000)])
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": [U1]}, S)
    idle(proxy, 30)
    entry = proxy.handle("anthropic", {"model": "claude-haiku-5-5", "max_tokens": 9, "system": marked(), "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "model_switch" and entry.gap_seconds == pytest.approx(30, abs=1) and entry.cache_age_seconds is None


# -- 4, 5. no change is claimed against a turn that is not this turn's predecessor ----------------

def swarm(agents, turns=4, session="swarm"):
    """Same-type subagents in one named session, taking turns; each reads all its own previous turn left."""
    usages, order, size = [], [], {i: 3000 + 137 * i for i in range(agents)}
    for t in range(turns):
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
        entry = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(),
                                           "messages": last_marked(convs[i])}, {"x-prefixwise-session-id": session})[1]
        if t:
            later.append(entry)
    return proxy, later


def test_the_lane_remembers_sixty_four_conversations():
    assert LANE_THREADS == 64


@pytest.mark.parametrize("agents", [8, 9, 12, LANE_THREADS])
def test_interleaved_conversations_are_each_compared_with_their_own_previous_turn(agents):
    """Each conversation reads everything its own previous turn left. The lane remembered 8
    conversations: with nine interleaved, 26 of 27 such hits were `history_edit`, compared with
    another conversation's turn. It remembers 64 now."""
    proxy, later = swarm(agents)
    assert {e.cause for e in later} == {"none"} and sum(e.avoidable_tokens for e in later) == 0
    assert all("the cache served all of the reusable prefix" in e.evidence for e in later)


def test_a_lane_of_long_conversations_is_held_to_a_block_budget_but_never_below_eight(monkeypatch):
    """64 conversations of a thousand messages would be about 50 MB in one lane. The remembered turns
    share a budget of blocks; past it the least recently used go, down to the 8 a lane always kept."""
    from prefixwise import collector

    def lane_after(n, budget):
        monkeypatch.setattr(collector, "LANE_BLOCKS", budget)
        proxy = scripted([gpt(2300, 0)] * n)
        for i in range(n):
            proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [
                AGENT, {"role": "user", "content": f"document {i} " * 100},
                {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]}, S)
        (lane,) = [v for k, v in proxy.lane_threads.items() if k[0] == "s"]
        return lane

    assert collector.LANE_BLOCKS == 16000 and collector.LANE_THREADS_FLOOR == 8
    per_turn = len(next(iter(lane_after(1, 16000).threads.values()))[0].blocks)
    assert len(lane_after(20, 16000).threads) == 20                    # far inside the budget: all remembered
    assert len(lane_after(20, per_turn * 12).threads) == 12            # the budget holds twelve such turns
    assert len(lane_after(20, per_turn * 3).threads) == 8              # never fewer than eight, whatever their size
    assert len(lane_after(20, 0).threads) == 8
    kept = lane_after(20, per_turn * 12)
    assert kept.latest in kept.threads and len(kept.threads) == 12     # the newest are the ones kept


def test_twelve_interleaved_conversations_that_share_an_instruction_are_found_by_their_openings():
    """Unique documents and answers, the same instruction text after each answer: each conversation
    is found by its opening message, so the shared message plays no part. (With 8 remembered, 33 of
    these 36 full hits were `history_edit`.)"""
    def conv(i, turns):
        msgs = [{"role": "user", "content": f"document {i} " * 100}]
        for k in range(1, turns):
            msgs += [{"role": "assistant", "content": f"answer {i}.{k} " * 40}, {"role": "user", "content": "Continue with the next section."}]
        return msgs
    for headers in ({}, S):
        usages = [gpt(2300 + 300 * (n - 1), 0 if (n, i) == (1, 0) else 1920 if n == 1 else (2000 + 300 * (n - 1)) // 128 * 128)
                  for n in range(1, 5) for i in range(12)]
        proxy = scripted(usages)
        rows = [proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + conv(i, n)}, headers)[1]
                for n in range(1, 5) for i in range(12)]
        assert collections.Counter(r.cause for r in rows) == {"first_write": 1, "none": 47}
        assert sum(r.avoidable_tokens for r in rows) == 0


def forgotten(usage, session=S, stamp=False, second_turn=True):
    """Conversation 0 comes back after 64 others: the lane has dropped it. The lane's latest turn is
    another conversation's second turn (its first, with `second_turn` off). `stamp`: a per-request
    timestamp in the system prompt."""
    proxy = scripted([claude(write=3000, fresh=0)] * (LANE_THREADS + 1) + [claude(read=0 if stamp else 3000, write=3300 if stamp else 300, fresh=0)] * second_turn + [usage])
    sent = iter(range(3600))
    now = lambda: divmod(next(sent), 60)  # noqa: E731
    turn = lambda i, more=(): {"model": OPUS, "max_tokens": 9,  # noqa: E731
                               "system": marked(SYS + (" Now: 2026-11-26T09:%02d:%02dZ" % now() if stamp else "")),
                               "messages": last_marked([{"role": "user", "content": f"topic {i}"}, *more])}
    for i in range(LANE_THREADS + 1):
        proxy.handle("anthropic", turn(i), session)
    if second_turn:
        proxy.handle("anthropic", turn(LANE_THREADS, [{"role": "assistant", "content": "notes"}, {"role": "user", "content": "more"}]), session)
    return proxy, proxy.handle("anthropic", turn(0, [A1, GO]), session)[1]


@pytest.mark.parametrize("session", [S, {}])
def test_beyond_sixty_four_conversations_the_oldest_is_compared_with_the_latest_turn(session):
    """The limit, as it behaves: a conversation is known by its opening message, and the 65th
    remembered one drops the least recently used. That conversation's next turn finds no turn of
    its own and is compared with the lane's latest turn, another conversation's: `history_edit`,
    as on main, with or without a session id."""
    _, missed = forgotten(claude(write=3400, fresh=0), session)
    assert (missed.cause, missed.legit, missed.avoidable_tokens) == ("history_edit", False, 3300)
    _, hit = forgotten(claude(read=3000, write=400, fresh=0), session)   # it read all its own previous turn left
    assert (hit.cause, hit.legit, hit.avoidable_tokens) == ("history_edit", False, 300)


def test_beyond_the_limit_after_a_single_message_request_it_is_a_new_conversation():
    """The same, when the lane's latest turn is a single message: the rule for a request that
    follows a one-shot applies (a new conversation on the same tools and system prompt). It read
    nothing of the system prompt the latest request cached at the marker both carry: a miss of
    that shared prefix, judged by the routing rule."""
    _, entry = forgotten(claude(write=3400, fresh=0), second_turn=False)
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("routing_miss", None, 0)
    assert "a new conversation on the tools and system prompt the previous request cached" in entry.evidence
    _, hit = forgotten(claude(read=3000, write=400, fresh=0), second_turn=False)
    assert hit.cause == "none"


def test_a_conversation_whose_previous_turn_is_not_at_hand_still_gets_the_lanes_causes():
    """Tools and the system prompt are the lane's: a change in them stands against any of its
    turns, so it is named exactly as on main, with the same avoidable tokens."""
    _, entry = forgotten(claude(write=3400, fresh=0), stamp=True)
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("volatile_system", False, 3300)


def test_twelve_users_with_a_timestamp_in_the_system_prompt_are_volatile_system():
    """12 users interleaved in one session with no id, a per-request timestamp in the system prompt,
    nothing ever read: every turn after the first is `volatile_system`, each compared with its own
    conversation's previous turn."""
    users, turns = 12, 4
    proxy = scripted([claude(write=2500 + 300 * t, fresh=0) for t in range(turns) for _ in range(users)])
    got, n = [], 0
    for t in range(1, turns + 1):
        for u in range(users):
            n += 1
            system = marked(SYS + f" Now: 2026-11-26T09:{n // 60:02d}:{n % 60:02d}Z")
            got.append(proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system,
                                                  "messages": chat(f"user{u}", t)}, {})[1])
    assert collections.Counter(e.cause for e in got) == {"first_write": 1, "volatile_system": 47}
    assert all(e.legit is False for e in got[1:])
    assert sum(e.avoidable_tokens for e in got) == 128_300   # the shorter prompt of each pair, unread


def test_a_reordered_tool_list_is_named_whatever_conversation_it_is_compared_with():
    """A first turn is compared with another user's turn, a later one with its own previous turn;
    the tools are the lane's either way."""
    tool = lambda name: {"name": name, "description": f"{name} things", "input_schema": {"type": "object", "properties": {}}}  # noqa: E731
    tools = [tool("search"), tool("read"), tool("write")]
    users, turns = 12, 3
    proxy = scripted([claude(write=2500 + 300 * t, fresh=0) for t in range(turns) for _ in range(users)])
    got, n = [], 0
    for t in range(1, turns + 1):
        for u in range(users):
            n += 1
            order = tools if (t + u) % 2 else list(reversed(tools))   # each user flips the order every turn
            got.append(proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "tools": order, "system": marked(),
                                                  "messages": chat(f"user{u}", t)}, {})[1])
    assert collections.Counter(e.cause for e in got) == {"first_write": 1, "tool_reserialize": 35}
    assert sum(e.avoidable_tokens for e in got) == 11 * 2500 + 12 * (2500 + 2800)


def test_a_compaction_sent_with_its_event_is_a_compaction_however_many_conversations_the_lane_holds():
    proxy = scripted([claude(write=3000, fresh=0)] * 8 + [claude(read=3000, write=600, fresh=0),
                                                         claude(read=2000, write=500, fresh=0)])
    turn = lambda msgs, **kw: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": last_marked(msgs), **kw}  # noqa: E731
    for i in range(8):
        proxy.handle("anthropic", turn([{"role": "user", "content": f"topic {i}"}]), S)
    long = [{"role": "user", "content": "topic 7"}, {"role": "assistant", "content": "a long answer " * 200},
            {"role": "user", "content": "and then?"}]
    proxy.handle("anthropic", turn(long), S)
    summary = [{"role": "user", "content": "Summary of the conversation so far: topic 7, a long answer."},
               {"role": "assistant", "content": "understood"}, {"role": "user", "content": "go on"}]
    entry = proxy.handle("anthropic", turn(summary, prefixwise_compaction_event=True), S)[1]
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("compaction_rewrite", True, 0)


def rewritten_history(headers):
    """A conversation of two turns, then a request whose every message is another: a full rewrite."""
    proxy = scripted([claude(write=3000, fresh=0), claude(read=3000, write=400, fresh=0), claude(read=2000, write=1500, fresh=0)])
    turn = lambda msgs: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": last_marked(msgs)}  # noqa: E731
    proxy.handle("anthropic", turn([U1]), headers)
    proxy.handle("anthropic", turn([U1, A1, GO]), headers)
    other = [{"role": "user", "content": "something else entirely"}, {"role": "assistant", "content": "sure"},
             {"role": "user", "content": "carry on"}]
    return proxy, proxy.handle("anthropic", turn(other), headers)[1]


@pytest.mark.parametrize("headers", [S, {}])
def test_a_request_of_several_messages_with_another_opening_is_a_history_edit(headers):
    """An unannounced rewrite of every message is reported, with the tokens it cost, in a named
    session and in one with no id alike, as on main. (A rule that took such a request for "another
    conversation" where conversations mix hid trimmed histories, and is gone.)"""
    proxy, entry = rewritten_history(headers)
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("history_edit", False, 3400 - 2000)
    assert "judged by its own read" not in entry.evidence and entry.divergence_role == "messages"


def test_an_edited_opening_is_an_edit_even_where_conversations_mix():
    """The first message differs and every message after it is the compared turn's: one
    conversation whose opening was edited."""
    proxy = scripted([claude(write=3000, fresh=0), claude(read=3000, write=400, fresh=0), claude(read=2000, write=1800, fresh=0)])
    turn = lambda msgs: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": last_marked(msgs)}  # noqa: E731
    proxy.handle("anthropic", turn([U1]), {})
    proxy.handle("anthropic", turn([U1, A1, GO]), {})
    edited = [{"role": "user", "content": "start (edited)"}, A1, GO, {"role": "assistant", "content": "done"},
              {"role": "user", "content": "next"}]
    entry = proxy.handle("anthropic", turn(edited), {})[1]
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("history_edit", False, 3400 - 2000)


def test_in_a_named_session_a_single_message_unlike_the_latest_turn_is_still_an_edit():
    """A single message with another opening is a new conversation only in a session that names no
    conversation. In a named one it could as well be the conversation compacted without its event,
    and stays reported, as on main."""
    proxy = scripted([claude(write=3000, fresh=0), claude(read=3000, write=400, fresh=0), claude(read=2000, write=1000, fresh=0)])
    turn = lambda msgs: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": last_marked(msgs)}  # noqa: E731
    proxy.handle("anthropic", turn([U1]), S)
    proxy.handle("anthropic", turn([U1, A1, GO]), S)
    entry = proxy.handle("anthropic", turn([{"role": "user", "content": "one new message"}]), S)[1]
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("history_edit", False, 1000)


def test_no_detail_and_no_lifetime_evidence_on_a_row_judged_by_its_own_read():
    """An earlier point of its conversation, on OpenAI: it was not compared with the turn at hand,
    so that turn's prompt_cache_key and timing say nothing about it. No detail ("prompt_cache_key
    differs from the previous turn's"), and no evidence of how long the cache lasts, hit or not."""
    system = {"role": "system", "content": "You are a support agent. " * 300}
    conv = lambda n: [system, {"role": "user", "content": "a question"}] + [  # noqa: E731
        m for i in range(1, n) for m in ({"role": "assistant", "content": f"answer {i}"}, {"role": "user", "content": f"follow-up {i}"})]
    for read, cause in ((0, "unknown"), (1920, "none")):
        proxy = scripted([gpt(2000, 0), gpt(2100, 1920), gpt(2000, read)])
        send = lambda n, key: proxy.handle("openai", {"model": "gpt-5.6-terra", "prompt_cache_key": key,  # noqa: E731
                                                      "messages": conv(n)}, S)[1]
        send(1, "key-a"), send(2, "key-a")
        idle(proxy, 900)
        entry = send(1, "key-b")   # the first turn again, under another cache key
        assert (entry.cause, entry.detail, entry.legit) == (cause, None, None)
        assert "judged by its own read" in entry.evidence and "prompt_cache_key" not in entry.evidence
        assert lifetime_rows(proxy) == [None, 1, None]   # the second turn's hit counts; the rewind, hit or not, does not


def test_twelve_one_shots_with_no_session_id_are_never_unknown():
    """The headline case: independent one-shot requests with no session id. Once the lane had seen
    8 openings every later one was `unknown`."""
    proxy = scripted([gpt(2500, 0)] * 12)
    got = [proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [
        {"role": "system", "content": "Classify the document as spam or not spam."},
        {"role": "user", "content": f"Document {i}: " + f"lorem {i} ipsum " * 800}]}, {})[1] for i in range(12)]
    assert [e.cause for e in got] == ["first_write"] * 12
    doc = lambda i: last_marked([{"role": "user", "content": f"Document {i}: " + f"alpha {i} " * 900}])  # noqa: E731
    proxy = scripted([claude(write=5000, fresh=0)] + [claude(read=2000, write=3000, fresh=0)] * 5
                     + [claude(read=2000, fresh=600)] * 6)
    got = [proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(),
                                      "messages": doc(i) if i < 6 else [{"role": "user", "content": f"short question {i}"}]}, {})[1]
           for i in range(12)]
    assert [e.cause for e in got] == ["first_write"] * 6 + ["none"] * 6


def test_a_gateway_log_of_forty_users_has_no_unknown_first_turns():
    """40 users through one gateway with no session id, never more than 3 at once, three turns
    each, every read a hit: 32 of the 40 first turns were `unknown` ("no longer remembered")."""
    edge, sink, events = Edge(), collector(), []
    for u in range(40):
        start, held = T0 + 20 * u, 2000 if u else 0   # a user arrives every 20 s and stays about 50 s
        for n in range(1, 4):
            prompt = 2000 + 300 * n + u
            sent = start + 18 * (n - 1)
            events.append((sent + 6, sent, u, n, claude(read=held, write=prompt - held, fresh=0)))
            held = prompt
    rows = []
    for done, sent, u, n, usage in sorted(events, key=lambda event: event[0]):
        body = {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat(f"user{u}", n)}
        rows.append((n, sink.ingest(edge.record("anthropic", body, {}, sent, done, usage))[1]))
    first = [e for n, e in rows if n == 1]   # each later user's first turn reads the shared system prompt
    assert collections.Counter(e.cause for e in first) == {"first_write": 1, "none": 39}
    assert {e.cause for n, e in rows if n > 1} == {"none"} and sum(e.avoidable_tokens for _, e in rows) == 0


def late_delivery_records():
    edge = Edge()

    def record(n, usage):  # turn n: sent at T0 + 10n, answered 5 s later
        body = {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("u", n)}
        return edge.record("anthropic", body, S, T0 + 10 * n, T0 + 10 * n + 5, usage)

    return record, {1: record(1, claude(write=3000, fresh=0)), 2: record(2, claude(read=3000, write=500, fresh=0)),
                    3: record(3, claude(read=3500, write=500, fresh=0)), 4: record(4, claude(read=4000, write=500, fresh=0))}


def test_a_record_delivered_late_is_judged_by_its_own_read():
    """Turn 2's record reaches the collector after turn 3's. It read all 3,000 tokens turn 1 left; it
    was compared with turn 3 and called `history_edit`. Nor may it become the conversation's latest."""
    record, recs = late_delivery_records()
    sink = collector()
    got = {n: sink.ingest(recs[n])[1] for n in (1, 3, 2, 4)}
    assert got[2].cause == "none" and got[2].avoidable_tokens == 0
    assert "its record arrived late, after a later turn's; it was sent 10.0 s before that turn" in got[2].evidence
    assert got[2].detail is None and got[2].divergence_role is None
    assert got[4].cause == "none" and "4000 of 4000" in got[4].evidence   # still compared with turn 3
    missed = collector()
    late = record(2, claude(read=0, write=3500, fresh=0))
    assert [missed.ingest(r)[1].cause for r in (recs[1], recs[3], late)][2] == "unknown"


def test_a_rewind_is_an_earlier_point_too_and_becomes_the_latest_turn():
    """The user goes back to turn 2 and sends it again, after turn 3: its messages open turn 3's,
    and nothing in it changed. No change is claimed (it was `history_edit`), the evidence does not
    say it arrived late, and the conversation goes on from it."""
    sink, edge = collector(), Edge()
    turn = lambda n, sent, usage: sink.ingest(edge.record(  # noqa: E731
        "anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("u", n)}, S, sent, sent + 5, usage))[1]
    turn(1, T0, claude(write=3000, fresh=0))
    turn(2, T0 + 10, claude(read=3000, write=500, fresh=0))
    turn(3, T0 + 20, claude(read=3500, write=500, fresh=0))
    again = turn(2, T0 + 60, claude(read=3500, fresh=0))
    assert (again.cause, again.legit, again.avoidable_tokens) == ("none", None, 0)
    assert "arrived late" not in again.evidence and "sent 40.0 s after that turn" in again.evidence
    assert "rewound to an earlier turn, or an earlier turn was sent again" in again.evidence
    cold = collector()
    for n, sent in ((1, T0), (2, T0 + 10), (3, T0 + 20)):
        cold.ingest(edge.record("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("u", n)},
                                S, sent, sent + 5, claude(write=3000 + 500 * n, fresh=0)))
    missed = cold.ingest(edge.record("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("u", 2)},
                                     S, T0 + 60, T0 + 65, claude(write=3500, fresh=0)))[1]
    assert (missed.cause, missed.avoidable_tokens) == ("unknown", 0)   # under half of its own prompt: not known
    # The rewound turn is now the conversation's latest: a new third turn is compared with it.
    msgs = chat("u", 2, mark=False) + [{"role": "assistant", "content": "another answer"}, {"role": "user", "content": "another way"}]
    nxt = sink.ingest(edge.record("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": last_marked(msgs)},
                                  S, T0 + 70, T0 + 75, claude(read=3500, write=300, fresh=0)))[1]
    assert nxt.cause == "none" and "3500 of 3500" in nxt.evidence


def rewound(usages, tools_for=None):
    """Three turns, then turn 2 again, with a per-request timestamp in the system prompt."""
    proxy, n, rows = scripted(usages), [0], []
    for turns in (1, 2, 3, 2):
        n[0] += 1
        body = {"model": OPUS, "max_tokens": 9, "system": marked(SYS + f" Now: 2026-11-26T09:00:{n[0]:02d}Z"),
                "messages": chat("u", turns)}
        rows.append(proxy.handle("anthropic", body, S)[1])
    return rows


def test_an_earlier_point_is_still_compared_for_its_tools_and_system_prompt():
    """A rewind to turn 2 whose system prompt carries a new timestamp: the timestamp is why nothing
    was read, whatever the conversation did. It was `unknown` with 0 avoidable; main says
    `volatile_system`, 3,500."""
    rows = rewound([claude(write=3000, fresh=0), claude(write=3500, fresh=0), claude(write=4000, fresh=0), claude(write=3500, fresh=0)])
    assert (rows[3].cause, rows[3].legit, rows[3].avoidable_tokens) == ("volatile_system", False, 3500)
    assert rows[3].divergence_role == "system" and "judged by its own read" not in rows[3].evidence
    # A change the cache absorbed is weighed as always.
    hit = rewound([claude(write=3000, fresh=0), claude(write=3500, fresh=0), claude(write=4000, fresh=0), claude(read=3500, fresh=0)])
    assert hit[3].cause == "none" and hit[3].secondary_causes == "volatile_system"


def test_a_late_record_with_a_changed_system_prompt_is_judged_as_any_request_in_flight():
    """Delivered late, sent before the turn it is compared with had finished, and it read nothing:
    with the system prompt changed it goes through the same checks as any other request, and the
    one for a request in flight leaves it unjudged, as on main."""
    edge, sink, n = Edge(), collector(), [0]

    def record(turns, sent, usage):
        n[0] += 1
        body = {"model": OPUS, "max_tokens": 9, "system": marked(SYS + f" Now: 2026-11-26T09:00:{n[0]:02d}Z"),
                "messages": chat("u", turns)}
        return edge.record("anthropic", body, S, T0 + sent, T0 + sent + 5, usage)

    one, two, three = (record(1, 10, claude(write=3000, fresh=0)), record(2, 20, claude(write=3500, fresh=0)),
                       record(3, 30, claude(write=4000, fresh=0)))
    sink.ingest(one), sink.ingest(three)
    late = sink.ingest(two)[1]
    assert late.cause == "unknown" and "had finished" in late.evidence and late.avoidable_tokens == 0


def test_an_earlier_point_that_read_all_the_conversation_caches_is_a_hit():
    """Only the system prompt is marked, so 2,000 tokens is all this conversation ever has in the
    cache. A rewind that reads them read everything there was, though it is under half of its
    prompt: `none`, as on main, not `unknown`. Reading less than that is not known."""
    def rewind(read):
        proxy = scripted([claude(write=2000, fresh=3000), claude(read=2000, fresh=3500), claude(read=read, fresh=5000 - read)])
        turn = lambda n: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("u", n, mark=False)}  # noqa: E731
        proxy.handle("anthropic", turn(2), S)
        proxy.handle("anthropic", turn(3), S)
        return proxy.handle("anthropic", turn(2), S)[1]
    entry = rewind(2000)
    assert (entry.cause, entry.avoidable_tokens) == ("none", 0) and "an earlier point of this conversation" in entry.evidence
    assert "under half of its prompt, but all that the compared turn left in the cache" in entry.evidence
    assert rewind(1000).cause == "unknown"


def test_an_earlier_point_after_a_turn_too_short_to_cache_is_below_the_minimum():
    """The rule that needs no comparison still applies: the compared turn was below the model's
    minimum, so nothing was cached for any turn of this conversation to read."""
    proxy = scripted([claude(fresh=700), claude(fresh=800), claude(fresh=700)])
    turn = lambda msgs: {"model": "claude-sonnet-4-6", "max_tokens": 9, "system": marked("You are brief."), "messages": msgs}  # noqa: E731
    proxy.handle("anthropic", turn([U1]), S)
    proxy.handle("anthropic", turn([U1, A1, GO]), S)
    entry = proxy.handle("anthropic", turn([U1]), S)[1]
    assert entry.cause == "below_min_prefix" and "previous request was 800 tokens" in entry.evidence


def test_an_earlier_point_within_the_clock_allowance_is_not_called_late():
    """Late means sent before the compared turn was, by more than the clocks of two edges may
    differ (1 s). Half a second before is not that: it is not dropped from the lane."""
    record, recs = late_delivery_records()
    edge, sink = Edge(), collector()
    body = lambda n: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("u", n)}  # noqa: E731
    sink.ingest(recs[1]), sink.ingest(recs[3])                     # turn 3 was sent at T0 + 30
    close = sink.ingest(edge.record("anthropic", body(2), S, T0 + 29.5, T0 + 36, claude(read=3000, write=500, fresh=0)))[1]
    assert close.cause == "none" and "arrived late" not in close.evidence and "within the clock allowance" in close.evidence
    lane = sink.lane_threads[("s", OPUS)]
    assert lane.threads[lane.latest][0].prompt_tokens == 3500      # it is the conversation's latest turn now
    far = collector()
    far.ingest(recs[1]), far.ingest(recs[3]), far.ingest(recs[2])  # sent 10 s before turn 3: late
    lane = far.lane_threads[("s", OPUS)]
    assert lane.threads[lane.latest][0].prompt_tokens == 4000      # turn 3 stays the latest


def test_a_row_judged_by_its_own_read_is_not_scored_against_the_providers_reason():
    """The provider's reason compares the pair the provider chose; a verdict that compared nothing
    neither agrees nor disagrees with it."""
    record, recs = late_delivery_records()
    late = record(2, claude(read=0, write=3500, fresh=0))
    late.native_reason = recs[3].native_reason = "messages_changed"
    sink = collector()
    sink.ingest(recs[1]), sink.ingest(recs[3])
    scored = (sink.scorer.total_misses, sink.scorer.total_with_native)
    assert sink.ingest(late)[1].cause == "unknown"
    assert (sink.scorer.total_misses, sink.scorer.total_with_native) == scored


def test_a_stitched_turn_is_never_an_earlier_point():
    """A Responses turn that continues a stored response sends only its new input. When that input
    repeats the conversation's opening words, it is still the next turn, not a rewind to the first."""
    edge, sink = Edge(), collector()
    base = {"model": "gpt-5.6", "instructions": "You are helpful. " * 400}
    rec = lambda body, sent, usage, rid: sink.ingest(edge.record(  # noqa: E731
        "openai", body, S, T0 + sent, T0 + sent + 5, usage, api="responses", response_id=rid))[1]
    go_on = [{"role": "user", "content": "continue"}]
    rec({**base, "input": go_on}, 0, responses_usage(2000, 0), "resp_1")
    rec({**base, "previous_response_id": "resp_1", "input": go_on}, 10, responses_usage(2050, 1920), "resp_2")
    entry = rec({**base, "previous_response_id": "resp_2", "input": go_on}, 20, responses_usage(2100, 1920), "resp_3")
    assert entry.cause == "none" and "earlier point" not in entry.evidence and "judged by its own read" not in entry.evidence


def test_a_truncated_history_with_a_compaction_event_is_still_a_compaction():
    proxy = scripted([claude(write=3000, fresh=0), claude(read=3000, write=500, fresh=0), claude(read=2000, write=1000, fresh=0)])
    turn = lambda n, **kw: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("u", n), **kw}  # noqa: E731
    proxy.handle("anthropic", turn(2), S)
    proxy.handle("anthropic", turn(3), S)
    entry = proxy.handle("anthropic", turn(2, prefixwise_compaction_event=True), S)[1]
    assert entry.cause == "compaction_rewrite" and entry.legit is True


def test_interleaved_conversations_delivered_in_completion_order_are_not_late():
    """X1 is sent at 0 and answered at 60; Y1 (10 to 12) and Y2 (20 to 25) finish first; X2 (70 to
    75) reads all X1 left. A clock-only rule called X1 "late" against Y2, another conversation's
    turn, and dropped it from the lane, so X2 was `history_edit` with 300 avoidable tokens."""
    for headers in (S, {}):
        edge, sink = Edge(), collector()
        body = lambda who, n: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat(who, n)}  # noqa: E731
        got = {}
        for name, who, n, sent, done, usage in (("Y1", "y", 1, 10, 12, claude(write=2300, fresh=0)),
                                                ("Y2", "y", 2, 20, 25, claude(read=2300, write=300, fresh=0)),
                                                ("X1", "x", 1, 0, 60, claude(read=2000, write=300, fresh=0)),
                                                ("X2", "x", 2, 70, 75, claude(read=2300, write=300, fresh=0))):
            got[name] = sink.ingest(edge.record("anthropic", body(who, n), headers, T0 + sent, T0 + done, usage))[1]
        assert "its record arrived late" not in got["X1"].evidence and "earlier point" not in got["X1"].evidence
        if headers:   # in a named session a first message unlike the latest turn's reads as an edit, as on main
            assert (got["X1"].cause, got["X1"].avoidable_tokens) == ("history_edit", 300)
        else:         # with no session id it is a new conversation, which read the shared system prompt
            assert (got["X1"].cause, got["X1"].avoidable_tokens) == ("none", 0)
        assert got["X2"].cause == "none" and "2300 of 2300" in got["X2"].evidence and got["X2"].avoidable_tokens == 0


def six_users(headers):
    """Six users through one gateway, four turns each, answers that overlap other users' turns,
    every turn a hit; records delivered in the order the answers completed."""
    edge, sink, events = Edge(), collector(), []
    for u in range(6):
        t, held = T0 + 3 * u, 2000 if u else 0   # a later user's first turn reads the shared system prompt
        for n in range(1, 5):
            took = 4 + (u * 5 + n * 3) % 11
            prompt = 2000 + 300 * n + 17 * u
            events.append((t + took, t, u, n, claude(read=held, write=prompt - held, fresh=0)))
            held = prompt
            t += took + 2 + (u + n) % 3
    out = []
    for done, sent, u, n, usage in sorted(events, key=lambda event: event[0]):
        body = {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat(f"user{u}", n)}
        out.append(sink.ingest(edge.record("anthropic", body, headers, sent, done, usage))[1])
    return out


def test_a_gateway_replay_of_six_overlapping_users_calls_no_record_late():
    """8 of these 24 rows said "arrived late", and the false `history_edit` rows went from main's
    to more. With no session id every row is a hit or the first write; in one named session the
    verdicts are main's (two first turns still read as edits of another user's conversation: the
    known limit of a session that names several conversations)."""
    rows = six_users({})
    assert not any("its record arrived late" in e.evidence for e in rows)
    assert collections.Counter(e.cause for e in rows) == {"first_write": 1, "none": 23}
    assert sum(e.avoidable_tokens for e in rows) == 0
    named = six_users(S)
    assert not any("its record arrived late" in e.evidence for e in named)
    assert collections.Counter(e.cause for e in named) == {"first_write": 1, "none": 21, "history_edit": 2}
    assert sum(e.avoidable_tokens for e in named) == 736   # as on main


def test_two_edges_with_skewed_clocks_do_not_make_every_other_turn_late():
    """Two edges whose clocks are 5 s apart, a turn every 3 s, alternating. Every turn through the
    slower clock looked "sent before the turn it is compared with was sent": it was called late and
    never became the conversation's latest."""
    edge, sink, held, rows = Edge(), collector(), 0, []
    for n in range(1, 9):
        sent = T0 + 3 * n - (5 if n % 2 == 0 else 0)   # even turns: the edge whose clock is 5 s behind
        prompt = 2000 + 300 * n
        body = {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("k", n)}
        rows.append(sink.ingest(edge.record("anthropic", body, S, sent, sent + 1, claude(read=held, write=prompt - held, fresh=0)))[1])
        held = prompt
    assert [e.cause for e in rows] == ["first_write"] + ["none"] * 7
    # each turn was compared with the one before it, whichever edge that came through
    assert [e.evidence for e in rows[1:]] == [f"the cache served all of the reusable prefix ({2000 + 300 * n} of {2000 + 300 * n} tokens)"
                                              for n in range(1, 8)]


def responses_usage(prompt, cached):
    return {"input_tokens": prompt, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": 3}


def test_a_late_responses_record_is_still_found_by_the_turn_that_continues_it():
    """A late record does not become its conversation's latest turn, but a later turn that names it
    as previous_response_id must still find it: that turn was `unknown`, "a response this proxy
    never saw"."""
    edge, sink = Edge(), collector()
    base = {"model": "gpt-5.6", "instructions": "You are helpful. " * 400}
    items = [{"role": "user", "content": "first question"}]
    longer = items + [{"role": "assistant", "content": "an answer"}, {"role": "user", "content": "second question"}]
    rec = lambda body, sent, usage, rid: edge.record("openai", body, S, T0 + sent, T0 + sent + 5, usage,  # noqa: E731
                                                     api="responses", response_id=rid)
    one = rec({**base, "input": items}, 0, responses_usage(2000, 0), "resp_1")
    two = rec({**base, "input": longer}, 10, responses_usage(2100, 1920), "resp_2")
    cont = rec({**base, "previous_response_id": "resp_1", "input": [{"role": "user", "content": "another way"}]},
               20, responses_usage(2050, 1920), "resp_3")
    got = [sink.ingest(r)[1] for r in (two, one, cont)]   # resp_1's record arrives after resp_2's
    assert "its record arrived late" in got[1].evidence
    assert got[2].cause == "none" and "never saw" not in got[2].evidence


# -- a conversation whose history is rewritten is still reported -----------------------------------

def trimmed(headers, system=AGENT, first=gpt(3300, 0), later=gpt(3300, 1408), turns=20, keep=6):
    """One agent that keeps only its last `keep` messages (a sliding window): every turn after the
    window fills opens with another message, and only the system prompt is read back."""
    hist, proxy, rows = [{"role": "user", "content": "task " + "detail " * 200}], scripted([first] + [later] * (turns - 1)), []
    for t in range(turns):
        if t:
            hist += [{"role": "assistant", "content": f"step {t} " * 120}, {"role": "user", "content": f"output {t} " * 120}]
        rows.append(proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [system] + hist[-keep:]}, headers)[1])
    return proxy, rows


@pytest.mark.parametrize("headers", [{}, S])
def test_a_sliding_window_is_a_history_edit_with_or_without_a_session_id(headers):
    """Dropping the oldest messages gives every turn another opening. Each is compared with the
    lane's latest turn, its own previous one, and is a `history_edit`, as on main. (Taken for
    "another conversation" these were 17 `unknown` rows and 0 avoidable tokens.)"""
    proxy, rows = trimmed(headers)
    assert collections.Counter(r.cause for r in rows) == {"first_write": 1, "routing_miss": 2, "history_edit": 17}
    assert sum(r.avoidable_tokens for r in rows) == 32_164   # as on main
    assert not any("judged by its own read" in r.evidence or "another conversation" in r.evidence for r in rows)


@pytest.mark.parametrize("headers", [{}, S])
def test_a_trimmed_history_behind_a_large_system_prompt_is_not_a_hit(headers):
    """With a 6,000-token system prompt read on every turn, more than half of each prompt comes from
    the cache: judged by its own read each trimmed turn was `none`, waste shown as a hit."""
    big = {"role": "system", "content": "You are a coding agent. " * 1500}
    _, rows = trimmed(headers, system=big, first=gpt(8000, 0), later=gpt(8000, 6016))
    assert collections.Counter(r.cause for r in rows) == {"first_write": 1, "none": 2, "history_edit": 17}
    assert sum(r.avoidable_tokens for r in rows) == 33_728   # as on main


@pytest.mark.parametrize("headers", [{}, S])
@pytest.mark.parametrize("event", [False, True])
def test_a_summary_followed_by_a_kept_tail_is_a_history_edit_or_a_compaction(headers, event):
    hist = [{"role": "user", "content": "task " + "detail " * 200}]
    for t in range(1, 6):
        hist += [{"role": "assistant", "content": f"step {t} " * 120}, {"role": "user", "content": f"output {t} " * 120}]
    proxy = scripted([gpt(3300, 0), gpt(3300, 1408)])
    proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + hist}, headers)
    compacted = ([{"role": "user", "content": "Summary of the work so far: steps 1 to 3 are done."}] + hist[-4:]
                 + [{"role": "assistant", "content": "step 6"}, {"role": "user", "content": "output 6"}])
    body = {"model": "gpt-5.6-terra", "messages": [AGENT] + compacted, **({"prefixwise_compaction_event": True} if event else {})}
    entry = proxy.handle("openai", body, headers)[1]
    if event:
        assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("compaction_rewrite", True, 0)
    else:
        assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("history_edit", False, 3300 - 1408)   # as on main


def beside(others, headers, turns=12):
    """Agent A keeps its last 6 messages; `others` agents beside it only append. One request each,
    in turn; every request reads the system prompt and nothing more. Returns A's rows and the rest."""
    def agent(name, keep):
        hist = [{"role": "user", "content": f"{name} task " + "detail " * 200}]
        for t in range(1, turns + 1):
            yield hist[-keep:]
            hist += [{"role": "assistant", "content": f"{name} step {t} " * 120}, {"role": "user", "content": f"{name} output {t} " * 120}]
    agents = [agent("A", 6)] + [agent(f"B{i}", 99) for i in range(others)]
    proxy, a, rest = scripted([gpt(3300, 1408)] * (turns * len(agents))), [], []
    for _ in range(turns):
        for i, messages in enumerate(agents):
            entry = proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + next(messages)}, headers)[1]
            (rest if i else a).append(entry)
    return a, rest


@pytest.mark.parametrize("headers", [{}, S])
@pytest.mark.parametrize("others", [0, 1, 8])
def test_a_trimming_agent_beside_other_conversations_is_reported_on_every_trimmed_turn(others, headers):
    """A trimmed turn has another opening, so it is not found by its opening and is compared with
    the lane's latest turn, which beside other conversations is someone else's. It is a
    `history_edit` all the same, each with the 1,892 tokens main counts: 9 trimmed turns of 12.
    (Taken for "another conversation", they were `unknown` with 0 avoidable as soon as one other
    conversation shared the lane.) The appending agents are compared with their own turns."""
    a, rest = beside(others, headers)
    assert [r.cause for r in a] == ["first_write", "routing_miss", "routing_miss"] + ["history_edit"] * 9
    assert sum(r.avoidable_tokens for r in a) == 17_028 and {r.avoidable_tokens for r in a[3:]} == {1892}
    assert {r.cause for r in rest} <= {"first_write", "routing_miss"} and sum(r.avoidable_tokens for r in rest) == 0


@pytest.mark.parametrize("headers", [{}, S])
def test_conversations_that_share_messages_and_one_first_seen_late_are_as_on_main(headers):
    """Four conversations, each `[document, "Understood.", the same instruction, ...]`, all full
    hits; the fourth is first seen at its second turn. A rule that filed a rewrite in the compared
    turn's place let the late one take another's thread: 3 `history_edit` rows where main has 1.
    Filing is main's again: the rows are main's (the first turns of the second and third
    conversation read as edits of the one before, the known limit of a lane without ids for
    conversations that open with several messages), and no later turn is blamed."""
    def conv(i, n):
        msgs = [{"role": "user", "content": f"document {i} " * 100}, {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "Now summarise the document in three sentences."}]
        for k in range(1, n):
            msgs += [{"role": "assistant", "content": f"summary {i}.{k} " * 40}, {"role": "user", "content": "Shorter, please."}]
        return msgs
    order = [(0, 1), (1, 1), (2, 1), (3, 2), (0, 2), (1, 2), (2, 2), (3, 3), (0, 3), (1, 3), (2, 3), (3, 4)]
    size = lambda i, n: 2300 + 250 * (n - 1) + 7 * i  # noqa: E731
    usages, seen = [], set()
    for i, n in order:
        read = size(i, n - 1) if (i, n - 1) in seen or (i, n) == (3, 2) else 1920 if seen else 0
        usages.append(gpt(size(i, n), read // 128 * 128))
        seen.add((i, n))
    proxy = scripted(usages)
    rows = [proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + conv(i, n)}, headers)[1] for i, n in order]
    assert [r.cause for r in rows] == ["first_write", "history_edit", "history_edit"] + ["none"] * 9   # as on main
    assert sum(r.avoidable_tokens for r in rows) == 767


def test_a_compaction_event_is_a_compaction_where_a_new_conversation_would_be_read():
    """After a single-message request, or as a single message with no session id, another opening
    is a new conversation; with a compaction event it is the compaction it says it is."""
    for headers, first, second in (
            (S, [{"role": "user", "content": "one question"}],
             [{"role": "user", "content": "Summary of the question so far."}, {"role": "assistant", "content": "noted"}, {"role": "user", "content": "go on"}]),
            ({}, [{"role": "user", "content": "a task"}, {"role": "assistant", "content": "working"}, {"role": "user", "content": "and?"}],
             [{"role": "user", "content": "Summary of the task so far, in one message."}])):
        for event, cause in ((True, "compaction_rewrite"), (False, "first_write")):
            proxy = scripted([gpt(3300, 0), gpt(3300, 0)])
            proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + first}, headers)
            entry = proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + second,
                                            **({"prefixwise_compaction_event": True} if event else {})}, headers)[1]
            assert entry.cause == cause


def test_repeated_compactions_with_their_event_are_each_a_compaction():
    proxy = scripted([gpt(3300, 0)] + [gpt(3300, 1408)] * 10)
    for n in range(10):
        msgs = [{"role": "user", "content": f"Summary number {n} of the work so far."}, {"role": "assistant", "content": f"noted {n}"},
                {"role": "user", "content": f"continue from {n}"}]
        entry = proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + msgs,
                                        **({"prefixwise_compaction_event": True} if n else {})}, S)[1]
        assert entry.cause == ("compaction_rewrite" if n else "first_write")


def test_a_compaction_event_in_a_lane_of_two_conversations_takes_nobodys_place():
    """Agent a compacts (a summary alone, with its event) while the lane's latest turn is agent
    b's. It is a compaction whatever it was compared with; b's turn stays in the lane, and b's
    next turn is compared with it."""
    proxy = scripted([gpt(3000, 0), gpt(3000, 1408), gpt(3300, 2944), gpt(2500, 1408), gpt(3600, 3200)])
    send = lambda msgs, **kw: proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + msgs, **kw}, S)[1]  # noqa: E731
    a = [{"role": "user", "content": "task for agent a " * 50}]
    b = [{"role": "user", "content": "task for agent b " * 50}]
    send(a), send(b)
    b += [{"role": "assistant", "content": "b did step 1 " * 50}, {"role": "user", "content": "b: go on"}]
    send(b)
    summary = [{"role": "user", "content": "Summary of agent a's work so far."}, {"role": "assistant", "content": "noted"},
               {"role": "user", "content": "a: continue"}]
    assert send(summary, prefixwise_compaction_event=True).cause == "compaction_rewrite"
    b += [{"role": "assistant", "content": "b did step 2 " * 50}, {"role": "user", "content": "b: and then?"}]
    back = send(b)
    assert back.cause == "none" and "compared with this conversation's last turn" in back.evidence


def test_a_side_call_of_another_type_does_not_take_the_conversations_place():
    """A side call with its own system prompt that carries the end of the conversation (a "title
    this" call over the last messages) is a new conversation of another type; the conversation's
    next turn is still compared with its own previous turn."""
    proxy = scripted([gpt(3000, 0), gpt(3300, 2944), gpt(900, 0), gpt(3600, 3200)])
    send = lambda system, msgs: proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [system] + msgs}, S)[1]  # noqa: E731
    a = [{"role": "user", "content": "task for agent a " * 50}]
    send(AGENT, a)
    a += [{"role": "assistant", "content": "a did step 1 " * 50}, {"role": "user", "content": "a: go on"}]
    send(AGENT, a)
    titler = {"role": "system", "content": "Write a short title for the exchange below."}
    side = send(titler, [{"role": "user", "content": "Here is the end of a conversation."}] + a[-2:])
    assert side.cause == "first_write"
    a += [{"role": "assistant", "content": "a did step 2 " * 50}, {"role": "user", "content": "a: and then?"}]
    back = send(AGENT, a)
    assert back.cause == "none" and "compared with this conversation's last turn" in back.evidence


def test_another_conversations_first_turn_does_not_take_this_ones_place_in_the_lane():
    """A second agent's first request (several messages, none of them the first agent's) reads as
    an edit in a named session, as on main; the first agent's next turn is still compared with its
    own previous turn."""
    proxy = scripted([gpt(3000, 0), gpt(3300, 2944), gpt(3200, 1408), gpt(3600, 3200)])
    send = lambda msgs: proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [AGENT] + msgs}, S)[1]  # noqa: E731
    a = [{"role": "user", "content": "task for agent a " * 50}]
    send(a)
    a += [{"role": "assistant", "content": "a did step 1 " * 50}, {"role": "user", "content": "a: go on"}]
    send(a)
    b = [{"role": "user", "content": "task for agent b " * 50}, {"role": "assistant", "content": "b did step 1 " * 50},
         {"role": "user", "content": "b: go on"}]
    assert send(b).cause == "history_edit"
    a += [{"role": "assistant", "content": "a did step 2 " * 50}, {"role": "user", "content": "a: and then?"}]
    back = send(a)
    assert back.cause == "none" and "compared with this conversation's last turn" in back.evidence
    assert len(proxy.lane_threads[("s", "gpt-5.6-terra")].threads) == 2


@pytest.mark.parametrize("headers", [S, {}])
def test_two_conversations_that_share_a_message_do_not_take_each_others_place(headers):
    """Two users, each first seen in mid-conversation (after a restart), who both once answered
    "yes". Each is filed under its own opening and compared with its own previous turn, as on main
    (every row a hit)."""
    system = {"role": "system", "content": "You are a support agent. " * 300}

    def conv(who, n):
        msgs = [{"role": "user", "content": f"question from {who} " * 20},
                {"role": "assistant", "content": f"shall I go on, {who}? " * 20}, {"role": "user", "content": "yes"}]
        for i in range(n):
            msgs += [{"role": "assistant", "content": f"answer {who}.{i} " * 20}, {"role": "user", "content": f"follow-up {who}.{i}"}]
        return msgs

    proxy = scripted([gpt(2000, 0)] + [gpt(2200, 1920)] * 9)
    rows = [proxy.handle("openai", {"model": "gpt-5.6-terra", "messages": [system] + conv(who, step)}, headers)[1]
            for step in range(5) for who in ("a", "b")]
    assert [r.cause for r in rows] == ["first_write"] + ["none"] * 9 and sum(r.avoidable_tokens for r in rows) == 0
    (lane,) = proxy.lane_threads.values()
    assert len(lane.threads) == 2


def test_a_trimming_agent_adds_a_remembered_conversation_with_every_trimmed_turn():
    """A conversation is known by its opening message, and every trimmed turn has a new one: the
    lane files each as one more conversation (the known cost of trimming: 64 trimmed turns fill a
    lane)."""
    proxy, _ = trimmed(S, turns=12)
    (lane,) = proxy.lane_threads.values()
    assert len(lane.threads) == 10   # the first three turns share an opening; each of the nine after has its own


# -- 6, 7. rules that must look at what was, or could still be, in the cache ----------------------

def unmarked(gap, ttl_header=None):
    proxy = scripted([claude(fresh=3000), claude(fresh=3100)])
    headers = {**S, **({"x-prefixwise-cache-ttl": ttl_header} if ttl_header else {})}
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": [U1]}, headers)
    idle(proxy, gap)
    return proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": [U1, A1, GO]}, headers)[1]


@pytest.mark.parametrize("gap", [10, 600, 3500])
def test_no_markers_within_the_longest_lifetime_is_still_a_missing_marker(gap):
    """With no cache_control the request states no lifetime; the 5 minutes the edge records for it
    is only what a marker would have had by default. 10 minutes later a marker asking for 1 hour
    would still have been read: 3,000 avoidable tokens, as on main, not an expiry."""
    entry = unmarked(gap)
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("breakpoint_misplaced", False, 3000)
    assert entry.cache_ttl_seconds == 300


@pytest.mark.parametrize("gap, ttl_header", [(3700, None), (7200, None), (3700, "86400")])
def test_no_markers_after_the_longest_lifetime_is_an_expiry_not_waste(gap, ttl_header):
    """No cache_control anywhere: 3,000 avoidable tokens, even two hours later, when no marker could
    have kept the entry alive. The longest a marker can ask for is 1 hour, whatever lifetime the
    edge was told."""
    late = unmarked(gap, ttl_header)
    assert (late.cause, late.legit, late.avoidable_tokens) == ("ttl_expired", True, 0)
    assert "breakpoint_misplaced" in (late.secondary_causes or "") and "exceeds the longest cache TTL 3600s" in late.evidence


def test_the_no_marker_expiry_needs_nothing_read():
    """Something read after two hours means an entry was alive: not an expiry."""
    body = {"model": OPUS, "system": SYS, "messages": [U1]}
    prev = snap(blocks_of(body), 3000, 0, has_breakpoint=False)
    curr = lambda cached: snap(blocks_of({**body, "messages": [U1, A1, GO]}), 3100, cached, has_breakpoint=False,  # noqa: E731
                               gap_seconds=7200, cache_age_seconds=7200)
    assert classify(prev, curr(0)).cause is MissCause.TTL_EXPIRED
    assert classify(prev, curr(500)).cause is MissCause.BREAKPOINT_MISPLACED


def test_a_marker_after_a_volatile_block_keeps_the_lifetime_it_stated():
    """(The proxy never sets this fact today; the classifier's rule is tested on its own.) A marker
    that is there states its lifetime: 10 minutes after a 5-minute marker the entry was gone."""
    body = {"model": OPUS, "system": SYS, "messages": [U1]}
    grown = {**body, "messages": [U1, A1, GO]}
    prev = snap(blocks_of(body), 3000, 0)
    verdict = lambda age: classify(prev, snap(blocks_of(grown), 3100, 0, breakpoint_after_volatile=True,  # noqa: E731
                                              gap_seconds=age, cache_age_seconds=age))
    assert verdict(200).cause is MissCause.BREAKPOINT_MISPLACED
    late = verdict(600)
    assert late.cause is MissCause.TTL_EXPIRED and "exceeds the cache TTL 300s" in late.evidence
    assert MissCause.BREAKPOINT_MISPLACED in late.secondary


def stamped(first, second, system_marked=(False, False), engine=False, **kw):
    """Two turns whose system prompt differs in a timestamp; cache_control on the system prompt of
    the turns named in `system_marked`."""
    proxy = scripted([first, second], **kw)
    if engine:
        proxy.engine, proxy.engine_providers = "vllm", {"anthropic"}
    system = lambda s, mark: marked(SYS + f" Now: 2026-11-26T09:00:{s}Z") if mark else SYS + f" Now: 2026-11-26T09:00:{s}Z"  # noqa: E731
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system("01", system_marked[0]), "messages": [U1]}, S)
    return proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system("09", system_marked[1]),
                                      "messages": [U1, A1, GO]}, S)[1]


def test_a_change_in_a_prompt_with_no_markers_says_both_need_fixing():
    entry = stamped(claude(fresh=3000), claude(fresh=3100))
    assert entry.cause == "volatile_system"   # still the first thing to fix: a marker on it would buy unread writes
    assert "also carries no cache_control" in entry.evidence and "breakpoint_misplaced" in entry.secondary_causes
    assert entry.avoidable_tokens == 3000


def test_the_no_marker_note_needs_a_previous_turn_that_wrote_nothing_and_no_engine():
    """After a turn that did write (it was marked), the change is the whole story. Behind a
    self-hosted engine no marker is needed at all."""
    wrote = stamped(claude(write=3000, fresh=0), claude(fresh=3100), system_marked=(True, False))
    assert wrote.cause == "volatile_system" and "also carries no cache_control" not in wrote.evidence
    assert "breakpoint_misplaced" not in (wrote.secondary_causes or "")
    engine = stamped(claude(fresh=3000), claude(fresh=3100), engine=True)
    assert engine.cause == "volatile_system" and "also carries no cache_control" not in engine.evidence


def below_minimum(usages, bodies=None, gaps=()):
    """Sonnet 4.6 (minimum 1,024): the only marker is on a 900-token system prompt."""
    proxy = scripted(usages)
    msgs = [{"role": "user", "content": "context " * 1500}]
    got = []
    for i in range(len(usages)):
        body = {"model": "claude-sonnet-4-6", "max_tokens": 9, "system": marked("You are a careful agent. " * 150), "messages": msgs}
        if bodies and bodies[i]:
            body = bodies[i](body)
        got.append(proxy.handle("anthropic", body, S)[1])
        if i < len(gaps):
            idle(proxy, gaps[i])
        msgs = msgs + [A1, {"role": "user", "content": f"go {i}"}]
    return got


def test_a_turn_that_left_nothing_in_the_cache_is_not_a_routing_miss():
    """Anthropic reports nothing written and nothing read on every turn: what the marker covers is
    too small to cache. There was nothing for a router to miss."""
    got = below_minimum([claude(fresh=2500), claude(fresh=2600), claude(fresh=2700)])
    assert [e.cause for e in got] == ["first_write", "below_min_prefix", "below_min_prefix"]
    assert "left nothing in the cache" in got[1].evidence and "1024" in got[1].evidence
    assert all(e.legit is None and e.avoidable_tokens == 0 for e in got[1:])


def test_nothing_left_in_the_cache_cannot_expire_or_be_retried():
    """Ten minutes after a turn that left nothing, nothing had expired; and repeating that turn at
    once is not a retry that missed."""
    got = below_minimum([claude(fresh=2500), claude(fresh=2600)], gaps=[600])
    assert got[1].cause == "below_min_prefix" and "ttl_expired" not in (got[1].secondary_causes or "")
    proxy = scripted([claude(fresh=2500), claude(fresh=2500)])
    body = {"model": "claude-sonnet-4-6", "max_tokens": 9, "system": marked("You are a careful agent. " * 150),
            "messages": [{"role": "user", "content": "context " * 1500}]}
    proxy.handle("anthropic", body, S)
    retry = proxy.handle("anthropic", body, S)[1]
    assert (retry.cause, retry.legit, retry.avoidable_tokens) == ("below_min_prefix", None, 0)


@pytest.mark.parametrize("gap", [5, 600])
def test_with_markers_and_nothing_held_a_change_had_nothing_to_lose(gap):
    """The previous turn's marked prefix was too small to cache: it read 0 and wrote 0. A timestamp
    that then changed cost nothing, and was charged 2,500 avoidable tokens. It is not waste, so
    there is no waste verdict with nothing avoidable, and no expiry of an entry that never was."""
    stamp = lambda s: (lambda body: {**body, "system": marked("You are a careful agent. " * 150 + f" Now: 2026-11-26T09:00:{s}Z")})  # noqa: E731
    got = below_minimum([claude(fresh=2500), claude(fresh=2600)], bodies=[stamp("01"), stamp("09")], gaps=[gap])
    entry = got[1]
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("below_min_prefix", None, 0)
    assert entry.secondary_causes == "volatile_system" and "the volatile_system after it cost nothing" in entry.evidence


def test_left_nothing_is_claimed_only_when_nothing_was_read_now():
    """The previous turn left nothing, yet this one read 1,200 tokens: something was in the cache
    after all (another conversation's write). The rule does not apply, with or without a change."""
    stamp = lambda s: (lambda body: {**body, "system": marked("You are a careful agent. " * 150 + f" Now: 2026-11-26T09:00:{s}Z")})  # noqa: E731
    changed = below_minimum([claude(fresh=2500), claude(read=1200, fresh=1400)], bodies=[stamp("01"), stamp("09")])[1]
    assert (changed.cause, changed.legit, changed.avoidable_tokens) == ("volatile_system", False, 2500 - 1200)
    same = below_minimum([claude(fresh=2500), claude(read=1200, fresh=1400)])[1]
    assert same.cause == "routing_miss" and "read 1,200 of the 2,500-token unchanged prefix" in same.evidence


def test_this_requests_ignored_markers_are_named_after_a_turn_that_left_nothing():
    """The previous turn's only marker covered too little. This request marks its last block, 12,000
    tokens in, and still writes and reads nothing: its markers were ignored, and the detail says so."""
    grow = lambda body: {**body, "messages": last_marked(body["messages"][:1] + [A1, {"role": "user", "content": "more context " * 4000}])}  # noqa: E731
    entry = below_minimum([claude(fresh=2500), claude(fresh=12000)], bodies=[None, grow])[1]
    assert (entry.cause, entry.detail) == ("routing_miss", "markers_ignored")


def test_markers_that_were_ignored_are_not_a_prefix_below_the_minimum():
    """A 12,000-token prompt with cache_control on its last block that writes 0 and reads 0: the
    marker covered the whole prompt, far above any minimum, so it was ignored (a gateway stripping
    it). That is main's `routing_miss` with the detail `markers_ignored`, not `below_min_prefix`;
    and a change in such a prompt keeps its avoidable tokens."""
    def run(change=False):
        proxy = scripted([claude(fresh=12000), claude(fresh=12100)])
        system = lambda s: SYS * 5 + (f" Now: 2026-11-26T09:00:{s}Z" if change else "")  # noqa: E731
        proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system("01"), "messages": last_marked([U1])}, S)
        return proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system("09"),
                                          "messages": last_marked([U1, A1, GO])}, S)[1]
    same = run()
    assert (same.cause, same.detail, same.avoidable_tokens) == ("routing_miss", "markers_ignored", 0)
    assert "below the model's minimum" not in same.evidence
    changed = run(change=True)
    assert (changed.cause, changed.detail, changed.legit, changed.avoidable_tokens) == ("volatile_system", "markers_ignored", False, 12000)


def unhonoured(change=False):
    """Opus (minimum 512): cache_control only on a 2,000-token system prompt, and the usage says 0
    read and 0 written on every turn. The marked part is far above the minimum."""
    proxy = scripted([claude(fresh=6000), claude(fresh=6100)])
    system = lambda s: marked(SYS + (f" Now: 2026-11-26T09:00:{s}Z" if change else ""))  # noqa: E731
    context = [{"role": "user", "content": "context " * 4000}]
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system("01"), "messages": context}, S)
    return proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system("09"), "messages": context + [A1, GO]}, S)[1]


@pytest.mark.parametrize("change", [False, True])
def test_a_marked_part_far_above_the_minimum_that_wrote_nothing_is_not_called_too_small(change):
    """Nothing was in the cache, so nothing is waste. But "what its markers covered was below the
    model's minimum cacheable size, 512 tokens" was false, and the advice (mark a longer prefix)
    wrong: the markers were not honoured. The cause is not known, and the evidence says what is."""
    entry = unhonoured(change)
    assert (entry.cause, entry.legit, entry.avoidable_tokens, entry.detail) == ("unknown", None, 0, None)
    assert "carried cache_control on about 1,4" in entry.evidence and "above the model's minimum cacheable size (512 tokens)" in entry.evidence
    assert "Its markers were not honoured" in entry.evidence and "below the model's minimum" not in entry.evidence
    assert entry.secondary_causes == ("volatile_system" if change else None)
    if change:
        assert "the volatile_system after it cost nothing" in entry.evidence


def test_the_true_below_minimum_case_says_how_large_the_marked_part_is():
    got = below_minimum([claude(fresh=2500), claude(fresh=2600)])
    assert got[1].cause == "below_min_prefix"
    assert "the part its markers covered, about 5" in got[1].evidence and "below the model's minimum cacheable size (1024 tokens)" in got[1].evidence


def test_the_marked_part_is_weighed_against_twice_the_minimum():
    """The estimate is a pair, at least and at most. At least twice the minimum: the markers were
    not honoured. At most under twice the minimum: too small, or too close to tell. With no
    estimate, or bounds on either side of that line, neither is claimed. None of them is waste."""
    body = {"model": OPUS, "system": SYS, "messages": [U1]}
    grown = blocks_of({**body, "messages": [U1, A1, GO]})
    verdict = lambda marked: classify(  # noqa: E731
        snap(blocks_of(body), 3000, 0, write_reported=True, marked_tokens=marked), snap(grown, 3100, 0))
    small = verdict((1023, 1023))   # Opus 5.5: minimum 512
    assert small.cause is MissCause.BELOW_MIN_PREFIX and "about 1,023 tokens" in small.evidence and "or not clearly above it" in small.evidence
    at = verdict((1024, 1024))
    assert at.cause is MissCause.UNKNOWN and at.legit is None and at.score is False
    assert "on about 1,024 tokens" in at.evidence and "not honoured" in at.evidence
    assert "on at least about 1,024 tokens" in verdict((1024, 3000)).evidence
    assert "at most about 1,023 tokens" in verdict((0, 1023)).evidence
    for unsure in (None, (500, 3000)):   # no estimate, or one that does not decide
        blind = verdict(unsure)
        assert blind.cause is MissCause.BELOW_MIN_PREFIX and "not honoured" not in blind.evidence
        assert ("the part its markers covered, whose size is not known, was below the model's minimum cacheable size "
                "(512 tokens), or was not cached") in blind.evidence


def test_the_tokens_a_request_marked():
    """The estimate behind those verdicts: everything up to and including the block that carries
    the last marker, as a share of the request's bytes, times its prompt tokens. Exact in bytes for
    a marker on a tool or on the last system block; only bounds elsewhere."""
    from prefixwise.details import body_facts, marked_tokens
    tool = lambda name, mark=False: {"name": name, "description": "d" * 400, "input_schema": {"type": "object"},  # noqa: E731
                                     **({"cache_control": {"type": "ephemeral"}} if mark else {})}
    text = lambda n, mark=False: {"type": "text", "text": "x" * n, **({"cache_control": {"type": "ephemeral"}} if mark else {})}  # noqa: E731
    msgs = [{"role": "user", "content": "y" * 4000}]

    def marked_of(body, prompt=2000):
        return marked_tokens(body_facts("anthropic", body), blocks_of(body), prompt)

    def sizes(body):
        out = collections.Counter()
        for b in blocks_of(body):
            out[b.role] += b.byte_length
        return out, sum(out.values())

    assert marked_of({"model": OPUS, "system": "x" * 1000, "messages": msgs}) is None                 # no marker
    body = {"model": OPUS, "tools": [tool("a"), tool("b")], "system": [text(1000, True)], "messages": msgs}
    by_role, total = sizes(body)
    exact = int(2000 * (by_role["tools"] + by_role["system"]) / total)     # the tools come before the system prompt
    assert marked_of(body) == (exact, exact) and by_role["tools"] > 0
    assert marked_of(body, prompt=0) is None                                                           # no usage
    # The last marker counts, not the first: one on the first tool, one on the system prompt.
    both = {**body, "tools": [tool("a", True), tool("b")]}
    by_role, total = sizes(both)
    assert marked_of(both) == (int(2000 * (by_role["tools"] + by_role["system"]) / total),) * 2
    first_tool = {"model": OPUS, "tools": [tool("a", True), tool("b")], "system": "x" * 1000, "messages": msgs}
    last_tool = {"model": OPUS, "tools": [tool("a"), tool("b", True)], "system": "x" * 1000, "messages": msgs}
    one = blocks_of(first_tool)[0].byte_length                                 # a marker on a tool covers it and those before
    by_role, total = sizes(first_tool)
    assert marked_of(first_tool) == (int(2000 * one / total),) * 2 and one > 0
    by_role, total = sizes(last_tool)
    assert marked_of(last_tool) == (int(2000 * by_role["tools"] / total),) * 2
    # A marker on an earlier system block: between the tools alone and the tools with the whole system prompt.
    early = {"model": OPUS, "tools": [tool("a")], "system": [text(20, True), text(4000)], "messages": msgs}
    by_role, total = sizes(early)
    assert marked_of(early) == (int(2000 * by_role["tools"] / total), int(2000 * (by_role["tools"] + by_role["system"]) / total))
    # A marker inside the conversation: between tools and system, and the whole request.
    inside = {"model": OPUS, "system": "x" * 1000, "messages": [
        {"role": "user", "content": [text(2000, True)]}, {"role": "assistant", "content": "z" * 2000}, {"role": "user", "content": "go"}]}
    by_role, total = sizes(inside)
    assert marked_of(inside) == (int(2000 * by_role["system"] / total), 2000)
    # Bytes that are not text's (an image in base64): no estimate.
    picture = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 400_000}}
    seen = {"model": OPUS, "system": [text(12_000, True)], "messages": [{"role": "user", "content": [picture, text(40)]}]}
    assert marked_of(seen, prompt=4600) is None and marked_of(seen, prompt=200_000) is not None


def after_nothing(system, messages, prompt, model=OPUS):
    """A turn whose markers left nothing (0 read, 0 written), then the same conversation one turn on."""
    proxy = scripted([claude(fresh=prompt), claude(fresh=prompt + 100)])
    proxy.handle("anthropic", {"model": model, "max_tokens": 9, "system": system, "messages": messages}, S)
    return proxy.handle("anthropic", {"model": model, "max_tokens": 9, "system": system, "messages": messages + [A1, GO]}, S)[1]


def test_a_marker_on_a_small_first_block_of_the_system_prompt_is_not_called_unhonoured():
    """The marker sits on a 20-token block; a 4,000-token block follows it, unmarked. Placing the
    marker by its position among the system prompt's blocks made that "about 2,000 tokens, above the
    minimum: its markers were not honoured". What it covered is not known from the sizes kept, so
    neither is claimed."""
    block = lambda text, mark=False: {"type": "text", "text": text, **({"cache_control": {"type": "ephemeral"}} if mark else {})}  # noqa: E731
    entry = after_nothing([block("You are brief. " * 5, True), block("Reference material. " * 800)], [U1], 4100)
    assert entry.cause == "below_min_prefix" and "not honoured" not in entry.evidence
    assert "whose size is not known, was below the model's minimum cacheable size (512 tokens), or was not cached" in entry.evidence


def test_the_last_marker_is_the_one_that_counts():
    """A marker on the first tool and one on the system prompt: the marked part ends at the system
    prompt, 2,000 tokens in, not at the tool."""
    tools = [{"name": "search", "description": "search things", "input_schema": {"type": "object", "properties": {}},
              "cache_control": {"type": "ephemeral"}}]
    proxy = scripted([claude(fresh=6000), claude(fresh=6100)])
    body = lambda msgs: {"model": OPUS, "max_tokens": 9, "tools": tools, "system": marked(), "messages": msgs}  # noqa: E731
    context = [{"role": "user", "content": "context " * 4000}]
    proxy.handle("anthropic", body(context), S)
    entry = proxy.handle("anthropic", body(context + [A1, GO]), S)[1]
    assert entry.cause == "unknown" and "Its markers were not honoured" in entry.evidence


def test_with_an_image_in_the_prompt_the_size_of_the_marked_part_is_not_estimated():
    """A 3,000-token marked system prompt and a 400 kB image: by bytes the system prompt is 3% of the
    request, "about 150 tokens, below the minimum". Bytes of base64 say nothing about tokens, so
    there is no estimate and the wording claims neither."""
    picture = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 400_000}}
    look = [{"role": "user", "content": [picture, {"type": "text", "text": "What is in this picture?"}]}]
    entry = after_nothing(marked(SYS + SYS[:5000]), look, 4600)
    assert entry.cause == "below_min_prefix" and "about" not in entry.evidence
    assert "whose size is not known, was below the model's minimum cacheable size (512 tokens), or was not cached" in entry.evidence
    words = after_nothing(marked(SYS + SYS[:5000]), [{"role": "user", "content": "What is in this text? " * 400}], 4600)
    assert words.cause == "unknown" and "not honoured" in words.evidence   # the same prompt in text: an estimate, far above the minimum


def test_a_whole_prompt_marker_that_wrote_nothing_is_not_called_below_the_minimum():
    """Under the size the `markers_ignored` detail needs, the same holds: a marker on the last block
    covers the whole prompt, which was not below the minimum (that is the rule for a short previous
    prompt), so "what its markers covered was below the minimum" would be false."""
    proxy = scripted([claude(fresh=3000), claude(fresh=3100)])
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": last_marked([U1])}, S)
    entry = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": last_marked([U1, A1, GO])}, S)[1]
    assert entry.cause == "routing_miss" and entry.detail is None


def no_write_count(read=0, fresh=6000):
    """Usage in Anthropic's shape from a server that reports no cache writes at all."""
    return {"input_tokens": fresh, "output_tokens": 5, "cache_read_input_tokens": read}


@pytest.mark.parametrize("engine", [False, True])
def test_a_server_that_reports_no_writes_did_not_leave_nothing(engine):
    """vLLM or SGLang behind /v1/messages, or an Anthropic-compatible API: the client sends
    cache_control and the usage never carries a write count. Its 0 is "not reported": a timestamp
    in the system prompt keeps its 6,000 avoidable tokens, and an unchanged prompt that missed is
    not `below_min_prefix`."""
    first = no_write_count() if not engine else claude(fresh=6000)   # (an engine that does send the field as 0)
    entry = stamped(first, no_write_count(fresh=6100) if not engine else claude(fresh=6100), system_marked=(True, True), engine=engine)
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("volatile_system", False, 6000)
    proxy = scripted([first, no_write_count(fresh=6100) if not engine else claude(fresh=6100)])
    if engine:
        proxy.engine, proxy.engine_providers = "vllm", {"anthropic"}
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": [U1]}, S)
    same = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": [U1, A1, GO]}, S)[1]
    assert same.cause == ("evicted" if engine else "routing_miss")


def test_whose_markers_count_when_only_one_turn_is_marked():
    """Previous turn unmarked, this one marked, a timestamp changed: the change is waste, counted
    as on main (the previous turn left nothing because it had no marker, not because nothing could
    be cached). Previous turn marked below the minimum, this one unmarked, nothing changed: the
    missing marker is the waste, counted as what a marked, stable prompt would have reused."""
    small = "You are a careful agent. " * 150
    proxy = scripted([claude(fresh=2500), claude(fresh=2600)])
    body = lambda system, msgs: {"model": "claude-sonnet-4-6", "max_tokens": 9, "system": system, "messages": msgs}  # noqa: E731
    context = [{"role": "user", "content": "context " * 1500}]
    proxy.handle("anthropic", body(small + " Now: 2026-11-26T09:00:01Z", context), S)
    entry = proxy.handle("anthropic", body(marked(small + " Now: 2026-11-26T09:00:09Z"), context + [A1, GO]), S)[1]
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("volatile_system", False, 2500)
    proxy = scripted([claude(fresh=2500), claude(fresh=2600)])
    proxy.handle("anthropic", body(marked(small), context), S)
    entry = proxy.handle("anthropic", body(small, context + [A1, GO]), S)[1]
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("breakpoint_misplaced", False, 2500)


# -- timing: the age of a cache entry wherever a lifetime is judged --------------------------------

def one_hour_asked(sent, done, breakpoint_far=False):
    """The first turn asks for a 1-hour cache and Anthropic keeps its write for 5 minutes; the next
    turn reads nothing."""
    proxy = scripted([claude(write=3000), claude(write=3100)])
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(ttl="1h"), "messages": [U1]}, S)
    shift(proxy, sent=sent, done=done)
    return proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(ttl="1h"), "messages": [U1, A1, GO]}, S)[1]


def test_ttl_shortened_counts_the_previous_turns_generation_time():
    """60 s of generation and 280 s idle: the 5-minute entry was 340 s old. It was `routing_miss`
    with no detail, because only the idle time was compared with 5 minutes."""
    entry = one_hour_asked(sent=340, done=280)
    assert (entry.cause, entry.detail, entry.legit) == ("ttl_expired", "ttl_shortened", True)
    assert "this request was sent 340 s after it was" in entry.evidence
    young = one_hour_asked(sent=290, done=230)
    assert (young.cause, young.detail) == ("routing_miss", None)


def test_lookback_exceeded_needs_an_entry_younger_than_five_minutes():
    """The detail says a write was alive and out of reach in blocks, and claims it only within 5
    minutes, when any write is alive. 280 s of generation and 60 s idle is 340 s: not claimed."""
    def run(sent, done):
        hour = lambda write: {**claude(write=write), "cache_creation": {"ephemeral_1h_input_tokens": write}}  # noqa: E731
        proxy = scripted([hour(3000), hour(3600)])
        first = [{"role": "user", "content": [{"type": "text", "text": "start", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]
        proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": first}, S)
        shift(proxy, sent=sent, done=done)
        more = [{"role": "user", "content": [{"type": "text", "text": "start"}]}]
        for i in range(15):
            more += [{"role": "assistant", "content": f"step {i}"}, {"role": "user", "content": f"go {i}"}]
        more[-1] = {"role": "user", "content": [{"type": "text", "text": "go 14", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}
        return proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": more}, S)[1]
    near = run(sent=60, done=55)
    assert (near.cause, near.detail) == ("breakpoint_misplaced", "lookback_exceeded")
    old = run(sent=340, done=60)
    assert (old.cause, old.detail) == ("routing_miss", None)


@pytest.mark.parametrize("stamps, detail", [(("", ""), "shared_prefix_not_reused"),
                                            ((" Now: 2026-11-26T09:00:01Z", " Now: 2026-11-26T09:07:30Z"), "volatile_prefix_at_start")])
def test_another_sessions_write_is_timed_from_when_its_request_was_sent(monkeypatch, stamps, detail):
    """Both details say another session's entry was within 5 minutes. That request was sent 340 s
    ago and took 280 s to answer: its entry is 340 s old, not 60."""
    def run(generation, idle_for):
        clock = [1_800_000_000.0]
        monkeypatch.setattr(px.time, "time", lambda: clock[0])
        proxy = PrefixwiseProxy(upstream_url="http://unused.invalid", api_key="k")
        usages, takes = [claude(write=2000, fresh=100), claude(write=2000, fresh=100)], [generation, 1]

        def forward(provider, body, request_id, session_id, client_headers, api="chat"):
            clock[0] += takes.pop(0)
            return 200, {"id": "msg_" + request_id[:8], "usage": usages.pop(0)}

        proxy._forward = forward
        body = lambda stamp: {"model": OPUS, "max_tokens": 9, "system": marked(SYS + stamp), "messages": [U1]}  # noqa: E731
        proxy.handle("anthropic", body(stamps[0]), {"x-prefixwise-session-id": "one"})
        clock[0] += idle_for
        return proxy.handle("anthropic", body(stamps[1]), {"x-prefixwise-session-id": "two"})[1]
    fresh = run(5, 60)
    assert (fresh.cause, fresh.detail) == ("first_write", detail) and "65 s earlier" in fresh.evidence
    stale = run(280, 60)
    assert (stale.cause, stale.detail) == ("first_write", None)


def test_a_call_recorded_with_only_its_end_time_was_not_sent_after_it_ended(tmp_path):
    """SDK `Recorder.record(ended_at=...)` with no latency: the send time was "now", later than the
    end. Two turns 10 minutes apart became `routing_miss`, "gap 0s within TTL 300s"."""
    from prefixwise.sdk import Recorder
    rec = Recorder(str(tmp_path / "sdk.db"))
    turn = lambda msgs: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": msgs}  # noqa: E731
    reply = lambda usage: {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "usage": usage}  # noqa: E731
    rec.record("anthropic", turn([U1]), reply(claude(write=3000)), session_id="s", ended_at=T0)
    entry = rec.record("anthropic", turn([U1, A1, GO]), reply(claude(write=3100)), session_id="s", ended_at=T0 + 600)
    assert entry.cause == "ttl_expired" and entry.gap_seconds == pytest.approx(600) and entry.cache_age_seconds == pytest.approx(600)
    assert rec.proxy.session_facts["s"].sent_at == pytest.approx(T0 + 600)   # the edge's own record of when it was sent
    rec.close()


def test_the_collector_never_takes_a_request_as_sent_after_it_ended():
    """A record from any edge whose send time is later than its end time: the end time stands."""
    edge, sink = Edge(), collector()
    turn = lambda n: {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": chat("u", n)}  # noqa: E731
    first = edge.record("anthropic", turn(1), S, T0 + 5000, T0, claude(write=3000))   # "sent" long after it ended
    second = edge.record("anthropic", turn(2), S, T0 + 5600, T0 + 600, claude(write=3300))
    sink.ingest(first)
    entry = sink.ingest(second)[1]
    assert entry.cause == "ttl_expired" and entry.gap_seconds == pytest.approx(600) and entry.cache_age_seconds == pytest.approx(600)


def test_a_stitched_turns_age_is_counted_from_the_turn_it_continues():
    """Two turns continue the same response. The second one's cache entry is as old as the response
    it continues, not as the other branch, which is the conversation's latest turn."""
    edge, sink = Edge(), collector()
    base = {"model": "gpt-5.6", "instructions": "You are helpful. " * 400}
    headers = {**S, "x-prefixwise-cache-ttl": "300"}
    rec = lambda body, sent, usage, rid: sink.ingest(edge.record(  # noqa: E731
        "openai", body, headers, T0 + sent, T0 + sent + 5, usage, api="responses", response_id=rid))[1]
    rec({**base, "input": "first question"}, 0, responses_usage(2000, 0), "resp_1")
    rec({**base, "previous_response_id": "resp_1", "input": "one way"}, 350, responses_usage(2050, 0), "resp_2")
    entry = rec({**base, "previous_response_id": "resp_1", "input": "another way"}, 400, responses_usage(2050, 0), "resp_3")
    assert entry.cache_age_seconds == pytest.approx(400)
    assert entry.cause == "ttl_expired" and "400s since the previous turn was sent" in entry.evidence


def test_the_one_hour_advice_counts_an_entrys_age_not_the_idle_gap():
    """A 5-minute entry that expired: a 1-hour cache would have kept it if it was under an hour old.
    Sent 3,700 s before, answered 3,500 s before: the idle gap is under the hour, the entry is not."""
    def ledger_with(age, gap):
        ledger = Ledger(":memory:")
        for cause, a, g in (("first_write", None, None), ("ttl_expired", age, gap)):
            ledger.record(LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model=OPUS, cause=cause,
                                      prompt_tokens=3000, cached_tokens=0, avoidable_tokens=0, avoidable_cost=None,
                                      legit=True, cache_write_tokens=3000, gap_seconds=g, cache_age_seconds=a,
                                      cache_ttl_seconds=300.0))
        return ledger
    assert [a["misses"] for a in ttl_advice(ledger_with(340, 60))] == [1]
    assert ttl_advice(ledger_with(3700, 3500)) == []
    assert [a["misses"] for a in ttl_advice(ledger_with(None, 600))] == [1]   # an older row: only the gap is known


# -- the record on the wire ------------------------------------------------------------------------

def test_session_shared_is_left_out_of_a_record_when_false():
    """An older collector refuses a record with a field it does not know. Most records come from
    named sessions: those must still be accepted."""
    edge = Edge()
    body = {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": [U1]}
    named = edge.record("anthropic", body, S, T0, T0 + 1, claude(write=3000))
    shared = edge.record("anthropic", body, {}, T0, T0 + 1, claude(write=3000))
    assert named.session_shared is False and "session_shared" not in named.to_dict()
    assert shared.session_shared is True and shared.to_dict()["session_shared"] is True
    assert TurnRecord.from_dict(named.to_dict()).session_shared is False
    assert TurnRecord.from_dict(shared.to_dict()).session_shared is True


def test_a_new_conversations_evidence_never_says_the_blocks_are_equal():
    """The rules a new conversation passes through on its way speak of what it shares, not of an
    unchanged prompt."""
    one = lambda text: blocks_of({"model": "gpt-5.6-terra", "messages": [{"role": "system", "content": SYS},  # noqa: E731
                                                                         {"role": "user", "content": text}]}, "openai")
    prev = snap(one("ticket 1"), 3000, 0, provider="openai", model="gpt-5.6-terra", params={"tool_choice": "a"})
    changed = classify(prev, snap(one("ticket 2"), 3000, 0, provider="openai", model="gpt-5.6-terra", params={"tool_choice": "b"}))
    assert changed.cause is MissCause.PARAM_RENDERED and "blocks equal" not in changed.evidence
    assert "a new conversation on the same tools and system prompt; prompt-affecting params changed: ['tool_choice']" in changed.evidence
    silent = classify(prev, snap(one("ticket 2"), 0, 0, provider="openai", model="gpt-5.6-terra", params={"tool_choice": "a"}))
    assert silent.cause is MissCause.UNKNOWN and silent.evidence.startswith("a new conversation in this session but the provider returned no usage")
    same = classify(prev, snap(one("ticket 1") + one("ticket 1")[-1:], 0, 0, provider="openai", model="gpt-5.6-terra",
                               params={"tool_choice": "a"}))
    assert same.evidence.startswith("blocks unchanged but the provider returned no usage")
