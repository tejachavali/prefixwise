"""Principal review (2026-11-08): each finding is a scenario that must hold."""

import time

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.fingerprint import Fingerprinter
from prefixwise.ledger import avoidable_for_turn
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, extract_blocks, make_handler
from test_proxy import FakeUpstream, sse, start, stream

FP = Fingerprinter()
TOOLS = [{"name": "read", "description": "read a file", "input_schema": {"type": "object"}}]


def snap(system="You are careful.", tools=TOOLS, messages=None, prompt=40_000, cached=0, gap=10.0, ttl=300.0,
         model="claude-sonnet-5-5", **kw):
    body = {"model": model, "system": system, "tools": tools,
            "messages": messages or [{"role": "user", "content": "hello"}]}
    return RequestSnapshot(model=model, provider="anthropic", blocks=FP.fingerprint_blocks(extract_blocks(body, "anthropic")),
                           prompt_tokens=prompt, cached_tokens=cached, gap_seconds=gap, ttl_seconds=ttl,
                           has_breakpoint=True, **kw)


def history(n, edit_at=None):
    msgs = []
    for i in range(n):
        msgs += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    if edit_at is not None:
        msgs[edit_at] = {"role": "user", "content": "question edited"}
    return msgs + [{"role": "user", "content": "next"}]


# -- 1. partial hits: the cause is the edit, the price only the part it cost ------------------

def test_a_late_edit_with_a_partial_hit_prices_only_the_delta():
    prev = snap(messages=history(10), prompt=39_000)
    curr = snap(messages=history(11, edit_at=14), prompt=40_000, cached=35_000)
    got = classify(prev, curr)
    assert got.cause == MissCause.HISTORY_EDIT and got.legit is False
    assert avoidable_for_turn(got.legit, 40_000, 35_000, 39_000) == 4_000  # not the whole turn


def test_a_change_that_cost_nothing_is_a_hit():
    """The cache served all of the prefix the previous turn could share: the change was after
    it, or never reached the rendered prompt."""
    prev = snap(system="You are careful. Today is 2026-11-07.", prompt=39_000)
    curr = snap(system="You are careful. Today is 2026-11-08.", prompt=40_000, cached=39_000)
    got = classify(prev, curr)
    assert got.cause == MissCause.NONE and MissCause.VOLATILE_SYSTEM in got.secondary


def test_an_expired_cache_is_the_cause_even_when_the_prompt_also_changed():
    prev = snap(system="You are careful. Today is 2026-11-07.")
    curr = snap(system="You are careful. Today is 2026-11-08.", gap=900.0, cached=0)
    got = classify(prev, curr)
    assert got.cause == MissCause.TTL_EXPIRED and got.legit is True
    assert MissCause.VOLATILE_SYSTEM in got.secondary


# -- 2. too short to cache wins over any change -----------------------------------------------

def test_too_short_to_cache_is_not_blamed_on_a_volatile_system_prompt():
    prev = snap(system="You are careful. Today is 2026-11-07.", prompt=500)
    curr = snap(system="You are careful. Today is 2026-11-08.", prompt=800, cached=0)
    got = classify(prev, curr)
    assert got.cause == MissCause.BELOW_MIN_PREFIX
    assert avoidable_for_turn(got.legit, 800, 0, 500) == 0


def test_tools_come_before_the_system_prompt_on_hosted_providers():
    prev = snap(system="A", tools=TOOLS)
    curr = snap(system="B", tools=TOOLS + [{"name": "write", "input_schema": {"type": "object"}}])
    assert classify(prev, curr).cause == MissCause.TOOL_CATALOG_CHANGE  # tools are earlier in the prefix
    curr.engine = True  # a self-hosted chat template renders the system prompt first
    assert classify(prev, curr).cause == MissCause.SYSTEM_EDIT


# -- 3. interleaved conversations in one session --------------------------------------------

def body(system, opening, turn):
    msgs = [{"role": "user", "content": opening}]
    for i in range(turn - 1):
        msgs += [{"role": "assistant", "content": f"a{i}"}, {"role": "user", "content": f"u{i}"}]
    return {"model": "claude-sonnet-5-5", "max_tokens": 8, "messages": msgs,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}


def run(proxy, *bodies):
    h = {"x-prefixwise-session-id": "orchestrator"}
    return [proxy.handle("anthropic", b, h)[0] for b in bodies]


def realistic(p):
    """An upstream whose cache reads nothing on a conversation's first turn, most of it after."""
    ids = iter(range(1, 10**6))

    def forward(provider, b, *a, **k):
        cached = 0 if len(b["messages"]) == 1 else 1800
        return 200, {"id": f"msg_{next(ids)}", "usage": {"input_tokens": 2000 - cached,
                                                         "cache_read_input_tokens": cached, "output_tokens": 1}}
    p._forward = forward
    return p


def test_interleaved_subagents_are_each_compared_with_their_own_last_turn():
    p = realistic(PrefixwiseProxy(upstream_url="http://unused", api_key="k"))
    a, b = ("You are the code reviewer.", "review x"), ("You are the test writer.", "test x")
    got = run(p, body(*a, 1), body(*b, 1), body(*a, 2), body(*b, 2), body(*a, 3))
    assert [r["cause"] for r in got] == ["first_write", "first_write", "none", "none", "none"]
    assert "new conversation" in got[1]["evidence"] and "came between" in got[2]["evidence"]


def test_a_rewritten_opening_with_the_same_system_prompt_is_not_a_new_thread():
    """A compaction, or an edit of the first message, keeps the system prompt: compared with the
    latest turn as before, so waste is never hidden as a 'new conversation'."""
    p = PrefixwiseProxy(mock=True)
    got = run(p, body("You are careful.", "original", 2), body("You are careful.", "rewritten", 2))
    assert got[1]["cause"] == "history_edit"


# -- 4. the gap a cache entry had to survive: from the last response to this send -----------

def test_the_gap_is_measured_to_the_send_not_to_the_response():
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    payload = {"id": "m", "usage": {"input_tokens": 100, "cache_read_input_tokens": 1900, "output_tokens": 1}}
    slow = []

    def forward(*a, **k):
        time.sleep(0.6 if slow else 0)  # this request's own generation time
        return 200, payload
    p._forward = forward
    h = {"x-prefixwise-session-id": "g"}
    p.handle("anthropic", body("You are careful.", "q", 1), h)
    lane = p.lane_threads[("g", "claude-sonnet-5-5")]
    snap0, done, facts, rid = lane.threads[lane.latest]
    lane.threads[lane.latest] = (snap0, done - 100, facts, rid)
    slow.append(1)
    _, entry = p.handle("anthropic", body("You are careful.", "q", 2), h)
    assert 99.9 <= entry.gap_seconds < 100.4  # not ~100.6: generation time is not idle time


# -- 6. a stream ends for the client before Prefixwise classifies it -------------------------

def test_a_stream_is_not_held_open_while_it_is_classified():
    done = {"id": "c1", "model": "gpt-5.6", "object": "chat.completion.chunk", "choices": [],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}}
    up = FakeUpstream(stream_lines=sse((None, {"id": "c1", "choices": [{"delta": {"content": "hi"}}]}),
                                       (None, done)) + [b"data: [DONE]\n\n"])
    proxy = PrefixwiseProxy(upstream_url=up.url)
    slow = proxy.finalize_safely

    def finalize_slowly(*a, **k):
        time.sleep(1.5)
        return slow(*a, **k)
    proxy.finalize_safely = finalize_slowly
    server = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    port = start(server)
    try:
        t0 = time.time()
        status, raw, _ = stream(port, "/v1/chat/completions",
                                {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}], "stream": True})
        assert status == 200 and b"[DONE]" in raw
        assert time.time() - t0 < 1.0  # the client did not wait for the classification
    finally:
        time.sleep(1.6)
        server.shutdown()
        up.server.shutdown()


# -- independent review of the fixes above: each must hold too ------------------------------

def test_a_timestamp_in_the_system_prompt_is_not_a_new_conversation():
    """One-shot calls, a different question each time, a fresh timestamp in the system prompt:
    the same system prompt beyond the date, so volatile_system (waste), never first_write."""
    p = realistic(PrefixwiseProxy(upstream_url="http://unused", api_key="k"))
    got = run(p, *[body(f"You are careful. Now: 2026-11-08T12:0{i}:00Z", f"question {i}", 1) for i in range(4)])
    assert [r["cause"] for r in got] == ["first_write"] + ["volatile_system"] * 3


def test_the_main_thread_is_remembered_however_long_a_subagent_runs():
    p = realistic(PrefixwiseProxy(upstream_url="http://unused", api_key="k"))
    main, sub = ("You are the orchestrator.", "build it"), ("You are the explorer.", "look around")
    got = run(p, body(*main, 1), body(*main, 2), *[body(*sub, i) for i in range(1, 8)], body(*main, 3))
    assert got[-1]["cause"] == "none" and "came between" in got[-1]["evidence"]


def test_many_parallel_subagents_of_one_type_are_each_followed():
    p = realistic(PrefixwiseProxy(upstream_url="http://unused", api_key="k"))
    agents = [("You are an explorer.", f"area {k}") for k in range(6)]
    turns = [body(*a, 1) for a in agents] + [body(*a, n) for n in (2, 3) for a in agents]
    got = run(p, *turns)
    assert all(r["cause"] == "none" for r in got[6:])  # every later turn: its own conversation's


def test_one_conversation_keeps_one_snapshot_and_busy_lanes_stay():
    p = realistic(PrefixwiseProxy(upstream_url="http://unused", api_key="k"))
    run(p, *[body("You are careful.", "q", n) for n in range(1, 30)])
    assert len(p.lane_threads[("orchestrator", "claude-sonnet-5-5")].threads) == 1  # memory as before
    p.lane_threads.limit = 2
    for session in ("A", "B", "A", "C"):  # A stays active, B goes quiet
        p.handle("anthropic", body("You are careful.", "q", 1), {"x-prefixwise-session-id": session})
    assert ("A", "claude-sonnet-5-5") in p.lane_threads and ("B", "claude-sonnet-5-5") not in p.lane_threads


def test_diagnostics_are_chained_to_the_requests_own_conversation():
    p = realistic(PrefixwiseProxy(upstream_url="http://unused", api_key="k"))
    sent, real = [], p._forward

    def spy(provider, b, *a, **k):
        sent.append(b.get("diagnostics"))
        return real(provider, b, *a, **k)
    p._forward = spy
    a, b = ("You are the code reviewer.", "review x"), ("You are the test writer.", "test x")
    first = run(p, body(*a, 1))[0]
    run(p, body(*b, 1), body(*a, 2))
    assert sent[1] == {"previous_message_id": None}  # a new conversation: like a first turn
    assert sent[2] == {"previous_message_id": first["response"]["id"]}  # A's own turn, not B's


def test_a_conversations_next_request_waits_for_its_streamed_turn_to_be_recorded():
    import threading
    p = PrefixwiseProxy(mock=True)
    h = {"x-prefixwise-session-id": "w"}
    first = p.prepare("anthropic", body("You are careful.", "q", 1), h)
    pending = p.begin_recording(first)  # its stream ended for the client; not recorded yet
    other = p.prepare("anthropic", body("You are the tester.", "another task", 1), h)
    assert other  # another conversation of the session is not held up
    done = []
    t = threading.Thread(target=lambda: done.append(p.prepare("anthropic", body("You are careful.", "q", 2), h)))
    t.start()
    time.sleep(0.2)
    assert not done  # the same conversation's next turn waits for the record
    p.end_recording(first, pending)
    t.join(2)
    assert done


def test_the_end_of_a_stream_is_recognised_in_every_format():
    from prefixwise.proxy import StreamAccumulator
    for provider, api, last in (("openai", "chat", b"data: [DONE]\n"),
                                ("anthropic", "chat", b'data: {"type": "message_stop"}\n'),
                                ("openai", "responses", b'data: {"type": "response.completed", "response": {}}\n')):
        acc = StreamAccumulator(provider, api=api)
        acc.feed(b'data: {"type": "ping"}\n')
        assert not acc.done
        acc.feed(last)
        assert acc.done, (provider, api)


def responses(p, prev=None, text="hi", instructions="You are careful."):
    b = {"model": "gpt-5.6", "instructions": instructions, "input": text}
    if prev:
        b["previous_response_id"] = prev
    return p.handle("openai", b, {"x-prefixwise-session-id": "r"}, api="responses")[0]


def test_stitched_responses_turns_stay_one_conversation_and_chain_to_the_turn_they_name():
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    sent, ids = [], iter(range(1, 100))

    def forward(provider, b, *a, **k):
        sent.append((b.get("prompt_cache_options") or {}).get("comparison_response_id"))
        return 200, {"id": f"resp_{next(ids)}", "usage": {"input_tokens": 2000, "output_tokens": 1,
                                                          "input_tokens_details": {"cached_tokens": 1800}}}
    p._forward = forward
    prev = responses(p)["response"]["id"]
    for i in range(10):
        prev = responses(p, prev, text=f"more {i}")["response"]["id"]
    assert len(p.lane_threads[("r", "gpt-5.6")].threads) == 1  # one conversation, not eleven
    fork_a = responses(p, "resp_3", text="branch a")
    responses(p, "resp_3", text="branch b")
    assert sent[-2:] == ["resp_3", "resp_3"]  # each branch compared with the turn it names
    assert fork_a["cause"] != "first_write"


def test_developer_messages_are_the_system_prompt():
    from prefixwise.proxy import extract_blocks
    b = {"model": "gpt-5.6", "messages": [{"role": "developer", "content": "You are terse."},
                                         {"role": "system", "content": "Today is Monday."},
                                         {"role": "user", "content": "hi"}]}
    roles = [r for r, _ in extract_blocks(b, "openai")]
    assert roles == ["system", "system", "messages"]
