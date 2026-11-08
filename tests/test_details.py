"""Sub-reasons (prefixwise.details), end to end through the proxy with scripted usage.

Each detail has a positive case and the false-positive scenarios found in review, which
must produce no claim.
"""

import copy

from prefixwise.proxy import PrefixwiseProxy

TOOLS = [{"name": f"t{i}", "description": "d", "input_schema": {"type": "object"}} for i in range(3)]
SYSTEM = [{"type": "text", "text": "You are a careful agent. " * 200}]
MODEL = "claude-haiku-5-5"


def proxy(usages):
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    queue = list(usages)

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        rid = ("msg_" if provider == "anthropic" else "chatcmpl-") + request_id[:8]
        return 200, {"id": rid, "usage": queue.pop(0)}
    p._forward = fake
    return p


def a_usage(read=0, write5=0, write1h=0, fresh=200):
    return {"input_tokens": fresh, "output_tokens": 5, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write5 + write1h,
            "cache_creation": {"ephemeral_5m_input_tokens": write5, "ephemeral_1h_input_tokens": write1h}}


def body(messages, ttl=None, model=MODEL, system_marker=True, last_marker=False):
    cc = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
    sys_ = copy.deepcopy(SYSTEM)
    if system_marker:
        sys_[-1]["cache_control"] = cc
    messages = copy.deepcopy(messages)
    if last_marker:
        last = messages[-1]
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        last["content"][-1]["cache_control"] = cc
    return {"model": model, "max_tokens": 64, "tools": TOOLS, "system": sys_, "messages": messages}


def turn(p, b, session="s"):
    return p.handle("anthropic", b, {"x-prefixwise-session-id": session})[1]


def idle(p, seconds, session="s", model=MODEL):
    """Pretend the previous response came `seconds` earlier."""
    history = p.lane_history[(session, model)]
    snap, done, facts = history[-1]
    history[-1] = (snap, done - seconds, facts)
    facts.done_at -= seconds
    p.last_seen[session] -= seconds


USER = {"role": "user", "content": "start"}
MORE = [USER, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]


# -- ttl_shortened -------------------------------------------------------------------

def test_ttl_shortened():
    p = proxy([a_usage(write5=6000), a_usage(write5=6100)])
    turn(p, body([USER], ttl="1h"))
    idle(p, 600)  # within the hour asked for, past the 5 minutes granted
    e = turn(p, body(MORE, ttl="1h"))
    assert e.cause == "ttl_expired" and e.detail == "ttl_shortened" and e.legit
    assert "asked for a 1-hour cache on every breakpoint" in e.evidence and "classifier: routing_miss" in e.evidence


def test_ttl_shortened_false_positives():
    # The hour was granted.
    p = proxy([a_usage(write1h=6000), a_usage(write1h=6100)])
    turn(p, body([USER], ttl="1h"))
    idle(p, 600)
    assert turn(p, body(MORE, ttl="1h")).detail is None
    # The previous request asked for 5 minutes; only this one asks for an hour.
    p = proxy([a_usage(write5=6000), a_usage(write5=6100)])
    turn(p, body([USER]))
    idle(p, 600)
    assert turn(p, body(MORE, ttl="1h")).detail is None
    # Mixed TTLs: a 1-hour system block and a 5-minute tail. The 5-minute write was asked for.
    p = proxy([a_usage(write5=6000), a_usage(write5=6100)])
    mixed = body([USER], ttl="1h")
    mixed["messages"] = [{"role": "user", "content": [{"type": "text", "text": "start",
                                                       "cache_control": {"type": "ephemeral"}}]}]
    turn(p, mixed)
    idle(p, 600)
    assert turn(p, body(MORE, ttl="1h")).detail is None
    # "ttl": "1h" inside a tool input is not a cache setting.
    p = proxy([a_usage(write5=6000), a_usage(write5=6100)])
    tricky = [USER, {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "t0",
                                                        "input": {"ttl": "1h"}}]}]
    turn(p, body(tricky))
    idle(p, 600)
    nxt = tricky + [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}]}]
    assert turn(p, body(nxt)).detail is None


# -- lookback_exceeded ---------------------------------------------------------------

def text_blocks(n, marker_last=True):
    out = [{"type": "text", "text": f"output {i}"} for i in range(n)]
    if marker_last:
        out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


GO = {"role": "user", "content": [{"type": "text", "text": "go"}]}
FIRST = [{"role": "user", "content": [{"type": "text", "text": "go", "cache_control": {"type": "ephemeral"}}]}]


def second_turn(blocks):
    return [GO, {"role": "assistant", "content": "working"}, {"role": "user", "content": blocks}]


def lookback_body(messages, model=MODEL):
    return {"model": model, "max_tokens": 64, "tools": TOOLS, "system": SYSTEM, "messages": messages}


def test_lookback_exceeded():
    p = proxy([a_usage(write5=6000), a_usage(write5=6500)])
    turn(p, lookback_body(FIRST))
    e = turn(p, lookback_body(second_turn(text_blocks(30))))
    assert e.cause == "breakpoint_misplaced" and e.detail == "lookback_exceeded" and e.legit is False


def test_lookback_false_positives():
    # A real expiry (1 hour idle) is not a misplaced breakpoint.
    p = proxy([a_usage(write5=6000), a_usage(write5=6500)])
    turn(p, lookback_body(FIRST))
    idle(p, 3600)
    e = turn(p, lookback_body(second_turn(text_blocks(30))))
    assert e.detail is None and e.cause != "breakpoint_misplaced"
    # Parallel tool calls: a run of tool_use (or tool_result) blocks is one position.
    p = proxy([a_usage(write5=6000), a_usage(write5=6500)])
    turn(p, lookback_body(FIRST))
    calls = [{"type": "tool_use", "id": f"c{i}", "name": "t0", "input": {}} for i in range(15)]
    results = [{"type": "tool_result", "tool_use_id": f"c{i}", "content": "r"} for i in range(15)]
    results[-1]["cache_control"] = {"type": "ephemeral"}
    msgs = [GO, {"role": "assistant", "content": calls}, {"role": "user", "content": results}]
    assert turn(p, lookback_body(msgs)).detail is None
    # The conversation was replaced, not extended: the prefix is not "unchanged".
    p = proxy([a_usage(write5=6000), a_usage(write5=6500)])
    turn(p, lookback_body([{"role": "user", "content": [{"type": "text", "text": "title this",
                                                         "cache_control": {"type": "ephemeral"}}]}]))
    assert turn(p, lookback_body(second_turn(text_blocks(30)))).detail != "lookback_exceeded"
    # Nothing was written before (markers stripped upstream): no entry to be out of reach of.
    p = proxy([a_usage(fresh=6000), a_usage(fresh=6500)])
    turn(p, lookback_body(FIRST))
    assert turn(p, lookback_body(second_turn(text_blocks(30)))).detail != "lookback_exceeded"


# -- markers_ignored -----------------------------------------------------------------

def test_markers_ignored():
    p = proxy([a_usage(fresh=9000)])
    e = turn(p, body([USER], system_marker=False, last_marker=True))
    assert e.cause == "first_write" and e.detail == "markers_ignored"


def test_markers_ignored_false_positives():
    p = proxy([a_usage(write5=9000)])  # written: the markers worked
    assert turn(p, body([USER], system_marker=False, last_marker=True)).detail is None
    p = proxy([a_usage(fresh=6000)])  # below 8,192 tokens: may be under a model's minimum
    assert turn(p, body([USER], system_marker=False, last_marker=True)).detail is None
    p = proxy([a_usage(fresh=60000)])  # breakpoint not on the last block: the prefix may be short
    assert turn(p, body(MORE)).detail is None


# -- fan-out and shared prefix -------------------------------------------------------

def test_sent_before_cache_written():
    p = proxy([a_usage(write5=6000), a_usage(write5=6000)])
    ctx_a = p.prepare("anthropic", body([USER]), {"x-prefixwise-session-id": "A"})  # still waiting
    e = turn(p, body([USER]), session="B")
    assert e.cause == "first_write" and e.detail == "sent_before_cache_written"
    p.finalize_safely(ctx_a, 200, {"id": "msg_a", "usage": a_usage(write5=6000)})


def test_fanout_false_positives():
    # Another model in flight: separate cache.
    p = proxy([a_usage(write5=6000)])
    p.prepare("anthropic", body([USER], model="claude-sonnet-5-5"), {"x-prefixwise-session-id": "A"})
    assert turn(p, body([USER]), session="B").detail is None
    # A mid-session miss is not a fan-out first request.
    p = proxy([a_usage(write5=6000), a_usage(fresh=6500)])
    turn(p, body([USER]))
    p.prepare("anthropic", body([USER]), {"x-prefixwise-session-id": "other"})
    assert turn(p, body(MORE)).detail != "sent_before_cache_written"
    # The other request had already started answering.
    p = proxy([a_usage(write5=6000)])
    ctx = p.prepare("anthropic", body([USER]), {"x-prefixwise-session-id": "A"})
    p.response_started(ctx)
    assert turn(p, body([USER]), session="B").detail is None
    # More than a second later: its prefill may be done.
    p = proxy([a_usage(write5=6000)])
    ctx = p.prepare("anthropic", body([USER]), {"x-prefixwise-session-id": "A"})
    p.prefix_inflight[ctx["facts"].prefix_key][ctx["request_id"]] -= 5
    assert turn(p, body([USER]), session="B").detail is None


def test_shared_prefix_not_reused():
    p = proxy([a_usage(write5=6000), a_usage(fresh=6200)])
    turn(p, body([USER]), session="A")  # wrote an entry ending at the system prompt
    e = turn(p, body([USER]), session="C")
    assert e.detail == "shared_prefix_not_reused"


def test_shared_prefix_false_positives():
    # The other session's only breakpoint was at the end of its messages: no entry for tools+system.
    p = proxy([a_usage(write5=6000), a_usage(fresh=6200)])
    turn(p, body([USER], system_marker=False, last_marker=True), session="A")
    assert turn(p, body([USER]), session="C").detail is None
    # It wrote nothing.
    p = proxy([a_usage(fresh=6000), a_usage(fresh=6200)])
    turn(p, body([USER]), session="A")
    assert turn(p, body([USER]), session="C").detail != "shared_prefix_not_reused"
    # Another model.
    p = proxy([a_usage(write5=6000), a_usage(fresh=6200)])
    turn(p, body([USER], model="claude-sonnet-5-5"), session="A")
    assert turn(p, body([USER]), session="C").detail is None


# -- thinking_stripped ---------------------------------------------------------------

THINK = {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "x"},
                                          {"type": "text", "text": "answer"}]}
STRIPPED = {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
NEXT = {"role": "user", "content": "next"}


def test_thinking_stripped():
    p = proxy([a_usage(write5=6000), a_usage(write5=6500)])
    turn(p, body([USER, THINK, NEXT]))
    e = turn(p, body([USER, STRIPPED, NEXT, {"role": "assistant", "content": "a"},
                      {"role": "user", "content": "more"}]))
    assert e.cause == "history_edit" and e.detail == "thinking_stripped"


def test_thinking_stripped_false_positives():
    # A model that strips earlier thinking itself: removing it is no cause.
    p = proxy([a_usage(write5=6000), a_usage(write5=6500)])
    turn(p, body([USER, THINK, NEXT], model="claude-haiku-4-5"))
    assert turn(p, body([USER, STRIPPED, NEXT], model="claude-haiku-4-5")).detail is None
    # The cache served it.
    p = proxy([a_usage(write5=6000), a_usage(read=6000, write5=500)])
    turn(p, body([USER, THINK, NEXT]))
    assert turn(p, body([USER, STRIPPED, NEXT])).detail is None


# -- cache_key_changed, report, API, leaks -------------------------------------------

def test_openai_cache_key_changed():
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    q = [{"prompt_tokens": 5000, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}] * 2
    p._forward = lambda *a, **k: (200, {"id": "chatcmpl-x", "usage": q.pop(0)})
    msgs = [{"role": "system", "content": "Be careful. " * 400}, {"role": "user", "content": "hi"}]
    p.handle("openai", {"model": "gpt-5.6", "messages": msgs, "prompt_cache_key": "a"}, {"x-prefixwise-session-id": "s"})
    _, e = p.handle("openai", {"model": "gpt-5.6", "messages": msgs + [{"role": "assistant", "content": "x"},
                                                                       {"role": "user", "content": "y"}],
                               "prompt_cache_key": "b"}, {"x-prefixwise-session-id": "s"})
    assert e.cause == "routing_miss" and e.detail == "cache_key_changed"


def test_an_ordinary_hit_has_no_detail():
    p = proxy([a_usage(write5=6000), a_usage(read=6000, write5=100)])
    turn(p, body([USER]))
    e = turn(p, body(MORE))
    assert e.cause == "none" and e.detail is None


def test_details_reach_the_report_and_the_api():
    from prefixwise import api, findings
    p = proxy([a_usage(fresh=9000), a_usage(fresh=9000)])
    turn(p, body([USER], system_marker=False, last_marker=True), session="x")
    turn(p, body([USER], system_marker=False, last_marker=True), session="y")
    f = findings.findings(p.ledger)
    first = next(e for e in f["expected"] if e["cause"] == "first_write")
    assert first["details"][0]["detail"] == "markers_ignored" and first["details"][0]["requests"] == 2
    assert "gateway" in findings.render(f)
    status, out = api.get(p, "/api/v1/breakdown", "by=detail")
    assert status == 200 and any(r["detail"] == "markers_ignored" for r in out["rows"])
    status, out = api.get(p, "/api/v1/requests", "detail=markers_ignored")
    assert status == 200 and len(out["requests"]) == 2


def test_a_rejected_stream_body_releases_its_prefix():
    p = proxy([])
    p.handle("anthropic", body([USER]) | {"stream": True}, {"x-prefixwise-session-id": "A"})
    assert all(not flying for flying in p.prefix_inflight.values())


def test_dropping_thinking_on_a_model_that_drops_it_itself_is_no_edit():
    # Haiku 4.5 keeps only the turn in progress: the API strips earlier thinking either way.
    p = proxy([a_usage(write5=6000), a_usage(read=6000, write5=300)])
    turn(p, body([USER, THINK, NEXT], model="claude-haiku-4-5"))
    e = turn(p, body([USER, STRIPPED, NEXT, {"role": "assistant", "content": "a"},
                      {"role": "user", "content": "more"}], model="claude-haiku-4-5"))
    assert e.cause == "none" and e.avoidable_tokens == 0
