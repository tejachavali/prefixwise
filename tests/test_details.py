"""Sub-reasons (prefixwise.details), end to end through the proxy with scripted usage."""

import copy

from prefixwise.proxy import PrefixwiseProxy

TOOLS = [{"name": f"t{i}", "description": "d", "input_schema": {"type": "object"}} for i in range(3)]
SYSTEM = [{"type": "text", "text": "You are a careful agent. " * 200}]


def proxy(usages):
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    queue = list(usages)

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        u = queue.pop(0)
        rid = ("msg_" if provider == "anthropic" else "chatcmpl-") + request_id[:8]
        return 200, {"id": rid, "usage": u}
    p._forward = fake
    return p


def a_usage(read=0, write5=0, write1h=0, fresh=200):
    return {"input_tokens": fresh, "output_tokens": 5, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write5 + write1h,
            "cache_creation": {"ephemeral_5m_input_tokens": write5, "ephemeral_1h_input_tokens": write1h}}


def body(messages, ttl=None, tools=TOOLS, system=SYSTEM):
    cc = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
    sys_ = copy.deepcopy(system)
    sys_[-1]["cache_control"] = cc
    return {"model": "claude-haiku-5-5", "max_tokens": 64, "tools": tools, "system": sys_, "messages": messages}


def turn(p, b, session="s"):
    res, entry = p.handle("anthropic", b, {"x-prefixwise-session-id": session})
    return entry


def back_in_time(p, seconds, session="s", model="claude-haiku-5-5"):
    p.lane_seen[(session, model)] -= seconds
    p.last_seen[session] -= seconds


USER = {"role": "user", "content": "start"}


def test_ttl_shortened_promotes_a_routing_miss_to_ttl_expired():
    p = proxy([a_usage(write5=6000), a_usage(write5=6100)])
    turn(p, body([USER], ttl="1h"))
    back_in_time(p, 600)  # 10 minutes: within the hour asked for, past the 5 minutes granted
    e = turn(p, body([USER, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}], ttl="1h"))
    assert e.cause == "ttl_expired" and e.detail == "ttl_shortened" and e.legit
    assert "asked for a 1-hour cache" in e.evidence


def test_a_granted_hour_is_not_called_shortened():
    p = proxy([a_usage(write1h=6000), a_usage(write1h=6100)])
    turn(p, body([USER], ttl="1h"))
    back_in_time(p, 600)
    e = turn(p, body([USER, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}], ttl="1h"))
    assert e.detail != "ttl_shortened"


def blocks(n, start=0):
    return [{"type": "text", "text": f"tool output {i}"} for i in range(start, start + n)]


def test_lookback_exceeded_is_a_misplaced_breakpoint():
    first = [{"role": "user", "content": blocks(1)[:0] + [{"type": "text", "text": "go", "cache_control": {"type": "ephemeral"}}]}]
    p = proxy([a_usage(write5=6000), a_usage(write5=6500), a_usage(read=6000, write5=500)])
    turn(p, body(first))
    # 30 new blocks in one turn, breakpoint only on the last: the previous write is out of reach.
    many = blocks(30)
    many[-1]["cache_control"] = {"type": "ephemeral"}
    second = [{"role": "user", "content": [{"type": "text", "text": "go"}]},
              {"role": "assistant", "content": "working"}, {"role": "user", "content": many}]
    e = turn(p, body(second))
    assert e.cause == "breakpoint_misplaced" and e.detail == "lookback_exceeded" and e.legit is False
    assert "Add a breakpoint at the start of each turn" in e.evidence
    # A short turn within reach that hits is left alone.
    third = second + [{"role": "assistant", "content": "done"},
                      {"role": "user", "content": [{"type": "text", "text": "and?", "cache_control": {"type": "ephemeral"}}]}]
    assert turn(p, body(third)).detail is None


def test_markers_ignored_when_nothing_is_written_or_read():
    p = proxy([a_usage(fresh=6000)])
    e = turn(p, body([USER]))
    assert e.cause == "first_write" and e.detail == "markers_ignored"
    p = proxy([a_usage(write5=6000)])
    assert turn(p, body([USER])).detail is None  # written: the markers worked


def test_sent_before_cache_written_and_shared_prefix_not_reused():
    p = proxy([a_usage(write5=6000), a_usage(write5=6000), a_usage(fresh=200, write5=6000)])
    # Session A is still in flight when B is sent (a parallel fan-out).
    ctx_a = p.prepare("anthropic", body([USER]), {"x-prefixwise-session-id": "A"})
    e = turn(p, body([USER]), session="B")
    assert e.cause == "first_write" and e.detail == "sent_before_cache_written"
    p.finalize_safely(ctx_a, 200, {"id": "msg_a", "usage": a_usage(write5=6000)})
    # A finished; C starts with the same tools and system prompt yet reads nothing.
    e = turn(p, body([USER]), session="C")
    assert e.detail == "shared_prefix_not_reused"


def test_thinking_stripped_from_history():
    p = proxy([a_usage(write5=6000), a_usage(write5=6500)])
    think = {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "x"},
                                              {"type": "text", "text": "answer"}]}
    turn(p, body([USER, think, {"role": "user", "content": "next"}]))
    stripped = {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
    e = turn(p, body([USER, stripped, {"role": "user", "content": "next"}, {"role": "assistant", "content": "a"},
                      {"role": "user", "content": "more"}]))
    assert e.cause == "history_edit" and e.detail == "thinking_stripped"


def test_openai_cache_key_changed():
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    usages = [{"prompt_tokens": 5000, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}] * 2
    q = list(usages)
    p._forward = lambda *a, **k: (200, {"id": "chatcmpl-x", "usage": q.pop(0)})
    msgs = [{"role": "system", "content": "Be careful. " * 400}, {"role": "user", "content": "hi"}]
    p.handle("openai", {"model": "gpt-5.6", "messages": msgs, "prompt_cache_key": "a"}, {"x-prefixwise-session-id": "s"})
    res, e = p.handle("openai", {"model": "gpt-5.6", "messages": msgs + [{"role": "assistant", "content": "x"},
                                                                         {"role": "user", "content": "y"}],
                                 "prompt_cache_key": "b"}, {"x-prefixwise-session-id": "s"})
    assert e.cause == "routing_miss" and e.detail == "cache_key_changed"


def test_an_ordinary_hit_has_no_detail():
    p = proxy([a_usage(write5=6000), a_usage(read=6000, write5=100)])
    turn(p, body([USER]))
    e = turn(p, body([USER, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]))
    assert e.cause == "none" and e.detail is None


def test_details_reach_the_report_and_the_api():
    from prefixwise import api, findings
    p = proxy([a_usage(fresh=6000), a_usage(fresh=6000)])
    turn(p, body([USER]), session="x")
    turn(p, body([USER]), session="y")
    f = findings.findings(p.ledger)
    first = next(e for e in f["expected"] if e["cause"] == "first_write")
    assert first["details"][0]["detail"] == "markers_ignored" and first["details"][0]["requests"] == 2
    assert "check that no gateway strips the markers" in findings.render(f)
    status, out = api.get(p, "/api/v1/breakdown", "by=detail")
    assert status == 200 and any(r["detail"] == "markers_ignored" for r in out["rows"])
    status, out = api.get(p, "/api/v1/requests", "detail=markers_ignored")
    assert status == 200 and len(out["requests"]) == 2


def test_markers_ignored_needs_a_breakpoint_covering_the_prompt():
    # A breakpoint on a short first tool only: that prefix may be below the minimum, so no claim.
    tools = [dict(TOOLS[0], cache_control={"type": "ephemeral"})] + TOOLS[1:]
    b = {"model": "claude-haiku-5-5", "max_tokens": 64, "tools": tools, "system": SYSTEM,
         "messages": [USER, {"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]}
    p = proxy([a_usage(fresh=6000)])
    assert turn(p, b).detail is None


def test_thinking_stripped_needs_a_real_miss():
    p = proxy([a_usage(write5=6000), a_usage(read=6000, write5=500)])
    think = {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "x"},
                                              {"type": "text", "text": "answer"}]}
    turn(p, body([USER, think, {"role": "user", "content": "next"}]))
    stripped = {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
    e = turn(p, body([USER, stripped, {"role": "user", "content": "next"}]))
    assert e.detail is None  # the cache served it
