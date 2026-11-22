"""PrefixwiseCallback: real LiteLLM calls to a mock server; the ledger shows what was recorded."""

import asyncio
import threading
import time

import pytest

litellm = pytest.importorskip("litellm")

from prefixwise.litellm_callback import PrefixwiseCallback
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler
from prefixwise.sdk import Recorder

SYSTEM = "You are careful. " * 300


@pytest.fixture(scope="module")
def base():
    srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(PrefixwiseProxy(mock=True)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(scope="module")
def _registered():
    # LiteLLM copies `callbacks` into its own lists on first use: register once, swap the ledger per test.
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    saved = litellm.callbacks
    litellm.callbacks = [c]
    yield c
    litellm.callbacks = saved


@pytest.fixture
def cb(_registered):
    _registered.recorder = Recorder(":memory:", agent_id="litellm")
    _registered._warned = False
    return _registered


def rows(cb, n, timeout=10.0):
    """LiteLLM runs success callbacks off the calling thread: wait for n rows."""
    deadline = time.monotonic() + timeout
    while True:
        with cb.recorder.ledger.reader() as c:
            out = c.execute("SELECT provider, session_id, agent_id, task_id, cause, prompt_tokens, cached_tokens "
                            "FROM entries ORDER BY id").fetchall()
        if len(out) >= n or time.monotonic() > deadline:
            return out
        time.sleep(0.05)


def call(base, model, messages, **kw):
    api_base = base if model.startswith("anthropic/") else base + "/v1"
    return litellm.completion(model=model, api_base=api_base, api_key="sk-test", messages=messages, **kw)


def test_anthropic_through_litellm(base, cb):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "hi"}]
    meta = {"task_id": "t1", "agent_id": "coder"}
    call(base, "anthropic/claude-haiku-5-5", msgs, litellm_session_id="s1", metadata=meta)
    call(base, "anthropic/claude-haiku-5-5", msgs + [{"role": "assistant", "content": "ok"},
                                                    {"role": "user", "content": "go"}],
         litellm_session_id="s1", metadata=meta)
    out = rows(cb, 2)
    assert [r[:5] for r in out] == [("anthropic", "s1", "coder", "t1", "first_write"),
                                    ("anthropic", "s1", "coder", "t1", "none")]
    assert out[1][5:7] == (2000, 2000)  # LiteLLM's prompt_tokens includes cache reads: not counted twice


def test_openai_through_litellm_session_from_metadata(base, cb):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    call(base, "openai/gpt-5.6", msgs, metadata={"session_id": "m1"})
    call(base, "openai/gpt-5.6", msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}],
         metadata={"session_id": "m1"})
    out = rows(cb, 2)
    assert [(r[0], r[1], r[2], r[4]) for r in out] == [("openai", "m1", "litellm", "first_write"),
                                                       ("openai", "m1", "litellm", "none")]


def test_session_falls_back_to_user(base, cb):
    call(base, "openai/gpt-5.6", [{"role": "user", "content": "hi"}], user="u-7")
    assert rows(cb, 1)[0][1] == "u-7"


def test_async_path(base, cb):
    async def go():
        await litellm.acompletion(model="openai/gpt-5.6", api_base=base + "/v1", api_key="sk-test",
                                  messages=[{"role": "user", "content": "hi"}], litellm_session_id="as")
        await asyncio.sleep(1)  # LiteLLM logs in a background task; a real app keeps its loop running
    asyncio.run(go())
    assert [r[1] for r in rows(cb, 1)] == ["as"]


def _stream(base, model, messages, **kw):
    """Read a LiteLLM stream to its end, sync, as an agent does."""
    return list(call(base, model, messages, stream=True, **kw))


async def _astream(base, model, messages, **kw):
    api_base = base if model.startswith("anthropic/") else base + "/v1"
    r = await litellm.acompletion(model=model, api_base=api_base, api_key="sk-test", messages=messages,
                                  stream=True, **kw)
    return [c async for c in r]


def _released(cb, timeout=5.0):
    """The kept body is let go just after the row is written (the callback's `finally`)."""
    deadline = time.monotonic() + timeout
    while cb._stream_bodies and time.monotonic() < deadline:
        time.sleep(0.02)
    return not cb._stream_bodies


def test_an_openai_stream_with_usage_is_recorded_like_a_call(base, cb):
    msgs = [{"role": "system", "content": SYSTEM + "stream"}, {"role": "user", "content": "hi"}]
    usage = {"stream_options": {"include_usage": True}}
    assert _stream(base, "openai/gpt-5.6", msgs, litellm_session_id="st", **usage)
    assert _stream(base, "openai/gpt-5.6", msgs + [{"role": "assistant", "content": "ok"},
                                                   {"role": "user", "content": "more"}],
                   litellm_session_id="st", **usage)
    got = rows(cb, 2)
    assert [r[4] for r in got] == ["first_write", "none"] and got[1][6] > 0  # the second read the cache


def test_an_async_openai_stream_is_recorded(base, cb):
    msgs = [{"role": "system", "content": SYSTEM + "astream"}, {"role": "user", "content": "hi"}]
    asyncio.run(_astream(base, "openai/gpt-5.6", msgs, litellm_session_id="ast",
                         stream_options={"include_usage": True}))
    assert causes(cb, 1) == ["first_write"]


def test_a_stream_without_usage_is_not_recorded_and_says_why(base, cb, capsys):
    """OpenAI-format streams carry usage only when asked: zero prompt tokens is not a miss."""
    cb._warned_stream_usage = False
    for _ in range(2):
        assert _stream(base, "openai/gpt-5.6", [{"role": "user", "content": "hi"}], litellm_session_id="nu")
    time.sleep(0.5)
    assert rows(cb, 1, timeout=0.5) == []
    assert capsys.readouterr().err.count('stream_options={"include_usage": True}') == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_claude_stream_is_recorded_with_the_body_litellm_sent(base, cb, asynchronous):
    """Claude streams carry usage. The body LiteLLM sent is classified: on a sync stream it is the one
    logged before the call (the success callback has none), on an async one the JSON LiteLLM sent."""
    bodies = []
    record = cb.recorder.record
    cb.recorder.record = lambda provider, body, *a, **k: (bodies.append(body), record(provider, body, *a, **k))[1]
    msgs = [{"role": "system", "content": SYSTEM + f"claude stream {asynchronous}"}, {"role": "user", "content": "hi"}]
    kw = dict(litellm_session_id=f"cs{asynchronous}",
              cache_control_injection_points=[{"location": "message", "role": "system"}])
    if asynchronous:
        asyncio.run(_astream(base, "anthropic/claude-haiku-5-5", msgs, **kw))
    else:
        _stream(base, "anthropic/claude-haiku-5-5", msgs, **kw)
    assert causes(cb, 1) == ["first_write"]
    assert bodies[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}  # the injected breakpoint
    assert _released(cb)  # nothing kept once recorded


def test_recording_never_breaks_litellm(base, cb, capsys):
    cb.recorder.record = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire"))
    for _ in range(2):
        r = call(base, "openai/gpt-5.6", [{"role": "user", "content": "hi"}])
        assert r.usage.prompt_tokens == 2000
    time.sleep(0.5)
    assert capsys.readouterr().err.count("LiteLLM call not recorded") == 1


def test_proxy_server_config_loads_one_shared_callback(tmp_path, monkeypatch):
    """config.yaml `callbacks: prefixwise.litellm_callback.prefixwise_callback`, loaded the way the
    LiteLLM proxy server loads it."""
    get_instance_fn = pytest.importorskip("litellm.proxy.types_utils.utils").get_instance_fn
    import prefixwise.litellm_callback as mod
    monkeypatch.setenv("PREFIXWISE_LEDGER_PATH", str(tmp_path / "ledger.db"))
    vars(mod).pop("prefixwise_callback", None)  # not delattr: hasattr would create it here
    first = get_instance_fn("prefixwise.litellm_callback.prefixwise_callback")
    assert isinstance(first, PrefixwiseCallback) and first.recorder.agent_id == "litellm"
    assert get_instance_fn("prefixwise.litellm_callback.prefixwise_callback") is first
    assert (tmp_path / "ledger.db").exists()


def causes(cb, n):
    return [r[4] for r in rows(cb, n)]


def test_answers_from_litellms_own_cache_are_not_recorded(base, cb):
    from litellm.caching.caching import Cache
    saved = litellm.cache
    litellm.cache = Cache(type="local")
    try:
        msgs = [{"role": "user", "content": "cache me " + SYSTEM}]
        for _ in range(3):
            call(base, "anthropic/claude-haiku-5-5", msgs, caching=True, litellm_session_id="lc")
    except ImportError as exc:  # LiteLLM's response cache needs its proxy-server extras
        pytest.skip(f"LiteLLM's response cache is not installed here ({exc.name}); "
                    "test_a_cache_hit_is_never_recorded covers the callback")
    finally:
        litellm.cache = saved
    time.sleep(0.5)
    assert causes(cb, 1) == ["first_write"]  # one call reached the provider, not three


def test_parameters_and_injected_breakpoints_reach_the_classifier(base, cb):
    """With real LiteLLM: the recorded body is the one it sent, `thinking` and the breakpoint
    it injected (cache_control_injection_points) included."""
    bodies = []
    record = cb.recorder.record
    cb.recorder.record = lambda provider, body, *a, **k: (bodies.append(body), record(provider, body, *a, **k))[1]
    call(base, "anthropic/claude-haiku-5-5", [{"role": "system", "content": SYSTEM + "params"},
                                              {"role": "user", "content": "hi"}],
         thinking={"type": "enabled", "budget_tokens": 2048},
         cache_control_injection_points=[{"location": "message", "role": "system"}])
    deadline = time.monotonic() + 10
    while not bodies and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "thinking" in bodies[0]
    assert bodies[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_a_changed_tool_result_is_named_as_such(base, cb):
    tool_turn = [{"role": "assistant", "content": None, "tool_calls": [
                     {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "t1", "content": "file contents " * 200}]
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {}}}}]
    head = [{"role": "system", "content": [{"type": "text", "text": SYSTEM + "tools",
                                            "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "read it"}]
    call(base, "anthropic/claude-haiku-5-5", head + tool_turn, tools=tools, litellm_session_id="tr")
    edited = [tool_turn[0], {**tool_turn[1], "content": "[cleared]"}]
    call(base, "anthropic/claude-haiku-5-5", head + edited + [{"role": "user", "content": "next"}],
         tools=tools, litellm_session_id="tr")
    assert causes(cb, 2)[1] == "tool_result_mutation"


class _Spy:
    def __init__(self):
        self.calls = []

    def record(self, provider, body, response, **kw):
        self.calls.append((provider, body, response, kw))


def _kwargs(**over):
    kw = {"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "hi"}],
          "litellm_params": {"custom_llm_provider": "anthropic", "metadata": {}}, "call_type": "completion"}
    kw.update(over)
    return kw


def _response(usage, model="claude-haiku-5-5"):
    return {"id": "r1", "model": model, "usage": usage}


def test_the_body_litellm_sent_is_what_is_classified():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    sent = {"model": "claude-haiku-5-5", "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "thinking": {"type": "adaptive"}}
    c._record(_kwargs(additional_args={"complete_input_dict": sent}),
              _response({"prompt_tokens": 10, "completion_tokens": 1}), None, None)
    provider, body, _, _ = spy.calls[0]
    assert provider == "anthropic" and body == sent  # injected breakpoints and every parameter included


def test_claude_on_bedrock_is_classified_by_anthropics_rules():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    usage = {"prompt_tokens": 3010, "completion_tokens": 5, "cache_creation_input_tokens": 3000,
             "cache_read_input_tokens": 0,
             "prompt_tokens_details": {"cache_creation_token_details": {"ephemeral_5m_input_tokens": 0,
                                                                        "ephemeral_1h_input_tokens": 3000}}}
    c._record(_kwargs(model="bedrock/anthropic.claude-haiku-5-5",
                      litellm_params={"custom_llm_provider": "bedrock", "metadata": {"session_id": "b1"}}),
              _response(usage, "anthropic.claude-haiku-5-5"), None, None)
    provider, body, response, kw = spy.calls[0]
    assert provider == "anthropic" and body["model"] == "anthropic.claude-haiku-5-5"
    assert response["usage"] == {"input_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 3000,
                                 "output_tokens": 5, "cache_creation": {"ephemeral_5m_input_tokens": 0,
                                                                        "ephemeral_1h_input_tokens": 3000}}
    assert kw["genai_provider"] == "aws.bedrock" and kw["session_id"] == "b1" and not kw["session_is_user"]


def test_redacted_messages_without_a_body_are_not_recorded(capsys):
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    for _ in range(2):
        c._record(_kwargs(messages=[{"role": "user", "content": "redacted-by-litellm"}]),
                  _response({"prompt_tokens": 10, "completion_tokens": 1}), None, None)
    assert spy.calls == [] and capsys.readouterr().err.count("redacted") == 1


def test_a_user_id_is_a_session_but_not_a_conversation():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    c._record(_kwargs(model="gpt-5.6", user="alice", litellm_params={"custom_llm_provider": "groq", "metadata": {}}),
              _response({"prompt_tokens": 10, "completion_tokens": 1}, "gpt-5.6"), None, None)
    provider, _, _, kw = spy.calls[0]
    assert provider == "openai" and kw["session_id"] == "alice" and kw["session_is_user"]
    assert kw["genai_provider"] == "groq"


def test_a_cache_hit_is_never_recorded():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    c._record(_kwargs(cache_hit=True), _response({"prompt_tokens": 10, "completion_tokens": 1}), None, None)
    c._record(_kwargs(cache_hit=None), _response({"prompt_tokens": 10, "completion_tokens": 1}), None, None)
    assert len(spy.calls) == 1


# LiteLLM runs sync success callbacks on a thread pool: a session's turns can arrive out of order
# (seen in CI on Python 3.10: turn 1, compared with turn 2, read as rewritten history).

def _kw(call, messages):
    return {"litellm_call_id": call, "call_type": "completion", "model": "gpt-5.6", "messages": messages,
            "litellm_params": {"litellm_session_id": "race", "custom_llm_provider": "openai"}}


def _resp(prompt, cached):
    return {"id": "r", "model": "gpt-5.6", "usage": {"prompt_tokens": prompt, "completion_tokens": 1,
                                                    "prompt_tokens_details": {"cached_tokens": cached}}}


def test_turns_are_recorded_in_send_order_when_callbacks_race():
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    first = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    second = first + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
    c.log_pre_api_call("gpt-5.6", first, _kw("c1", first))
    c.log_pre_api_call("gpt-5.6", second, _kw("c2", second))
    late = threading.Thread(target=c.log_success_event, args=(_kw("c2", second), _resp(2000, 1800), None, None))
    late.start()  # turn 2's callback first: it must wait for turn 1's
    time.sleep(0.2)
    assert rows(c, 0) == []
    c.log_success_event(_kw("c1", first), _resp(1800, 0), None, None)
    late.join(5)
    assert [r[4] for r in rows(c, 2)] == ["first_write", "none"]


def test_a_failed_or_lost_call_does_not_hold_the_session():
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    c.log_pre_api_call("gpt-5.6", msgs, _kw("failed", msgs))
    c.log_pre_api_call("gpt-5.6", msgs, _kw("ok", msgs))
    c.log_failure_event(_kw("failed", msgs), None, None, None)
    t0 = time.monotonic()
    c.log_success_event(_kw("ok", msgs), _resp(1800, 0), None, None)
    assert time.monotonic() - t0 < 1 and [r[4] for r in rows(c, 1)] == ["first_write"]
    assert c._turns == {} and not c._key  # nothing left behind


def test_other_conversations_never_hold_a_turn():
    # Parallel subagents need no order.
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    other = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "another task"}]
    c.log_pre_api_call("gpt-5.6", other, _kw("subagent", other))  # in flight, never completes here
    c.log_pre_api_call("gpt-5.6", msgs, _kw("turn", msgs))
    t0 = time.monotonic()
    c.log_success_event(_kw("turn", msgs), _resp(1800, 0), None, None)
    assert time.monotonic() - t0 < 0.5 and len(rows(c, 1)) == 1


def test_a_stream_abandoned_unread_holds_the_next_turn_briefly_and_is_let_go():
    """A stream the client stops reading may never send a callback: like any lost call, the next turn
    of its conversation waits at most ORDER_WAIT_SECONDS, and the stream's kept body goes with it."""
    import prefixwise.litellm_callback as mod
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    c.log_pre_api_call("gpt-5.6", msgs, {**_kw("stream", msgs), "stream": True,
                                         "additional_args": {"complete_input_dict": {"model": "gpt-5.6",
                                                                                     "messages": msgs}}})
    assert "stream" in c._stream_bodies
    c.log_pre_api_call("gpt-5.6", msgs, _kw("turn", msgs))
    t0 = time.monotonic()
    c.log_success_event(_kw("turn", msgs), _resp(1800, 0), None, None)
    assert time.monotonic() - t0 < mod.ORDER_WAIT_SECONDS + 0.5 and len(rows(c, 1)) == 1
    c.log_pre_api_call("gpt-5.6", msgs, _kw("later", msgs))  # long after: the lost stream is forgotten
    with c._order:
        c._sweep(time.monotonic() + mod.ORDER_FORGET_SECONDS + 1)
    assert "stream" not in c._stream_bodies


def test_abandoned_calls_are_forgotten_and_tracking_is_capped(monkeypatch):
    import prefixwise.litellm_callback as mod

    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    msgs = [{"role": "user", "content": "hi"}]
    clock = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    for i in range(50):  # one-call sessions whose callback never came
        c.log_pre_api_call("gpt-5.6", msgs, {**_kw(f"lost{i}", msgs),
                                             "litellm_params": {"litellm_session_id": f"s{i}"}})
    clock[0] += mod.ORDER_FORGET_SECONDS + 1
    c.log_pre_api_call("gpt-5.6", msgs, _kw("fresh", msgs))
    assert list(c._key) == ["fresh"] and len(c._turns) == 1
    monkeypatch.setattr(mod, "ORDER_MAX_CALLS", 10)
    for i in range(30):
        c.log_pre_api_call("gpt-5.6", msgs, {**_kw(f"x{i}", msgs), "litellm_params": {"litellm_session_id": f"y{i}"}})
    assert len(c._key) <= 10


def test_a_long_generation_keeps_its_place(monkeypatch):
    # A turn that generates for a minute is still waited for: it becomes due only when the next
    # turn of its conversation is sent, and is forgotten only if its callback is late after that.
    import prefixwise.litellm_callback as mod

    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    msgs = [{"role": "user", "content": "hi"}]
    clock = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    c.log_pre_api_call("gpt-5.6", msgs, _kw("long", msgs))
    clock[0] += 60
    c.log_pre_api_call("gpt-5.6", msgs, {**_kw("noise", msgs), "litellm_params": {"litellm_session_id": "other"}})
    c.log_pre_api_call("gpt-5.6", msgs, _kw("next", msgs))
    with c._order:
        assert not c._first(_kw("next", msgs))  # still behind the long turn
    clock[0] += 2 * mod.ORDER_WAIT_SECONDS + 1
    c.log_pre_api_call("gpt-5.6", msgs, _kw("third", msgs))  # the long turn's callback never came
    assert "long" not in c._key


def test_a_cancelled_async_callback_releases_its_turn():
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    msgs = [{"role": "user", "content": "hi"}]
    c.log_pre_api_call("gpt-5.6", msgs, _kw("a", msgs))
    c.log_pre_api_call("gpt-5.6", msgs, _kw("b", msgs))

    async def go():
        waiting = asyncio.ensure_future(c.async_log_success_event(_kw("b", msgs), _resp(1800, 0), None, None))
        await asyncio.sleep(0.05)  # b waits for a
        waiting.cancel()  # LiteLLM's logging worker cancels slow callbacks
        with pytest.raises(asyncio.CancelledError):
            await waiting
    asyncio.run(go())
    assert "b" not in c._key and "a" in c._key


def test_conversation_key_survives_moved_markers_and_content_shape():
    # Review: clients put cache_control on the last message, which on turn 1 is the opening one;
    # and content can be a string one turn and text blocks the next.
    from prefixwise.litellm_callback import _conversation

    kw = _kw("x", [])
    marked = [{"role": "user", "content": [{"type": "text", "text": "fix the bug",
                                            "cache_control": {"type": "ephemeral"}}]}]
    plain = [{"role": "user", "content": "fix the bug"},
             {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
    blocks = [{"role": "user", "content": [{"type": "text", "text": "fix the bug"}]}]
    assert _conversation("m", marked, kw) == _conversation("m", plain, kw) == _conversation("m", blocks, kw)
    assert _conversation("m", [{"role": "user", "content": "another"}], kw) != _conversation("m", plain, kw)


# Review of the stream support: LiteLLM's own estimate is not usage, and every stream keeps its body.

def _stream_kw(**over):
    kw = {"litellm_call_id": "s1", "call_type": "completion", "model": "gpt-5.6", "stream": True,
          "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}],
          "litellm_params": {"litellm_session_id": "est", "custom_llm_provider": "openai"}}
    kw.update(over)
    return kw


def _estimate(kw):
    return litellm.token_counter(model=kw["model"], messages=kw["messages"])


def test_a_stream_whose_usage_was_not_asked_for_is_not_recorded(capsys):
    """No usage in the stream: LiteLLM fills in its tokenizer's estimate, which would read as a miss."""
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    kw = _stream_kw()  # no stream_options
    c._record(kw, {"id": "r", "model": "gpt-5.6", "usage": {"prompt_tokens": _estimate(kw), "completion_tokens": 3}},
              None, None)
    assert spy.calls == [] and "LiteLLM estimated it" in capsys.readouterr().err


def test_a_stream_asked_for_usage_that_got_litellms_estimate_is_not_recorded():
    """Asked for usage, but the server ignored stream_options: the count is LiteLLM's estimate."""
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    kw = _stream_kw(stream_options={"include_usage": True})
    c._record(kw, {"id": "r", "model": "gpt-5.6", "usage": {"prompt_tokens": _estimate(kw), "completion_tokens": 3}},
              None, None)
    assert spy.calls == []
    c._record(kw, {"id": "r", "model": "gpt-5.6", "usage": {"prompt_tokens": _estimate(kw) + 7, "completion_tokens": 3}},
              None, None)  # the provider's own count differs from LiteLLM's tokenizer
    c._record(kw, {"id": "r", "model": "gpt-5.6", "usage": {"prompt_tokens": _estimate(kw), "completion_tokens": 3,
                                                            "prompt_tokens_details": {"cached_tokens": 64}}},
              None, None)  # an estimate never reports cached tokens
    assert len(spy.calls) == 2


def test_openais_own_count_is_believed_even_when_it_equals_litellms_tokenizer():
    """OpenAI's prompt count can equal LiteLLM's tiktoken estimate exactly; its usage carries
    prompt_tokens_details even with nothing cached, which the estimate never does. A first turn or
    a real miss must not be skipped."""
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    kw = _stream_kw(stream_options={"include_usage": True})
    c._record(kw, {"id": "r", "model": "gpt-5.6", "usage": {"prompt_tokens": _estimate(kw), "completion_tokens": 3,
                                                            "prompt_tokens_details": {"cached_tokens": 0}}},
              None, None)
    assert len(spy.calls) == 1


def test_a_claude_stream_with_its_cache_fields_is_believed():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    c._record(_kwargs(stream=True), _response({"prompt_tokens": 9, "completion_tokens": 1,
                                               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}),
              None, None)
    assert len(spy.calls) == 1


def test_a_sessionless_sync_claude_stream_keeps_the_body_litellm_sent(base, cb):
    """Review: a sync stream's success callback has no body; without a session the body logged before
    the call was not kept, the injected breakpoint was lost, and the miss could be blamed on it."""
    bodies = []
    record = cb.recorder.record
    cb.recorder.record = lambda provider, body, *a, **k: (bodies.append(body), record(provider, body, *a, **k))[1]
    msgs = [{"role": "system", "content": SYSTEM + "sessionless"}, {"role": "user", "content": "hi"}]
    _stream(base, "anthropic/claude-haiku-5-5", msgs,
            cache_control_injection_points=[{"location": "message", "role": "system"}])
    deadline = time.monotonic() + 10
    while not bodies and time.monotonic() < deadline:
        time.sleep(0.05)
    assert bodies and bodies[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert _released(cb)


def test_the_json_litellm_sent_on_an_anthropic_stream_is_read():
    """An async Anthropic stream carries the body it sent as a JSON string."""
    import json
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    sent = {"model": "claude-haiku-5-5", "stream": True,
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    c._record(_kwargs(stream=True, additional_args={"complete_input_dict": json.dumps(sent)}),
              _response({"prompt_tokens": 900, "completion_tokens": 1, "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0}), None, None)
    assert spy.calls and spy.calls[0][1] == sent


def test_kept_stream_bodies_are_let_go_after_the_forget_time():
    import prefixwise.litellm_callback as mod
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    body = {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]}
    c.log_pre_api_call("gpt-5.6", body["messages"], {"litellm_call_id": "old", "stream": True, "call_type": "completion",
                                                     "additional_args": {"complete_input_dict": body}})
    assert "old" in c._stream_bodies  # kept although the call has no session
    c._stream_bodies["old"] = (body, time.monotonic() - mod.ORDER_FORGET_SECONDS - 1)
    c.log_pre_api_call("gpt-5.6", body["messages"], {"litellm_call_id": "new", "stream": True, "call_type": "completion",
                                                     "additional_args": {"complete_input_dict": body}})
    assert list(c._stream_bodies) == ["new"]


def test_the_estimate_check_never_reaches_the_network(monkeypatch):
    """Review: token_counter fetches high-detail image URLs to size them and downloads Hugging Face
    tokenizers (llama-3 and others), inside LiteLLM's logging and on the event loop for async calls.
    Neither happens: such an image prompt is not counted (and not recorded), such a model's count is
    taken as reported."""
    import litellm.utils as lu
    seen = []

    def counter(**kw):
        seen.append(kw)
        return 10

    monkeypatch.setattr(litellm, "token_counter", counter)
    images = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/c.png",
                                                                            "detail": "high"}}]}]
    assert PrefixwiseCallback._stream_usage_reported({"prompt_tokens": 99}, "gpt-5.6", images, 99, {}) is False
    llama = "meta-llama/Llama-3.1-8B-Instruct"
    assert lu.huggingface_tokenizer_kind(llama) is not None
    hi = [{"role": "user", "content": "hi"}]
    # Review: not counted, so believed only when usage was asked for (unasked, a server sends none
    # and what is there is LiteLLM's estimate).
    assert PrefixwiseCallback._stream_usage_reported({"prompt_tokens": 10}, llama, hi, 10, {}, {}) is False
    assert PrefixwiseCallback._stream_usage_reported({"prompt_tokens": 10}, llama, hi, 10, {},
                                                     {"stream_options": {"include_usage": True}}) is True
    assert PrefixwiseCallback._stream_usage_reported({"prompt_tokens": 10}, llama, hi, 10, {},
                                                     {"optional_params": {"stream_options": {"include_usage": True}}}) is True
    assert seen == []  # never counted: no image fetched, no tokenizer downloaded


def test_an_image_counted_locally_is_compared_as_litellm_counts_it():
    """Review: counting images at a default size differed from LiteLLM's own estimate, so an estimate
    with an image passed as real usage. Auto-detail and base64 images are counted as LiteLLM does."""
    import base64
    png = "data:image/png;base64," + base64.b64encode(
        bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                      "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")).decode()
    for url in (png, "https://example.com/c.png"):  # auto detail: no fetch
        msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                             {"type": "image_url", "image_url": {"url": url}}]}]
        estimate = litellm.token_counter(model="gpt-5.6", messages=msgs)
        assert PrefixwiseCallback._stream_usage_reported({"prompt_tokens": estimate}, "gpt-5.6", msgs, estimate, {}) \
            is False
        assert PrefixwiseCallback._stream_usage_reported({"prompt_tokens": estimate + 3}, "gpt-5.6", msgs,
                                                         estimate + 3, {}) is True


def test_a_high_detail_image_inside_a_tool_result_is_not_fetched(monkeypatch):
    """Review: the counter walks a tool_result's own content list, so a nested image must count too."""
    seen = []
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: seen.append(kw) or 10)
    nested = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/c.png", "detail": "high"}}]}]}]
    assert PrefixwiseCallback._stream_usage_reported({"prompt_tokens": 800}, "gpt-4o", nested, 800, {}) is False
    assert seen == []


# LiteLLM's agent_eval labels each model call of an evaluation run (agent_eval_run, agent_eval_task,
# agent_eval_target). Without them every run fell into one session, and each new task or trial was
# compared with the last call of the one before and named history_edit (checked with a real
# agent_eval run on LiteLLM 1.112.0.dev2: 5 of 12 calls).

def _eval_kw(where, run, task="task0", target="tool_loop:openai/gpt-5.6", call="c1", msgs=None):
    labels = {"agent_eval_suite": "s", "agent_eval_run": run, "agent_eval_task": task,
              "agent_eval_target": target, "agent_eval_trial": 0}
    meta = labels if where == "metadata" else {"spend_logs_metadata": labels}
    return {"litellm_call_id": call, "call_type": "acompletion", "model": "gpt-5.6",
            "messages": msgs or [{"role": "user", "content": "Task 0: read the note"}],
            "litellm_params": {"custom_llm_provider": "openai", "metadata": meta}}


@pytest.mark.parametrize("where", ["metadata", "spend_logs_metadata"])
def test_an_agent_eval_run_is_its_session_with_its_task_and_target(where):
    """Direct (the call's metadata) and through the LiteLLM proxy server (spend_logs_metadata)."""
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    c._record(_eval_kw(where, "run-a"), _resp(1800, 0), None, None)
    _, _, _, kw = spy.calls[0]
    assert kw["session_id"] == "run-a" and kw["task_id"] == "task0"
    assert kw["agent_id"] == "tool_loop:openai/gpt-5.6" and kw["session_is_user"] is False


def test_two_trials_of_one_task_are_two_conversations_not_an_edit():
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    first = [{"role": "user", "content": "Task 0: read the note"}]
    longer = first + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
    for run, call, msgs, cached in (("run-a", "a1", first, 0), ("run-a", "a2", longer, 1800),
                                    ("run-b", "b1", first, 0)):
        kw = _eval_kw("spend_logs_metadata", run, call=call, msgs=msgs)
        c.log_pre_api_call("gpt-5.6", msgs, kw)
        c.log_success_event(kw, _resp(1800, cached), None, None)
    assert [r[4] for r in rows(c, 3)] == ["first_write", "none", "first_write"]


def test_a_session_the_caller_named_wins_over_the_evaluation_run():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    kw = _eval_kw("metadata", "run-a")
    kw["litellm_params"]["litellm_session_id"] = "mine"
    c._record(kw, _resp(1800, 0), None, None)
    assert spy.calls[0][3]["session_id"] == "mine"


def test_the_eval_judge_is_recorded_apart_from_the_agent():
    """Review: agent_eval's judge (checks.py, agent_eval_stage "judge") carries the run's labels; its
    call is not the target agent's turn and must not join the agent's conversation."""
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    kw = _eval_kw("metadata", "run-a")
    kw["litellm_params"]["metadata"]["agent_eval_stage"] = "judge"
    c._record(kw, _resp(1800, 0), None, None)
    _, _, _, rec = spy.calls[0]
    assert rec["session_id"] == "run-a:judge" and rec["agent_id"] == "agent_eval_judge"
    assert rec["task_id"] == "task0"


# LiteLLM's Anthropic Messages API: litellm.anthropic.messages, and the proxy server's /v1/messages
# (Claude Code's route). The body LiteLLM sent is logged before the call and is gone by the success
# callback, streamed or not.

def _messages_kw(call="m1", sent=None, **over):
    kw = {"litellm_call_id": call, "call_type": "anthropic_messages", "model": "claude-haiku-5-5",
          "messages": [{"role": "user", "content": "hi"}],
          "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
          "litellm_params": {"custom_llm_provider": "anthropic", "metadata": {}, "litellm_session_id": "cc1"},
          "additional_args": {"complete_input_dict": sent if sent is not None else {}}}
    kw.update(over)
    return kw


def test_a_messages_api_call_is_classified_with_the_body_logged_before_it():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    sent = {"model": "claude-haiku-5-5", "max_tokens": 50, "system": [{"type": "text", "text": "s"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi",
                                                      "cache_control": {"type": "ephemeral"}}]}]}
    c.log_pre_api_call("claude-haiku-5-5", [], _messages_kw(sent=sent))
    assert "m1" in c._stream_bodies  # kept although not streamed
    usage = {"prompt_tokens": 1716, "completion_tokens": 5, "cache_read_input_tokens": 1708,
             "cache_creation_input_tokens": 0, "prompt_tokens_details": {"cached_tokens": 1708}}
    c.log_success_event(_messages_kw(), _response(usage), None, None)  # complete_input_dict cleared by then
    provider, body, response, kw = spy.calls[0]
    assert provider == "anthropic" and body == sent and kw["session_id"] == "cc1"
    assert response["usage"]["cache_read_input_tokens"] == 1708 and response["usage"]["input_tokens"] == 8
    assert not c._stream_bodies  # let go once recorded


# The bodies LiteLLM 1.112.0.dev2 sent for one Messages API request (a system prompt and the last
# message marked, a tool call and its result), captured before sending.
_CONVERSE = {"messages": [{"role": "user", "content": [{"text": "q1"}]},
                          {"role": "assistant", "content": [{"toolUse": {"input": {"cmd": "ls"}, "name": "bash", "toolUseId": "t1"}}]},
                          {"role": "user", "content": [{"toolResult": {"content": [{"text": "a.py"}], "toolUseId": "t1"}},
                                                       {"text": "next"}, {"cachePoint": {"type": "default"}}]}],
             "inferenceConfig": {"maxTokens": 20},
             "system": [{"text": "SYSTEM PROMPT"}, {"cachePoint": {"type": "default"}}],
             "toolConfig": {"tools": [{"toolSpec": {"inputSchema": {"json": {"type": "object"}}, "name": "bash",
                                                    "description": "run"}}]}}
_OPENROUTER = {"model": "anthropic/claude-sonnet-4.5",
               "messages": [{"role": "system", "content": [{"type": "text", "text": "SYSTEM PROMPT",
                                                            "cache_control": {"type": "ephemeral"}}]},
                            {"role": "user", "content": "q1"},
                            {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function",
                                                                  "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"}}]},
                            {"role": "tool", "tool_call_id": "t1", "content": "a.py"},
                            {"role": "user", "content": [{"type": "text", "text": "next", "cache_control": {"type": "ephemeral"}}]}],
               "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}, "description": "run"}}]}
_RESPONSES = {"model": "gpt-4o",
              "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "SYSTEM PROMPT"}]},
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q1"}]},
                        {"type": "function_call", "call_id": "t1", "name": "bash", "arguments": "{\"cmd\": \"ls\"}"},
                        {"type": "function_call_output", "call_id": "t1", "output": "a.py"}],
              "max_output_tokens": 20, "tools": [{"type": "function", "name": "bash", "parameters": {"type": "object"}}]}


def _marks(body):
    """Where the body carries cache_control: (part, index) pairs."""
    out = [("system", i) for i, b in enumerate(body.get("system") or []) if "cache_control" in b]
    out += [("tools", i) for i, t in enumerate(body.get("tools") or []) if "cache_control" in t]
    for n, m in enumerate(body["messages"]):
        content = m.get("content")
        out += [(f"messages[{n}]", i) for i, b in enumerate(content if isinstance(content, list) else [])
                if isinstance(b, dict) and "cache_control" in b]
    return out


def test_claude_on_bedrock_converse_is_classified_from_the_converse_body():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    usage = {"prompt_tokens": 10, "completion_tokens": 1, "cache_read_input_tokens": 0}
    c._record(_messages_kw(sent=_CONVERSE, model="converse/anthropic.claude-sonnet-4-5", system=None,
                           litellm_params={"custom_llm_provider": "bedrock", "metadata": {}}),
              _response(usage), None, None)
    provider, body, _, _ = spy.calls[0]
    assert provider == "anthropic"
    assert body["system"] == [{"type": "text", "text": "SYSTEM PROMPT", "cache_control": {"type": "ephemeral"}}]
    assert body["tools"] == [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}]
    assert [b["type"] for m in body["messages"] for b in m["content"]] == ["text", "tool_use", "tool_result", "text"]
    assert _marks(body) == [("system", 0), ("messages[2]", 1)]  # each cachePoint on the block before it


def test_claude_sent_as_a_chat_completion_keeps_its_breakpoints():
    # OpenRouter: LiteLLM translated the call to a chat completion, cache_control markers included.
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    usage = {"prompt_tokens": 10, "completion_tokens": 1,
             "prompt_tokens_details": {"cached_tokens": 4, "cache_creation_tokens": 6}}
    c._record(_messages_kw(sent=_OPENROUTER, model="anthropic/claude-sonnet-4.5", system=None,
                           litellm_params={"custom_llm_provider": "openrouter", "metadata": {}}),
              _response(usage), None, None)
    provider, body, response, _ = spy.calls[0]
    assert provider == "anthropic" and response["usage"]["cache_creation_input_tokens"] == 6
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert _marks(body)[-1][0] == f"messages[{len(body['messages']) - 1}]"  # the last message's breakpoint
    # Bedrock invoke under an inference-profile ARN: no "claude" in the name, but Anthropic's body.
    invoke = {"anthropic_version": "bedrock-2023-05-31", "system": [{"type": "text", "text": "s"}],
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    arn = "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc"
    c._record(_messages_kw(sent=invoke, model=arn, litellm_params={"custom_llm_provider": "bedrock", "metadata": {}}),
              _response(usage), None, None)
    assert spy.calls[1][:2] == ("anthropic", {**invoke, "model": arn})


def test_an_openai_model_through_the_messages_api_is_classified_as_a_responses_call():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    usage = {"prompt_tokens": 1300, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 1024}}
    c._record(_messages_kw(sent=_RESPONSES, model="gpt-4o", litellm_params={"custom_llm_provider": "openai", "metadata": {}}),
              _response(usage, "gpt-4o"), None, None)
    provider, body, response, kw = spy.calls[0]
    assert (provider, kw["api"], body) == ("openai", "responses", _RESPONSES)
    assert response["usage"] == {"input_tokens": 1300, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 1024}}


def test_a_system_prompt_sent_as_the_opening_developer_item_is_the_system_prompt():
    # LiteLLM 1.112 sends a Messages API system prompt to the Responses API as the first input item.
    rec = Recorder(":memory:")
    def turn(system, items, cached):
        body = {**_RESPONSES, "input": [{**_RESPONSES["input"][0], "content": [{"type": "input_text", "text": system}]}] + items}
        rec.record("openai", body, {"id": "r", "model": "gpt-4o", "usage": {
            "input_tokens": 3000, "output_tokens": 5, "input_tokens_details": {"cached_tokens": cached}}},
            api="responses", session_id="dev")
    history = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q " * 400}]}]
    turn("You are careful. " * 200, history, 0)
    turn("You are careful. " * 200, history + history, 2900)
    turn("You are concise. " * 200, history + history + history, 0)
    with rec.ledger.reader() as cur:
        causes = [r[0] for r in cur.execute("SELECT cause FROM entries ORDER BY id")]
    assert causes == ["first_write", "none", "system_edit"]


def test_another_model_through_the_messages_api_is_recorded_only_as_a_chat_completion():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    usage = {"prompt_tokens": 1216, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0},
             "cache_creation_input_tokens": 0}
    chat = {"model": "qwen3", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]}
    vllm = {"custom_llm_provider": "hosted_vllm", "metadata": {}, "litellm_session_id": "v1"}
    c._record(_messages_kw(sent=chat, model="qwen3", litellm_params=vllm), _response(usage, "qwen3"), None, None)
    assert [(p, b) for p, b, _, _ in spy.calls] == [("openai", chat)]
    # The provider's own Messages API (DeepSeek's): Anthropic's format, not Anthropic's cache rules,
    # with or without a system prompt (LiteLLM says which route it took).
    deepseek = {"custom_llm_provider": "deepseek", "metadata": {}, "litellm_session_id": "d1"}
    for native in ({"model": "deepseek-chat", "system": "s", "messages": [{"role": "user", "content": "hi"}]},
                   {"model": "deepseek-chat", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}):
        c._record(_messages_kw(sent=native, model="deepseek-chat", litellm_params=deepseek),
                  _response(usage, "deepseek-chat"), None, None)
    # No body at all (its kept copy let go): nothing to classify faithfully.
    c._record(_messages_kw(), _response(usage), None, None)
    assert len(spy.calls) == 1


def test_anthropics_metadata_user_id_is_the_session_when_nothing_else_names_one():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    user = "user_ab12_account_cd34_session_0f9e"  # Claude Code's: it names the session
    c._record(_messages_kw(sent={"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "hi"}]},
                           litellm_params={"custom_llm_provider": "anthropic", "metadata": {"user_id": user}}),
              _response({"prompt_tokens": 10, "completion_tokens": 1}), None, None)
    _, _, _, kw = spy.calls[0]
    assert kw["session_id"] == user and kw["session_is_user"]
    # A chat completion's metadata is LiteLLM's own: a user_id there is not taken as a session.
    c._record(_kwargs(model="gpt-5.6", litellm_params={"custom_llm_provider": "openai", "metadata": {"user_id": "alice"}}),
              _response({"prompt_tokens": 10, "completion_tokens": 1}, "gpt-5.6"), None, None)
    assert spy.calls[1][3]["session_id"] is None


def test_usage_in_anthropics_own_shape_is_read():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    usage = {"input_tokens": 8, "cache_read_input_tokens": 1708, "cache_creation_input_tokens": 20, "output_tokens": 5}
    c._record(_messages_kw(sent={"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "hi"}]}),
              _response(usage), None, None)
    _, _, response, _ = spy.calls[0]
    assert response["usage"] == {"input_tokens": 8, "cache_read_input_tokens": 1708,
                                 "cache_creation_input_tokens": 20, "output_tokens": 5}


def test_a_failed_messages_api_call_lets_its_body_go():
    c = PrefixwiseCallback(recorder=_Spy())
    c.log_pre_api_call("claude-haiku-5-5", [], _messages_kw(sent={"model": "m", "messages": []}))
    c.log_failure_event(_messages_kw(), None, None, None)
    assert not c._stream_bodies and not c._key


async def _messages(base, session, turns, stream=False, model="anthropic/claude-haiku-5-5"):
    system = [{"type": "text", "text": SYSTEM + "messages api", "cache_control": {"type": "ephemeral"}}]
    msgs = [{"role": "user", "content": "hi"}]
    api_base = base if model.startswith("anthropic/") else base + "/v1"
    for i in range(turns):
        r = await litellm.anthropic.messages.acreate(model=model, max_tokens=20, system=system, messages=msgs,
                                                     api_base=api_base, api_key="sk-test", stream=stream,
                                                     litellm_session_id=session,
                                                     cache_control_injection_points=[{"location": "message", "index": -1}])
        if stream:
            async for _ in r:
                pass
        msgs = msgs + [{"role": "assistant", "content": [{"type": "text", "text": f"ok {i}"}]},
                       {"role": "user", "content": f"go {i}"}]
    await asyncio.sleep(1)  # LiteLLM logs in a background task


@pytest.fixture
def bodies(cb, monkeypatch):
    """The bodies the callback hands the recorder."""
    seen, record = [], cb.recorder.record

    def keep(provider, body, response, **kw):
        seen.append(body)
        return record(provider, body, response, **kw)

    monkeypatch.setattr(cb.recorder, "record", keep)
    return seen


def _marked_last(body):
    content = body["messages"][-1]["content"]
    return isinstance(content, list) and "cache_control" in content[-1]


def test_litellms_messages_api_is_recorded(base, cb, bodies):
    asyncio.run(_messages(base, "msgs", 2))
    out = rows(cb, 2)
    assert [(r[0], r[1], r[4]) for r in out] == [("anthropic", "msgs", "first_write"), ("anthropic", "msgs", "none")]
    assert all(_marked_last(b) for b in bodies)  # the body LiteLLM sent: with the breakpoint it injected
    assert _released(cb)


def test_another_model_through_litellms_messages_api_is_recorded_as_sent(base, cb, bodies):
    asyncio.run(_messages(base, "msgs-chat", 2, model="hosted_vllm/qwen3"))
    out = rows(cb, 2)
    assert [(r[0], r[1], r[4]) for r in out] == [("openai", "msgs-chat", "first_write"), ("openai", "msgs-chat", "none")]
    assert bodies[0]["messages"][0]["role"] == "system"  # the chat completion LiteLLM translated it to
    assert _released(cb)


def test_a_streamed_messages_api_call_is_recorded(base, cb, bodies):
    pytest.importorskip("backoff")  # LiteLLM's Messages API streams through its proxy package
    asyncio.run(_messages(base, "msgs-stream", 2, stream=True))
    out = rows(cb, 2)
    assert [(r[0], r[1], r[4]) for r in out] == [("anthropic", "msgs-stream", "first_write"),
                                                 ("anthropic", "msgs-stream", "none")]
    assert all(_marked_last(b) for b in bodies)
    assert _released(cb)


def _stitched_causes(developer: bool):
    """Prefixwise's own path: a Responses turn opened by a developer item, then one continuing it."""
    rec = Recorder(":memory:")
    opening = [{"type": "message", "role": "developer", "content": "Sandbox: read-only. " * 50}] if developer else []
    first = {"model": "gpt-5.6", "instructions": "You are a coding agent. " * 100,
             "input": opening + [{"type": "message", "role": "user", "content": "fix it " * 300}]}
    rec.record("openai", first, {"id": "resp_1", "model": "gpt-5.6", "usage": {
        "input_tokens": 3000, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 0}}},
        api="responses", session_id="st")
    nxt = {"model": "gpt-5.6", "instructions": first["instructions"], "previous_response_id": "resp_1",
           "input": [{"type": "message", "role": "user", "content": "go on"}]}
    rec.record("openai", nxt, {"id": "resp_2", "model": "gpt-5.6", "usage": {
        "input_tokens": 3010, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 0}}},
        api="responses", session_id="st")
    with rec.ledger.reader() as cur:
        return [r[0] for r in cur.execute("SELECT cause FROM entries ORDER BY id")]


def test_a_continued_responses_turn_keeps_the_developer_items_its_input_opened_with():
    with_dev, without = _stitched_causes(True), _stitched_causes(False)
    assert with_dev == without and "system_edit" not in with_dev


def test_claude_in_anthropics_format_is_classified_as_sent_whatever_the_route():
    # A /v1/messages passthrough deployment: Anthropic's body, though LiteLLM has no native config for it.
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    sent = {"model": "claude-sonnet-4-5", "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]}
    c._record(_messages_kw(sent=sent, model="claude-sonnet-4-5", litellm_params={"custom_llm_provider": "openai", "metadata": {}}),
              _response({"prompt_tokens": 10, "completion_tokens": 1}), None, None)
    assert spy.calls[0][:2] == ("anthropic", sent)


def test_openrouter_claude_keeps_its_parameters_its_name_and_its_one_hour_writes():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    sent = {**_OPENROUTER, "thinking": {"type": "enabled", "budget_tokens": 1024}, "tool_choice": "auto",
            "messages": [{**m, "content": [{**b, "cache_control": {"type": "ephemeral", "ttl": "1h"}} for b in m["content"]]}
                         if isinstance(m.get("content"), list) else m for m in _OPENROUTER["messages"]]}
    usage = {"prompt_tokens": 10, "completion_tokens": 1,
             "prompt_tokens_details": {"cached_tokens": 4, "cache_creation_tokens": 6}}
    c._record(_messages_kw(sent=sent, model="anthropic/claude-sonnet-4.5", system=None,
                           litellm_params={"custom_llm_provider": "openrouter", "metadata": {}}),
              _response(usage, "anthropic/claude-sonnet-4.5"), None, None)
    _, body, response, _ = spy.calls[0]
    assert body["thinking"] == sent["thinking"] and body["tool_choice"] == "auto"
    assert body["model"] == "claude-sonnet-4-5" and response["model"] == "claude-sonnet-4-5"
    assert response["usage"]["cache_creation"] == {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 6}


def test_the_converse_body_keeps_reasoning_system_tools_and_every_breakpoint():
    from prefixwise.litellm_callback import _anthropic_from_converse
    body = _anthropic_from_converse({
        "messages": [{"role": "user", "content": [{"text": "a"}]},
                     {"role": "assistant", "content": [{"cachePoint": {"type": "default"}},
                                                       {"reasoningContent": {"reasoningText": {"text": "t", "signature": "s"}}},
                                                       {"reasoningContent": {"redactedContent": "eA=="}}, {"text": "x"}]}],
        "toolConfig": {"tools": [{"toolSpec": {"name": "a", "inputSchema": {"json": {}}}}, {"systemTool": {"name": "web"}},
                                 {"cachePoint": {"type": "default", "ttl": "1h"}}]},
        "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 2048}}}, "claude-sonnet-4-5")
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}  # a cachePoint opening the next message
    assert [b["type"] for b in body["messages"][1]["content"]] == ["thinking", "redacted_thinking", "text"]
    assert body["messages"][1]["content"][0] == {"type": "thinking", "thinking": "t", "signature": "s"}
    assert body["tools"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}  # on the system tool before it
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_another_model_in_converse_or_anthropics_format_is_not_recorded():
    spy = _Spy()
    c = PrefixwiseCallback(recorder=spy)
    usage = {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}
    c._record(_messages_kw(sent=_CONVERSE, model="converse/amazon.nova-pro-v1:0",
                           litellm_params={"custom_llm_provider": "bedrock", "metadata": {}}), _response(usage, "nova"), None, None)
    anthropic_body = {"model": "glm-4.6", "system": "s", "messages": [{"role": "user", "content": "hi"}]}
    c._record(_messages_kw(sent=anthropic_body, model="glm-4.6", litellm_params={"custom_llm_provider": "openai", "metadata": {}}),
              _response(usage, "glm-4.6"), None, None)
    assert spy.calls == []
