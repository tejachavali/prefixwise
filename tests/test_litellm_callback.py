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
