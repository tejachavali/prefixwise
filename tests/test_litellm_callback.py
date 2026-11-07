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
    assert out[1][5:7] == (2000, 1000)  # LiteLLM's prompt_tokens includes cache reads: not counted twice


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


def test_streams_pass_through_unrecorded(base, cb):
    chunks = list(call(base, "openai/gpt-5.6", [{"role": "user", "content": "hi"}], stream=True))
    assert chunks
    time.sleep(0.5)
    assert rows(cb, 1, timeout=0.5) == []


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
