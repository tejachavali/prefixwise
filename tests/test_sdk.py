"""prefixwise.wrap: the real OpenAI and Anthropic SDKs, talking to a mock server over HTTP."""

import asyncio
import threading

import pytest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

import prefixwise
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler
from prefixwise.sdk import Recorder


@pytest.fixture(scope="module")
def server():
    srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(PrefixwiseProxy(mock=True)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def rows(rec):
    with rec.ledger.reader() as c:
        return c.execute("SELECT provider, api, session_id, agent_id, task_id, cause, prompt_tokens, "
                         "cached_tokens FROM entries ORDER BY id").fetchall()


SYSTEM = "You are careful. " * 300


def test_openai_chat_and_responses(server):
    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    with prefixwise.session("s1", task_id="t1"):
        r1 = client.chat.completions.create(model="gpt-5.6", messages=msgs)
        r2 = client.chat.completions.create(model="gpt-5.6", messages=msgs + [
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}])
    assert r1.usage.prompt_tokens == 2000 and r2.id  # the SDK's own objects, unchanged
    out = rows(rec)
    assert [(r[0], r[1], r[2], r[4], r[5]) for r in out] == [
        ("openai", "chat", "s1", "t1", "first_write"), ("openai", "chat", "s1", "t1", "none")]
    client.responses.create(model="gpt-5.6", instructions=SYSTEM, input="hi")
    assert rows(rec)[-1][1] == "responses"


def test_anthropic_messages(server):
    rec = Recorder(":memory:")
    client = prefixwise.wrap(anthropic.Anthropic(base_url=server, api_key="sk-ant-test"), recorder=rec,
                             agent_id="my-agent")
    body = dict(model="claude-haiku-5-5", max_tokens=16,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": "hi"}])
    with prefixwise.session("a1"):
        client.messages.create(**body)
        client.messages.create(**{**body, "messages": body["messages"] + [
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]})
    assert [(r[0], r[2], r[3], r[5]) for r in rows(rec)] == [
        ("anthropic", "a1", "my-agent", "first_write"), ("anthropic", "a1", "my-agent", "none")]


def test_async_clients(server):
    rec = Recorder(":memory:")
    oa = prefixwise.wrap(openai.AsyncOpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    an = prefixwise.wrap(anthropic.AsyncAnthropic(base_url=server, api_key="sk-ant-test"), recorder=rec)

    async def go():
        with prefixwise.session("async"):
            await oa.chat.completions.create(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}])
            await an.messages.create(model="claude-haiku-5-5", max_tokens=8,
                                     messages=[{"role": "user", "content": "hi"}])
    asyncio.run(go())
    assert [r[0] for r in rows(rec)] == ["openai", "anthropic"]


def test_streams_pass_through_unrecorded(server):
    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    chunks = list(client.chat.completions.create(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}],
                                                 stream=True))
    assert chunks and rows(rec) == []


def test_recording_can_never_break_a_call(server, capsys):
    rec = Recorder(":memory:")
    rec.proxy.prepare = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire"))
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    for _ in range(2):
        r = client.chat.completions.create(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}])
        assert r.usage.prompt_tokens == 2000
    assert capsys.readouterr().err.count("could not record a call") == 1


def test_wrap_is_idempotent_and_rejects_other_objects(server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = Recorder(":memory:")
    client = openai.OpenAI(base_url=server + "/v1", api_key="sk-test")
    prefixwise.wrap(prefixwise.wrap(client, recorder=rec), recorder=rec)
    client.chat.completions.create(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}])
    assert len(rows(rec)) == 1  # wrapped once, recorded once
    with pytest.raises(TypeError):
        prefixwise.wrap(object())
    assert not (tmp_path / "prefixwise.db").exists()  # rejected before any ledger is created


def test_fixes_are_never_applied_to_what_is_recorded(monkeypatch):
    monkeypatch.setenv("PREFIXWISE_FIXES", "canonical_tools")
    assert Recorder(":memory:").proxy.fixes == ()


def causes(rec):
    return [r[5] for r in rows(rec)]


def test_sdk_objects_in_the_history_are_recorded(server):
    """The documented tool-use patterns put SDK objects into the next request."""
    rec = Recorder(":memory:")
    oa = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    msgs = [{"role": "system", "content": SYSTEM + "objects"}, {"role": "user", "content": "hi"}]
    with prefixwise.session("objs"):
        r = oa.chat.completions.create(model="gpt-5.6", messages=msgs)
        msgs += [r.choices[0].message, {"role": "user", "content": "more"}]  # a ChatCompletionMessage
        oa.chat.completions.create(model="gpt-5.6", messages=msgs, timeout=30.0, extra_headers={"x-a": "1"})
    an = prefixwise.wrap(anthropic.Anthropic(base_url=server, api_key="sk-ant-test"), recorder=rec)
    body = dict(model="claude-haiku-5-5", max_tokens=16, messages=[{"role": "user", "content": "hi"}],
                system=[{"type": "text", "text": SYSTEM + "anthropic objects", "cache_control": {"type": "ephemeral"}}])
    with prefixwise.session("objs-a"):
        r = an.messages.create(**body, tools=anthropic.NOT_GIVEN)
        an.messages.create(**{**body, "messages": body["messages"] + [
            {"role": "assistant", "content": r.content}, {"role": "user", "content": "go"}]})  # TextBlocks
    assert causes(rec) == ["first_write", "none", "first_write", "none"]


def test_recorded_request_is_what_went_on_the_wire(server):
    """extra_body is sent as part of the body; extra_headers, timeout and NOT_GIVEN are not."""
    rec = Recorder(":memory:")
    seen = []
    prepare = rec.proxy.prepare
    rec.proxy.prepare = lambda provider, body, *a, **k: (seen.append(body), prepare(provider, body, *a, **k))[1]
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    client.chat.completions.create(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}],
                                   extra_body={"prompt_cache_key": "k1"}, extra_headers={"x-a": "1"},
                                   timeout=30.0, temperature=openai.NOT_GIVEN)
    assert seen[0] == {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}],
                       "prompt_cache_key": "k1"}


def test_raw_response_parse_and_beta_are_recorded(server):
    rec = Recorder(":memory:")
    oa = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    raw = oa.chat.completions.with_raw_response.create(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}])
    assert raw.parse().usage.prompt_tokens == 2000  # the caller can still parse it
    oa.chat.completions.parse(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}])
    an = prefixwise.wrap(anthropic.Anthropic(base_url=server, api_key="sk-ant-test"), recorder=rec)
    an.beta.messages.create(model="claude-haiku-5-5", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
    assert [r[0] for r in rows(rec)] == ["openai", "openai", "anthropic"]


def test_recorder_ignores_the_proxys_engine_settings(monkeypatch):
    monkeypatch.setenv("PREFIXWISE_ENGINE", "vllm")
    monkeypatch.setenv("PREFIXWISE_KV_EVENTS", "tcp://127.0.0.1:1")
    rec = Recorder(":memory:")
    assert rec.proxy.engine is None and rec.proxy.engine_events is None


def test_a_classifier_failure_warns_once_and_still_leaves_a_row(server, capsys):
    rec = Recorder(":memory:")
    rec.proxy._finalize_locked = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug"))
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    for _ in range(3):
        client.chat.completions.create(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}])
    err = capsys.readouterr().err
    assert err.count("prefixwise:") == 1 and "could not record a call" in err
    with rec.ledger.reader() as c:
        assert [r[0] for r in c.execute("SELECT error FROM entries")] == ["not_recorded"] * 3
