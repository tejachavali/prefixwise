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
