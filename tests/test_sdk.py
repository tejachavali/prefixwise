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
    rec.proxy.collector.ingest = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug"))
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    for _ in range(3):
        client.chat.completions.create(model="gpt-5.6", messages=[{"role": "user", "content": "hi"}])
    err = capsys.readouterr().err
    assert err.count("prefixwise:") == 1 and "could not record a call" in err
    with rec.ledger.reader() as c:
        assert [r[0] for r in c.execute("SELECT error FROM entries")] == ["not_recorded"] * 3


# Streamed calls -------------------------------------------------------------------------------


def errors(rec):
    with rec.ledger.reader() as c:
        return [r[0] for r in c.execute("SELECT error FROM entries ORDER BY id")]


MSGS = [{"role": "system", "content": SYSTEM + "streams"}, {"role": "user", "content": "hi"}]
NEXT = MSGS + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
ANTH = dict(model="claude-haiku-5-5", max_tokens=16,
            system=[{"type": "text", "text": SYSTEM + "anthropic streams", "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "hi"}])
ANTH_NEXT = {**ANTH, "messages": ANTH["messages"] + [{"role": "assistant", "content": "ok"},
                                                     {"role": "user", "content": "go"}]}


def test_openai_chat_stream_is_recorded_and_the_caller_sees_its_own_chunks(server):
    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    opts = {}
    with prefixwise.session("cs"):
        stream = client.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True, stream_options=opts)
        assert isinstance(stream, openai.Stream) and stream.response.status_code == 200  # still the SDK's
        chunks = list(stream)
        stream.close()  # after the end: no second row
        again = list(client.chat.completions.create(model="gpt-5.6", messages=NEXT, stream=True))
    # Prefixwise asked for usage; the usage-only chunk it caused never reaches the caller.
    assert opts == {} and chunks and all(c.choices for c in chunks + again)
    assert [(r[0], r[1], r[2], r[5], r[6], r[7]) for r in rows(rec)] == [
        ("openai", "chat", "cs", "first_write", 2000, 0), ("openai", "chat", "cs", "none", 2000, 2000)]


def test_a_caller_that_asked_for_usage_gets_its_usage_chunk(server):
    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    chunks = list(client.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True,
                                                 stream_options={"include_usage": True}))
    assert chunks[-1].usage.prompt_tokens == 2000 and not chunks[-1].choices
    # A caller that turned usage off is left as it asked: recorded, with no usage to judge.
    list(client.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True,
                                        stream_options={"include_usage": False}))
    assert [r[6] for r in rows(rec)] == [2000, 0]


def test_stream_helpers_are_recorded_once(server):
    """chat.completions.stream and responses.stream call create: one row each, not two."""
    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    with prefixwise.session("helpers"):
        with client.chat.completions.stream(model="gpt-5.6", messages=MSGS) as s:
            assert s.get_final_completion().choices[0].message.content == "[prefixwise mock]"
        with client.responses.stream(model="gpt-5.6", instructions=SYSTEM, input="hi") as s:
            assert s.get_final_response().id.startswith("resp_")
    assert [(r[0], r[1], r[2]) for r in rows(rec)] == [("openai", "chat", "helpers"),
                                                       ("openai", "responses", "helpers")]
    assert errors(rec) == [None, None]


def test_openai_responses_stream(server):
    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    with prefixwise.session("rs"):
        own = SYSTEM + "responses streams"  # the mock server's cache is shared across this module
        events = list(client.responses.create(model="gpt-5.6", instructions=own, input="hi", stream=True))
        list(client.responses.create(model="gpt-5.6", instructions=own, input="hi", stream=True))  # the same prompt again
    assert events[-1].type == "response.completed"
    assert [(r[1], r[2], r[5], r[6], r[7]) for r in rows(rec)] == [
        ("responses", "rs", "first_write", 2000, 0), ("responses", "rs", "none", 2000, 2000)]


def test_anthropic_create_stream_and_messages_stream(server):
    rec = Recorder(":memory:")
    client = prefixwise.wrap(anthropic.Anthropic(base_url=server, api_key="sk-ant-test"), recorder=rec,
                             agent_id="streamer")
    with prefixwise.session("as"):
        events = list(client.messages.create(**ANTH, stream=True))
        with client.messages.stream(**ANTH_NEXT) as s:
            assert s.get_final_text() == "[prefixwise mock]"
        with client.beta.messages.stream(**{**ANTH_NEXT, "messages": ANTH_NEXT["messages"] + [
                {"role": "assistant", "content": "ok"}, {"role": "user", "content": "again"}]}) as s:
            for _ in s:
                pass
    assert [e.type for e in events][0] == "message_start" and events[-1].type == "message_stop"
    assert [(r[0], r[2], r[3], r[5], r[6]) for r in rows(rec)] == [
        ("anthropic", "as", "streamer", "first_write", 2000), ("anthropic", "as", "streamer", "none", 2000),
        ("anthropic", "as", "streamer", "none", 2000)]
    assert [r[7] for r in rows(rec)][1:] == [2000, 2000]  # cache reads from message_start


def test_a_stream_left_early_is_a_failure_row_never_a_guess(server):
    rec = Recorder(":memory:")
    oa = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    an = prefixwise.wrap(anthropic.Anthropic(base_url=server, api_key="sk-ant-test"), recorder=rec)
    stream = oa.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True)
    next(stream)
    stream.close()
    stream.close()
    with oa.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True) as stream:
        next(stream)
    with an.messages.stream(**ANTH) as s:
        next(iter(s))  # message_start only
    assert errors(rec) == ["client_disconnected"] * 3


def test_a_recording_bug_never_breaks_a_stream(server, capsys):
    rec = Recorder(":memory:")
    rec.proxy.prepare = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire"))
    oa = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    an = prefixwise.wrap(anthropic.Anthropic(base_url=server, api_key="sk-ant-test"), recorder=rec)
    for _ in range(2):
        assert list(oa.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True))
        with an.messages.stream(**ANTH) as s:
            assert s.get_final_text()
        s = oa.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True)
        next(s)
        s.close()
    assert capsys.readouterr().err.count("could not record a call") == 1


def test_async_streams_and_the_session_across_yields(server):
    """The session is the one current when the call was made, even when the stream is consumed
    in another task, after the session block ended, with other sessions' calls in between."""
    rec = Recorder(":memory:")
    oa = prefixwise.wrap(openai.AsyncOpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    an = prefixwise.wrap(anthropic.AsyncAnthropic(base_url=server, api_key="sk-ant-test"), recorder=rec)

    async def tokens(stream):
        async for chunk in stream:  # an async generator: yields to the event loop between chunks
            await asyncio.sleep(0)
            yield chunk

    async def go():
        with prefixwise.session("early", task_id="t-early"):
            pending = await oa.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True)
        assert isinstance(pending, openai.AsyncStream)

        async def other():
            with prefixwise.session("other"):
                async with await oa.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True) as s:
                    return [c async for c in s]
        with prefixwise.session("late"):
            consumed, other_chunks = await asyncio.gather(
                asyncio.create_task(_collect(tokens(pending))), other())
            assert consumed and other_chunks and all(c.choices for c in consumed)
            async with oa.chat.completions.stream(model="gpt-5.6", messages=NEXT) as s:
                async for _ in s:
                    pass
            r = [e async for e in await oa.responses.create(model="gpt-5.6", input="hi", stream=True)]
            assert r[-1].type == "response.completed"
            async with oa.responses.stream(model="gpt-5.6", input="hi") as s:
                await s.get_final_response()
            [e async for e in await an.messages.create(**ANTH, stream=True)]
            async with an.messages.stream(**ANTH_NEXT) as s:
                assert await s.get_final_text() == "[prefixwise mock]"
            s = await oa.chat.completions.create(model="gpt-5.6", messages=MSGS, stream=True)
            await s.__anext__()
            await s.close()
            await s.aclose()

    async def _collect(gen):
        return [c async for c in gen]

    asyncio.run(go())
    got = sorted((r[2], r[4], r[0], r[1]) for r in rows(rec))
    assert got == sorted([("early", "t-early", "openai", "chat"), ("other", None, "openai", "chat"),
                          ("late", None, "openai", "chat"), ("late", None, "openai", "responses"),
                          ("late", None, "openai", "responses"), ("late", None, "anthropic", "chat"),
                          ("late", None, "anthropic", "chat"), ("late", None, "openai", "chat")])
    assert sorted(e or "" for e in errors(rec)) == [""] * 7 + ["client_disconnected"]


def test_a_server_that_rejects_stream_options_is_asked_once(tmp_path):
    """Some OpenAI-compatible servers refuse stream_options: the call is retried as the caller
    made it, and that server is not asked again."""
    httpx = pytest.importorskip("httpx")
    import json as _json
    seen = []

    def handler(request):
        body = _json.loads(request.content)
        seen.append("stream_options" in body)
        if "stream_options" in body:
            return httpx.Response(400, json={"error": {"message": "Unrecognized request argument: stream_options"}})
        chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                 "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}
        sse = f"data: {_json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url="http://local.invalid/v1", api_key="k", max_retries=0,
                                           http_client=httpx.Client(transport=httpx.MockTransport(handler))),
                             recorder=rec)
    for _ in range(2):
        assert [c.choices[0].delta.content for c in
                client.chat.completions.create(model="m", messages=MSGS, stream=True)] == ["hi"]
    assert seen == [True, False, False]
    assert errors(rec) == [None, None]  # recorded; no usage, so no verdict is claimed from it


# Found by review: raw responses, extra_body, helpers left early, cancellation, error events ------


def _sse_openai(handler_seen, with_usage_when_asked=True):
    import json as _json
    httpx = pytest.importorskip("httpx")

    def handler(request):
        body = _json.loads(request.content)
        handler_seen.append(body)
        base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
        chunks = [{**base, "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
                  {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
        if with_usage_when_asked and (body.get("stream_options") or {}).get("include_usage"):
            chunks.append({**base, "choices": [], "usage": {"prompt_tokens": 2000, "completion_tokens": 1,
                                                            "prompt_tokens_details": {"cached_tokens": 0}}})
        sse = "".join(f"data: {_json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handler)


def test_raw_and_streaming_responses_are_left_untouched():
    httpx = pytest.importorskip("httpx")
    seen = []
    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url="http://local.invalid/v1", api_key="k", max_retries=0,
                                           http_client=httpx.Client(transport=_sse_openai(seen))), recorder=rec)
    chunks = list(client.chat.completions.with_raw_response.create(model="m", messages=MSGS, stream=True).parse())
    with client.chat.completions.with_streaming_response.create(model="m", messages=MSGS, stream=True) as r:
        text = b"".join(r.iter_bytes())
    assert len(chunks) == 2 and all(c.choices for c in chunks) and b'"usage"' not in text
    assert ["stream_options" in b for b in seen] == [False, False] and rows(rec) == []


def test_usage_asked_for_through_extra_body_reaches_the_caller():
    httpx = pytest.importorskip("httpx")
    seen = []
    rec = Recorder(":memory:")
    client = prefixwise.wrap(openai.OpenAI(base_url="http://local.invalid/v1", api_key="k", max_retries=0,
                                           http_client=httpx.Client(transport=_sse_openai(seen))), recorder=rec)
    chunks = list(client.chat.completions.create(model="m", messages=MSGS, stream=True,
                                                 extra_body={"stream_options": {"include_usage": True}}))
    assert len(chunks) == 3 and chunks[-1].usage.prompt_tokens == 2000
    assert [r[6] for r in rows(rec)] == [2000]


def test_helpers_left_early_are_failure_rows(server):
    rec = Recorder(":memory:")
    oa = prefixwise.wrap(openai.OpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)
    with oa.chat.completions.stream(model="gpt-5.6", messages=MSGS) as s:
        next(iter(s))
    with oa.responses.stream(model="gpt-5.6", input="hi") as s:
        next(iter(s))
    aoa = prefixwise.wrap(openai.AsyncOpenAI(base_url=server + "/v1", api_key="sk-test"), recorder=rec)

    async def go():
        async with aoa.chat.completions.stream(model="gpt-5.6", messages=MSGS) as s:
            async for _ in s:
                break
        async with aoa.chat.completions.stream(model="gpt-5.6", messages=MSGS) as s:  # to the end: a row
            async for _ in s:
                pass
    asyncio.run(go())
    assert errors(rec) == ["client_disconnected"] * 3 + [None]


def test_a_cancelled_stream_is_the_caller_stopping_not_an_upstream_failure():
    from prefixwise.sdk import _AsyncStreamTap, _Call, _StreamTap, _Tap
    rec = Recorder(":memory:")

    class Stalls:  # the next chunk never comes: the caller's timeout cancels the wait
        async def __anext__(self):
            raise asyncio.CancelledError

    class Interrupted:
        def __next__(self):
            raise KeyboardInterrupt

    async def go():
        stream = _AsyncStreamTap(Stalls(), _Tap(_Call(rec, "openai", "chat", {"model": "m", "messages": MSGS},
                                                      None), False))
        with pytest.raises(asyncio.CancelledError):
            await stream.__anext__()
    asyncio.run(go())
    with pytest.raises(KeyboardInterrupt):
        next(_StreamTap(Interrupted(), _Tap(_Call(rec, "openai", "chat", {"model": "m", "messages": MSGS}, None),
                                            False)))
    assert errors(rec) == ["client_disconnected", "client_disconnected"]


def test_an_error_event_mid_stream_is_an_upstream_failure():
    """Anthropic sends an `error` event (overloaded) after message_start: the SDK raises from the
    stream, and the row says the stream broke, not that the caller left."""
    import json as _json
    try:
        import httpx2 as httpx  # the anthropic SDK's own HTTP client, where it has one
    except ImportError:
        httpx = pytest.importorskip("httpx")

    def handler(request):
        start = {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                 "model": "claude-haiku-5-5", "content": [], "stop_reason": None,
                 "usage": {"input_tokens": 2000, "output_tokens": 0}}}
        err = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        sse = f"event: message_start\ndata: {_json.dumps(start)}\n\nevent: error\ndata: {_json.dumps(err)}\n\n"
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    rec = Recorder(":memory:")
    an = prefixwise.wrap(anthropic.Anthropic(base_url="http://local.invalid", api_key="k", max_retries=0,
                                             http_client=httpx.Client(transport=httpx.MockTransport(handler))),
                         recorder=rec)
    with pytest.raises(anthropic.APIStatusError):
        with an.messages.stream(**ANTH) as s:
            for _ in s:
                pass
    with pytest.raises(anthropic.APIStatusError):
        for _ in an.messages.create(**ANTH, stream=True):
            pass
    with pytest.raises(RuntimeError):  # the caller's own error inside the block: the caller left
        with an.messages.stream(**ANTH) as s:
            next(iter(s))
            raise RuntimeError("mine")
    assert errors(rec) == ["upstream_dropped", "upstream_dropped", "client_disconnected"]
