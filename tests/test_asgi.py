"""prefixwise.asgi.PrefixwiseMiddleware: the edge inside the engine's own server (raw ASGI 3)."""

import asyncio
import json
import types

import pytest

import prefixwise
from prefixwise.asgi import PrefixwiseMiddleware, api_server_count, kv_events_from_argv

SYSTEM = "You are a careful assistant. " * 200


def openai_payload(cached=0, prompt=2000):
    return {"id": "chatcmpl-1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": 1, "total_tokens": prompt + 1,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def anthropic_payload(read=0, write=0, fresh=2000):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": fresh, "cache_read_input_tokens": read, "cache_creation_input_tokens": write,
                      "output_tokens": 1}}


class FakeEngine:
    """An ASGI app standing in for vLLM: answers inference routes from a script, records what it
    received (the body after the middleware), and serves GET /version like the real one."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []  # (path, parsed body)
        self.state = types.SimpleNamespace()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            raise AssertionError("http only")
        if scope["method"] == "GET":
            body = json.dumps({"version": "0.33.0"}).encode()
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        raw = b""
        while True:
            m = await receive()
            raw += m.get("body") or b""
            if not m.get("more_body"):
                break
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = raw
        self.seen.append((scope["path"], body, dict(scope["headers"])))
        kind, payload = self.responses.pop(0)
        if kind == "json":
            data = json.dumps(payload).encode()
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
            await send({"type": "http.response.body", "body": data[:10], "more_body": True})
            await send({"type": "http.response.body", "body": data[10:], "more_body": False})
        elif kind == "error":
            data = json.dumps(payload).encode()
            await send({"type": "http.response.start", "status": 400, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": data})
        elif kind == "sse":  # payload: list of raw chunks (bytes), split arbitrarily across messages
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
            for i, chunk in enumerate(payload):
                await send({"type": "http.response.body", "body": chunk, "more_body": i < len(payload) - 1})


def call(app, method, path, body=None, headers=None, chunks=2):
    """Drive an ASGI app; returns (status, headers dict, body bytes, list of body messages)."""
    raw = json.dumps(body).encode() if isinstance(body, dict) else (body or b"")
    parts = [raw]
    if raw and chunks > 1:
        n = max(1, len(raw) // chunks)
        parts = [raw[i:i + n] for i in range(0, len(raw), n)]
    messages = [{"type": "http.request", "body": p, "more_body": i < len(parts) - 1} for i, p in enumerate(parts)]
    scope = {"type": "http", "method": method, "path": path, "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
             + [(b"content-length", str(len(raw)).encode())], "app": getattr(app, "_fake", None)}
    out = {"status": None, "headers": None, "body": b"", "bodies": []}

    async def receive():
        if messages:
            return messages.pop(0)
        await asyncio.sleep(3600)

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"], out["headers"] = message["status"], {k.decode(): v.decode() for k, v in message["headers"]}
        else:
            out["body"] += message.get("body") or b""
            out["bodies"].append(message.get("body") or b"")

    asyncio.run(app(scope, receive, send))
    return out["status"], out["headers"], out["body"], out["bodies"]


def middleware(engine, **kw):
    mw = PrefixwiseMiddleware(engine, ledger=":memory:", engine=kw.pop("engine_name", None), kv_events=kw.pop("kv", ""),
                              argv=kw.pop("argv", []), env=kw.pop("env", {}), **kw)
    mw._fake = engine  # scope["app"] in the tests: what Starlette would put there
    return mw


def test_non_streaming_turns_get_verdict_headers_and_rows_in_each_api():
    eng = FakeEngine([("json", openai_payload(0)), ("json", openai_payload(2000)), ("json", anthropic_payload(write=2000)),
                      ("json", anthropic_payload(read=2000, fresh=0))])
    mw = middleware(eng)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    st, h, body, _ = call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs}, {"x-prefixwise-session-id": "s"})
    assert st == 200 and json.loads(body)["id"] == "chatcmpl-1" and h["x-prefixwise-cause"] == "first_write"
    more = msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
    st, h, body, _ = call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": more}, {"x-prefixwise-session-id": "s"})
    assert h["x-prefixwise-cause"] == "none" and h["x-prefixwise-legit"] == "none" and h["x-prefixwise-avoidable-tokens"] == "0"
    assert h["x-prefixwise-request-id"]
    # Anthropic route, two turns: a first write then a hit
    a = {"model": "m", "max_tokens": 5, "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
         "messages": [{"role": "user", "content": "hi"}]}
    st, h, _, _ = call(mw, "POST", "/v1/messages", a, {"x-prefixwise-session-id": "a"})
    assert h["x-prefixwise-cause"] == "first_write"
    a["messages"] += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
    st, h, _, _ = call(mw, "POST", "/v1/messages", a, {"x-prefixwise-session-id": "a"})
    assert h["x-prefixwise-cause"] == "none"
    rows = mw.proxy.ledger.recent(10)
    assert [r["cause"] for r in rows] == ["none", "first_write", "none", "first_write"]
    assert {r["provider"] for r in rows} == {"openai", "anthropic"}
    # the engine received each body intact (the middleware re-emitted it) with a matching content-length
    assert all(b["model"] == "m" for _, b, _ in eng.seen)
    assert all(int(hs[b"content-length"]) == len(json.dumps(b).encode()) for _, b, hs in eng.seen)


def test_other_requests_pass_straight_through():
    eng = FakeEngine([])
    mw = middleware(eng)
    st, h, body, _ = call(mw, "GET", "/version")
    assert st == 200 and json.loads(body) == {"version": "0.33.0"} and "x-prefixwise-request-id" not in h
    # a POST elsewhere, and a malformed inference body, both reach the engine untouched
    eng.responses = [("json", {"ok": True}), ("json", {"ok": True})]
    st, h, _, _ = call(mw, "POST", "/v1/embeddings", {"input": "x"})
    assert "x-prefixwise-request-id" not in h and eng.seen[-1][1] == {"input": "x"}
    st, h, _, _ = call(mw, "POST", "/v1/chat/completions", b"not json", chunks=1)
    assert "x-prefixwise-request-id" not in h and eng.seen[-1][1] == b"not json" and mw.proxy.ledger.recent(5) == []


def test_streams_are_relayed_as_they_pass_and_recorded_at_the_end():
    def sse(*events, done=True):
        out = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        return out + (b"data: [DONE]\n\n" if done else b"")
    base = {"id": "chatcmpl-9", "object": "chat.completion.chunk", "model": "m"}
    stream = sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "he"}}]},
                 {**base, "choices": [{"index": 0, "delta": {"content": "llo"}}]},
                 {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                 {**base, "choices": [], "usage": openai_payload(2000)["usage"]})
    cut = len(stream) // 3
    eng = FakeEngine([("json", openai_payload(0)), ("sse", [stream[:cut], stream[cut:2 * cut], stream[2 * cut:]])])
    mw = middleware(eng)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs}, {"x-prefixwise-session-id": "s"})
    more = msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
    st, h, body, bodies = call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": more, "stream": True},
                               {"x-prefixwise-session-id": "s"})
    assert st == 200 and h["x-prefixwise-request-id"] and "x-prefixwise-cause" not in h  # the verdict comes after the stream
    assert len(bodies) == 3  # relayed chunk by chunk, not buffered
    text = body.decode()
    assert '"content": "he"' in text and text.rstrip().endswith("data: [DONE]")
    # the client did not ask for usage: the middleware asked the engine for it and kept the usage chunk to itself
    sent = eng.seen[-1][1]
    assert sent["stream_options"] == {"include_usage": True}
    assert '"usage"' not in text
    rows = mw.proxy.ledger.recent(5)
    assert rows[0]["cause"] == "none" and rows[0]["stream"] == 1 and rows[0]["prompt_tokens"] == 2000
    # a client that asked for usage keeps its chunk
    eng.responses = [("sse", [stream])]
    st, h, body, _ = call(mw, "POST", "/v1/chat/completions",
                          {"model": "m", "messages": more, "stream": True, "stream_options": {"include_usage": True}},
                          {"x-prefixwise-session-id": "s"})
    assert '"usage"' in body.decode()


def test_anthropic_streams_and_error_responses_are_recorded():
    def ev(name, obj):
        return f"event: {name}\ndata: {json.dumps(obj)}\n\n".encode()
    stream = (ev("message_start", {"type": "message_start", "message": {"id": "msg_s", "type": "message", "role": "assistant",
                                                                        "model": "m", "content": [],
                                                                        "usage": {"input_tokens": 0, "cache_read_input_tokens": 2000,
                                                                                  "cache_creation_input_tokens": 0, "output_tokens": 0}}})
              + ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
              + ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}})
              + ev("content_block_stop", {"type": "content_block_stop", "index": 0})
              + ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}})
              + ev("message_stop", {"type": "message_stop"}))
    eng = FakeEngine([("json", anthropic_payload(write=2000)), ("sse", [stream[:40], stream[40:]]),
                      ("error", {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})])
    mw = middleware(eng)
    a = {"model": "m", "max_tokens": 5, "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
         "messages": [{"role": "user", "content": "hi"}]}
    call(mw, "POST", "/v1/messages", a, {"x-prefixwise-session-id": "a"})
    a["messages"] += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
    st, h, body, _ = call(mw, "POST", "/v1/messages", {**a, "stream": True}, {"x-prefixwise-session-id": "a"})
    assert st == 200 and body == stream  # Anthropic streams always carry usage: relayed byte for byte
    assert "stream_options" not in eng.seen[-1][1]
    rows = mw.proxy.ledger.recent(5)
    assert rows[0]["cause"] == "none" and rows[0]["cached_tokens"] == 2000 and rows[0]["stream"] == 1
    # an error from the engine: relayed as is, a row without a verdict
    st, h, body, _ = call(mw, "POST", "/v1/messages", {**a, "stream": True}, {"x-prefixwise-session-id": "a"})
    assert st == 400 and json.loads(body)["error"]["message"] == "bad" and h["x-prefixwise-request-id"]
    row = mw.proxy.ledger.recent(1)[0]
    assert row["status"] == 400 and row["cause"] is None and row["error"] == "invalid_request_error"


def test_kv_events_come_from_the_engine_client_once_and_argv_is_the_last_resort(monkeypatch):
    pytest.importorskip("zmq")
    pytest.importorskip("msgspec")
    started = []

    def fake_start(sources, full_attention=False, ranks=None):
        started.append((sources, ranks))
        mw.proxy.engine_events = types.SimpleNamespace(discard=lambda t: None, request_sent=lambda s: 1,
                                                        request_done=lambda *a: (None, None))
    eng = FakeEngine([("json", openai_payload(0)), ("json", openai_payload(0))])
    # vLLM hands back KVEventsConfig dataclasses keyed by int rank, not dicts
    eng.state.engine_client = types.SimpleNamespace(get_kv_event_sources=lambda: {
        0: types.SimpleNamespace(enable_kv_cache_events=True, publisher="zmq", endpoint="tcp://*:5557", topic=""),
        1: types.SimpleNamespace(enable_kv_cache_events=True, publisher="zmq", endpoint="tcp://*:5558", topic="")})
    mw = middleware(eng, engine_name="vllm", kv="auto")
    mw.proxy._start_events = fake_start
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs}, {"x-prefixwise-session-id": "s"})
    call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs}, {"x-prefixwise-session-id": "t"})
    assert started == [([("rank 0", "tcp://127.0.0.1:5557"), ("rank 1", "tcp://127.0.0.1:5558")], [0, 1])]  # once
    assert any("subscribed to the engine's KV events" in n for n in mw.notes)
    # the engine prep went into the body the engine saw: session id and the full report mode
    _, sent, _ = eng.seen[-1]
    assert sent["session_id"] == "t" and sent["vllm_xargs"] == {"kv_cache_report_mode": "full"}
    # no engine client: the command line is read
    eng2 = FakeEngine([("json", openai_payload(0))])
    mw2 = middleware(eng2, engine_name="vllm", kv="auto",
                     argv=["vllm", "serve", "m", "--kv-events-config", '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5559"}'])
    got = []
    mw2.proxy._start_events = lambda sources, full_attention=False, ranks=None: got.append(sources)
    call(mw2, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs}, {"x-prefixwise-session-id": "s"})
    assert got == [[("rank 0", "tcp://127.0.0.1:5559")]]
    # nothing anywhere: said once, evictions stay inferred
    eng3 = FakeEngine([("json", openai_payload(0)), ("json", openai_payload(0))])
    mw3 = middleware(eng3, engine_name="vllm", kv="auto")
    call(mw3, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs})
    call(mw3, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs})
    assert sum("publishes no KV events" in n for n in mw3.notes) == 1 and mw3.proxy.engine_events is None


def test_a_client_that_leaves_mid_stream_a_raising_app_and_a_silent_app_are_failed_rows_that_release_the_turn():
    class Leaves(FakeEngine):  # Starlette's StreamingResponse returns without the last message on disconnect
        async def __call__(self, scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
            await send({"type": "http.response.body", "body": b"data: {\"id\": \"chatcmpl-1\", \"choices\": []}\n\n", "more_body": True})

    class Raises(FakeEngine):
        async def __call__(self, scope, receive, send):
            raise RuntimeError("engine died")

    class Silent(FakeEngine):
        async def __call__(self, scope, receive, send):
            return
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    body = {"model": "m", "messages": msgs, "stream": True}
    mw = middleware(Leaves([]))
    st, h, _, _ = call(mw, "POST", "/v1/chat/completions", body, {"x-prefixwise-session-id": "s"})
    row = mw.proxy.ledger.recent(1)[0]
    assert st == 200 and row["status"] == 499 and row["error"] == "client_disconnected" and row["cause"] is None
    assert mw.proxy._recording == {}  # the next turn of this conversation does not wait
    mw = middleware(Raises([]))
    with pytest.raises(RuntimeError):
        call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs}, {"x-prefixwise-session-id": "s"})
    row = mw.proxy.ledger.recent(1)[0]
    assert row["status"] == 500 and row["error"] == "app_error" and mw.proxy._recording == {}
    mw = middleware(Silent([]))
    call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs})
    assert mw.proxy.ledger.recent(1)[0]["error"] == "no_response"
    # a body over the cap passes through unexamined
    import prefixwise.asgi as asgi_mod
    eng = FakeEngine([("json", openai_payload(0))])
    mw = middleware(eng)
    old = asgi_mod.MAX_BODY_BYTES
    asgi_mod.MAX_BODY_BYTES = 10
    try:
        st, h, _, _ = call(mw, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs})
    finally:
        asgi_mod.MAX_BODY_BYTES = old
    assert st == 200 and "x-prefixwise-request-id" not in h and mw.proxy.ledger.recent(1) == []


def test_sglang_in_process_declines_kv_events_with_a_note():
    mw = PrefixwiseMiddleware(FakeEngine([]), ledger=":memory:", engine="sglang", kv_events="tcp://127.0.0.1:5557",
                              argv=[], env={})
    assert mw.proxy.engine == "sglang" and mw.proxy.engine_events is None
    assert any("run the proxy" in n for n in mw.notes)


def test_argv_helpers():
    assert kv_events_from_argv(["vllm", "serve", "m", "--kv-events-config",
                                '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}']) == "tcp://127.0.0.1:5557"
    assert kv_events_from_argv(["--kv-events-config={\"publisher\": \"zmq\", \"endpoint\": \"tcp://0.0.0.0:6000\"}"]) == "tcp://127.0.0.1:6000"
    assert kv_events_from_argv(["--kv-events-config", '{"publisher": "null"}']) is None
    assert kv_events_from_argv(["--kv-events-config", "not json"]) is None
    assert kv_events_from_argv(["vllm", "serve"]) is None
    assert api_server_count(["vllm", "serve", "--api-server-count", "4"]) == 4
    assert api_server_count(["-asc=2"]) == 2 and api_server_count(["vllm"]) == 1 and api_server_count(["-asc", "x"]) == 1


def test_configuration_from_the_environment_and_the_multi_worker_warning(tmp_path, capsys):
    env = {"PREFIXWISE_LEDGER": str(tmp_path / "l.db"), "PREFIXWISE_ENGINE": "sglang", "PREFIXWISE_FIX": "canonical_tools",
           "PREFIXWISE_KV_EVENTS": ""}
    mw = PrefixwiseMiddleware(FakeEngine([]), env=env, argv=["sglang", "--api-server-count", "2"])
    assert mw.proxy.engine == "sglang" and mw.proxy.ledger.path == str(tmp_path / "l.db")
    assert "canonical_tools" in (mw.proxy.fixes or ())
    assert any("--api-server-count 2" in n and "PREFIXWISE_COLLECTOR" in n for n in mw.notes)
    # with a collector configured, no warning
    env["PREFIXWISE_COLLECTOR"] = "http://collector:8081"
    mw = PrefixwiseMiddleware(FakeEngine([]), env=env, argv=["vllm", "-asc", "3"])
    assert not any("api-server-count" in n for n in mw.notes) and getattr(mw.proxy.collector, "remote", False)
    mw.proxy.close()
    # the package exports it lazily, for `--middleware prefixwise.PrefixwiseMiddleware`
    assert prefixwise.PrefixwiseMiddleware is PrefixwiseMiddleware
    with pytest.raises(AttributeError):
        prefixwise.NoSuchThing  # noqa: B018
