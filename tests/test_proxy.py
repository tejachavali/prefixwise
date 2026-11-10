import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prefixwise.proxy import (PrefixwiseHTTPServer, PrefixwiseProxy, has_cache_breakpoint, make_handler,
                              normalize_usage)


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def run_proxy(**kw):
    proxy = PrefixwiseProxy(**kw)
    server = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    return proxy, server, start(server)


def post(port, path, body, headers=None):
    """Return (status, json body, response headers); never raises on 4xx/5xx."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                 data=json.dumps(body).encode() if not isinstance(body, bytes) else body,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), e.headers


class FakeUpstream:
    """Records what the proxy forwards; replies with a scripted status/payload."""

    def __init__(self, status=200, payload=None, stream_lines=None):
        self.status, self.payload, self.requests = status, payload, []
        self.stream_lines = stream_lines
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                if outer.stream_lines is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for line in outer.stream_lines:
                        self.wfile.write(line)
                        self.wfile.flush()
                    return
                data = json.dumps(outer.payload).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{start(self.server)}"


OPENAI_OK = {"id": "chatcmpl-1", "object": "chat.completion",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
             "usage": {"prompt_tokens": 2000, "completion_tokens": 1,
                       "prompt_tokens_details": {"cached_tokens": 1800}}}
BODY = {"model": "gpt-5.6", "session_id": "s1", "agent_id": "a1",
        "messages": [{"role": "system", "content": "You are helpful."},
                     {"role": "user", "content": "hi"}]}


def test_mock_first_then_second_turn_verdict_in_headers():
    proxy, server, port = run_proxy()
    try:
        status, first, h1 = post(port, "/v1/chat/completions", BODY)
        assert status == 200 and first["object"] == "chat.completion"  # a real completion shape
        assert h1["X-Prefixwise-Cause"] == "first_write"
        _, _, h2 = post(port, "/v1/chat/completions", BODY)
        assert h2["X-Prefixwise-Cause"] == "none"  # mock caches 1,000 of 2,000 tokens
        assert h2["X-Prefixwise-Avoidable-Tokens"] == "0"
        assert proxy.ledger.tokens_per_successful_task() is None  # no task_success yet
    finally:
        server.shutdown()


def test_passthrough_strips_prefixwise_fields_and_sets_ids():
    up = FakeUpstream(payload=OPENAI_OK)
    proxy, server, port = run_proxy(upstream_url=up.url, api_key="sk-test")
    try:
        status, resp, headers = post(port, "/v1/chat/completions",
                                     {**BODY, "prefixwise_compaction_event": True})
        assert status == 200 and resp == OPENAI_OK  # upstream payload, unchanged
        fwd = up.requests[0]
        assert fwd["path"] == "/v1/chat/completions"
        for k in ("session_id", "agent_id", "prefixwise_compaction_event", "prompt_cache_options"):
            assert k not in fwd["body"]
        assert fwd["headers"]["authorization"] == "Bearer sk-test"
        assert "x-session-id" not in fwd["headers"]  # Prefixwise sends no session key of its own
        assert fwd["headers"]["x-request-id"] == headers["X-Prefixwise-Request-Id"]
        assert proxy.ledger.rollup_by_cause()[0]["cause"] == "first_write"
    finally:
        server.shutdown()
        up.server.shutdown()


def test_client_auth_passed_through_and_session_header():
    up = FakeUpstream(payload=OPENAI_OK)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        body = {k: v for k, v in BODY.items() if k not in ("session_id", "agent_id")}
        post(port, "/v1/chat/completions", body,
             {"Authorization": "Bearer client-key", "X-Prefixwise-Session-Id": "hdr-session"})
        fwd = up.requests[0]
        assert fwd["headers"]["authorization"] == "Bearer client-key"
        assert "x-session-id" not in fwd["headers"]  # the Prefixwise id is not turned into a routing key
        assert "x-prefixwise-session-id" not in fwd["headers"]
    finally:
        server.shutdown()
        up.server.shutdown()


def test_anthropic_auth_headers_and_usage():
    payload = {"id": "msg_1", "type": "message", "content": [],
               "usage": {"input_tokens": 50, "cache_read_input_tokens": 1900,
                         "cache_creation_input_tokens": 50, "output_tokens": 1}}
    up = FakeUpstream(payload=payload)
    proxy, server, port = run_proxy(upstream_url=up.url, api_key="sk-ant")
    try:
        body = {"model": "claude-sonnet-5-5", "session_id": "a",
                "system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "hi"}]}
        post(port, "/v1/messages", body)
        status, resp, h = post(port, "/v1/messages", body)
        fwd = up.requests[0]
        assert fwd["path"] == "/v1/messages"
        assert fwd["headers"]["x-api-key"] == "sk-ant"
        assert "authorization" not in fwd["headers"]
        assert fwd["headers"]["anthropic-version"]
        assert fwd["body"]["diagnostics"] == {"previous_message_id": None}
        assert up.requests[1]["body"]["diagnostics"] == {"previous_message_id": "msg_1"}
        # Breakpoint on the system block counts; 1,900 of 2,000 read from cache.
        assert h["X-Prefixwise-Cause"] == "none"
        entry = proxy.ledger.conn.execute(
            "SELECT prompt_tokens, cached_tokens FROM entries ORDER BY id DESC").fetchone()
        assert entry == (2000, 1900)
    finally:
        server.shutdown()
        up.server.shutdown()


def test_upstream_error_returned_unchanged_and_not_recorded():
    err = {"error": {"message": "Unrecognized request argument", "type": "invalid_request_error"}}
    up = FakeUpstream(status=400, payload=err)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        status, resp, headers = post(port, "/v1/chat/completions", BODY)
        assert status == 400 and resp == err
        assert "X-Prefixwise-Cause" not in headers
        assert proxy.ledger.rollup_by_cause() == []
        assert "s1" not in proxy.last_snapshot  # next good turn is still a first write
    finally:
        server.shutdown()
        up.server.shutdown()


def test_unreachable_upstream_is_502():
    proxy, server, port = run_proxy(upstream_url="http://127.0.0.1:1")
    try:
        status, resp, _ = post(port, "/v1/chat/completions", BODY)
        assert status == 502 and "unreachable" in resp["error"]["message"]
    finally:
        server.shutdown()


def test_bad_json_and_unknown_path():
    proxy, server, port = run_proxy()
    try:
        assert post(port, "/v1/chat/completions", b"{not json")[0] == 400
        assert post(port, "/v1/embeddings", BODY)[0] == 404
    finally:
        server.shutdown()


def test_normalize_usage_and_breakpoints():
    assert normalize_usage("anthropic", {"input_tokens": 10, "cache_read_input_tokens": 80,
                                         "cache_creation_input_tokens": 10}) == (100, 80)
    assert normalize_usage("openai", {"prompt_tokens": 100,
                                      "prompt_tokens_details": {"cached_tokens": 64}}) == (100, 64)
    assert has_cache_breakpoint({"tools": [{"name": "t", "cache_control": {"type": "ephemeral"}}]})
    assert has_cache_breakpoint({"cache_control": {"type": "ephemeral"}})
    assert not has_cache_breakpoint({"system": "x", "messages": [{"role": "user", "content": "hi"}]})



def stream(port, path, body, headers=None):
    """POST a streaming request; return (status, raw body bytes, headers)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read(), r.headers


def get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def recorded(port, request_id, timeout=5.0):
    """A streamed request's row. The proxy ends the stream for the client first and records it
    right after (so the client never waits on classification): poll until it is there."""
    deadline = time.monotonic() + timeout
    while True:
        status, row = get(port, f"/requests/{request_id}")
        if status != 404 or time.monotonic() > deadline:
            return status, row
        time.sleep(0.02)


def sse(*events):
    out = []
    for name, obj in events:
        if name:
            out.append(f"event: {name}\n".encode())
        out += [b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n", b"\n"]
    return out


OPENAI_STREAM = sse(
    (None, {"id": "c1", "choices": [{"index": 0, "delta": {"content": "hel"}}], "usage": None}),
    (None, {"id": "c1", "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
            "usage": None}),
    (None, {"id": "c1", "choices": [], "usage": {"prompt_tokens": 2000, "completion_tokens": 2,
                                                 "prompt_tokens_details": {"cached_tokens": 1800}}}),
    (None, b"[DONE]"),
)


def test_openai_stream_relayed_usage_injected_and_hidden():
    up = FakeUpstream(stream_lines=OPENAI_STREAM)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        body = {**BODY, "stream": True, "task_id": "task-1"}
        status, raw, headers = stream(port, "/v1/chat/completions", body)
        assert status == 200 and headers["Content-Type"] == "text/event-stream"
        # Proxy asked for usage; the client did not, so the usage chunk is hidden.
        assert up.requests[0]["body"]["stream_options"] == {"include_usage": True}
        assert "task_id" not in up.requests[0]["body"]
        assert b'"hel"' in raw and b'"lo"' in raw and b"data: [DONE]" in raw
        assert b"prompt_tokens" not in raw
        status, row = recorded(port, headers['X-Prefixwise-Request-Id'])
        assert status == 200
        assert (row["cause"], row["prompt_tokens"], row["cached_tokens"], row["completion_tokens"],
                row["task_id"]) == ("first_write", 2000, 1800, 2, "task-1")
    finally:
        server.shutdown()
        up.server.shutdown()


def test_openai_stream_usage_kept_when_client_asked():
    up = FakeUpstream(stream_lines=OPENAI_STREAM)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        body = {**BODY, "stream": True, "stream_options": {"include_usage": True}}
        _, raw, _ = stream(port, "/v1/chat/completions", body)
        assert b"prompt_tokens" in raw  # the client asked for it, so it gets it
    finally:
        server.shutdown()
        up.server.shutdown()


def test_anthropic_stream_usage_from_start_and_delta():
    events = sse(
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_9", "type": "message", "content": [],
            "usage": {"input_tokens": 50, "cache_read_input_tokens": 1900,
                      "cache_creation_input_tokens": 50, "output_tokens": 1}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "hi"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 7}}),
        ("message_stop", {"type": "message_stop"}),
    )
    up = FakeUpstream(stream_lines=events)
    proxy, server, port = run_proxy(upstream_url=up.url, api_key="k")
    try:
        body = {"model": "claude-sonnet-5-5", "session_id": "a", "stream": True,
                "system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "hi"}]}
        _, raw, h = stream(port, "/v1/messages", body)
        assert b"stream_options" not in json.dumps(up.requests[0]["body"]).encode()
        assert raw.count(b"event: ") == 4  # every event relayed
        row = proxy.ledger.get_request(h["X-Prefixwise-Request-Id"])
        assert (row["prompt_tokens"], row["cached_tokens"], row["completion_tokens"]) == (2000, 1900, 7)
        stream(port, "/v1/messages", body)  # second turn chains diagnostics from the stream id
        assert up.requests[1]["body"]["diagnostics"] == {"previous_message_id": "msg_9"}
    finally:
        server.shutdown()
        up.server.shutdown()


def test_stream_upstream_error_is_json_and_not_recorded():
    err = {"error": {"message": "bad", "type": "invalid_request_error"}}
    up = FakeUpstream(status=400, payload=err)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        status, resp, _ = post(port, "/v1/chat/completions", {**BODY, "stream": True})
        assert status == 400 and resp == err
        assert proxy.ledger.rollup_by_cause() == []  # no verdict
        row = proxy.ledger.recent(1)[0]
        assert (row["status"], row["error"], row["stream"]) == (400, "invalid_request_error", 1)
    finally:
        server.shutdown()
        up.server.shutdown()


def test_mock_stream_then_task_outcome_metric():
    proxy, server, port = run_proxy()
    try:
        hdr = {"X-Prefixwise-Task-Id": "fix-bug-1"}
        _, raw1, h1 = stream(port, "/v1/chat/completions", {**BODY, "stream": True}, hdr)
        _, raw2, h2 = stream(port, "/v1/messages", {"model": "c", "session_id": "m", "stream": True,
                                                     "messages": [{"role": "user", "content": "hi"}]}, hdr)
        assert b"[prefixwise mock]" in raw1 and b"[DONE]" in raw1 and b"usage" not in raw1
        assert b"event: message_stop" in raw2
        assert recorded(port, h1['X-Prefixwise-Request-Id'])[1]["cause"] == "first_write"
        # a stream is recorded after it ends for the client: wait for the second row too, or the
        # ledger below may hold one request (seen on CI, Python 3.10)
        assert recorded(port, h2['X-Prefixwise-Request-Id'])[0] == 200
        assert post(port, "/tasks", {"task_id": "fix-bug-1"})[0] == 400  # success required
        assert post(port, "/tasks", {"task_id": "fix-bug-1", "success": True})[0] == 200
        _, ledger = get(port, "/ledger")
        # Two requests of 2,000 prompt + 1 completion tokens, one successful task.
        assert ledger["tasks"] == {"resolved": 1, "successful": 1}
        assert ledger["tokens_per_successful_task"] == 4002.0
        assert get(port, "/requests/nope")[0] == 404
    finally:
        server.shutdown()


def test_concurrent_requests_all_recorded_and_sessions_consistent():
    from concurrent.futures import ThreadPoolExecutor

    proxy, server, port = run_proxy()
    try:
        def turn(i):
            body = {**BODY, "session_id": f"s{i % 5}"}
            return post(port, "/v1/chat/completions", body)[2]["X-Prefixwise-Cause"]

        with ThreadPoolExecutor(max_workers=16) as pool:
            causes = list(pool.map(turn, range(100)))
        # Exactly one first write per session, however the threads interleave.
        assert causes.count("first_write") == 5
        # A turn racing its session's first write can see no cache yet (the
        # mock decides cached tokens before the first turn is recorded), as a
        # real cache would; it is a miss (an identical turn within 60 s is
        # RETRY_REPLAY), never a second first write.
        # A turn sent while the compared turn was still in flight is `unknown` (it could not have
        # read that turn's write: a parallel request, or a late record).
        assert set(causes) <= {"first_write", "none", "routing_miss", "retry_replay", "unknown"}
        assert sum(r["requests"] for r in proxy.ledger.rollup_by_cause()) == 100
    finally:
        server.shutdown()


def test_session_state_is_bounded():
    from prefixwise.proxy import _SessionMap

    m = _SessionMap(3)
    for k in "abcd":
        m[k] = 1
    assert list(m) == ["b", "c", "d"]


def responses_payload(rid, input_tokens, cached, diag=None):
    p = {"id": rid, "object": "response", "status": "completed", "output": [],
         "usage": {"input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": cached},
                   "output_tokens": 3}}
    if diag:
        p["prompt_cache_diagnostics"] = diag
    return p


class ScriptedUpstream(FakeUpstream):
    """Replies with the next payload from a list, one per request."""

    def __init__(self, payloads):
        super().__init__(payload=None)
        self.queue = list(payloads)
        orig = self.server.RequestHandlerClass.do_POST
        outer = self

        def do_post(handler):
            outer.payload = outer.queue.pop(0)
            return orig(handler)
        self.server.RequestHandlerClass.do_POST = do_post


def test_responses_forwarding_chaining_and_usage():
    up = ScriptedUpstream([responses_payload("resp_1", 2000, 0),
                           responses_payload("resp_2", 2100, 1900,
                                             {"type": "cache_hit", "reason": None})])
    proxy, server, port = run_proxy(upstream_url=up.url, api_key="sk")
    try:
        body = {"model": "gpt-5.6", "instructions": "You are helpful.", "input": "hi",
                "session_id": "r1"}
        _, resp, h1 = post(port, "/v1/responses", body)
        assert resp["id"] == "resp_1" and h1["X-Prefixwise-Cause"] == "first_write"
        assert up.requests[0]["path"] == "/v1/responses"
        assert "session_id" not in up.requests[0]["body"]
        assert "prompt_cache_options" not in up.requests[0]["body"]  # nothing to compare yet
        body2 = {**body, "input": [{"role": "user", "content": "hi"},
                                   {"role": "assistant", "content": "hello"},
                                   {"role": "user", "content": "next"}]}
        _, _, h2 = post(port, "/v1/responses", body2)
        assert up.requests[1]["body"]["prompt_cache_options"] == {"comparison_response_id": "resp_1"}
        assert h2["X-Prefixwise-Cause"] == "none"
        row = proxy.ledger.get_request(h2["X-Prefixwise-Request-Id"])
        assert (row["prompt_tokens"], row["cached_tokens"], row["completion_tokens"]) == (2100, 1900, 3)
    finally:
        server.shutdown()
        up.server.shutdown()


def age(proxy, session, model, when):
    """Pretend the session's previous response on `model` ended at `when`."""
    proxy.last_seen[session] = when
    lane = proxy.lane_threads[(session, model)]
    snap, _, facts, rid = lane.threads[lane.latest]
    lane.threads[lane.latest] = (snap, when, facts, rid)


def test_responses_previous_response_id_is_stitched_not_history_edit():
    up = ScriptedUpstream([responses_payload("resp_a", 2000, 0),
                           responses_payload("resp_b", 2050, 1950),
                           responses_payload("resp_c", 2100, 0),
                           responses_payload("resp_d", 2100, 0, {"type": "cache_miss",
                                                                 "reason": "input_changed"})])
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        base = {"model": "gpt-5.6", "instructions": "You are helpful.", "session_id": "r2"}
        post(port, "/v1/responses", {**base, "input": "first question"})
        # Server-held history: only the new item is sent.
        _, _, h = post(port, "/v1/responses", {**base, "previous_response_id": "resp_a",
                                                "input": "follow-up"})
        assert h["X-Prefixwise-Cause"] == "none"
        # Instructions are not carried over by previous_response_id; changing them is an edit.
        _, _, h = post(port, "/v1/responses", {**base, "instructions": "You are terse.",
                                                "previous_response_id": "resp_b", "input": "more"})
        assert h["X-Prefixwise-Cause"] == "system_edit"
        # A parent this proxy never saw: say so, do not guess HISTORY_EDIT.
        _, _, h = post(port, "/v1/responses", {**base, "previous_response_id": "resp_elsewhere",
                                                "input": "x"})
        assert h["X-Prefixwise-Cause"] == "unknown"
        row = proxy.ledger.get_request(h["X-Prefixwise-Request-Id"])
        assert row["native_reason"] == "input_changed"
    finally:
        server.shutdown()
        up.server.shutdown()


def test_responses_stream_usage_from_completed_event():
    done = responses_payload("resp_s", 2000, 1500)
    events = sse(("response.created", {"type": "response.created",
                                       "response": {**done, "status": "in_progress", "usage": None}}),
                 ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hi"}),
                 ("response.completed", {"type": "response.completed", "response": done}))
    up = FakeUpstream(stream_lines=events)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        _, raw, h = stream(port, "/v1/responses", {"model": "gpt-5.6", "input": "hi", "stream": True})
        assert "stream_options" not in up.requests[0]["body"]  # Responses streams always carry usage
        assert raw.count(b"event: ") == 3
        row = proxy.ledger.get_request(h["X-Prefixwise-Request-Id"])
        assert (row["prompt_tokens"], row["cached_tokens"], row["completion_tokens"]) == (2000, 1500, 3)
    finally:
        server.shutdown()
        up.server.shutdown()


def test_tool_result_mutation_through_proxy():
    proxy, server, port = run_proxy()
    try:
        msgs = [{"role": "system", "content": "You are an agent."},
                {"role": "user", "content": "run the tests"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                                      "function": {"name": "sh", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "3 passed in 0.41s"}]
        post(port, "/v1/chat/completions", {"model": "m", "session_id": "tr", "messages": msgs})
        # The harness re-renders the earlier tool output (timing changed): waste.
        msgs2 = msgs[:3] + [{**msgs[3], "content": "3 passed in 0.39s"}]
        _, _, h = post(port, "/v1/chat/completions", {"model": "m", "session_id": "tr", "messages": msgs2})
        assert h["X-Prefixwise-Cause"] == "tool_result_mutation" and h["X-Prefixwise-Legit"] == "false"
    finally:
        server.shutdown()


def test_anthropic_ttl_expired_and_one_hour_ttl():
    import time as _time

    payload = {"id": "msg", "usage": {"input_tokens": 2000, "cache_read_input_tokens": 0,
                                      "cache_creation_input_tokens": 0}}
    up = FakeUpstream(payload=payload)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        def body(ttl=None, session="ttl"):
            cc = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
            return {"model": "claude-sonnet-5-5", "session_id": session,
                    "system": [{"type": "text", "text": "x", "cache_control": cc}],
                    "messages": [{"role": "user", "content": "hi"}]}
        post(port, "/v1/messages", body())
        idle = _time.time() - 600  # 10 minutes idle > 5 min default
        age(proxy, "ttl", "claude-sonnet-5-5", idle)
        _, _, h = post(port, "/v1/messages", body())
        assert h["X-Prefixwise-Cause"] == "ttl_expired"
        # Same 10-minute gap in a session that asked for a 1 h cache: not expiry.
        post(port, "/v1/messages", body("1h", "ttl1h"))
        age(proxy, "ttl1h", "claude-sonnet-5-5", _time.time() - 600)
        _, _, h = post(port, "/v1/messages", body("1h", "ttl1h"))
        assert h["X-Prefixwise-Cause"] == "routing_miss"
    finally:
        server.shutdown()
        up.server.shutdown()


def test_fork_header_marks_first_turn():
    proxy, server, port = run_proxy()
    try:
        _, _, h = post(port, "/v1/chat/completions", {**BODY, "session_id": "child"},
                       {"X-Prefixwise-Fork-Of": "s1"})
        assert h["X-Prefixwise-Cause"] == "fork_boundary" and h["X-Prefixwise-Avoidable-Tokens"] == "0"
    finally:
        server.shutdown()


def test_moving_anthropic_breakpoint_is_not_a_history_edit():
    cc = {"type": "ephemeral"}
    payloads = [{"id": "m1", "usage": {"input_tokens": 10, "cache_read_input_tokens": 0,
                                       "cache_creation_input_tokens": 2000}},
                {"id": "m2", "usage": {"input_tokens": 60, "cache_read_input_tokens": 2000,
                                       "cache_creation_input_tokens": 40}}]
    up = ScriptedUpstream(payloads)
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        system = [{"type": "text", "text": "You are helpful.", "cache_control": cc}]
        t1 = {"model": "claude-haiku-5-5", "session_id": "bp", "max_tokens": 8, "system": system,
              "messages": [{"role": "user", "content": [{"type": "text", "text": "q1", "cache_control": cc}]}]}
        # Standard pattern: the breakpoint moves to the newest message.
        t2 = {**t1, "messages": [{"role": "user", "content": [{"type": "text", "text": "q1"}]},
                                 {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": [{"type": "text", "text": "q2",
                                                               "cache_control": cc}]}]}
        post(port, "/v1/messages", t1)
        _, _, h = post(port, "/v1/messages", t2)
        assert h["X-Prefixwise-Cause"] == "none"
        assert "cache_control" in json.dumps(up.requests[1]["body"])  # forwarded untouched
    finally:
        server.shutdown()
        up.server.shutdown()


def test_comparison_id_merged_into_client_cache_options():
    up = ScriptedUpstream([responses_payload("resp_1", 2000, 0), responses_payload("resp_2", 2000, 1900)])
    proxy, server, port = run_proxy(upstream_url=up.url)
    try:
        body = {"model": "m", "input": "hi", "session_id": "o", "prompt_cache_options": {"mode": "explicit"}}
        post(port, "/v1/responses", body)
        post(port, "/v1/responses", body)
        assert up.requests[1]["body"]["prompt_cache_options"] == {"mode": "explicit",
                                                                  "comparison_response_id": "resp_1"}
    finally:
        server.shutdown()
        up.server.shutdown()
