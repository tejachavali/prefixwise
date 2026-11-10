"""SGLang with --enable-session-radix-cache: a body session_id decides what survives memory pressure
(KV no session holds is evicted first), and SGLang reads the session from the body only. Measured on
0.5.23 (experiments/e_route/runs/2026-11-10-colab-l4-sglang-session-header.md): 4,000 of 4,012 cached
with a body session_id, 3 without; with the flag off, 3 either way.

So Prefixwise passes a client's body session_id through to SGLang, and when a client names its session
only in an x-session-id header, an eviction gets the detail session_header_dropped. The cause stays
evicted: the miss is an eviction."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prefixwise.details import Facts, detail_for
from prefixwise.findings import findings, render
from prefixwise.proxy import PrefixwiseProxy

SYSTEM = "You are a careful assistant. " * 400


class Upstream:
    """An SGLang-like server: replies with the usage it is given, records what it received."""

    def __init__(self, usages, refuse_session=False):
        self.seen, self.usages = [], list(usages)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                outer.seen.append({"x-session-id": self.headers.get("x-session-id"), "body": body})
                if refuse_session and "session_id" in body:
                    b = b'{"error": {"message": "unknown field: session_id"}}'
                    self.send_response(400)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                    return
                prompt, cached = outer.usages.pop(0) if outer.usages else (10, 0)
                reply = {"id": "chatcmpl-x", "model": "m",
                         "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                         "usage": {"prompt_tokens": prompt, "completion_tokens": 1,
                                   "prompt_tokens_details": {"cached_tokens": cached}}}
                b = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def _sglang(up, radix=True):
    """A proxy in front of SGLang whose /server_info showed (or not) --enable-session-radix-cache. The
    join stays off, as _check_sglang leaves it with that flag; evictions are inferred."""
    p = PrefixwiseProxy(upstream_url=up.url, mock=False)
    p.engine, p.sglang_join, p.sglang_session_radix = "sglang", False, radix
    return p


def _turn(p, n, headers=None, **body):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    if n == 2:
        messages += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "and then?"}]
    return p.handle("openai", {"model": "m", "messages": messages, "max_tokens": 32, **body}, headers or {})


# The live run's turn 2 after the flood: (prompt tokens, cached tokens).
TURN1, MISS = (4002, 0), (4012, 3)


def test_the_header_is_forwarded_and_a_header_only_session_after_a_flood_is_evicted_with_the_detail():
    up = Upstream([TURN1, MISS])
    p = _sglang(up)
    try:
        _turn(p, 1, {"x-session-id": "conv-1"})
        _, entry = _turn(p, 2, {"x-session-id": "conv-1"})
        assert [s["x-session-id"] for s in up.seen] == ["conv-1", "conv-1"]  # the router still gets its key
        assert all("session_id" not in s["body"] for s in up.seen)  # nothing invented for the client
        assert entry.cause == "evicted"  # the miss is an eviction: the cause is not replaced
        assert entry.detail == "session_header_dropped"
        assert "x-session-id header, which SGLang does not read" in entry.evidence
    finally:
        up.server.shutdown()


def test_a_clients_body_session_id_reaches_sglang_unchanged_even_with_the_join_off():
    """Before: stripped as a Prefixwise field unless the join was on, which it never is with session
    radix cache. That alone is the live run's 3-of-4,012 row."""
    up = Upstream([TURN1, (4012, 4000)])
    p = _sglang(up)
    try:
        _turn(p, 1, {"x-session-id": "conv-1"}, session_id="s-sa")
        _, entry = _turn(p, 2, {"x-session-id": "conv-1"}, session_id="s-sa")
        assert [s["body"].get("session_id") for s in up.seen] == ["s-sa", "s-sa"]
        assert [s["x-session-id"] for s in up.seen] == ["conv-1", "conv-1"]
        assert entry.cause == "none" and entry.session_id == "s-sa"
    finally:
        up.server.shutdown()


def test_without_the_flag_a_body_session_id_stays_a_prefixwise_field():
    """There it changes no cache hit (live: 3 of 4,012 either way), so the old contract holds."""
    up = Upstream([TURN1])
    p = _sglang(up, radix=False)
    try:
        _, entry = _turn(p, 1, {"x-session-id": "conv-1"}, session_id="s-1")
        assert "session_id" not in up.seen[0]["body"] and entry.session_id == "s-1"
    finally:
        up.server.shutdown()


def test_a_server_that_refuses_the_field_gets_the_request_without_it():
    up = Upstream([TURN1, TURN1, TURN1], refuse_session=True)
    p = _sglang(up)
    try:
        _, entry = _turn(p, 1, {"x-session-id": "conv-1"}, session_id="s-1")
        assert "session_id" in up.seen[0]["body"] and "session_id" not in up.seen[1]["body"]  # resent without
        assert entry is not None and entry.error is None
        _turn(p, 1, {"x-session-id": "conv-2"}, session_id="s-2")  # refused twice: the server does not take it
        _turn(p, 1, {"x-session-id": "conv-3"}, session_id="s-3")
        assert [("session_id" in s["body"]) for s in up.seen] == [True, False, True, False, False]
    finally:
        up.server.shutdown()


def test_with_the_join_on_the_clients_header_value_is_the_body_session_id():
    p = PrefixwiseProxy(mock=True)
    p.engine, p.sglang_join, p.sglang_session_radix = "sglang", True, False

    class Events:
        def request_sent(self, sid):
            return 1

        def discard(self, token):
            pass

    p.engine_events = Events()
    ctx = p.prepare("openai", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    {"x-session-id": "conv-9"})
    assert ctx["forward_body"]["session_id"] == "conv-9"
    assert ctx["facts"].session_unprotected is False  # it reaches the worker
    p.response_started(ctx)


def test_a_hosted_api_still_gets_no_session_id_field():
    up = Upstream([(10, 0)])
    p = PrefixwiseProxy(upstream_url=up.url, mock=False)  # no engine: a hosted API rejects unknown fields
    try:
        _turn(p, 1, {"x-session-id": "conv-1"}, session_id="s-1")
        assert "session_id" not in up.seen[0]["body"]
    finally:
        up.server.shutdown()


def test_an_eviction_with_a_body_session_id_stays_evicted_with_no_detail():
    up = Upstream([TURN1, MISS])  # held by a session, and evicted anyway (session protection is soft)
    p = _sglang(up)
    try:
        _turn(p, 1, {"x-session-id": "conv-1"}, session_id="s-1")
        _, entry = _turn(p, 2, {"x-session-id": "conv-1"}, session_id="s-1")
        assert entry.cause == "evicted" and entry.detail is None
    finally:
        up.server.shutdown()


def test_without_session_radix_cache_the_session_changes_nothing_so_no_detail():
    """Live, default worker: 3 of 4,012 with or without a session id."""
    up = Upstream([TURN1, MISS])
    p = _sglang(up, radix=False)
    try:
        _turn(p, 1, {"x-session-id": "conv-1"})
        _, entry = _turn(p, 2, {"x-session-id": "conv-1"})
        assert entry.cause == "evicted" and entry.detail is None
    finally:
        up.server.shutdown()


def test_no_header_no_detail_and_a_hit_gets_none():
    up = Upstream([TURN1, MISS, TURN1, (4012, 4000)])
    p = _sglang(up)
    try:
        _turn(p, 1)
        _, entry = _turn(p, 2)  # no session named at all: nothing was dropped
        assert entry.cause == "evicted" and entry.detail is None
        _turn(p, 1, {"x-session-id": "conv-2"})
        _, entry = _turn(p, 2, {"x-session-id": "conv-2"})
        assert entry.cause == "none" and entry.detail is None  # unprotected, but it survived: no claim
    finally:
        up.server.shutdown()


def test_the_detail_belongs_to_evictions_only():
    curr = Facts(provider="openai", session_unprotected=True)
    new_cause, detail, why = detail_for("evicted", None, curr, prompt_tokens=4012, cached_tokens=3, engine=True)
    assert new_cause is None and detail == "session_header_dropped" and "sglang#47763" in why
    for cause in ("routing_miss", "unknown", "none", "history_edit"):
        _, detail, _ = detail_for(cause, None, curr, prompt_tokens=4012, cached_tokens=3, engine=True)
        assert detail != "session_header_dropped"
    _, detail, _ = detail_for("evicted", None, Facts(provider="openai"), prompt_tokens=4012, cached_tokens=3,
                              engine=True)
    assert detail is None


def test_warned_once_and_the_report_gives_the_fix(capsys):
    up = Upstream([TURN1, MISS, TURN1, MISS])
    p = _sglang(up)
    try:
        for conv in ("conv-1", "conv-2"):
            _turn(p, 1, {"x-session-id": conv})
            _turn(p, 2, {"x-session-id": conv})
        err = capsys.readouterr().err
        assert err.count("SGLang does not read") == 1
        text = render(findings(p.ledger))
        assert "x-session-id header, which SGLang does not read" in text
        assert "body session_id" in text and "#45620" in text
    finally:
        up.server.shutdown()
