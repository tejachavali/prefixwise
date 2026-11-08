"""Several vLLM KV caches (data-parallel ranks, replicas behind a router): which one served each
request comes from the KV events themselves, so a request sent away from its cached prefix is a
confirmed routing miss, and nothing else is claimed without evidence."""

import http.server
import json
import threading
import time

import pytest

msgspec = pytest.importorskip("msgspec")
zmq = pytest.importorskip("zmq")

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.engines import NOTHING, EngineFleet
from prefixwise.proxy import PrefixwiseProxy
from test_engine_join import BODY, NEXT, ENC, BlockRemoved, EventBatch, decoded, stored, wait_for


def fleet(n=2):
    return EngineFleet([(f"rank {i}", f"tcp://127.0.0.1:{5557 + i}") for i in range(n)])


def turn(f, session, cached, served_by, *events, prompt=None, elsewhere=()):
    """Send the session's next request; `served_by` reports `events` while it is in flight."""
    token = f.request_sent(session)
    for e in events:
        f.sources[served_by].handle(decoded(e))
    for i, e in elsewhere:
        f.sources[i].handle(decoded(e))
    return f.request_done(token, cached, prompt)


def test_a_request_sent_away_from_its_cached_prefix_is_a_confirmed_routing_miss():
    f = fleet()
    assert turn(f, "s", 0, 0, stored([1, 2, 3, 4], "s")) == NOTHING  # first turn
    # Turn 2 went to rank 1, which had nothing (cached 0); rank 0 still holds all four blocks.
    assert turn(f, "s", 0, 1, stored([11, 12, 13, 14, 15], "s")) == (None, None, ("rank 0", "rank 1", 4))


def test_the_same_cache_twice_keeps_that_caches_own_verdict():
    f = fleet()
    turn(f, "s", 0, 1, stored([1, 2, 3, 4], "s"))
    f.sources[1].handle(decoded(BlockRemoved(block_hashes=[1], medium="GPU")))
    assert turn(f, "s", 0, 1, stored([1, 2, 3, 4, 5], "s")) == (1, 0)  # evicted where it was cached
    assert turn(f, "s", 0, 1, stored([1, 2, 3, 4, 5, 6], "s")) == (0, 5)  # resident: not an eviction


def test_no_claim_without_evidence():
    f = fleet(3)
    turn(f, "s", 0, 0, stored([1, 2, 3, 4], "s"))
    # The previous cache lost the block where the read stopped: not a routing miss to claim.
    f.sources[0].handle(decoded(BlockRemoved(block_hashes=[1], medium="GPU")))
    assert turn(f, "s", 0, 2, stored([21, 22], "s")) == NOTHING
    turn(f, "t", 0, 0, stored([31, 32, 33], "t"))
    assert turn(f, "t", 0, 1) == NOTHING  # no cache reported the request: which one served it is unknown
    turn(f, "u", 0, 0, stored([41, 42], "u"))
    assert turn(f, "u", 0, 1, stored([51], "u"), elsewhere=[(2, stored([52], "u"))]) == NOTHING  # two caches
    turn(f, "v", 0, 0, stored([61, 62], "v"))
    assert turn(f, "v", 2000, 1, stored([71], "v"), prompt=3000) == NOTHING  # a hit is never examined


def test_unsafe_residency_gets_no_claim():
    f = fleet()
    turn(f, "s", 0, 0, stored([1, 2, 3], "s"))
    token = f.request_sent("s")
    f.sources[0].stats["dropped_messages"] += 1  # a removal on the previous cache may have been lost
    f.sources[1].handle(decoded(stored([9], "s")))
    assert f.request_done(token, 0) == NOTHING
    g = fleet()
    turn(g, "s", 0, 0, stored([1, 2, 3], "s"))
    g.sources[0].hybrid = True  # sliding-window layers: a resident full-attention block proves nothing
    assert turn(g, "s", 0, 1, stored([9], "s")) == NOTHING
    h = fleet()
    turn(h, "s", 0, 0, stored([1, 2, 3], "s"))
    first, second = h.request_sent("s"), h.request_sent("s")  # two requests of the session in flight
    h.sources[1].handle(decoded(stored([9], "s")))
    assert h.request_done(second, 0) == NOTHING and h.request_done(first, 0) == NOTHING


def test_the_classifier_names_a_routing_miss_and_never_infers_an_eviction_across_caches():
    def snap(**kw):
        return RequestSnapshot(blocks=[], prompt_tokens=4000, cached_tokens=0, model="m", engine=True, **kw)
    prev = snap()
    routed = classify(prev, snap(engine_caches=2, routed=("rank 0", "rank 1", 4)))
    assert routed.cause == MissCause.ROUTING_MISS and "served by rank 1, while rank 0" in routed.evidence
    unknown = classify(prev, snap(engine_caches=2))
    assert unknown.cause == MissCause.UNKNOWN and "2 KV caches" in unknown.evidence
    assert classify(prev, snap()).cause == MissCause.EVICTED  # one engine: still inferred, as before


# -- over the wire: two vLLM replicas' KV-event streams --------------------------------------


@pytest.fixture
def publishers():
    pubs = []

    def make():
        pub = zmq.Context.instance().socket(zmq.PUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        seq = iter(range(1, 10**6))

        def send(*events):
            pub.send_multipart([b"kv-events", next(seq).to_bytes(8, "big"),
                                ENC.encode(EventBatch(ts=time.time(), events=list(events)))])
        send.port = port
        send.endpoint = f"tcp://127.0.0.1:{port}"
        pubs.append(pub)
        return send
    yield make
    for pub in pubs:
        pub.close(0)


def run_two_turns(p, a, b):
    """Turn 1 is served by replica a, turn 2 by replica b."""
    seen, cached = [], [0, 0]

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        seen.append(dict(body))
        return 200, {"id": request_id, "usage": {"prompt_tokens": 4000, "completion_tokens": 1,
                                                 "prompt_tokens_details": {"cached_tokens": cached.pop(0)}}}
    p._forward = fake
    for send, src in ((a, p.engine_events.sources[0]), (b, p.engine_events.sources[1])):
        assert wait_for(lambda: (send(), src.snapshot()["batches"] > 0)[1])  # subscriptions propagated
    h = {"x-prefixwise-session-id": "agent-7"}
    try:
        assert p.handle("openai", dict(BODY), h)[0]["cause"] == "first_write"
        a(stored([111, 222, 333], "agent-7"))
        assert wait_for(lambda: p.engine_events.sources[0].snapshot()["stored"] >= 3)
        orig = p._forward

        def second(*args, **kw):  # replica b serves turn 2 and reports its blocks while it runs
            b(stored([444, 555], "agent-7"))
            wait_for(lambda: p.engine_events.sources[1].snapshot()["stored"] >= 2)
            return orig(*args, **kw)
        p._forward = second
        res = p.handle("openai", dict(NEXT), h)[0]
        assert seen[1]["session_id"] == "agent-7"
        return res
    finally:
        p.engine_events.stop()


def test_replicas_behind_a_router_by_comma_separated_endpoints(publishers):
    a, b = publishers(), publishers()
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k", engine="vllm",
                        kv_events=f"{a.endpoint}, {b.endpoint}")
    assert p.engine_caches == 2
    res = run_two_turns(p, a, b)
    assert res["cause"] == "routing_miss" and f"served by 127.0.0.1:{b.port}" in res["evidence"]


def test_auto_discovers_every_data_parallel_rank(publishers):
    a, b = publishers(), publishers()
    ranks = {"0": {"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": f"tcp://*:{a.port}", "topic": ""},
             "1": {"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": f"tcp://*:{b.port}", "topic": ""}}

    class Vllm(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            out = json.dumps(ranks if self.path == "/kv_event_sources" else {}).encode()
            self.send_response(200 if self.path == "/kv_event_sources" else 404)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Vllm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        p = PrefixwiseProxy(upstream_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k",
                            engine="vllm", kv_events="auto")
    finally:
        srv.shutdown()
    assert p.engine_caches == 2 and [s.endpoint for s in p.engine_events.sources] == [a.endpoint, b.endpoint]
    res = run_two_turns(p, a, b)
    assert res["cause"] == "routing_miss" and "served by rank 1, while rank 0" in res["evidence"]


def test_auto_without_the_endpoint_says_so_and_stays_inferred():
    p = PrefixwiseProxy(upstream_url="http://127.0.0.1:9/v1", api_key="k", engine="vllm", kv_events="auto")
    assert p.engine_events is None and any("/kv_event_sources" in n for n in p.engine_notes)


def test_several_endpoints_are_vllm_only_for_now():
    with pytest.raises(ValueError):
        PrefixwiseProxy(upstream_url="http://unused", engine="sglang", kv_events="tcp://a:1,tcp://b:2")
    with pytest.raises(ValueError):
        PrefixwiseProxy(upstream_url="http://unused", engine="sglang", kv_events="auto")
