"""SGLang with several data-parallel ranks: each rank has its own cache and KV-event stream
(port base + rank, batches stamped with the rank). SGLang reports only the blocks a request
newly stores, tagged with its session, and its block hashes are the same on every rank, so the
rank that served a request is the one that stored the session's blocks from where its read
stopped, and a routing miss is confirmed by the very block it computed still being held by the
rank that served the previous turn."""

import socket
import time

import pytest

msgspec = pytest.importorskip("msgspec")
zmq = pytest.importorskip("zmq")

from prefixwise.engines import NOTHING, EngineFleet, SGLangEvents
from prefixwise.proxy import PrefixwiseProxy
from test_sglang_events import BODY, ENC, INFO, EventBatch, removed, server_info, stored, wait_for


def fleet(n=2, full_attention=True):
    return EngineFleet([(f"rank {r}", f"tcp://127.0.0.1:{6000 + r}") for r in range(n)],
                       make=lambda endpoint: SGLangEvents(endpoint, full_attention=full_attention))


def deliver(f, rank, *events, stamp=None):
    f.sources[rank].handle(msgspec.msgpack.decode(ENC.encode(
        EventBatch(ts=time.time(), events=list(events), attn_dp_rank=rank if stamp is None else stamp))))


def turn(f, session, cached, rank, *events, prompt=None, late=()):
    token = f.request_sent(session)
    for r, e in late:
        deliver(f, r, e)
    if events:
        deliver(f, rank, *events)
    return f.request_done(token, cached, prompt)


def test_a_turn_sent_to_another_rank_is_a_confirmed_routing_miss():
    f = fleet()
    assert turn(f, "s", 0, 0, stored(100, 10, session="s")) == NOTHING  # first turn, on rank 0
    # Turn 2 on rank 1: nothing cached there, so it computes the same blocks from token 0.
    assert turn(f, "s", 0, 1, stored(100, 15, session="s")) == (None, None, ("rank 0", "rank 1", 10))


def test_the_same_rank_twice_keeps_its_own_verdicts():
    f = fleet()
    turn(f, "s", 0, 1, stored(100, 10, session="s"))
    deliver(f, 1, removed(105, 5))
    evicted, resident = turn(f, "s", 5, 1, stored(105, 7, parent=104, session="s"))[:2]
    assert evicted and resident == 0


def test_another_block_where_the_read_stopped_is_not_a_routing_miss():
    """Rank 1 holds a shared system prompt (tokens 0-4) from another session; turn 2 was
    rendered differently from token 5. Rank 0 still holding its token 5 proves nothing."""
    f = fleet()
    turn(f, "s", 0, 0, stored(100, 10, session="s"))
    deliver(f, 1, stored(100, 5, session="other"))
    assert turn(f, "s", 5, 1, stored(205, 6, parent=104, session="s")) == NOTHING


def test_the_previous_turns_late_store_is_not_mistaken_for_this_one():
    """SGLang publishes a request's finish store after its response: it can land after the
    next turn was sent. It starts where the previous turn ended, not where this one stopped."""
    f = fleet()
    turn(f, "s", 0, 0, stored(100, 10, session="s"))
    late = [(0, stored(110, 3, parent=109, session="s"))]  # turn 1's generated tokens
    assert turn(f, "s", 0, 1, stored(100, 15, session="s"), late=late)[2][:2] == ("rank 0", "rank 1")


def test_no_claim_without_evidence():
    f = fleet()
    turn(f, "s", 0, 0, stored(100, 10, session="s"))
    assert turn(f, "s", 0, 1) == NOTHING  # nothing stored from the stop point: rank unknown
    g = fleet(full_attention=False)  # sliding-window layers: a resident block proves nothing
    turn(g, "s", 0, 0, stored(100, 10, session="s"))
    assert turn(g, "s", 0, 1, stored(100, 15, session="s")) == NOTHING
    h = fleet()
    turn(h, "s", 0, 0, stored(100, 10, session="s"))
    deliver(h, 0, removed(100, 10))  # the previous rank evicted it too
    assert turn(h, "s", 0, 1, stored(100, 15, session="s")) == NOTHING
    k = fleet()
    turn(k, "s", 0, 0, stored(100, 10, session="s"))
    assert turn(k, "s", 6, 1, stored(106, 4, parent=105, session="s"), prompt=10) == NOTHING  # a hit


def test_a_crossed_stream_is_caught_by_its_rank_stamp():
    f = fleet()
    for r, src in enumerate(f.sources):
        src.rank = r
    turn(f, "s", 0, 0, stored(100, 10, session="s"))
    token = f.request_sent("s")
    deliver(f, 0, removed(100, 10), stamp=1)  # rank 1's events arriving on rank 0's endpoint
    deliver(f, 1, stored(100, 15, session="s"))
    assert f.sources[0].snapshot()["wrong_rank"] == 1
    assert f.request_done(token, 0) == NOTHING  # rank 0's residency cannot be trusted any more


# -- over the wire: one SGLang server, two ranks, discovered from /server_info ------------------


def free_pair():
    for _ in range(50):
        base = 20000 + int(time.time() * 1000) % 30000
        socks = []
        try:
            for port in (base, base + 1):
                s = socket.socket()
                s.bind(("127.0.0.1", port))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    pytest.skip("no two consecutive free ports")


@pytest.fixture
def ranks():
    base = free_pair()
    pubs, seqs = [], []
    for r in range(2):
        pub = zmq.Context.instance().socket(zmq.PUB)
        pub.bind(f"tcp://127.0.0.1:{base + r}")
        pubs.append(pub)
        seqs.append(iter(range(1, 10**6)))

    def send(r, *events):
        pubs[r].send_multipart([b"kv", next(seqs[r]).to_bytes(8, "big"),
                                ENC.encode(EventBatch(ts=time.time(), events=list(events), attn_dp_rank=r))])
    send.base = base
    yield send
    for pub in pubs:
        pub.close(0)


def info_for(base, **over):
    return {**INFO, "dp_size": 2, **over,
            "kv_events": {"publisher": "zmq", "endpoint_host": "*", "endpoint_port_base": base, "topic": "",
                          "block_size": 1, "dp_size": 2}}


def test_auto_follows_every_rank_and_confirms_a_routing_miss(ranks):
    srv = server_info(info_for(ranks.base))
    p = PrefixwiseProxy(upstream_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k",
                        engine="sglang", kv_events="auto", full_attention=True)
    try:
        assert p.engine_caches == 2 and [s.rank for s in p.engine_events.sources] == [0, 1]
        assert [s.endpoint for s in p.engine_events.sources] == [f"tcp://127.0.0.1:{ranks.base + r}"
                                                                for r in range(2)]
        for r, src in enumerate(p.engine_events.sources):
            assert wait_for(lambda: (ranks(r), src.snapshot()["batches"] > 0)[1])
        served, seen = [0, 1], []

        def fake(provider, body, request_id, session_id, client_headers, api="chat"):
            seen.append(dict(body))
            r = served[len(seen) - 1]
            ranks(r, stored(100, 10 + 5 * (len(seen) - 1), session=body.get("session_id")))
            wait_for(lambda: p.engine_events.sources[r].snapshot()["stored"] >= 10)
            return 200, {"id": request_id, "usage": {"prompt_tokens": 4000, "completion_tokens": 1,
                                                     "prompt_tokens_details": {"cached_tokens": 0}}}
        p._forward = fake
        h = {"x-prefixwise-session-id": "agent-7"}
        assert p.handle("openai", dict(BODY), h)[0]["cause"] == "first_write"
        assert seen[0]["session_id"] == "agent-7"
        nxt = {**BODY, "messages": BODY["messages"] + [{"role": "assistant", "content": "ok"},
                                                       {"role": "user", "content": "more"}]}
        res = p.handle("openai", nxt, h)[0]
        assert res["cause"] == "routing_miss" and "served by rank 1, while rank 0" in res["evidence"]
    finally:
        if p.engine_events is not None:
            p.engine_events.stop()
        srv.shutdown()


@pytest.mark.parametrize("over,why", [({"disaggregation_mode": "prefill"}, "disaggregation"),
                                      ({"nnodes": 2}, "several nodes"),
                                      ({"enable_session_radix_cache": True}, "session-radix-cache")])
def test_setups_where_ranks_cannot_be_followed_are_refused(ranks, over, why):
    srv = server_info(info_for(ranks.base, **over))
    try:
        p = PrefixwiseProxy(upstream_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k",
                            engine="sglang", kv_events="auto")
    finally:
        srv.shutdown()
    assert p.engine_events is None and not p.sglang_join
    assert why in " ".join(p.engine_notes) and "evictions stay inferred" in " ".join(p.engine_notes)


def test_several_ranks_with_one_endpoint_points_to_auto(ranks):
    srv = server_info(info_for(ranks.base))
    try:
        p = PrefixwiseProxy(upstream_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k",
                            engine="sglang", kv_events=f"tcp://127.0.0.1:{ranks.base}")
    finally:
        srv.shutdown()
        p.engine_events.stop()
    assert not p.sglang_join and "--kv-events auto" in " ".join(p.engine_notes)
