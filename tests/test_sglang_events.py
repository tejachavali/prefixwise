"""Engine join for SGLang: its KV events confirm an eviction or rule it out.

Wire types mirrored from sglang/srt/disaggregation/kv_events.py (v0.5.23): a
positional batch [ts, events, attn_dp_rank] of type-tagged maps. SGLang reports
only the blocks a request newly stores, tagged with the body session_id, and by
default one block is one token (--page-size 1).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

msgspec = pytest.importorskip("msgspec")  # pip install 'prefixwise[engines]'
zmq = pytest.importorskip("zmq")

from prefixwise.engines import SGLangEvents
from prefixwise.proxy import PrefixwiseProxy


class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list
    attn_dp_rank: int | None = None


class KVCacheEvent(msgspec.Struct, omit_defaults=True, gc=False, tag=True):
    pass


class BlockStored(KVCacheEvent):
    block_hashes: list
    parent_block_hash: int | None
    token_ids: list
    block_size: int
    lora_id: int | None
    medium: str | None = None
    cache_salt: str | None = None
    session_id: str | None = None
    lora_name: str | None = None


class BlockRemoved(KVCacheEvent):
    block_hashes: list
    medium: str | None = None


class AllBlocksCleared(KVCacheEvent):
    pass


ENC = msgspec.msgpack.Encoder()


def stored(first, n, parent=None, session=None, page=1, medium="GPU"):
    """One coalesced BlockStored: n blocks of `page` tokens, hashes first..first+n-1."""
    return BlockStored(block_hashes=list(range(first, first + n)), parent_block_hash=parent,
                       token_ids=[7] * (n * page), block_size=page, lora_id=None, medium=medium,
                       session_id=session)


def removed(first, n, medium="GPU"):
    return BlockRemoved(block_hashes=list(range(first, first + n)), medium=medium)


def batch(*events):
    return msgspec.msgpack.decode(ENC.encode(EventBatch(ts=time.time(), events=list(events))))


def turn(ev, session, cached, *events, prompt=None):
    token = ev.request_sent(session)
    for e in events:
        ev.handle(batch(e))
    return ev.request_done(token, cached, prompt)


def two_turns(ev, *between):
    """Turn 1 stores tokens 0-9 (hashes 100-109); turn 2 reuses them and stores
    tokens 10-14 (hashes 110-114). Returns after `between` is delivered."""
    turn(ev, "s", 0, stored(100, 10, session="s"))
    turn(ev, "s", 10, stored(110, 5, parent=109, session="s"), prompt=12)
    for e in between:
        ev.handle(batch(e))


def test_confirmed_where_the_read_stopped_was_evicted():
    ev = SGLangEvents()
    two_turns(ev, removed(110, 5), removed(105, 5))  # leaves first, as SGLang evicts
    # Turn 3 read 5 tokens: it stopped at token 5, which was removed with everything after it.
    assert turn(ev, "s", 5) == (10, 0)


def test_not_an_eviction_needs_a_full_attention_model():
    ev = SGLangEvents(full_attention=True)
    two_turns(ev)
    assert turn(ev, "s", 5) == (0, 15)  # token 5 is still on the GPU: the tokens changed there
    ev = SGLangEvents()  # undeclared: SWA or Mamba layers may have been freed without events
    two_turns(ev)
    assert turn(ev, "s", 5) == (None, None)


def test_removals_after_the_stop_point_do_not_confirm():
    ev = SGLangEvents(full_attention=True)
    two_turns(ev, removed(110, 5))  # turn 2's output, which a template may never reuse
    assert turn(ev, "s", 5) == (0, 15)


def test_eviction_while_the_request_waits_counts():
    ev = SGLangEvents()
    two_turns(ev)
    assert turn(ev, "s", 0, removed(110, 5), removed(100, 10)) == (15, 0)


def test_a_prefix_stored_by_another_session_is_on_the_chain():
    # A shared system prompt (tokens 0-9) stored by session "a"; "b" stored only its own tail.
    ev = SGLangEvents()
    turn(ev, "a", 0, stored(100, 10, session="a"))
    turn(ev, "b", 10, stored(200, 4, parent=109, session="b"), prompt=14)
    ev.handle(batch(removed(200, 4), removed(100, 10)))
    assert turn(ev, "b", 0) == (14, 0)


def test_a_split_node_loses_only_its_lower_part():
    # SGLang splits a node without an event; evicting the lower half removes a suffix of the run.
    ev = SGLangEvents(full_attention=True)
    turn(ev, "s", 0, stored(100, 10, session="s"))
    turn(ev, "s", 10, stored(110, 2, parent=109, session="s"), prompt=12)
    ev.handle(batch(removed(110, 2), removed(106, 4)))
    assert turn(ev, "s", 6) == (6, 0)  # stopped at token 6: removed
    turn(ev, "s", 0, stored(300, 3, session="s"))  # a fresh chain: tokens 0-2
    assert turn(ev, "s", 1) == (0, 3)  # stopped at 1, still resident


def test_both_halves_of_a_split_node_removed_in_one_event():
    ev = SGLangEvents(full_attention=True)
    two_turns(ev)
    ev.handle(batch(BlockRemoved(block_hashes=list(range(110, 115)) + list(range(106, 110)) + list(range(100, 106)),
                                 medium="GPU")))
    assert turn(ev, "s", 0) == (15, 0)  # token 0 went too, though it was listed last


def test_paged_blocks_and_positions():
    ev = SGLangEvents()
    turn(ev, "s", 0, stored(100, 4, session="s", page=16))  # tokens 0-63
    turn(ev, "s", 64, stored(104, 2, parent=103, session="s", page=16), prompt=90)  # 64-95
    ev.handle(batch(removed(104, 2), removed(102, 2)))
    assert turn(ev, "s", 32) == (4, 0)  # token 32 is block 2


def test_hits_unplaced_chains_and_no_chain_are_not_judged():
    ev = SGLangEvents(full_attention=True)
    two_turns(ev)
    assert turn(ev, "s", 900, prompt=1000) == (None, None)  # a hit (e.g. HiCache restored it)
    ev = SGLangEvents(full_attention=True)
    turn(ev, "s", 0, stored(500, 4, parent=499, session="s"))  # parent stored before we subscribed
    assert turn(ev, "s", 1) == (None, None)
    assert turn(SGLangEvents(), "new", 0) == (None, None)
    two = SGLangEvents(full_attention=True)
    two_turns(two)
    assert turn(two, "s", 40) == (None, None)  # read stopped beyond the previous chain


def test_host_tier_events_are_ignored():
    ev = SGLangEvents(full_attention=True)
    two_turns(ev, removed(100, 10, medium="CPU_PINNED"), stored(100, 10, medium="CPU_PINNED"))
    assert turn(ev, "s", 5) == (0, 15)


def test_cleared_restarted_concurrent_and_lost_messages():
    ev = SGLangEvents(full_attention=True)
    two_turns(ev, AllBlocksCleared())
    assert turn(ev, "s", 5) == (15, 0)  # /flush_cache: an eviction in effect

    ev = SGLangEvents(full_attention=True)
    two_turns(ev)
    ev.handle(batch(), seq=5)
    ev.handle(batch(), seq=1)  # sequence restarted: the engine restarted
    assert turn(ev, "s", 5) == (15, 0)

    ev = SGLangEvents(full_attention=True)
    two_turns(ev)
    t1 = ev.request_sent("s")
    t2 = ev.request_sent("s")  # two requests of one session in flight
    assert ev.request_done(t1, 5) == (None, None)
    ev.discard(t2)

    ev = SGLangEvents(full_attention=True)
    two_turns(ev)
    token = ev.request_sent("s")
    ev.handle(batch(), seq=1)
    ev.handle(batch(), seq=4)  # two messages lost: a removal may be among them
    assert ev.request_done(token, 5) == (None, None)


def test_overflow_judges_nothing():
    ev = SGLangEvents(full_attention=True, max_alive=12)
    two_turns(ev)  # 15 blocks > 12: removals must have been missed
    assert ev.stats["overflows"] == 1
    assert turn(ev, "s", 5) == (None, None)


# -- over the wire, through the proxy -------------------------------------------------

BODY = {"model": "Qwen/Qwen3-4B", "messages": [{"role": "system", "content": "Be brief. " * 200},
                                               {"role": "user", "content": "hi"}]}
NEXT = {**BODY, "messages": BODY["messages"] + [{"role": "assistant", "content": "Hello."},
                                                {"role": "user", "content": "and then?"}]}


@pytest.fixture
def publisher():
    pub = zmq.Context.instance().socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    seq = iter(range(1, 10**6))

    def send(*events):
        pub.send_multipart([b"", next(seq).to_bytes(8, "big"),
                            ENC.encode(EventBatch(ts=time.time(), events=list(events), attn_dp_rank=0))])
    send.endpoint = f"tcp://127.0.0.1:{port}"
    yield send
    pub.close(0)


def server_info(info):
    """A stand-in for SGLang's GET /server_info."""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            server_info.auth = self.headers.get("Authorization")
            ok = self.path == "/server_info" and info is not None
            data = json.dumps(info if ok else {}).encode()
            self.send_response(200 if ok else 401)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


INFO = {"version": "0.5.23", "enable_session_radix_cache": False, "kv_events": {"publisher": "zmq"},
        "internal_states": [{"memory_usage": {"token_capacity": 32768, "token_capacity_swa": None}}]}


def proxy(publisher, cached, info=INFO, full_attention=True):
    srv = server_info(info)
    p = PrefixwiseProxy(upstream_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k",
                        engine="sglang", kv_events=publisher.endpoint, full_attention=full_attention)
    seen, cached = [], list(cached)

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        seen.append(dict(body))
        if len(seen) > 1 and p.engine_store:  # the engine stores this request's blocks as it finishes
            publisher(*p.engine_store)
        return 200, {"id": request_id, "usage": {"prompt_tokens": 4000, "completion_tokens": 1,
                                                 "prompt_tokens_details": {"cached_tokens": cached.pop(0)}}}
    p._forward = fake
    p._srv = srv
    p.engine_store = [stored(9000, 3000, session="agent-7")]
    return p, seen


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def over_the_wire(publisher, cached, *between, own_store=True, **kw):
    p, seen = proxy(publisher, cached, **kw)
    if not own_store:
        p.engine_store = []
    try:
        assert wait_for(lambda: (publisher(), p.engine_events.snapshot()["batches"] > 0)[1])
        h = {"x-prefixwise-session-id": "agent-7"}
        assert p.handle("openai", dict(BODY), h)[0]["cause"] == "first_write"
        assert seen[0]["session_id"] == "agent-7"  # SGLang stamps it on BlockStored
        publisher(stored(1000, 3000, session="agent-7"), *between)
        assert wait_for(lambda: p.engine_events.snapshot()["stored"] >= 3000)
        return p.handle("openai", dict(NEXT), h)[0], p
    finally:
        p.engine_events.stop()
        p._srv.shutdown()


def test_over_the_wire_confirmed_eviction(publisher):
    res, _ = over_the_wire(publisher, [0, 0], removed(1000, 3000))
    assert res["cause"] == "evicted" and res["evidence"].startswith("confirmed")


def test_over_the_wire_still_resident_is_not_an_eviction(publisher):
    res, _ = over_the_wire(publisher, [0, 0])
    assert res["cause"] == "unknown" and res["evidence"].startswith("not an eviction")


def test_over_the_wire_no_claim_until_the_requests_own_store_arrives(publisher):
    # SGLang publishes a step's events after its output: an empty socket proves nothing.
    res, _ = over_the_wire(publisher, [0, 0], own_store=False)
    assert res["cause"] == "evicted" and res["evidence"].startswith("inferred")


def test_over_the_wire_sliding_window_model_keeps_the_inference(publisher):
    info = {**INFO, "internal_states": [{"memory_usage": {"token_capacity_swa": 8192}}]}
    res, p = over_the_wire(publisher, [0, 0], info=info)
    assert res["cause"] == "evicted" and res["evidence"].startswith("inferred")
    assert any("sliding-window" in n for n in p.engine_notes)


def test_session_radix_cache_and_old_engines_get_no_session_id(publisher):
    p, seen = proxy(publisher, [0], info={**INFO, "enable_session_radix_cache": True})
    try:
        p.handle("openai", dict(BODY), {"x-prefixwise-session-id": "x"})
        assert "session_id" not in seen[0]
    finally:
        p.engine_events.stop()
        p._srv.shutdown()
    p, _ = proxy(publisher, [], info={**INFO, "version": "0.5.19", "kv_events": None})
    p.engine_events.stop()
    p._srv.shutdown()
    assert any("0.5.20" in n for n in p.engine_notes) and any("--kv-events-config" in n for n in p.engine_notes)


def test_server_info_gates_the_join(publisher):
    def sent_session(info, **kw):
        p, seen = proxy(publisher, [0], info=info, **kw)
        try:
            p.handle("openai", dict(BODY), {"x-prefixwise-session-id": "x"})
            return "session_id" in seen[0], p
        finally:
            p.engine_events.stop()
            p._srv.shutdown()

    ok, p = sent_session(INFO)
    assert ok and server_info.auth == "Bearer k"  # SGLang --api-key protects /server_info
    assert p.engine_events.max_alive == 2_000_000  # 32k-token cache: the default cap holds
    big = {**INFO, "page_size": 1, "internal_states": [{"memory_usage": {"token_capacity": 3_000_000}}]}
    ok, p = sent_session(big)
    assert ok and p.engine_events.max_alive > 3_000_000 and any("MB" in n for n in p.engine_notes)
    for info in ({**INFO, "dp_size": 2}, {**INFO, "kv_events": {"dp_size": 4}}, None):
        ok, p = sent_session(info)  # several ranks, or /server_info unreadable: no session id
        assert not ok and "evictions stay inferred" in " ".join(p.engine_notes)


def test_n_choices_are_not_judged(publisher):
    p, seen = proxy(publisher, [0])
    try:
        p.handle("openai", dict(BODY, n=2), {"x-prefixwise-session-id": "x"})
        assert "session_id" not in seen[0]
    finally:
        p.engine_events.stop()
        p._srv.shutdown()


def test_review_findings():
    # A hash on two subtrees (requests with different extra_key): neither chain is judged.
    ev = SGLangEvents(full_attention=True)
    turn(ev, "a", 0, stored(100, 10, session="a"))
    turn(ev, "b", 0, stored(100, 10, session="b"))
    ev.handle(batch(removed(100, 10)))
    assert turn(ev, "a", 5) == (None, None) and turn(ev, "b", 5) == (None, None)

    # Blocks stored again (another session, HiCache load-back) before the request: present.
    ev = SGLangEvents(full_attention=True)
    two_turns(ev, removed(110, 5), removed(100, 10), stored(100, 10, session="other"))
    assert turn(ev, "s", 5) == (0, 15)
    # ...but stored again by the request itself, after it was sent: still an eviction.
    ev = SGLangEvents(full_attention=True)
    two_turns(ev, removed(110, 5), removed(100, 10))
    assert turn(ev, "s", 5, stored(100, 10, session="s")) == (15, 0)

    # An overflow while a check waits does not crash it.
    ev = SGLangEvents(full_attention=True, max_alive=20)
    two_turns(ev)
    token = ev.request_sent("s")
    ev.handle(batch(stored(500, 30)))
    assert ev.request_done(token, 5) == (None, None)

    # Chains kept for sessions are bounded by a token budget.
    ev = SGLangEvents(max_held_tokens=1000)
    for k in range(10):
        ev.handle(batch(stored(10_000 * (k + 1), 300, session=f"s{k}"), removed(10_000 * (k + 1), 300)))
    assert ev.snapshot()["chain_tokens_held"] <= 1000 and ev.snapshot()["sessions_tracked"] <= 4

    # Deep chains are walked once, not once per removed run.
    ev = SGLangEvents()
    turn(ev, "s", 0, *[stored(k + 1, 1, parent=k or None, session="s") for k in range(9000)])
    ev.handle(batch(AllBlocksCleared()))
    t = time.time()
    assert turn(ev, "s", 3) == (9000, 0) and time.time() - t < 0.5
