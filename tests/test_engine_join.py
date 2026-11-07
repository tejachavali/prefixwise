"""Engine join: vLLM / SGLang KV events turn an engine eviction into `evicted`."""

import time

import pytest

msgspec = pytest.importorskip("msgspec")  # pip install 'prefixwise[engines]'
zmq = pytest.importorskip("zmq")

from prefixwise.engines import EngineEvents
from prefixwise.proxy import PrefixwiseProxy


# The publisher's own wire types, mirrored from vllm/distributed/kv_events.py
# (SGLang's are the same shape): an array-like batch of type-tagged maps.
class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list
    data_parallel_rank: int | None = None


class KVCacheEvent(msgspec.Struct, omit_defaults=True, gc=False, tag=True):
    pass


class BlockStored(KVCacheEvent):
    block_hashes: list
    parent_block_hash: int | bytes | None
    token_ids: list
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    session_id: str | None = None


class BlockRemoved(KVCacheEvent):
    block_hashes: list
    medium: str | None
    group_idx: int | None = None


class AllBlocksCleared(KVCacheEvent):
    pass


ENC = msgspec.msgpack.Encoder()


def stored(hashes, session=None, medium="GPU", group=None, kind=None):
    return BlockStored(block_hashes=hashes, parent_block_hash=None, token_ids=[1] * 16 * len(hashes),
                       block_size=16, lora_id=None, medium=medium, lora_name=None, group_idx=group,
                       kv_cache_spec_kind=kind, session_id=session)


def decoded(*events):
    return msgspec.msgpack.decode(ENC.encode(EventBatch(ts=time.time(), events=list(events))))


def test_judges_only_the_blocks_of_the_previous_request():
    ev = EngineEvents()
    ev.request_sent("s")
    ev.handle(decoded(stored([1, 2, 3], "s")))  # turn 1 used blocks 1-3
    assert ev.request_sent("s") == (0, 3)  # turn 2 sent: all resident
    ev.handle(decoded(stored([1, 2, 3, 4], "s")))  # turn 2 reused 1-3, stored 4 (report mode "full")
    ev.handle(decoded(BlockRemoved(block_hashes=[3], medium="GPU")))
    assert ev.request_sent("s") == (1, 3)  # turn 3 sent: one of turn 2's blocks gone


def test_stale_blocks_from_older_turns_do_not_confirm_an_eviction():
    # Review finding: after a template re-render, the old chain is orphaned and
    # evicted normally; that must not be read as this turn's eviction.
    ev = EngineEvents()
    ev.request_sent("s")
    ev.handle(decoded(stored([1, 2], "s")))
    ev.request_sent("s")
    ev.handle(decoded(stored([9, 10], "s")))  # turn 2 rendered differently: new chain
    ev.handle(decoded(BlockRemoved(block_hashes=[1, 2], medium="GPU")))  # old chain evicted
    assert ev.request_sent("s") == (0, 2)  # turn 2's blocks are all there: not an eviction


def test_hybrid_models_track_full_attention_only_and_offload_is_ignored():
    ev = EngineEvents()
    ev.request_sent("s")
    ev.handle(decoded(stored([5], "s", group=0, kind="full_attention"),
                      stored([5], "s", group=1, kind="sliding_window")))
    ev.handle(decoded(BlockRemoved(block_hashes=[5], medium="GPU", group_idx=1)))  # routine SW free
    ev.handle(decoded(BlockRemoved(block_hashes=[5], medium="CPU", group_idx=0)))  # LMCache tier
    assert ev.request_sent("s") == (0, 1)


def test_restart_lost_messages_and_bounds():
    ev = EngineEvents(max_sessions=2, max_chain=2)
    ev.request_sent("s")
    ev.handle(decoded(stored([1, 2, 3], "s")), seq=10)
    ev.handle(decoded(), seq=1)  # sequence restarted: the engine restarted, its cache is gone
    assert ev.request_sent("s") == (2, 0)  # chain capped at 2 blocks, both gone
    ev.handle(decoded(stored([4], "s")), seq=2)
    ev.handle(decoded(), seq=5)  # two messages lost: a removal may be among them
    assert ev.request_sent("s") == (0, 0)  # residency unknown: no "not an eviction" claim
    ev.handle(decoded(stored([6], "s")), seq=6)
    assert ev.request_sent("s") == (0, 1)  # trust comes back for the next clean window
    for name in ("a", "b", "c"):
        ev.request_sent(name)
    assert ev.snapshot()["sessions_tracked"] == 2


BODY = {"model": "Qwen/Qwen3-4B", "messages": [{"role": "system", "content": "You are a careful assistant. " * 40},
                                               {"role": "user", "content": "hi"}]}
NEXT = {**BODY, "messages": BODY["messages"] + [{"role": "assistant", "content": "Hello."},
                                                {"role": "user", "content": "and then?"}]}


def proxy_with(kv_events, cached_seq, engine="vllm"):
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k", engine=engine, kv_events=kv_events)
    seen, cached = [], list(cached_seq)

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        seen.append(dict(body))
        return 200, {"id": request_id, "usage": {"prompt_tokens": 4000, "completion_tokens": 1,
                                                 "prompt_tokens_details": {"cached_tokens": cached.pop(0)}}}

    p._forward = fake
    return p, seen


@pytest.fixture
def publisher():
    pub = zmq.Context.instance().socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    seq = iter(range(1, 10**6))

    def send(*events):
        pub.send_multipart([b"kv-events", next(seq).to_bytes(8, "big"),
                            ENC.encode(EventBatch(ts=time.time(), events=list(events)))])
    send.endpoint = f"tcp://127.0.0.1:{port}"
    send.raw = lambda payload: pub.send_multipart([b"kv-events", next(seq).to_bytes(8, "big"), payload])
    yield send
    pub.close(0)


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def warm(send, ev):
    # ZMQ PUB/SUB drops messages until the subscription has propagated.
    assert wait_for(lambda: (send(), ev.snapshot()["batches"] > 0)[1])


def two_turns(publisher, cached, *between):
    """Turn 1; vLLM reports its blocks; then `between` events; then turn 2."""
    p, seen = proxy_with(publisher.endpoint, cached)
    try:
        warm(publisher, p.engine_events)
        h = {"x-prefixwise-session-id": "agent-7"}
        assert p.handle("openai", dict(BODY, vllm_xargs={"priority": 1}), h)[0]["cause"] == "first_write"
        assert seen[0]["session_id"] == "agent-7"  # vLLM stamps it on BlockStored
        assert seen[0]["vllm_xargs"] == {"priority": 1, "kv_cache_report_mode": "full"}  # client's kept
        publisher(stored([111, 222, 333], "agent-7"), *between)
        assert wait_for(lambda: p.engine_events.snapshot()["stored"] >= 3)
        time.sleep(0.1)
        return p.handle("openai", dict(NEXT), h)[0]
    finally:
        p.engine_events.stop()


def test_over_the_wire_confirmed_eviction(publisher):
    res = two_turns(publisher, [0, 0], BlockRemoved(block_hashes=[111, 222], medium="GPU"))
    assert res["cause"] == "evicted"


def test_over_the_wire_blocks_still_resident_is_not_an_eviction(publisher):
    assert two_turns(publisher, [0, 0])["cause"] == "unknown"


def test_over_the_wire_offload_restored_prefix_is_a_hit(publisher):
    # LMCache: the GPU copy was evicted, the prefix came back from CPU memory.
    res = two_turns(publisher, [0, 3900], BlockRemoved(block_hashes=[111], medium="GPU"))
    assert res["cause"] == "none"


def test_malformed_payload_does_not_stop_the_subscriber(publisher):
    ev = EngineEvents(publisher.endpoint).start()
    try:
        warm(publisher, ev)
        publisher.raw(b"\xc1 not msgpack")
        assert wait_for(lambda: ev.snapshot()["decode_errors"] == 1)
        publisher(stored([1], "s"))  # still consuming
        assert wait_for(lambda: ev.snapshot()["stored"] == 1)
    finally:
        ev.stop()


def test_engine_settings_are_validated():
    with pytest.raises(ValueError):
        PrefixwiseProxy(engine="tgi")
    with pytest.raises(ValueError):  # SGLang events cannot be attributed to sessions
        PrefixwiseProxy(engine="sglang", kv_events="tcp://127.0.0.1:1")
    p, seen = proxy_with(None, [0, 0], engine="sglang")
    p.handle("openai", dict(BODY), {"x-prefixwise-session-id": "x"})
    assert "session_id" not in seen[0] and "vllm_xargs" not in seen[0]
    res = p.handle("openai", dict(NEXT), {"x-prefixwise-session-id": "x"})[0]
    assert res["cause"] == "evicted"  # one engine, no events: inferred
