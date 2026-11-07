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
    session_id: str | None = None


class BlockRemoved(KVCacheEvent):
    block_hashes: list
    medium: str | None


class AllBlocksCleared(KVCacheEvent):
    pass


ENC = msgspec.msgpack.Encoder()


def stored(hashes, session=None, medium="GPU"):
    return BlockStored(block_hashes=hashes, parent_block_hash=None, token_ids=[1] * 16 * len(hashes),
                       block_size=16, lora_id=None, medium=medium, lora_name=None, session_id=session)


def decoded(*events):
    return msgspec.msgpack.decode(ENC.encode(EventBatch(ts=time.time(), events=list(events))))


def test_tracks_each_sessions_resident_and_removed_gpu_blocks():
    ev = EngineEvents()
    ev.handle(decoded(stored([b"a", b"b"], "s1"), stored([b"b", b"c"], "s2")))  # b shared
    assert ev.for_turn("s1") == (0, 2)
    ev.handle(decoded(BlockRemoved(block_hashes=[b"b"], medium="GPU")))
    ev.handle(decoded(BlockRemoved(block_hashes=[b"a"], medium="CPU")))  # offload tier (LMCache): ignored
    assert ev.for_turn("s1") == (1, 1)
    assert ev.for_turn("s2") == (1, 1)  # a shared block counts for every session that holds it
    assert ev.for_turn("s1") == (0, 1)  # removals are consumed per turn
    ev.handle(decoded(AllBlocksCleared()))
    assert ev.for_turn("s1") == (1, 0)


def test_unattributed_blocks_and_dropped_messages():
    ev = EngineEvents(max_blocks=2)
    ev.handle(decoded(stored([1, 2])), seq=1)  # no session id: counted, not tracked
    ev.handle(decoded(stored([3, 4, 5], "s")), seq=4)
    snap = ev.snapshot()
    assert snap["stored"] == 5 and snap["tracked_blocks"] == 2  # bounded
    assert snap["dropped_messages"] == 2
    assert ev.for_turn("s") == (0, 0)  # messages were lost: residency reported as unknown


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


def turn_pair(publisher, cached, *events):
    """First turn, then the given engine events, then the next turn. Returns the second result."""
    p, seen = proxy_with(publisher.endpoint, cached)
    try:
        warm(publisher, p.engine_events)
        h = {"x-prefixwise-session-id": "agent-7"}
        assert p.handle("openai", dict(BODY), h)[0]["cause"] == "first_write"
        assert seen[0]["session_id"] == "agent-7"  # vLLM stamps it on BlockStored
        before = p.engine_events.snapshot()["batches"]
        publisher(*events)
        assert wait_for(lambda: p.engine_events.snapshot()["batches"] > before)
        return p.handle("openai", dict(NEXT), h)[0]
    finally:
        p.engine_events.stop()


def test_over_the_wire_confirmed_eviction(publisher):
    res = turn_pair(publisher, [0, 0], stored([111, 222, 333], "agent-7"),
                    BlockRemoved(block_hashes=[111, 222], medium="GPU"))
    assert res["cause"] == "evicted"


def test_over_the_wire_blocks_still_resident_is_not_an_eviction(publisher):
    # The engine kept every block, yet the prefix missed: the token prefix differs (template).
    res = turn_pair(publisher, [0, 0], stored([111, 222, 333], "agent-7"))
    assert res["cause"] == "unknown"


def test_over_the_wire_offload_restored_prefix_is_a_hit(publisher):
    # LMCache: the GPU copy was evicted, the prefix came back from CPU memory.
    res = turn_pair(publisher, [0, 3900], stored([7], "agent-7"), BlockRemoved(block_hashes=[7], medium="GPU"))
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
    assert "session_id" not in seen[0]  # SGLang rejects session ids it did not open
    res = p.handle("openai", dict(NEXT), {"x-prefixwise-session-id": "x"})[0]
    assert res["cause"] == "evicted"  # one engine, no events: inferred


def test_engine_restart_clears_residency_and_lost_messages_stop_the_negative_claim():
    ev = EngineEvents()
    ev.handle(decoded(stored([1, 2], "s")), seq=10)
    ev.handle(decoded(), seq=1)  # sequence restarted: the engine restarted, its cache is gone
    assert ev.for_turn("s") == (2, 0)
    ev.handle(decoded(stored([3], "s")), seq=2)
    assert ev.for_turn("s") == (0, 1)
    ev.handle(decoded(), seq=5)  # two messages lost: a removal may be among them
    assert ev.for_turn("s") == (0, 0)  # residency unknown, so no "not an eviction" claim
