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


def test_attributed_eviction_counts_only_this_sessions_gpu_blocks():
    ev = EngineEvents()
    assert ev.for_turn("s1") == (0, 0, False)
    ev.handle_batch(decoded(stored([b"a", b"b"], "s1"), stored([b"c"], "s2")))
    ev.handle_batch(decoded(BlockRemoved(block_hashes=[b"a", b"c"], medium="GPU")))
    ev.handle_batch(decoded(BlockRemoved(block_hashes=[b"b"], medium="CPU")))  # offload tier, not GPU
    assert ev.for_turn("s1") == (1, 2, True)
    assert ev.for_turn("s1") == (0, 0, True)  # consumed
    assert ev.snapshot()["removed_offload"] == 1


def test_unattributed_engine_wide_count_since_this_sessions_last_turn():
    ev = EngineEvents()
    ev.for_turn("x")  # first turn sets the mark
    ev.handle_batch(decoded(stored([1, 2, 3]), BlockRemoved(block_hashes=[1, 2], medium=None)))
    assert ev.for_turn("x") == (0, 2, False)
    ev.handle_batch(decoded(AllBlocksCleared()))
    assert ev.for_turn("x")[1] >= 1


def test_seq_gaps_are_counted_as_dropped_messages():
    ev = EngineEvents()
    ev.handle_batch(decoded(), seq=1)
    ev.handle_batch(decoded(), seq=4)
    assert ev.snapshot()["dropped_messages"] == 2


BODY = {"model": "Qwen/Qwen3-4B", "messages": [{"role": "system", "content": "You are a careful assistant. " * 40},
                                               {"role": "user", "content": "hi"}]}
NEXT = {**BODY, "messages": BODY["messages"] + [{"role": "assistant", "content": "Hello."},
                                                {"role": "user", "content": "and then?"}]}


def proxy_with(engine, kv_events, cached_seq):
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k", engine=engine, kv_events=kv_events)
    seen, cached = [], list(cached_seq)

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        seen.append(dict(body))
        c = cached.pop(0)
        return 200, {"id": request_id, "usage": {"prompt_tokens": 4000, "completion_tokens": 1,
                                                 "prompt_tokens_details": {"cached_tokens": c}}}

    p._forward = fake
    return p, seen


@pytest.fixture
def publisher():
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    seq = iter(range(1, 10**6))

    def send(*events):
        pub.send_multipart([b"", next(seq).to_bytes(8, "big"), ENC.encode(EventBatch(ts=time.time(),
                                                                                     events=list(events)))])
    send.endpoint = f"tcp://127.0.0.1:{port}"
    send.raw = lambda payload: pub.send_multipart([b"", next(seq).to_bytes(8, "big"), payload])
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


def test_vllm_eviction_over_the_wire_is_named_evicted(publisher):
    p, seen = proxy_with("vllm", publisher.endpoint, [0, 0])
    try:
        warm(publisher, p.engine_events)
        h = {"x-prefixwise-session-id": "agent-7"}
        assert p.handle("openai", dict(BODY), h)[0]["cause"] == "first_write"
        assert seen[0]["session_id"] == "agent-7"  # stamped so vLLM attributes its BlockStored
        publisher(stored([111, 222, 333], "agent-7"))
        publisher(BlockRemoved(block_hashes=[111, 222], medium="GPU"))
        assert wait_for(lambda: p.engine_events.snapshot()["removed_gpu"] == 2)
        res = p.handle("openai", dict(NEXT), h)[0]
        assert res["cause"] == "evicted", res
        assert p.ledger.get_request(res["request_id"])["cause"] == "evicted"
    finally:
        p.engine_events.stop()


def test_sglang_gets_no_session_id_and_engine_wide_evidence(publisher):
    p, seen = proxy_with("sglang", publisher.endpoint, [0, 0])
    try:
        warm(publisher, p.engine_events)
        h = {"x-prefixwise-session-id": "agent-8"}
        p.handle("openai", dict(BODY), h)
        assert "session_id" not in seen[0]  # SGLang rejects session ids it did not open
        publisher(stored([5, 6]), BlockRemoved(block_hashes=[5, 6], medium="GPU"))
        assert wait_for(lambda: p.engine_events.snapshot()["removed_gpu"] == 2)
        assert p.handle("openai", dict(NEXT), h)[0]["cause"] == "evicted"
    finally:
        p.engine_events.stop()


def test_a_hit_wins_over_evictions_offload_restored_it(publisher):
    # LMCache / HiCache: the GPU copy was evicted but the prefix came back from CPU.
    p, _ = proxy_with("vllm", publisher.endpoint, [0, 3900])
    try:
        warm(publisher, p.engine_events)
        h = {"x-prefixwise-session-id": "agent-9"}
        p.handle("openai", dict(BODY), h)
        publisher(stored([7], "agent-9"), BlockRemoved(block_hashes=[7], medium="GPU"))
        assert wait_for(lambda: p.engine_events.snapshot()["removed_gpu"] == 1)
        assert p.handle("openai", dict(NEXT), h)[0]["cause"] == "none"
    finally:
        p.engine_events.stop()


def test_without_engine_events_the_same_miss_stays_a_routing_miss():
    p, seen = proxy_with(None, None, [0, 0])
    h = {"x-prefixwise-session-id": "agent-10"}
    p.handle("openai", dict(BODY), h)
    assert "session_id" not in seen[0]  # hosted APIs never get it
    assert p.handle("openai", dict(NEXT), h)[0]["cause"] == "routing_miss"


def test_malformed_payload_does_not_stop_the_subscriber(publisher):
    ev = EngineEvents(publisher.endpoint).start()
    try:
        warm(publisher, ev)
        publisher.raw(b"\xc1 not msgpack")
        assert wait_for(lambda: ev.snapshot()["decode_errors"] == 1)
        publisher(BlockRemoved(block_hashes=[1], medium="GPU"))  # still consuming
        assert wait_for(lambda: ev.snapshot()["removed_gpu"] == 1)
    finally:
        ev.stop()


def test_unknown_engine_is_refused():
    with pytest.raises(ValueError):
        PrefixwiseProxy(engine="tgi")
