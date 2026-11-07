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


def turn(ev, session, cached, *events):
    """Send the session's next request, deliver `events` while it is in flight,
    and judge it with the cached_tokens the engine reported."""
    token = ev.request_sent(session)
    for e in events:
        ev.handle(decoded(e))
    return ev.request_done(token, cached)


def test_judges_the_block_where_the_cache_read_stopped():
    ev = EngineEvents()
    assert turn(ev, "s", 0, stored([1, 2, 3, 4], "s")) == (None, None)  # first turn: nothing to judge
    # Turn 2 missed from block 0 (cached 0) and block 1 of turn 1 is gone, but
    # block 0, where the read stopped, is still there: not an eviction.
    ev.handle(decoded(BlockRemoved(block_hashes=[2], medium="GPU")))
    assert turn(ev, "s", 0, stored([1, 2, 3, 4, 5], "s")) == (0, 3)
    # Turn 3 stopped at block 2 (32 tokens of 16-token blocks) and block 2 was evicted.
    ev.handle(decoded(BlockRemoved(block_hashes=[3], medium="GPU")))
    assert turn(ev, "s", 32) == (1, 0)


def test_eviction_while_the_request_waits_in_the_queue_counts():
    # Review finding: removals after the send but before admission must count.
    ev = EngineEvents()
    turn(ev, "s", 0, stored([1, 2, 3], "s"))
    assert turn(ev, "s", 0, BlockRemoved(block_hashes=[1], medium="GPU")) == (1, 0)


def test_output_blocks_and_older_chains_cannot_confirm_an_eviction():
    # Review findings: a template that strips earlier output leaves those blocks
    # unused; their routine eviction must not be read as this miss's cause.
    ev = EngineEvents()
    turn(ev, "s", 0, stored([1, 2], "s"), stored([3, 4], "s"))  # prompt blocks 1-2, output blocks 3-4
    ev.handle(decoded(BlockRemoved(block_hashes=[3, 4], medium="GPU")))
    assert turn(ev, "s", 16) == (0, 2)  # read stopped at block 1 (still there): not an eviction


def test_hybrid_models_track_full_attention_only_and_offload_is_ignored():
    ev = EngineEvents()
    turn(ev, "s", 0, stored([5], "s", group=0, kind="full_attention"),
         stored([5], "s", group=1, kind="sliding_window"))
    ev.handle(decoded(BlockRemoved(block_hashes=[5], medium="GPU", group_idx=1)))  # routine SW free
    ev.handle(decoded(BlockRemoved(block_hashes=[5], medium="CPU", group_idx=0)))  # LMCache tier
    assert turn(ev, "s", 0) == (0, 1)


def test_restart_lost_messages_duplicates_and_bounds():
    ev = EngineEvents(max_sessions=2, max_chain=3)
    token = ev.request_sent("s")
    ev.handle(decoded(stored([1, 2], "s"), stored([1, 2, 3, 4], "s")), seq=10)  # re-reported while queued
    ev.request_done(token, 0)
    ev.handle(decoded(), seq=1)  # sequence restarted: the engine restarted, its cache is gone
    assert turn(ev, "s", 0) == (3, 0)  # chain deduplicated and capped at 3 blocks, all gone
    ev.handle(decoded(stored([7], "s")), seq=2)
    token = ev.request_sent("s")
    ev.handle(decoded(), seq=5)  # two messages lost while in flight: a removal may be among them
    assert ev.request_done(token, 0) == (None, None)  # no claim either way
    for name in ("a", "b", "c"):
        ev.handle(decoded(stored([9], name)), seq=6 + ord(name))
    assert ev.snapshot()["sessions_tracked"] == 2  # windows are bounded even for clients we never saw
    assert ev.request_done(12345, 0) == (None, None)  # unknown token


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


def test_a_non_dict_vllm_xargs_is_forwarded_untouched(publisher):
    p, seen = proxy_with(publisher.endpoint, [0])
    try:
        res = p.handle("openai", dict(BODY, vllm_xargs="client-value"), {"x-prefixwise-session-id": "z"})[0]
        assert res["status"] == 200 and seen[0]["vllm_xargs"] == "client-value"
    finally:
        p.engine_events.stop()
