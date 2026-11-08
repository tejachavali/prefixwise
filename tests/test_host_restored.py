"""host_restored: a hit whose prefix came back from a host or offload tier after a GPU eviction.

With --kv-events, a BlockRemoved on the GPU for blocks of the previous request, followed by a
response that still read them (vLLM CPU offload, LMCache, SGLang HiCache restore the blocks
without recomputing prefill), is `none` with detail `host_restored`: a hit that cost a restore,
and a sign the GPU cache alone is too small for the session.
"""

import pytest

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.details import Facts, detail_for
from prefixwise.fingerprint import Fingerprinter
from prefixwise.proxy import PrefixwiseProxy

msgspec = pytest.importorskip("msgspec")
pytest.importorskip("zmq")
from test_engine_join import BlockRemoved, EngineEvents, decoded, stored, turn  # noqa: E402
from test_sglang_events import SGLangEvents  # noqa: E402
from test_sglang_events import removed as sg_removed  # noqa: E402
from test_sglang_events import stored as sg_stored  # noqa: E402
from test_sglang_events import turn as sg_turn  # noqa: E402


def test_vllm_reports_restored_blocks_on_a_hit_and_nothing_for_output_blocks_or_offload_removals():
    ev = EngineEvents()
    turn(ev, "s", 0, stored([1, 2, 3, 4], "s"))  # turn 1: prompt blocks 1-4 (16 tokens each)
    ev.handle(decoded(BlockRemoved(block_hashes=[1, 2], medium="GPU")))  # evicted from the GPU while idle
    assert turn(ev, "s", 64, prompt=80) == (2, 0)  # turn 2 read all 64 tokens: blocks 1 and 2 came from elsewhere
    # output blocks evicted routinely do not count: the read did not cover them
    ev = EngineEvents()
    turn(ev, "s", 0, stored([1, 2], "s"), stored([3, 4], "s", parent=2))
    ev.handle(decoded(BlockRemoved(block_hashes=[3, 4], medium="GPU")))
    assert turn(ev, "s", 32, prompt=40) == (None, None)
    # a removal on the offload tier itself is not a GPU eviction
    ev = EngineEvents()
    turn(ev, "s", 0, stored([5, 6], "s"))
    ev.handle(decoded(BlockRemoved(block_hashes=[5], medium="CPU")))
    assert turn(ev, "s", 32, prompt=40) == (None, None)
    # a plain hit with nothing removed: nothing to say
    ev = EngineEvents()
    turn(ev, "s", 0, stored([7, 8], "s"))
    assert turn(ev, "s", 32, prompt=40) == (None, None)


def test_vllm_claims_nothing_from_a_contended_or_cut_stream():
    ev = EngineEvents()
    turn(ev, "s", 0, stored([1, 2, 3, 4], "s"))
    ev.handle(decoded(BlockRemoved(block_hashes=[1], medium="GPU")))
    a = ev.request_sent("s")
    b = ev.request_sent("s")  # a second request of the session in flight: concurrent
    assert ev.request_done(a, 64, 80) == (None, None)
    ev.discard(b)
    ev = EngineEvents()
    turn(ev, "s", 0, stored([1, 2, 3, 4], "s"))
    ev.handle(decoded(BlockRemoved(block_hashes=[1], medium="GPU")))
    token = ev.request_sent("s")
    ev.stats["dropped_messages"] += 1  # the stream lost messages after the send
    assert ev.request_done(token, 64, 80) == (None, None)


def test_sglang_reports_restored_blocks_on_a_hit():
    ev = SGLangEvents()
    sg_turn(ev, "s", 0, sg_stored(100, 10, session="s"))  # tokens 0-9, one token per block
    ev.handle(__import__("test_sglang_events").batch(sg_removed(100, 10)))  # all evicted from the GPU
    assert sg_turn(ev, "s", 10, prompt=12) == (10, 0)  # yet read back: restored from HiCache
    ev = SGLangEvents()
    sg_turn(ev, "s", 0, sg_stored(100, 10, session="s"))
    assert sg_turn(ev, "s", 10, prompt=12) == (None, None)  # nothing removed: a plain hit


def test_details_names_host_restored_only_on_a_hit_with_evicted_blocks():
    facts = Facts(provider="openai", model="m")
    new, detail, why = detail_for("none", None, facts, prompt_tokens=4000, cached_tokens=3900, evicted_blocks=3)
    assert new is None and detail == "host_restored" and "3 of the previous request's KV blocks" in why
    assert detail_for("none", None, facts, prompt_tokens=4000, cached_tokens=3900, evicted_blocks=None)[1] is None
    assert detail_for("none", None, facts, prompt_tokens=4000, cached_tokens=3900, evicted_blocks=0)[1] is None
    assert detail_for("evicted", None, facts, prompt_tokens=4000, cached_tokens=0, evicted_blocks=3)[1] is None


class _Join:
    """A stand-in engine join with a programmed verdict."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def request_sent(self, session_id):
        return 1

    def request_done(self, token, cached, prompt=None):
        return self.verdicts.pop(0) if self.verdicts else (None, None)

    def discard(self, token):
        pass


def _proxy(verdicts, caches=1):
    p = PrefixwiseProxy(upstream_url="http://unused", api_key="k")
    p.engine, p.engine_events, p.engine_caches = "vllm", _Join(verdicts), caches
    usages = [{"prompt_tokens": 4000, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}},
              {"prompt_tokens": 4200, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4000}}]

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        return 200, {"id": "chatcmpl-" + request_id[:8], "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                     "usage": usages.pop(0)}
    p._forward = fake
    return p


def test_the_proxy_records_a_restored_hit_as_none_with_the_detail_one_cache_only():
    body = {"model": "m", "messages": [{"role": "system", "content": "x " * 300}, {"role": "user", "content": "hi"}]}
    more = {**body, "messages": body["messages"] + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]}
    for caches, expected in ((1, "host_restored"), (2, None)):
        p = _proxy([(None, None), (3, 0)], caches)
        p.handle("openai", body, {"x-prefixwise-session-id": "s"})
        e = p.handle("openai", more, {"x-prefixwise-session-id": "s"})[1]
        assert e.cause == "none" and e.detail == expected, (caches, e.cause, e.detail, e.evidence)
        if expected:
            assert "host or offload tier" in e.evidence


def test_classifier_still_calls_a_hit_a_hit_when_blocks_were_evicted():
    fp = Fingerprinter()
    blocks = fp.fingerprint_blocks([("system", "x"), ("messages", {"role": "user", "content": "hi"})])
    prev = RequestSnapshot(model="m", provider="openai", blocks=blocks, prompt_tokens=4000, cached_tokens=0, engine=True)
    curr = RequestSnapshot(model="m", provider="openai", blocks=blocks, prompt_tokens=4200, cached_tokens=4000, engine=True,
                           evicted_blocks=3, resident_blocks=0)
    assert classify(prev, curr).cause is MissCause.NONE
