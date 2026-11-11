"""What SGLang's /server_info says about the cache itself, read with or without KV events.

- --disable-radix-cache (HiSparse asserts it; deterministic inference sets it on most attention
  backends): nothing is ever read back, so a miss is no eviction, no routing miss and no client edit.
  Before, an unchanged prefix that missed was an inferred eviction on every turn.
- uses_mamba_radix_cache: SGLang frees Mamba state without KV events, so --full-attention would let
  the events rule evictions out wrongly. It is ignored there.
- Fields that change the engine's cache key or rendered prompt while the blocks stay the same are
  compared, so such a miss is param_rendered, not an eviction."""

import pytest

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.details import body_facts
from prefixwise.fingerprint import Fingerprinter
from prefixwise.proxy import PrefixwiseProxy
from prefixwise.records import PARAM_KEYS, TurnRecord, param_fingerprint

pytest.importorskip("zmq")
pytest.importorskip("msgspec")
from test_sglang_events import INFO, server_info  # noqa: E402

BODY = {"model": "m", "messages": [{"role": "system", "content": "You are careful. " * 200},
                                   {"role": "user", "content": "hi"}]}
MORE = {**BODY, "messages": BODY["messages"] + [{"role": "assistant", "content": "ok"},
                                                {"role": "user", "content": "go on"}]}


def _proxy(info, **kw):
    srv = server_info(info)
    p = PrefixwiseProxy(upstream_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", engine="sglang", **kw)
    usages = [(4000, 0), (4020, 0)]

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        prompt, cached = usages.pop(0)
        return 200, {"id": request_id, "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                     "usage": {"prompt_tokens": prompt, "completion_tokens": 1,
                               "prompt_tokens_details": {"cached_tokens": cached}}}
    p._forward = fake
    return p, srv


def test_radix_cache_off_is_no_eviction():
    p, srv = _proxy({**INFO, "disable_radix_cache": True})
    try:
        assert p.no_prefix_cache and any("radix cache off" in n for n in p.engine_notes)
        p.handle("openai", BODY, {"x-prefixwise-session-id": "s"})
        e = p.handle("openai", MORE, {"x-prefixwise-session-id": "s"})[1]
        assert e.cause == "unknown" and "keeps no prefix cache" in e.evidence and e.avoidable_tokens == 0
    finally:
        srv.shutdown()


def test_with_the_radix_cache_on_the_same_miss_is_still_an_inferred_eviction():
    p, srv = _proxy(INFO)
    try:
        assert not p.no_prefix_cache
        p.handle("openai", BODY, {"x-prefixwise-session-id": "s"})
        e = p.handle("openai", MORE, {"x-prefixwise-session-id": "s"})[1]
        assert e.cause == "evicted"
    finally:
        srv.shutdown()


def test_the_fact_is_on_the_wire_only_when_set():
    rec = TurnRecord(provider="openai", api="chat", model="m", session_id="s", agent_id="a", request_id="r")
    assert "no_prefix_cache" not in rec.to_dict()["facts"]  # an older collector still reads it
    rec.facts.no_prefix_cache = True
    assert TurnRecord.from_dict(rec.to_dict()).facts.no_prefix_cache is True


def test_full_attention_is_ignored_on_a_mamba_model():
    p, srv = _proxy({**INFO, "uses_mamba_radix_cache": True}, full_attention=True)
    try:
        assert p._full_attention is False and any("Mamba" in n for n in p.engine_notes)
    finally:
        srv.shutdown()


def test_fields_that_change_the_engines_cache_key_are_compared():
    for k in ("reasoning_effort", "mm_processor_kwargs", "media_io_kwargs", "lora_path", "documents",
              "continue_final_message"):
        assert k in PARAM_KEYS
    fp = Fingerprinter()
    blocks = fp.fingerprint_blocks([("system", "x"), ("messages", {"role": "user", "content": "hi"})])

    def snap(effort):
        return RequestSnapshot(model="gpt-oss-20b", provider="openai", blocks=blocks, prompt_tokens=3000,
                               cached_tokens=0 if effort == "high" else 2900, engine=True, gap_seconds=5.0,
                               params={"reasoning_effort": param_fingerprint(effort)})
    assert classify(snap("low"), snap("high")).cause is MissCause.PARAM_RENDERED


def test_a_cache_salt_keeps_tenants_out_of_each_others_shared_prefix():
    a = body_facts("openai", {**BODY, "cache_salt": "tenant-a"}, model="m", credential="k")
    b = body_facts("openai", {**BODY, "cache_salt": "tenant-b"}, model="m", credential="k")
    same = body_facts("openai", {**BODY, "cache_salt": "tenant-a"}, model="m", credential="k")
    assert a.prefix_key != b.prefix_key and a.prefix_key == same.prefix_key
    assert body_facts("openai", BODY, model="m", credential="k").prefix_key == \
        body_facts("openai", BODY, model="m", credential="k").prefix_key


def test_radix_cache_off_applies_to_the_engines_provider_only():
    """prefixwise run with SGLang behind the OpenAI URL and Anthropic's API behind the other: a
    hosted Anthropic miss keeps its own cause."""
    p, srv = _proxy({**INFO, "disable_radix_cache": True})
    try:
        p.engine_providers = {"openai"}
        ctx = p.prepare("anthropic", {"model": "claude-x", "max_tokens": 5, "system": "s",
                                      "messages": [{"role": "user", "content": "hi"}]}, {})
        assert ctx["facts"].no_prefix_cache is False
    finally:
        srv.shutdown()


def test_radix_cache_off_drops_a_details_advice():
    """Two conversations whose system prompts differ only in a timestamp: with a cache, the second
    gets volatile_prefix_at_start and advice to move the timestamp; with none, there is nothing to
    move it for."""
    p, srv = _proxy({**INFO, "disable_radix_cache": True})
    try:
        for i, stamp in enumerate(("2026-11-11T10:00:00Z", "2026-11-11T10:05:00Z")):
            body = {"model": "m", "messages": [{"role": "system", "content": "You are careful. " * 200 + stamp},
                                               {"role": "user", "content": "hi"}]}
            e = p.handle("openai", body, {"x-prefixwise-session-id": f"s{i}"})[1]
        assert e.detail is None
    finally:
        srv.shutdown()
