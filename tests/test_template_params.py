"""A chat template's arguments are prompt-affecting parameters: a self-hosted engine renders the
prompt through the template, so `chat_template_kwargs` (enable_thinking on Qwen3) or a per-request
`chat_template` changes the rendered prefix although no block changed. Before, such a turn had
nothing to blame and read as `unknown` ("not an eviction") behind an engine with KV events."""

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.explain import Turn, explain_pair
from prefixwise.fingerprint import Fingerprinter
from prefixwise.records import PARAM_KEYS, param_fingerprint

FP = Fingerprinter()
BLOCKS = FP.fingerprint_blocks([("system", "Be careful."), ("messages", {"role": "user", "content": "hi"})])


def snap(prompt, cached, **params):
    return RequestSnapshot(model="m", provider="openai", blocks=BLOCKS, prompt_tokens=prompt, cached_tokens=cached,
                           engine=True, params={k: param_fingerprint(v) for k, v in params.items()})


def test_chat_template_kwargs_and_chat_template_are_compared():
    assert "chat_template_kwargs" in PARAM_KEYS and "chat_template" in PARAM_KEYS
    prev = snap(4000, 0, chat_template_kwargs={"enable_thinking": True})
    cold = snap(4200, 0, chat_template_kwargs={"enable_thinking": False})
    c = classify(prev, cold)
    assert c.cause is MissCause.PARAM_RENDERED and "chat_template_kwargs" in c.evidence
    # the cache absorbed it (the tail moved, the prefix did not): a hit, the change kept as secondary
    hit = snap(4200, 4000, chat_template_kwargs={"enable_thinking": False})
    c = classify(prev, hit)
    assert c.cause is MissCause.NONE and MissCause.PARAM_RENDERED in c.secondary
    # a per-request template
    assert classify(snap(4000, 0), snap(4200, 0, chat_template="{{ messages }}")).cause is MissCause.PARAM_RENDERED


def test_explain_sees_the_same_parameters():
    body = {"model": "m", "messages": [{"role": "system", "content": "x " * 300}, {"role": "user", "content": "hi"}]}
    a = Turn(body={**body, "chat_template_kwargs": {"enable_thinking": True}},
             usage={"prompt_tokens": 4000, "prompt_tokens_details": {"cached_tokens": 0}})
    b = Turn(body={**body, "chat_template_kwargs": {"enable_thinking": False}},
             usage={"prompt_tokens": 4000, "prompt_tokens_details": {"cached_tokens": 0}})
    assert explain_pair(a, b, "vllm").cause == "param_rendered"


def test_rounded_read_with_a_param_change_is_a_hit():
    """Live (vLLM, 16-token blocks, experiments/e_template): a prompt of 407 tokens read back as
    400, and a toggle of enable_thinking on top. The toggle renders only a 4-token tail stub, the
    prefix was served to within one block: a hit with the change as secondary, not a cause."""
    prev = snap(407, 0, chat_template_kwargs={"enable_thinking": True})
    curr = snap(489, 400, chat_template_kwargs={"enable_thinking": False})
    c = classify(prev, curr)
    assert c.cause is MissCause.NONE and MissCause.PARAM_RENDERED in c.secondary
    assert "whole cache blocks" in c.evidence
    # the same numbers without a change: the same hit; a previous prompt whose tail was its
    # generation prompt (never reusable) reads a block further short, still a hit
    assert classify(snap(407, 0), snap(489, 400)).cause is MissCause.NONE
    assert classify(snap(579, 0), snap(616, 560)).cause is MissCause.NONE
    # a read that stopped short of the prefix is not rounded up to a hit
    short = classify(prev, snap(489, 16, chat_template_kwargs={"enable_thinking": False}))
    assert short.cause is MissCause.PARAM_RENDERED
    assert classify(snap(50, 0), snap(70, 16)).cause is not MissCause.NONE
    assert classify(snap(240, 0), snap(300, 128)).cause is not MissCause.NONE  # 112 tokens recomputed
    # a count a whole block short is not a full read: OpenAI reports a full 2,048 as 2,048, not 1,920
    assert classify(snap(2048, 0, thinking="a"), snap(2300, 1920, thinking="b")).cause is MissCause.PARAM_RENDERED
    assert classify(snap(2048, 0), snap(2300, 1920)).cause is not MissCause.NONE or \
        "whole cache blocks" not in classify(snap(2048, 0), snap(2300, 1920)).evidence
    assert classify(snap(1100, 0, thinking="a"), snap(1300, 1024, thinking="b")).cause is MissCause.NONE
    # Anthropic reports exact counts: no slack there
    a_prev = RequestSnapshot(model="m", provider="anthropic", blocks=BLOCKS, prompt_tokens=4000, cached_tokens=0,
                             cache_written=4000, has_breakpoint=True, params={"thinking": "a"})
    a_curr = RequestSnapshot(model="m", provider="anthropic", blocks=BLOCKS, prompt_tokens=4200, cached_tokens=3990,
                             has_breakpoint=True, params={"thinking": "b"})
    assert classify(a_prev, a_curr).cause is MissCause.PARAM_RENDERED


def test_a_known_block_size_tightens_the_slack():
    """With the engine's block size on the record (from its KV events), the slack is that block plus
    the generation tail, not the 128-token ceiling: a divergence 112 tokens before the end of a
    2,000-token prompt is caught on vLLM (16-token blocks), while the live reads, up to 25 tokens
    short (a block and a stub), stay hits. On SGLang (page size 1) a 597-of-600 read is a hit."""
    def esnap(prompt, cached, block, **params):
        s = snap(prompt, cached, **params)
        s.block_size = block
        return s
    assert classify(esnap(579, 0, 16), esnap(616, 560, 16)).cause is MissCause.NONE
    assert classify(esnap(617, 0, 16), esnap(683, 592, 16)).cause is MissCause.NONE
    assert classify(esnap(600, 0, 1), esnap(640, 597, 1)).cause is MissCause.NONE
    far = classify(esnap(2000, 0, 16, thinking="a"), esnap(2200, 1888, 16, thinking="b"))
    assert far.cause is MissCause.PARAM_RENDERED
    # the same read with the block size unknown falls back to the 128-token ceiling: a hit
    assert classify(snap(2000, 0, thinking="a"), snap(2200, 1888, thinking="b")).cause is MissCause.NONE


def test_the_watcher_remembers_the_block_size_until_a_restart():
    from prefixwise.engines import EngineEvents, EngineFleet
    w = EngineEvents()
    assert w.block_size is None
    w._store({"block_hashes": [1, 2], "block_size": 16, "session_id": None})
    assert w.block_size == 16
    w._clear()  # the engine restarted: it may come back with another block size
    assert w.block_size is None
    fleet = EngineFleet([("tcp://a:1", "a"), ("tcp://b:2", "b")], make=lambda *a, **k: EngineEvents())
    assert fleet.block_size is None
    fleet.sources[0]._store({"block_hashes": [1], "block_size": 16, "session_id": None})
    assert fleet.block_size == 16  # a rank that has not stored yet does not count
    fleet.sources[1]._store({"block_hashes": [2], "block_size": 32, "session_id": None})
    assert fleet.block_size is None  # they differ: nothing is claimed
