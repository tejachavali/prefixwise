from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.fingerprint import Fingerprinter

FP = Fingerprinter()


def snap(model="gpt-5.6", provider="openai", system="You are helpful.", tools=None, **kw):
    blocks = []
    if tools is not None:
        for tool in tools:
            blocks.append(("tools", tool))
    blocks.append(("system", system))
    blocks.append(("messages", {"role": "user", "content": "hello"}))
    return RequestSnapshot(model=model, provider=provider, blocks=FP.fingerprint_blocks(blocks), **kw)


def test_first_write():
    assert classify(None, snap()).cause is MissCause.FIRST_WRITE


def test_model_switch_is_legit():
    c = classify(snap(model="gpt-5.6"), snap(model="claude-sonnet-4.5"))
    assert c.cause is MissCause.MODEL_SWITCH and c.legit is True


def test_volatile_system_not_system_edit():
    prev = snap(system="Today is 2026-11-05.")
    curr = snap(system="Today is 2026-11-06.")
    c = classify(prev, curr)
    assert c.cause is MissCause.VOLATILE_SYSTEM and c.legit is False


def test_system_edit():
    c = classify(snap(system="Be helpful."), snap(system="Be concise and helpful."))
    assert c.cause is MissCause.SYSTEM_EDIT


def test_tool_reserialize_vs_catalog_change():
    tools_a = [{"name": "search", "description": "search the web"},
               {"name": "calc", "description": "calculate"}]
    tools_b = tools_a + [{"name": "fetch", "description": "fetch a url"}]
    # Same multiset, different order: TOOL_RESERIALIZE, not a catalog change.
    prev = snap(tools=tools_a)
    curr = snap(tools=list(reversed(tools_a)))
    assert classify(prev, curr).cause is MissCause.TOOL_RESERIALIZE
    c = classify(prev, snap(tools=tools_b))
    assert c.cause is MissCause.TOOL_CATALOG_CHANGE and c.legit is True


def test_compaction_rewrite_needs_event():
    prev = snap()
    curr = snap()
    curr.blocks = FP.fingerprint_blocks([("system", "You are helpful."),
                                         ("messages", {"role": "user", "content": "different history"})])
    curr.compaction_event = True
    assert classify(prev, curr).cause is MissCause.COMPACTION_REWRITE
    curr.compaction_event = False
    assert classify(prev, curr).cause is MissCause.HISTORY_EDIT


def test_param_rendered():
    prev = snap()
    curr = snap()
    curr.params = {"tool_choice": "auto"}
    prev.params = {"tool_choice": "none"}
    assert classify(prev, curr).cause is MissCause.PARAM_RENDERED


def test_unknown_when_no_evidence():
    prev = snap(prompt_tokens=2000, cached_tokens=0)
    curr = snap(prompt_tokens=2000, cached_tokens=0)
    # Blocks equal, no param change, no TTL, no eviction: routing or unknown.
    assert classify(prev, curr).cause in (MissCause.ROUTING_MISS, MissCause.UNKNOWN)


def _conv(*msgs, system="You are helpful."):
    return [("system", system)] + [("messages", m) for m in msgs]


def test_append_only_growth_is_not_history_edit():
    u1 = {"role": "user", "content": "hi"}
    a1 = {"role": "assistant", "content": "hello"}
    u2 = {"role": "user", "content": "next question"}
    prev = RequestSnapshot(model="m", provider="vllm", blocks=FP.fingerprint_blocks(_conv(u1)),
                           prompt_tokens=2000, cached_tokens=0)
    curr = RequestSnapshot(model="m", provider="vllm", blocks=FP.fingerprint_blocks(_conv(u1, a1, u2)),
                           prompt_tokens=2400, cached_tokens=1900)
    assert classify(prev, curr).cause is MissCause.NONE


def test_append_with_cold_cache_is_not_history_edit():
    # Growth with cache_read 0 is a routing/unknown question, never HISTORY_EDIT.
    u1 = {"role": "user", "content": "hi"}
    prev = RequestSnapshot(model="m", provider="vllm", blocks=FP.fingerprint_blocks(_conv(u1)),
                           prompt_tokens=2000)
    curr = RequestSnapshot(model="m", provider="vllm",
                           blocks=FP.fingerprint_blocks(_conv(u1, {"role": "assistant", "content": "x"})),
                           prompt_tokens=2100, cached_tokens=0, gap_seconds=5)
    assert classify(prev, curr).cause is MissCause.ROUTING_MISS


def test_edit_inside_prior_prefix_is_history_edit():
    u1 = {"role": "user", "content": "hi"}
    a1 = {"role": "assistant", "content": "hello"}
    prev = RequestSnapshot(model="m", blocks=FP.fingerprint_blocks(_conv(u1, a1)))
    edited = {"role": "assistant", "content": "hello, edited"}
    curr = RequestSnapshot(model="m", blocks=FP.fingerprint_blocks(
        _conv(u1, edited, {"role": "user", "content": "more"})))
    assert classify(prev, curr).cause is MissCause.HISTORY_EDIT


def test_truncated_history_is_history_edit():
    u1 = {"role": "user", "content": "hi"}
    a1 = {"role": "assistant", "content": "hello"}
    prev = RequestSnapshot(model="m", blocks=FP.fingerprint_blocks(_conv(u1, a1)))
    curr = RequestSnapshot(model="m", blocks=FP.fingerprint_blocks(_conv(u1)))
    assert classify(prev, curr).cause is MissCause.HISTORY_EDIT



def _pair(gap=None, ttl=None, cached=0, **curr_kw):
    prev = snap(prompt_tokens=2000)
    curr = snap(prompt_tokens=2000, cached_tokens=cached, gap_seconds=gap, ttl_seconds=ttl, **curr_kw)
    return classify(prev, curr)


def test_ttl_expired_only_when_cache_read_low():
    assert _pair(gap=900, ttl=300).cause is MissCause.TTL_EXPIRED
    assert _pair(gap=900, ttl=300, cached=1900).cause is MissCause.NONE  # cache survived


def test_routing_miss_needs_a_short_gap():
    assert _pair(gap=90).cause is MissCause.ROUTING_MISS  # past the 60 s retry window
    assert _pair(gap=200, ttl=3600).cause is MissCause.ROUTING_MISS
    c = _pair(gap=1200)  # 20 min, TTL unknown: expiry or routing, do not guess
    assert c.cause is MissCause.UNKNOWN and "unknown TTL" in c.evidence
    assert _pair(gap=None).cause is MissCause.UNKNOWN


def test_evicted_only_when_cache_read_low():
    assert _pair(gap=30, evicted_block_hashes=["h1"]).cause is MissCause.EVICTED
    assert _pair(gap=30, cached=1900, evicted_block_hashes=["h1"]).cause is MissCause.NONE


def test_fork_boundary_on_first_turn_only():
    first = snap(fork_parent_session_id="parent")
    c = classify(None, first)
    assert c.cause is MissCause.FORK_BOUNDARY and c.legit is True
    later = snap(fork_parent_session_id="parent", prompt_tokens=2000, cached_tokens=1900)
    assert classify(snap(prompt_tokens=2000), later).cause is MissCause.NONE


def _with_tool_results(*results, compaction=False):
    blocks = [("system", "You are helpful."), ("messages", {"role": "user", "content": "run it"})]
    blocks += [("tool_results", {"role": "tool", "tool_call_id": f"c{i}", "content": r})
               for i, r in enumerate(results)]
    return RequestSnapshot(model="m", blocks=FP.fingerprint_blocks(blocks), compaction_event=compaction)


def test_tool_result_mutation_and_clearing():
    prev = _with_tool_results("output A", "output B")
    assert classify(prev, _with_tool_results("output A", "output B", "output C")).cause \
        is not MissCause.TOOL_RESULT_MUTATION  # appending a result is growth
    c = classify(prev, _with_tool_results("output A (re-run)", "output B"))
    assert c.cause is MissCause.TOOL_RESULT_MUTATION and c.legit is False
    c = classify(prev, _with_tool_results("[cleared]", "output B", compaction=True))
    assert c.cause is MissCause.COMPACTION_REWRITE and c.legit is True
