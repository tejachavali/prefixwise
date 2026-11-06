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
