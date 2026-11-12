"""Invariants of the classifier, checked over generated conversations.

Each case builds a real request body (tools with JSON schemas, a system prompt that may carry a
timestamp, a multi-turn conversation with tool calls and results) in Anthropic's or OpenAI's
shape, fingerprints it exactly as the proxy does, applies ONE known change for the next turn, and
checks that the verdict names that change. Seeded, so a failure reproduces: the case id names the
seed. These are the rules docs/ALGORITHM.md states; a failure here is a bug in one or the other."""

import copy
import json
import random

import pytest

from prefixwise.classifier import (MissCause, RequestSnapshot, classify, first_divergence, served_all)
from prefixwise.fingerprint import Fingerprinter
from prefixwise.ledger import avoidable_for_turn
from prefixwise.proxy import extract_blocks

FP = Fingerprinter()
MODELS = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-5"}
WORDS = ("search", "read", "write", "list", "grep", "fetch", "edit", "run", "plan", "test", "diff", "open")
WASTE = {MissCause.TOOL_RESERIALIZE, MissCause.VOLATILE_SYSTEM, MissCause.HISTORY_EDIT,
         MissCause.TOOL_RESULT_MUTATION, MissCause.RETRY_REPLAY, MissCause.BREAKPOINT_MISPLACED}
LEGIT = {MissCause.FIRST_WRITE, MissCause.MODEL_SWITCH, MissCause.TOOL_CATALOG_CHANGE,
         MissCause.COMPACTION_REWRITE, MissCause.FORK_BOUNDARY, MissCause.TTL_EXPIRED}
CHANGE_CAUSES = {MissCause.TOOL_RESERIALIZE, MissCause.TOOL_CATALOG_CHANGE, MissCause.VOLATILE_SYSTEM,
                 MissCause.SYSTEM_EDIT, MissCause.HISTORY_EDIT, MissCause.TOOL_RESULT_MUTATION,
                 MissCause.COMPACTION_REWRITE}
ROLE_OF = {MissCause.TOOL_RESERIALIZE: "tools", MissCause.TOOL_CATALOG_CHANGE: "tools",
           MissCause.VOLATILE_SYSTEM: "system", MissCause.SYSTEM_EDIT: "system",
           MissCause.HISTORY_EDIT: "messages", MissCause.TOOL_RESULT_MUTATION: "tool_results"}


# -- generating conversations ---------------------------------------------------------------

def _tool(rng, provider, name):
    props = {f"{w}_{i}": {"type": rng.choice(["string", "integer", "boolean"]),
                          "description": f"the {w} to use"} for i, w in enumerate(rng.sample(WORDS, rng.randint(2, 4)))}
    schema = {"type": "object", "properties": props, "required": sorted(props)[: rng.randint(2, len(props))]}
    desc = f"{name.capitalize()} things. " * rng.randint(1, 4)
    if provider == "anthropic":
        return {"name": name, "description": desc, "input_schema": schema}
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}


def _system(rng, stamp):
    text = "You are a careful coding agent. " * rng.randint(20, 60)
    return text + (f"Current time: {stamp}." if stamp else "")


def _conversation(rng, provider, turns):
    """`turns` user turns; each but the last followed by a tool call, its result and a reply."""
    msgs = []
    for t in range(turns):
        msgs.append({"role": "user", "content": f"Task step {t}: " + " ".join(rng.choices(WORDS, k=12))})
        if t == turns - 1:
            break
        call_id = f"call_{t}_{rng.randrange(10**6)}"
        out = "result: " + " ".join(rng.choices(WORDS, k=30))
        if provider == "anthropic":
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": call_id, "name": "search", "input": {"q": f"x{t}"}}]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": out}]})
        else:
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "search", "arguments": json.dumps({"q": f"x{t}"})}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": out})
        msgs.append({"role": "assistant", "content": "Done with step %d." % t})
    return msgs


def _body(rng, provider, n_tools, turns, stamp):
    names = rng.sample(WORDS, n_tools)
    tools = [_tool(rng, provider, n) for n in names]
    system = _system(rng, stamp)
    msgs = _conversation(rng, provider, turns)
    if provider == "anthropic":
        return {"model": MODELS[provider], "max_tokens": 64, "tools": tools, "system": system, "messages": msgs}
    return {"model": MODELS[provider], "tools": tools, "messages": [{"role": "system", "content": system}] + msgs}


def _append_turn(body):
    """The ordinary next turn: the assistant's reply and a new user message."""
    b = copy.deepcopy(body)
    b["messages"].append({"role": "assistant", "content": "Here is what I found."})
    b["messages"].append({"role": "user", "content": "Good, continue with the next part."})
    return b


def _system_text(body, provider):
    return body["system"] if provider == "anthropic" else body["messages"][0]["content"]


def _set_system(body, provider, text):
    if provider == "anthropic":
        body["system"] = text
    else:
        body["messages"][0]["content"] = text


def _conv_msgs(body, provider):
    return body["messages"] if provider == "anthropic" else body["messages"][1:]


# -- the changes: each returns (next body, the cause it must produce) or None if not applicable ----

def m_reorder_tools(rng, b, p):
    if len(b["tools"]) < 2:
        return None
    order = list(range(len(b["tools"])))
    while order == sorted(order):
        rng.shuffle(order)
    b["tools"] = [b["tools"][i] for i in order]
    return b, MissCause.TOOL_RESERIALIZE


def m_reorder_keys(rng, b, p):
    i = rng.randrange(len(b["tools"]))
    b["tools"][i] = dict(reversed(list(b["tools"][i].items())))
    return b, MissCause.TOOL_RESERIALIZE


def m_required_order(rng, b, p):
    i = rng.randrange(len(b["tools"]))
    schema = b["tools"][i]["input_schema"] if p == "anthropic" else b["tools"][i]["function"]["parameters"]
    if len(schema["required"]) < 2:
        return None
    schema["required"] = list(reversed(schema["required"]))
    return b, MissCause.TOOL_RESERIALIZE


def m_add_tool(rng, b, p):
    names = {t["name"] if p == "anthropic" else t["function"]["name"] for t in b["tools"]}
    free = [w for w in WORDS if w not in names]
    b["tools"].insert(rng.randrange(len(b["tools"]) + 1), _tool(rng, p, rng.choice(free) + "_new"))
    return b, MissCause.TOOL_CATALOG_CHANGE


def m_remove_tool(rng, b, p):
    if len(b["tools"]) < 2:
        return None
    b["tools"].pop(rng.randrange(len(b["tools"])))
    return b, MissCause.TOOL_CATALOG_CHANGE


def m_edit_tool(rng, b, p):
    i = rng.randrange(len(b["tools"]))
    t = b["tools"][i] if p == "anthropic" else b["tools"][i]["function"]
    t["description"] = t["description"] + " Now also handles archives."
    return b, MissCause.TOOL_CATALOG_CHANGE


def m_system_stamp(rng, b, p):
    text = _system_text(b, p)
    if "Current time: " not in text:
        return None
    _set_system(b, p, text.rsplit("Current time: ", 1)[0] + "Current time: 2026-11-12T10:%02d:%02dZ."
                % (rng.randrange(60), rng.randrange(60)))
    return b, MissCause.VOLATILE_SYSTEM


def m_system_word(rng, b, p):
    _set_system(b, p, _system_text(b, p).replace("careful", "thorough", 1))
    return b, MissCause.SYSTEM_EDIT


def m_edit_history(rng, b, p):
    msgs = _conv_msgs(b, p)
    users = [i for i, m in enumerate(msgs[:-1]) if m["role"] == "user" and isinstance(m["content"], str)]
    if len(msgs) < 3 or not users:
        return None
    msgs[rng.choice(users)]["content"] += " (edited)"
    return b, MissCause.HISTORY_EDIT


def m_edit_tool_result(rng, b, p):
    msgs = _conv_msgs(b, p)
    idx = [i for i, m in enumerate(msgs) if (m["role"] == "tool" if p == "openai" else
                                               isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result")]
    if not idx:
        return None
    m = msgs[rng.choice(idx)]
    if p == "openai":
        m["content"] += " [re-rendered]"
    else:
        m["content"][0]["content"] += " [re-rendered]"
    return b, MissCause.TOOL_RESULT_MUTATION


MUTATIONS = [m_reorder_tools, m_reorder_keys, m_required_order, m_add_tool, m_remove_tool, m_edit_tool,
             m_system_stamp, m_system_word, m_edit_history, m_edit_tool_result]


def _snap(body, provider, prompt, cached, gap=5.0, ttl=300.0, engine=False, **kw):
    return RequestSnapshot(model=body["model"], provider=provider,
                           blocks=FP.fingerprint_blocks(extract_blocks(body, provider)),
                           prompt_tokens=prompt, cached_tokens=cached, gap_seconds=gap, ttl_seconds=ttl,
                           has_breakpoint=True, engine=engine, **kw)


def _cases(n):
    for seed in range(n):
        rng = random.Random(seed)
        provider = rng.choice(["anthropic", "openai"])
        body = _body(rng, provider, rng.randint(2, 6), rng.randint(2, 5),
                     "2026-11-12T09:00:00Z" if rng.random() < 0.6 else None)
        mutation = rng.choice(MUTATIONS)
        yield seed, rng, provider, body, mutation


CASES = list(_cases(600))


# -- invariants -------------------------------------------------------------------------------

@pytest.mark.parametrize("seed,rng,provider,body,mutation", CASES, ids=[f"seed{c[0]}-{c[4].__name__}" for c in CASES])
def test_one_change_is_named_and_a_full_read_is_never_a_miss(seed, rng, provider, body, mutation):
    applied = mutation(rng, _append_turn(body), provider)
    if applied is None:
        pytest.skip("change not applicable to this conversation")
    nxt, expected = applied
    prev = _snap(body, provider, 20_000, 0)
    engine = provider == "openai" and rng.random() < 0.3  # behind vLLM/SGLang: system before tools
    # 1. A miss (nothing read): the verdict is exactly the change made, with its legitimacy.
    miss = classify(prev, _snap(nxt, provider, 20_400, 0, engine=engine))
    assert miss.cause is expected, (seed, mutation.__name__, miss.cause, miss.evidence)
    assert miss.legit is (False if expected in WASTE else True if expected in LEGIT else None)
    # 2. The ledger's "first changed" column agrees with the cause about where the prefix broke.
    div = first_divergence(prev.blocks, _snap(nxt, provider, 1, 0).blocks, engine=engine)
    if expected in ROLE_OF:
        assert div is not None and ROLE_OF[expected] in div["role"], (seed, expected, div)
    # 3. A partial read (half of the prefix): still the change, not a hit.
    part = classify(prev, _snap(nxt, provider, 20_400, 10_000, engine=engine))
    assert part.cause is expected and part.legit is miss.legit
    # 4. The cache served the whole reusable prefix: no miss is claimed, the change stays secondary.
    full = classify(prev, _snap(nxt, provider, 20_400, 20_000, engine=engine))
    assert full.cause is MissCause.NONE and expected in full.secondary and full.legit is None
    # 5. Avoidable tokens: only for waste, never more than what was reusable minus what was read.
    for verdict, cached in ((miss, 0), (part, 10_000), (full, 20_000)):
        avoid = avoidable_for_turn(verdict.legit, 20_400, cached, 20_000)
        assert 0 <= avoid <= 20_000 - cached
        assert (avoid > 0) == (verdict.legit is False and cached < 20_000)


@pytest.mark.parametrize("seed", range(200))
def test_appending_is_never_a_change(seed):
    rng = random.Random(10_000 + seed)
    provider = rng.choice(["anthropic", "openai"])
    body = _body(rng, provider, rng.randint(1, 6), rng.randint(1, 5), "2026-11-12T09:00:00Z")
    prev, nxt = _snap(body, provider, 20_000, 0), _append_turn(body)
    for cached in (0, 7_000, 20_000):
        v = classify(prev, _snap(nxt, provider, 20_400, cached))
        assert v.cause not in CHANGE_CAUSES and not (set(v.secondary) & CHANGE_CAUSES), (seed, cached, v)
    assert classify(prev, _snap(nxt, provider, 20_400, 20_000)).cause is MissCause.NONE
    div = first_divergence(prev.blocks, _snap(nxt, provider, 1, 0).blocks)
    assert div is not None and "appended" in div["kind"]


@pytest.mark.parametrize("seed", range(100))
def test_engine_evidence_decides_only_when_the_prompt_did_not_change(seed):
    rng = random.Random(20_000 + seed)
    body = _body(rng, "openai", rng.randint(2, 5), rng.randint(2, 4), None)
    prev, nxt = _snap(body, "openai", 20_000, 0, engine=True), _append_turn(body)
    curr = lambda **kw: _snap(nxt, "openai", 20_400, 0, engine=True, **kw)  # noqa: E731
    assert classify(prev, curr(evicted_blocks=12)).cause is MissCause.EVICTED
    assert classify(prev, curr(routed=("rank 0", "rank 1", 40))).cause is MissCause.ROUTING_MISS
    assert classify(prev, curr(evicted_blocks=0, resident_blocks=40)).cause is MissCause.UNKNOWN  # not an eviction
    applied = rng.choice([m_reorder_tools, m_system_word, m_edit_history])(rng, copy.deepcopy(nxt), "openai")
    if applied:  # a changed prompt is the cause, whatever the engine saw
        changed, expected = applied
        v = classify(prev, _snap(changed, "openai", 20_400, 0, engine=True, evicted_blocks=12))
        assert v.cause is expected
        v = classify(prev, _snap(changed, "openai", 20_400, 0, engine=True, routed=("rank 0", "rank 1", 40)))
        assert v.cause is expected


@pytest.mark.parametrize("seed", range(100))
def test_model_switch_and_first_turn_win_and_classification_is_deterministic(seed):
    rng = random.Random(30_000 + seed)
    provider = rng.choice(["anthropic", "openai"])
    body = _body(rng, provider, 3, 3, "2026-11-12T09:00:00Z")
    nxt = rng.choice(MUTATIONS)(rng, _append_turn(body), provider)
    nxt = nxt[0] if nxt else _append_turn(body)
    other = dict(nxt, model="claude-opus-4-6" if provider == "anthropic" else "gpt-5-mini")
    prev = _snap(body, provider, 20_000, 0)
    assert classify(prev, _snap(other, provider, 20_400, 0)).cause is MissCause.MODEL_SWITCH
    assert classify(None, _snap(nxt, provider, 20_400, 0)).cause is MissCause.FIRST_WRITE
    a = classify(prev, _snap(nxt, provider, 20_400, 0))
    b = classify(prev, _snap(nxt, provider, 20_400, 0))
    assert (a.cause, a.legit, a.evidence, a.secondary) == (b.cause, b.legit, b.evidence, b.secondary)


def test_served_all_is_monotonic_in_what_was_read():
    rng = random.Random(1)
    body = _body(rng, "openai", 3, 3, None)
    prev, nxt = _snap(body, "openai", 20_000, 0), _append_turn(body)
    flags = [served_all(prev, _snap(nxt, "openai", 20_400, c)) for c in range(0, 20_001, 100)]
    assert flags == sorted(flags)  # once a read counts as full, every larger read does too
    assert flags[-1] and not flags[0]


# -- two changes: the earliest one in the prompt is where the cache stopped matching ---------------

def _first_changed_cause(prev_body, nxt_body, provider, engine):
    """Ground truth: walk the two requests' blocks in the order the provider renders them and
    return the role of the first block that is not byte-identical (append-only growth excluded)."""
    pb = FP.fingerprint_blocks(extract_blocks(prev_body, provider))
    cb = FP.fingerprint_blocks(extract_blocks(nxt_body, provider))
    if engine:  # the chat template renders the system prompt before the tools
        key = {"system": 0, "tools": 1}
        pb = sorted(pb, key=lambda b: key.get(b.role, 2))
        cb = sorted(cb, key=lambda b: key.get(b.role, 2))
    rank = ({"system": 0, "tools": 1} if engine else {"tools": 0, "system": 1})
    for a, b in zip(pb, cb):
        if a.role != b.role:  # a block added or removed here: the earlier section is the one that changed
            return min(a.role, b.role, key=lambda r: rank.get(r, 2))
        if a.raw != b.raw:
            return a.role
    return None


@pytest.mark.parametrize("seed", range(300))
def test_with_two_changes_the_earliest_one_is_named(seed):
    rng = random.Random(40_000 + seed)
    provider = rng.choice(["anthropic", "openai"])
    engine = provider == "openai" and rng.random() < 0.4
    body = _body(rng, provider, rng.randint(2, 5), rng.randint(3, 5), "2026-11-12T09:00:00Z")
    first, second = rng.sample(MUTATIONS, 2)
    a = first(rng, _append_turn(body), provider)
    if a is None:
        pytest.skip("first change not applicable")
    b = second(rng, a[0], provider)
    if b is None:
        pytest.skip("second change not applicable")
    nxt = b[0]
    role = _first_changed_cause(body, nxt, provider, engine)
    if role is None or role not in ("tools", "system", "messages", "tool_results"):
        pytest.skip("the second change undid the first")
    v = classify(_snap(body, provider, 20_000, 0, engine=engine), _snap(nxt, provider, 20_400, 0, engine=engine))
    assert v.cause in CHANGE_CAUSES, (seed, v)
    assert ROLE_OF[v.cause] == role, (seed, first.__name__, second.__name__, role, v.cause)
    div = first_divergence(_snap(body, provider, 1, 0).blocks, _snap(nxt, provider, 1, 0).blocks, engine=engine)
    assert div is not None and role in div["role"], (seed, role, div)


# -- the fingerprints every verdict rests on ---------------------------------------------------

def _rand_json(rng, depth=0):
    kind = rng.random()
    if depth > 2 or kind < 0.3:
        return rng.choice([rng.randrange(1000), "word " + rng.choice(WORDS), True, None, 3.5])
    if kind < 0.6:
        return [_rand_json(rng, depth + 1) for _ in range(rng.randint(1, 4))]
    return {f"{rng.choice(WORDS)}_{i}": _rand_json(rng, depth + 1) for i in range(rng.randint(1, 4))}


def _reorder_keys(value, rng):
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {k: _reorder_keys(v, rng) for k, v in items}
    if isinstance(value, list):
        return [_reorder_keys(v, rng) for v in value]
    return value


@pytest.mark.parametrize("seed", range(300))
def test_hashes_ignore_exactly_what_they_should(seed):
    rng = random.Random(50_000 + seed)
    value = {"name": rng.choice(WORDS), "spec": _rand_json(rng)}
    a = FP.fingerprint_block("tools", 0, value)
    # Key order at any depth: canonical and masked equal (the same tool), raw may differ.
    b = FP.fingerprint_block("tools", 0, _reorder_keys(value, rng))
    assert (a.canonical, a.masked) == (b.canonical, b.masked)
    # Any change of content: canonical differs (a different tool), and so does raw.
    changed = copy.deepcopy(value)
    changed["spec_extra"] = rng.choice(WORDS)
    c = FP.fingerprint_block("tools", 0, changed)
    assert c.canonical != a.canonical and c.raw != a.raw
    # Volatile values in text: masked equal, raw and canonical differ; a word change: masked differs.
    stamp = lambda d, h, m: f"Today is 2026-11-{d:02d}T{h:02d}:{m:02d}:00Z, run {rng.randrange(16**8):08x}-aaaa-4bbb-8ccc-dddddddddddd"  # noqa: E731
    s1 = FP.fingerprint_block("system", 0, "You are careful. " + stamp(rng.randint(1, 28), rng.randrange(24), rng.randrange(60)))
    s2 = FP.fingerprint_block("system", 0, "You are careful. " + stamp(rng.randint(1, 28), rng.randrange(24), rng.randrange(60)))
    s3 = FP.fingerprint_block("system", 0, "You are thorough. " + stamp(1, 1, 1))
    assert s1.masked == s2.masked
    assert s3.masked != s1.masked


@pytest.mark.parametrize("seed", range(100))
def test_a_mid_conversation_developer_message_is_growth_not_a_system_edit(seed):
    """OpenAI-style agents append developer or system reminders mid-conversation. The provider
    renders them where they were sent, so appending one is growth; only the leading system and
    developer messages are the system prompt."""
    rng = random.Random(60_000 + seed)
    body = _body(rng, "openai", rng.randint(1, 4), rng.randint(2, 4), "2026-11-12T09:00:00Z")
    nxt = _append_turn(body)
    nxt["messages"].insert(len(nxt["messages"]) - 1, {"role": rng.choice(["developer", "system"]),
                                                      "content": "Reminder: keep answers short."})
    prev = _snap(body, "openai", 20_000, 0)
    for cached in (0, 20_000):
        v = classify(prev, _snap(nxt, "openai", 20_400, cached))
        assert v.cause not in CHANGE_CAUSES and not (set(v.secondary) & CHANGE_CAUSES), (seed, cached, v)
    # and the next turn, which repeats the reminder where it was, is still growth
    after = _append_turn(nxt)
    v = classify(_snap(nxt, "openai", 20_400, 0), _snap(after, "openai", 20_800, 0))
    assert v.cause not in CHANGE_CAUSES, (seed, v)


@pytest.mark.parametrize("seed", range(200))
def test_avoidable_never_exceeds_what_was_unread_of_the_shorter_prompt(seed):
    rng = random.Random(70_000 + seed)
    prev_p = rng.randint(1, 10**6)
    curr_p = rng.randint(1, 10**6)
    cached = rng.randint(0, curr_p)
    held = rng.choice([None, rng.randint(1, 10**6)])
    for legit in (False, True, None):
        avoid = avoidable_for_turn(legit, curr_p, cached, prev_p, held)
        assert 0 <= avoid <= max(0, min(prev_p, curr_p) - cached)
        if held:
            assert avoid <= max(0, held - cached)
        if legit is not False:
            assert avoid == 0
