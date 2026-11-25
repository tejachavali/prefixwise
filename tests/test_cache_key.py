"""--fix cache_key: a prompt_cache_key per conversation on OpenAI's own models, so conversations that
open with the same system prompt are not all routed to one cache machine (fixes.FIX_STATUS)."""
import json

from prefixwise.fixes import FIX_STATUS, cache_key_for, parse_fixes
from prefixwise.proxy import PrefixwiseProxy

SYSTEM = {"role": "system", "content": "You are a careful coding agent. " * 50}


def _turn(n, model="gpt-5.6", **extra):
    msgs = [SYSTEM, {"role": "user", "content": "fix the bug"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f"step {i}"}, {"role": "user", "content": "go on"}]
    return {"model": model, "messages": msgs, **extra}


def _proxy(fixes=("cache_key",)):
    p = PrefixwiseProxy(mock=True, fixes=list(fixes))
    sent = []
    real = p._forward

    def spy(provider, body, *a, **k):
        sent.append(json.loads(json.dumps(body)))
        return real(provider, body, *a, **k)

    p._forward = spy
    return p, sent


def test_each_conversation_gets_its_own_stable_key():
    p, sent = _proxy()
    for n in range(2):
        r, e = p.handle("openai", _turn(n), {"x-prefixwise-session-id": "alice"})
        assert "cache_key" in r["fixes_applied"] and "cache_key" in (e.fixes or "")
    p.handle("openai", _turn(0), {"x-prefixwise-session-id": "bob"})
    alice, alice2, bob = (b["prompt_cache_key"] for b in sent)
    assert alice == alice2 == cache_key_for("alice") and bob == cache_key_for("bob") and alice != bob
    assert alice.startswith("pw-") and "alice" not in alice  # a hash, not the session id


def test_never_over_the_clients_key_never_off_openai_never_without_a_session_and_off_by_default():
    p, sent = _proxy()
    p.handle("openai", _turn(0, prompt_cache_key="mine"), {"x-prefixwise-session-id": "s"})
    assert sent[-1]["prompt_cache_key"] == "mine"
    r, _ = p.handle("openai", _turn(0, model="deepseek-chat"), {"x-prefixwise-session-id": "s"})
    assert "prompt_cache_key" not in sent[-1] and "cache_key" not in r["fixes_applied"]
    p.handle("openai", _turn(0))  # no session id: nothing to key on
    assert "prompt_cache_key" not in sent[-1]
    p.handle("anthropic", {"model": "claude-sonnet-5", "max_tokens": 8, "system": "x",
                           "messages": [{"role": "user", "content": "q"}]}, {"x-prefixwise-session-id": "s"})
    assert "prompt_cache_key" not in sent[-1]
    off, sent_off = _proxy(fixes=())
    off.handle("openai", _turn(0), {"x-prefixwise-session-id": "s"})
    assert "prompt_cache_key" not in sent_off[-1]


def test_a_provider_that_refuses_the_field_gets_the_request_without_it():
    p = PrefixwiseProxy(mock=True, fixes=["cache_key"])
    real, sent = p._forward, []

    def refuse(provider, body, *a, **k):
        sent.append(json.loads(json.dumps(body)))
        if "prompt_cache_key" in body:
            return 400, {"error": {"message": "Unrecognized request argument supplied: prompt_cache_key"}}
        return real(provider, body, *a, **k)

    p._forward = refuse
    for _ in range(2):
        r, e = p.handle("openai", _turn(0), {"x-prefixwise-session-id": "s"})
        assert r["status"] == 200 and "cache_key" not in r["fixes_applied"] and "prompt_cache_key" not in sent[-1]
    assert ("openai", "prompt_cache_key") in p.rejected_fields
    p.handle("openai", _turn(1), {"x-prefixwise-session-id": "s"})
    assert "prompt_cache_key" not in sent[-1]  # no longer sent


def test_listed_as_recommended_not_proven():
    assert parse_fixes("cache_key") == ("cache_key",)
    assert "cache_key" not in parse_fixes("optimize")
    assert FIX_STATUS["cache_key"]["status"] == "recommended"


def test_a_rejected_key_is_not_in_the_record():
    """Found by review: the rejection undid the key in the request but not in the request's facts, so
    a later turn could be told its cache key changed although nothing sent had."""
    p = PrefixwiseProxy(mock=True, fixes=["cache_key"])
    ctx = p.prepare("openai", _turn(0), {"x-prefixwise-session-id": "s"})
    assert ctx["facts"].cache_key is not None and "prompt_cache_key" in ctx["forward_body"]
    err = {"error": {"message": "Unrecognized request argument supplied: prompt_cache_key"}}
    assert p._undo_rejected(ctx, 400, err)
    assert "prompt_cache_key" not in ctx["forward_body"] and ctx["facts"].cache_key is None
    assert "cache_key" not in ctx["fixes_applied"]


def test_a_user_id_alone_still_gets_a_key_and_the_key_is_keyed_when_a_secret_is_set():
    p, sent = _proxy()
    p.handle("openai", _turn(0, user="end-user-7"))  # only OpenAI's `user`: that user's conversations share a key
    assert sent[-1]["prompt_cache_key"] == cache_key_for("end-user-7")
    assert cache_key_for("alice", b"secret") != cache_key_for("alice")
    assert cache_key_for("alice", b"secret") == cache_key_for("alice", b"secret")
    keyed = PrefixwiseProxy(mock=True, fixes=["cache_key"], hmac_key=b"secret")
    seen = []
    real = keyed._forward
    keyed._forward = lambda prov, body, *a, **k: (seen.append(dict(body)), real(prov, body, *a, **k))[1]
    keyed.handle("openai", _turn(0), {"x-prefixwise-session-id": "alice"})
    assert seen[-1]["prompt_cache_key"] == cache_key_for("alice", b"secret")


def test_the_responses_api_gets_it_too():
    p, sent = _proxy()
    r, _ = p.handle("openai", {"model": "gpt-5.6", "input": [{"role": "user", "content": "hi"}],
                               "instructions": "be brief"}, {"x-prefixwise-session-id": "s"}, api="responses")
    assert sent[-1]["prompt_cache_key"] == cache_key_for("s") and "cache_key" in r["fixes_applied"]
