"""Observed cache lifetimes (#172): learned from hits and cold reads on an unchanged prefix, used only
when no TTL is known, and reported as observed, never promised."""
from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.findings import findings, render
from prefixwise.fingerprint import Fingerprinter
from prefixwise.lifetime import LifetimeTracker, Observed, estimate
from prefixwise.sdk import Recorder

FP = Fingerprinter()


def test_the_window_needs_evidence_on_each_side():
    assert estimate([], []) is None
    assert estimate([120, 300], [2000, 2500]) is None                 # two of each: not enough
    o = estimate([120, 300, 600, 30], [1500, 2000, 2500, 200])        # a cold read at 200 s is not past a hit
    assert o.alive_up_to == 600 and o.hits == 3                       # the 30 s hit says nothing about lifetime
    assert o.cold_from == 1500 and o.colds == 3
    o = estimate([120, 300, 600, 3000], [1500, 2000, 2500])           # a hit after every cold gap: no expiry claimed
    assert o.cold_from is None and o.alive_up_to == 3000
    assert "up to 50 min (4 turns)" in o.describe()


def test_the_tracker_keeps_models_apart():
    t = LifetimeTracker()
    for g in (120, 300, 600):
        t.observe("openai", "gpt-5", g, hit=True)
    assert t.estimate("openai", "gpt-5").alive_up_to == 600
    assert t.estimate("openai", "gpt-4.1") is None
    t.observe("openai", "gpt-5", 900, hit=True)
    assert t.estimate("openai", "gpt-5").alive_up_to == 900           # the cached estimate is refreshed


def _pair(gap, cached=0, ttl=None, observed=None):
    blocks = [FP.fingerprint_block("system", 0, "S" * 50), FP.fingerprint_block("user", 1, "U")]
    prev = RequestSnapshot(model="gpt-5", provider="openai", blocks=blocks, prompt_tokens=5000, cached_tokens=4000)
    curr = RequestSnapshot(model="gpt-5", provider="openai", blocks=blocks + [FP.fingerprint_block("user", 2, "V")],
                           prompt_tokens=5100, cached_tokens=cached, gap_seconds=gap, ttl_seconds=ttl,
                           observed_lifetime=observed)
    return classify(prev, curr)


OBS = Observed(alive_up_to=600, cold_from=1500, hits=5, colds=4)


def test_the_classifier_uses_the_window_only_when_no_ttl_is_known():
    c = _pair(400, observed=OBS)
    assert c.cause is MissCause.ROUTING_MISS and "observed lifetime" in c.evidence and c.cold_unchanged
    c = _pair(2000, observed=OBS)
    assert c.cause is MissCause.TTL_EXPIRED and c.legit is True and "past the observed lifetime" in c.evidence
    c = _pair(900, observed=OBS)  # between the two: still not distinguishable
    assert c.cause is MissCause.UNKNOWN and "observed on this model" in c.evidence
    assert _pair(400).cause is MissCause.UNKNOWN                      # nothing learned: as before
    assert _pair(400, ttl=3600, observed=OBS).cause is MissCause.ROUTING_MISS  # a stated TTL wins
    assert "TTL 3600s" in _pair(400, ttl=3600, observed=OBS).evidence
    assert _pair(2000, ttl=3600, observed=OBS).cause is MissCause.ROUTING_MISS
    assert _pair(2000, cached=4500, observed=OBS).cause is MissCause.NONE  # a hit is a hit


def _turns(rec, session, t0, gap, cached_second):
    system = {"role": "system", "content": "You are a careful agent. " * 400}
    first = {"model": "gpt-5", "messages": [system, {"role": "user", "content": "start " + session}]}
    second = {"model": "gpt-5", "messages": first["messages"] + [{"role": "assistant", "content": "ok"},
                                                              {"role": "user", "content": "next"}]}
    usage = lambda p, c: {"id": "x", "model": "gpt-5", "usage": {"prompt_tokens": p, "completion_tokens": 1,  # noqa: E731
                                                                 "prompt_tokens_details": {"cached_tokens": c}}}
    rec.record("openai", first, usage(5000, 0), session_id=session, latency_ms=100, ended_at=t0)
    return rec.record("openai", second, usage(5010, cached_second), session_id=session, latency_ms=100,
                      ended_at=t0 + gap + 0.1)


def test_learned_from_recorded_turns_and_reported(tmp_path):
    rec = Recorder(str(tmp_path / "l.db"))
    t = 1_800_000_000.0
    for i, gap in enumerate((120, 300, 600)):
        assert _turns(rec, f"hit{i}", t + i * 10_000, gap, 4990).cause == "none"
    for i, gap in enumerate((1500, 1800, 2400)):
        _turns(rec, f"cold{i}", t + 100_000 + i * 10_000, gap, 0)
    row = _turns(rec, "late", t + 200_000, 3000, 0)
    assert row.cause == "ttl_expired" and "past the observed lifetime" in row.evidence
    row = _turns(rec, "soon", t + 210_000, 450, 0)
    assert row.cause == "routing_miss" and "observed lifetime" in row.evidence
    f = findings(rec.ledger)
    obs = f["observed_lifetimes"]
    assert len(obs) == 1 and obs[0]["model"] == "gpt-5" and obs[0]["alive_up_to_seconds"] == 600
    text = render(f)
    assert "Observed cache lifetime, gpt-5 (openai): read after idle gaps up to 10 min" in text
    assert "not a promise" in text
    # A new collector on the same ledger starts from what was learned.
    again = Recorder(str(tmp_path / "l.db"))
    assert again.proxy.collector.lifetimes.estimate("openai", "gpt-5").alive_up_to == 600
