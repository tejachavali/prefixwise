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
    assert estimate([], [5, 10, 20, 400, 500, 600]) is None           # a cache that never hits is not expiring
    assert estimate([10, 20, 30, 40, 50], [400, 500, 600, 700, 800]) is None  # no hit after an idle gap
    hits = [120, 200, 300, 450, 600, 30]
    o = estimate(hits, [1500, 2000, 2500, 3000])                     # four colds: not enough
    assert o.alive_up_to == 120 and o.hits == 5 and o.cold_from is None  # the 5th longest idle hit
    o = estimate(hits, [1500, 2000, 2500, 3000, 3600, 200])          # a cold read at 200 s is not past a hit
    assert o.cold_from == 1500 and o.colds == 5
    o = estimate(hits + [4000], [1500, 2000, 2500, 3000, 3600])      # a hit after every cold gap: no expiry
    assert o.cold_from is None and o.alive_up_to == 200              # and one late hit does not stretch it
    o = estimate([70, 80, 90, 100, 110], [150, 160, 170, 180, 190])  # colds before 5 min prove nothing
    assert o.cold_from is None


def test_the_tracker_keeps_models_apart():
    t = LifetimeTracker()
    for g in (120, 200, 300, 450, 600):
        t.observe("openai", "gpt-5", g, hit=True)
    assert t.estimate("openai", "gpt-5").alive_up_to == 120
    assert t.estimate("openai", "gpt-4.1") is None
    for g in (700, 800, 900, 1000):
        t.observe("openai", "gpt-5", g, hit=True)
    assert t.estimate("openai", "gpt-5").alive_up_to == 600           # the cached estimate is refreshed


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
    def usage(p, c):
        return {"id": "x", "model": "gpt-5", "usage": {"prompt_tokens": p, "completion_tokens": 1,
                                                       "prompt_tokens_details": {"cached_tokens": c}}}
    rec.record("openai", first, usage(5000, 0), session_id=session, latency_ms=100, ended_at=t0)
    return rec.record("openai", second, usage(5010, cached_second), session_id=session, latency_ms=100,
                      ended_at=t0 + gap + 0.1)


def test_learned_from_recorded_turns_and_reported(tmp_path):
    rec = Recorder(str(tmp_path / "l.db"))
    t = 1_800_000_000.0
    for i, gap in enumerate((120, 200, 300, 450, 600)):
        assert _turns(rec, f"hit{i}", t + i * 10_000, gap, 4990).cause == "none"
    for i, gap in enumerate((1500, 1800, 2400, 2700, 3000)):
        _turns(rec, f"cold{i}", t + 100_000 + i * 10_000, gap, 0)
    row = _turns(rec, "late", t + 200_000, 3600, 0)
    assert row.cause == "ttl_expired" and "past the observed lifetime" in row.evidence
    assert row.lifetime_evidence == 0
    row = _turns(rec, "soon", t + 210_000, 100, 0)
    assert row.cause == "routing_miss"
    f = findings(rec.ledger)
    obs = f["observed_lifetimes"]
    assert len(obs) == 1 and obs[0]["model"] == "gpt-5" and obs[0]["alive_up_to_seconds"] == 120
    assert obs[0]["cold_from_seconds"] == 1500
    text = render(f)
    assert "Observed cache lifetime, gpt-5 (openai): read after idle gaps up to 2 min (5 turns)" in text
    assert "not a promise" in text
    # A new collector on the same ledger starts from exactly what was used.
    again = Recorder(str(tmp_path / "l.db"))
    assert again.proxy.collector.lifetimes.estimate("openai", "gpt-5") == \
        rec.proxy.collector.lifetimes.estimate("openai", "gpt-5")


def test_anthropic_rows_are_not_evidence(tmp_path):
    """Anthropic states its TTL: its turns teach nothing, and the report shows no window for them."""
    rec = Recorder(str(tmp_path / "l.db"))
    body = {"model": "claude-sonnet-5", "max_tokens": 8,
            "system": [{"type": "text", "text": "x " * 3000, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "hi"}]}
    resp = {"id": "m", "model": "claude-sonnet-5", "usage": {"input_tokens": 10, "output_tokens": 1,
                                                             "cache_read_input_tokens": 3000,
                                                             "cache_creation_input_tokens": 0}}
    for i in range(8):
        row = rec.record("anthropic", body, resp, session_id="a", latency_ms=50, ended_at=1_800_000_000 + i * 200)
        assert row.lifetime_evidence is None
    assert findings(rec.ledger)["observed_lifetimes"] == []
