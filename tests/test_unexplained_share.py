"""The misses with no cause are a number, not just a sentence: a share of the misses and of the
tokens they left uncached, so it can be watched over time and shows when the set of causes needs
another member (raised on r/LLMDevs)."""

from prefixwise.classifier import CACHE_NOT_REPORTED
from prefixwise.findings import findings, render, render_share
from prefixwise.ledger import Ledger, LedgerEntry


def _rec(led, cause, prompt, cached, legit=None, evidence="", avoidable=0):
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="openai", model="gpt-5", cause=cause,
                           legit=legit, prompt_tokens=prompt, cached_tokens=cached, avoidable_tokens=avoidable,
                           avoidable_cost=0.0, task_success=None, evidence=evidence))


def test_share_of_misses_and_of_uncached_tokens(tmp_path):
    led = Ledger(str(tmp_path / "l.db"))
    _rec(led, "first_write", 10_000, 0, legit=True)                 # a miss, explained
    _rec(led, "none", 10_100, 10_000)                                # a hit: not a miss
    _rec(led, "tool_reserialize", 10_200, 2_000, legit=False, avoidable=8_000)
    _rec(led, "unknown", 10_300, 300)                                # unexplained: 10,000 uncached
    f = findings(led)
    share = f["unexplained_share"]
    assert share["requests"] == 1 and share["of_misses"] == 3
    assert share["uncached_tokens"] == 10_000 and share["of_uncached_tokens"] == 10_000 + 8_200 + 10_000
    text = render(f)
    assert "Cause not determined for 1 of 3 misses (33.3%), 10,000 of 28,200 uncached prompt tokens (35.5%)" in text
    assert "cause not determined: 1 of 3 misses (33.3%)" in render_share(f)


def test_requests_on_a_server_that_reports_no_cache_are_not_counted_as_misses(tmp_path):
    led = Ledger(str(tmp_path / "l.db"))
    _rec(led, "first_write", 5_000, 0, legit=True)
    _rec(led, "unknown", 5_100, 0, evidence=CACHE_NOT_REPORTED)     # blind, not a known miss
    _rec(led, "unknown", 5_200, 100)
    share = findings(led)["unexplained_share"]
    assert share["requests"] == 1 and share["of_misses"] == 2
    assert share["uncached_tokens"] == 5_100 and share["of_uncached_tokens"] == 5_000 + 5_100


def test_nothing_unexplained_prints_nothing(tmp_path):
    led = Ledger(str(tmp_path / "l.db"))
    _rec(led, "first_write", 5_000, 0, legit=True)
    f = findings(led)
    assert f["unexplained_share"] is None
    assert "Cause not determined" not in render(f) and "cause not determined" not in render_share(f)


def test_unknowns_not_known_to_be_misses_are_left_out(tmp_path):
    """Found by review: a turn continuing a response the proxy never saw is `unknown` whatever it read
    (a full hit included), and so is a response with no usage; neither is known to be a miss."""
    from prefixwise.classifier import UNSEEN_PARENT
    led = Ledger(str(tmp_path / "l.db"))
    _rec(led, "first_write", 10_000, 0, legit=True)
    _rec(led, "unknown", 10_000, 9_900, evidence=UNSEEN_PARENT)                   # probably a hit
    _rec(led, "unknown", 0, 0, evidence="blocks unchanged but the provider returned no usage")
    f = findings(led)
    assert f["unexplained_share"] is None
    _rec(led, "unknown", 10_000, 100, evidence="prefix reusable but cache_read low")  # a real one
    share = findings(led)["unexplained_share"]
    assert share["requests"] == 1 and share["of_misses"] == 2
    assert share["uncached_tokens"] == 9_900 and share["of_uncached_tokens"] == 19_900


def test_blind_rows_are_found_with_text_appended_to_their_evidence(tmp_path):
    """Found by review: the collector appends to the evidence (an interleaved conversation, a detail),
    so the blind check must match the start, not the whole string."""
    led = Ledger(str(tmp_path / "l.db"))
    _rec(led, "first_write", 5_000, 0, legit=True)
    _rec(led, "unknown", 5_100, 0, evidence=CACHE_NOT_REPORTED + "; compared with this conversation's last turn")
    f = findings(led)
    assert f["cache_not_reported"] == 1 and f["unexplained_share"] is None
    assert "Cause not determined" not in render(f)


def test_the_window_applies_to_the_share(tmp_path):
    led = Ledger(str(tmp_path / "l.db"))
    _rec(led, "unknown", 1_000, 0, evidence="prefix reusable but cache_read low")
    _rec(led, "first_write", 1_000, 0, legit=True)
    import sqlite3
    with sqlite3.connect(str(tmp_path / "l.db")) as conn:  # place the two rows a day apart
        conn.execute("UPDATE entries SET recorded_at = 1000 WHERE cause = 'unknown'")
        conn.execute("UPDATE entries SET recorded_at = 90000 WHERE cause = 'first_write'")
    assert findings(led)["unexplained_share"]["of_misses"] == 2
    assert findings(led, since=50_000)["unexplained_share"] is None
    assert findings(led, until=50_000)["unexplained_share"]["of_misses"] == 1


def test_failed_requests_and_hits_are_not_misses(tmp_path):
    led = Ledger(str(tmp_path / "l.db"))
    _rec(led, "unknown", 1_000, 0, evidence="prefix reusable but cache_read low")
    _rec(led, "none", 1_000, 1_000)
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="openai", model="gpt-5", cause=None, legit=None,
                           prompt_tokens=0, cached_tokens=0, avoidable_tokens=0, avoidable_cost=0.0,
                           task_success=None, error="http_500", status=500))
    share = findings(led)["unexplained_share"]
    assert share["requests"] == 1 and share["of_misses"] == 1 and share["share_of_misses"] == 1.0
