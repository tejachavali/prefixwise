"""What an expected miss cost (#184): a miss with a legitimate cause is not waste, but it is on the
bill. On a real Claude Code session, resuming after the cache had expired was about a fifth of the
bill, and the report had no number for it."""
import sqlite3

import pytest

from prefixwise import metrics
from prefixwise.findings import findings, render
from prefixwise.importers import import_claude_code
from prefixwise.ledger import EXPECTED_MISS_CAUSES, Ledger, LedgerEntry, missed_for_turn

from test_classifier_verdicts import A1, GO, OPUS, S, U1, claude, idle, marked, scripted
from test_import_claude_code import build


def test_missed_tokens_are_what_the_cache_had_held_and_was_not_read():
    assert EXPECTED_MISS_CAUSES == {"ttl_expired", "evicted"}
    assert missed_for_turn("ttl_expired", True, 3200, 0, 3000) == 3000          # the previous prompt, not the new tail
    assert missed_for_turn("ttl_expired", True, 3200, 500, 3000) == 2500        # less what was read all the same
    assert missed_for_turn("ttl_expired", True, 3200, 0, 3000, 2000) == 2000    # no more than was left in the cache
    assert missed_for_turn("evicted", None, 3200, 0, 3000) == 3000
    assert missed_for_turn("ttl_expired", True, 2500, 0, 3000) == 2500          # a shorter prompt could reuse only itself
    assert missed_for_turn("ttl_expired", True, 3200, 0, None) is None          # previous size unknown: no figure
    # waste is counted in avoidable_tokens, never in both; other causes have no such figure
    assert missed_for_turn("ttl_expired", False, 3200, 0, 3000) is None
    for cause in ("none", "first_write", "model_switch", "compaction_rewrite", "tool_catalog_change",
                  "volatile_system", "routing_miss", "unknown", None):
        assert missed_for_turn(cause, True, 3200, 0, 3000) is None, cause


def resume(seconds, usage=None, **kw):
    """Two turns of one marked conversation, the second sent `seconds` after the first."""
    proxy = scripted([claude(write=3000), usage or claude(write=3100)], **kw)
    base = {"model": OPUS, "max_tokens": 9, "system": marked()}
    proxy.handle("anthropic", {**base, "messages": [U1]}, S)
    idle(proxy, seconds)
    return proxy, proxy.handle("anthropic", {**base, "messages": [U1, A1, GO]}, S)[1]


def test_a_resume_after_the_cache_expired_has_its_cost_and_is_not_waste():
    proxy, entry = resume(600)
    assert (entry.cause, entry.legit, entry.avoidable_tokens) == ("ttl_expired", True, 0)
    assert entry.missed_tokens == 3000                 # what the first turn wrote, all sent again
    # priced like avoidable tokens: what they were paid at (a cache write here) above a read
    assert entry.missed_cost == proxy.prices.avoidable_cost(OPUS, 3000, entry.prompt_tokens, "anthropic",
                                                           claude(write=3100))
    assert entry.missed_cost > 0 and entry.missed_cost < entry.cost
    row = proxy.ledger.get_request(entry.request_id)
    assert (row["missed_tokens"], row["missed_cost"]) == (3000, entry.missed_cost)
    # a part read all the same is not counted
    _, part = resume(600, claude(read=1000, write=2100))
    assert (part.cause, part.missed_tokens, part.avoidable_tokens) == ("ttl_expired", 2000, 0)


def test_hits_waste_and_first_turns_have_no_missed_figure():
    proxy, hit = resume(10, claude(read=3000, write=100))
    assert hit.cause == "none" and hit.missed_tokens is None and hit.missed_cost is None
    proxy, miss = resume(10)                           # read nothing, seconds later: not an expiry
    assert miss.cause != "ttl_expired" and miss.missed_tokens is None
    with proxy.ledger.reader() as conn:
        first = conn.execute("SELECT cause, missed_tokens, missed_cost FROM entries ORDER BY id LIMIT 1").fetchone()
    assert tuple(first) == ("first_write", None, None)


def test_an_unpriced_model_has_the_tokens_and_no_dollars():
    proxy = scripted([claude(write=3000), claude(write=3100)])
    base = {"model": "claude-unknown-9", "max_tokens": 9, "system": marked()}
    proxy.handle("anthropic", {**base, "messages": [U1]}, S)
    idle(proxy, 600)
    entry = proxy.handle("anthropic", {**base, "messages": [U1, A1, GO]}, S)[1]
    assert (entry.cause, entry.missed_tokens, entry.missed_cost, entry.cost) == ("ttl_expired", 3000, None, None)


def test_the_report_says_what_expected_misses_cost():
    proxy, entry = resume(600)
    found = findings(proxy.ledger, prices=proxy.prices)
    expired = next(e for e in found["expected"] if e["cause"] == "ttl_expired")
    assert expired["missed_tokens"] == 3000 and expired["missed_cost_usd"] == pytest.approx(entry.missed_cost)
    assert found["summary"]["missed_tokens"] == 3000
    assert found["expected"][0]["cause"] == "ttl_expired"          # what cost most comes first
    text = render(found)
    line = next(ln for ln in text.splitlines() if ln.startswith("Expected misses"))
    assert "3,000 tokens the cache had held were sent again" in line
    assert "more than reading them" in line and "% of the bill" in line
    assert "first request" in line and line.index("expired") < line.index("first request")
    assert "No avoidable waste" in text                            # it is still not waste
    row = metrics.summary(proxy.ledger)
    assert row["missed_tokens"] == 3000 and row["missed_cost_usd"] == pytest.approx(entry.missed_cost)
    assert {"missed_tokens", "missed_cost_usd"} <= {m["name"] for m in metrics.catalog()}


def test_rows_from_an_older_ledger_have_no_figure_and_the_report_says_nothing(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, agent_id TEXT, provider TEXT, model TEXT, cause TEXT, legit INTEGER,
        prompt_tokens INTEGER, cached_tokens INTEGER, avoidable_tokens INTEGER,
        avoidable_cost REAL, task_success INTEGER, native_reason TEXT, classifier_version TEXT)""")
    conn.execute("INSERT INTO entries (session_id, model, cause, legit, prompt_tokens, cached_tokens, avoidable_tokens)"
                 " VALUES ('s', 'claude-opus-5-5', 'ttl_expired', 1, 3200, 0, 0)")
    conn.commit()
    conn.close()
    led = Ledger(str(path))                                        # the columns are added on open
    with led.reader() as c:
        assert {"missed_tokens", "missed_cost"} <= {r[1] for r in c.execute("PRAGMA table_info(entries)")}
    line = next(ln for ln in render(findings(led)).splitlines() if ln.startswith("Expected misses"))
    assert "1 x too long since the previous turn" in line and "sent again" not in line
    led.record(LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model=OPUS, cause="ttl_expired",
                           prompt_tokens=3200, cached_tokens=0, avoidable_tokens=0, avoidable_cost=None,
                           legit=True, missed_tokens=3000))
    line = next(ln for ln in render(findings(led)).splitlines() if ln.startswith("Expected misses"))
    assert "2 x too long" in line and "3,000 tokens the cache had held were sent again" in line
    assert "$" not in line                                         # tokens with no dollars: none are invented


def test_the_claude_code_import_has_the_figure_too(tmp_path):
    build(tmp_path)
    led = Ledger()
    import_claude_code(tmp_path, led)
    with led.reader() as c:
        rows = c.execute("SELECT cause, legit, avoidable_tokens, missed_tokens, missed_cost, cost FROM entries "
                         "WHERE agent_id = 'main' ORDER BY recorded_at, id").fetchall()
    expired = [r for r in rows if r[0] == "ttl_expired"]
    assert len(expired) == 1
    cause, legit, avoidable, missed, missed_cost, cost = expired[0]
    # the turn before read 9,300 and wrote 200: 9,500 were in the cache and none was read
    assert (legit, avoidable, missed) == (1, 0, 9500)
    assert 0 < missed_cost < cost
    assert all(r[3] is None for r in rows if r[0] != "ttl_expired")
