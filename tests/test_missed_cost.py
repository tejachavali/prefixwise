"""What an expected miss cost (#184): an expiry is not waste, but it is on the bill. On a real Claude
Code session, resuming after the cache had expired was about a fifth of the bill, and the report had
no number for it. A figure is given only where it is true: the prompt had not changed, and the
previous turn is known to have left something in the cache."""
import sqlite3

import pytest

from prefixwise import metrics
from prefixwise.findings import _missed, findings, render
from prefixwise.importers import import_claude_code
from prefixwise.ledger import EXPECTED_MISS_CAUSES, Ledger, LedgerEntry, missed_for_turn
from prefixwise.pricing import PriceTable

from test_classifier_verdicts import A1, GO, OPUS, S, SYS, U1, claude, idle, marked, scripted
from test_import_claude_code import Log, build, usage

BASE = {"model": OPUS, "max_tokens": 9}
RATES = PriceTable.load("").rates_for(OPUS)   # USD per million tokens


def test_missed_tokens_are_what_the_cache_had_held_and_was_not_read():
    assert EXPECTED_MISS_CAUSES == {"ttl_expired"}
    assert missed_for_turn("ttl_expired", True, 3200, 0, 3000, 3000) == 3000    # the previous prompt, not the new tail
    assert missed_for_turn("ttl_expired", True, 3200, 500, 3000, 3000) == 2500  # less what was read all the same
    assert missed_for_turn("ttl_expired", True, 3200, 0, 3000, 2000) == 2000    # no more than was left in the cache
    assert missed_for_turn("ttl_expired", True, 2500, 0, 3000, 3000) == 2500    # a shorter prompt could reuse only itself
    assert missed_for_turn("ttl_expired", True, 3200, 3000, 3000, 3000) == 0
    # no figure where it would not be true
    assert missed_for_turn("ttl_expired", True, 3200, 0, None, 3000) is None    # previous size unknown
    assert missed_for_turn("ttl_expired", True, 3200, 0, 3000, None) is None    # what was left is not known
    assert missed_for_turn("ttl_expired", True, 3200, 0, 3000, 0) is None       # it left nothing: nothing expired
    assert missed_for_turn("ttl_expired", True, 3200, 0, 3000, 3000, unchanged=False) is None
    assert missed_for_turn("ttl_expired", False, 3200, 0, 3000, 3000) is None   # waste is in avoidable_tokens, never both
    for cause in ("none", "first_write", "model_switch", "compaction_rewrite", "tool_catalog_change", "evicted",
                  "volatile_system", "routing_miss", "unknown", None):
        assert missed_for_turn(cause, True, 3200, 0, 3000, 3000) is None, cause


def two(first, second, u1, u2, seconds):
    proxy = scripted([u1, u2])
    proxy.handle("anthropic", first, S)
    idle(proxy, seconds)
    return proxy, proxy.handle("anthropic", second, S)[1]


def resume(seconds, second_usage=None):
    """Two turns of one marked conversation, the second sent `seconds` after the first."""
    return two({**BASE, "system": marked(), "messages": [U1]}, {**BASE, "system": marked(), "messages": [U1, A1, GO]},
               claude(write=3000), second_usage or claude(write=3100), seconds)


def test_a_resume_after_the_cache_expired_has_its_cost_and_is_not_waste():
    proxy, entry = resume(600)
    assert (entry.cause, entry.legit, entry.avoidable_tokens, entry.secondary_causes) == ("ttl_expired", True, 0, None)
    assert entry.missed_tokens == 3000                 # what the first turn wrote, all sent again
    # they were written to the cache again: the 5-minute write rate, less the read they would have been
    assert entry.missed_cost == pytest.approx(3000 * (RATES["cache_write_5m"] - RATES["cache_read"]) / 1e6)
    assert 0 < entry.missed_cost < entry.cost
    row = proxy.ledger.get_request(entry.request_id)
    assert (row["missed_tokens"], row["missed_cost"]) == (3000, entry.missed_cost)
    _, part = resume(600, claude(read=1000, write=2100))                    # a part read all the same is not counted
    assert (part.cause, part.missed_tokens, part.avoidable_tokens) == ("ttl_expired", 2000, 0)


def test_no_figure_when_the_prompt_also_changed():
    """An expiry named for a turn whose prompt changed too: a live cache would not have served the
    changed part, and how much it would have served is not known."""
    replaced = two({**BASE, "system": marked(), "messages": [U1]},
                   {**BASE, "system": marked("Other. " * 400), "messages": [U1, A1, GO]},
                   claude(write=3000), claude(write=3100), 600)[1]
    another = two({**BASE, "system": marked(), "messages": [U1, A1, GO]},
                  {**BASE, "system": marked(), "messages": [{"role": "user", "content": "new task " + "paste " * 20000}]},
                  claude(write=30000), claude(write=23000), 600)[1]
    compacted = two({**BASE, "system": marked(), "messages": [U1, A1, GO]},
                    {**BASE, "system": marked(), "messages": [U1], "prefixwise_compaction_event": True},
                    claude(write=30000), claude(write=3100), 600)[1]
    for entry in (replaced, another, compacted):
        assert entry.cause == "ttl_expired" and entry.secondary_causes, entry.evidence
        assert entry.missed_tokens is None and entry.missed_cost is None


def test_no_figure_when_the_previous_turn_left_nothing():
    """No cache_control anywhere, resumed two hours later: nothing was ever in the cache."""
    entry = two({**BASE, "system": SYS, "messages": [U1]}, {**BASE, "system": SYS, "messages": [U1, A1, GO]},
                claude(fresh=3000), claude(fresh=3100), 7200)[1]
    assert entry.cause == "ttl_expired" and entry.missed_tokens is None and entry.missed_cost is None


def test_hits_waste_and_first_turns_have_no_missed_figure():
    proxy, hit = resume(10, claude(read=3000, write=100))
    assert hit.cause == "none" and hit.missed_tokens is None and hit.missed_cost is None
    proxy, miss = resume(10)                           # read nothing, seconds later: not an expiry
    assert miss.cause != "ttl_expired" and miss.missed_tokens is None
    with proxy.ledger.reader() as conn:
        first = conn.execute("SELECT cause, missed_tokens, missed_cost FROM entries ORDER BY id LIMIT 1").fetchone()
    assert tuple(first) == ("first_write", None, None)


def test_the_cost_is_the_write_rate_on_what_was_written_again_not_a_blend():
    """3,000 tokens written again at the 1-hour rate, beside 30,000 uncached: the re-sent prefix paid
    the write rate. The blended rate of everything not read would halve it."""
    table = PriceTable.load("")
    u = {"input_tokens": 30000, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 3000,
         "cache_creation": {"ephemeral_1h_input_tokens": 3000}}
    cost = table.missed_cost(OPUS, 3000, 33000, "anthropic", u)
    assert cost == pytest.approx(3000 * (RATES["cache_write_1h"] - RATES["cache_read"]) / 1e6)
    assert cost > 1.5 * table.avoidable_cost(OPUS, 3000, 33000, "anthropic", u)
    # more missed than written: the rest was sent as plain input
    more = table.missed_cost(OPUS, 5000, 33000, "anthropic", u)
    assert more == pytest.approx((3000 * (RATES["cache_write_1h"] - RATES["cache_read"])
                                  + 2000 * (RATES["input"] - RATES["cache_read"])) / 1e6)
    assert table.missed_cost(OPUS, 0, 33000, "anthropic", u) == 0.0
    assert table.missed_cost("claude-unknown-9", 3000, 33000, "anthropic", u) is None
    assert table.missed_cost(OPUS, 3000, 33000) == pytest.approx(3000 * (RATES["input"] - RATES["cache_read"]) / 1e6)
    # never more than those tokens cost on the bill
    assert cost <= 3000 * RATES["cache_write_1h"] / 1e6


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
    assert (expired["requests_missed"], expired["requests_missed_priced"]) == (1, 1)
    assert found["summary"]["missed_tokens"] == 3000
    assert found["expected"][0]["cause"] == "ttl_expired"          # what cost most comes first
    text = render(found)
    line = next(ln for ln in text.splitlines() if ln.startswith("Expected misses"))
    assert "expired (3,000 tokens the cache had held were sent again, $" in line     # every row has it: no "on N of them"
    assert "more than reading them, " in line and "% of the bill" in line
    assert "first request" in line and line.index("expired") < line.index("first request")
    assert "No avoidable waste" in text                            # it is still not waste
    row = metrics.summary(proxy.ledger)
    assert row["missed_tokens"] == 3000 and row["missed_cost_usd"] == pytest.approx(entry.missed_cost)
    assert (row["requests_missed"], row["requests_missed_priced"]) == (1, 1)
    assert {"missed_tokens", "missed_cost_usd", "requests_missed"} <= {m["name"] for m in metrics.catalog()}


def test_the_line_says_which_requests_the_figure_covers():
    """Four expiries: two with a figure and a price, one with a figure and no price, one with none.
    The tokens are those of three, the dollars of two, and the line says so."""
    item = {"requests": 4, "missed_tokens": 15000, "missed_cost_usd": 0.0542, "requests_missed": 3,
            "requests_missed_priced": 2}
    assert _missed(item, 0.5) == (" (on 3 of them, 15,000 tokens the cache had held were sent again, $0.0542 more "
                                  "than reading them on the 2 that are priced, 11% of the bill)")
    assert _missed({**item, "requests": 3, "requests_missed_priced": 3}, None) == (
        " (15,000 tokens the cache had held were sent again, $0.0542 more than reading them)")
    assert _missed({**item, "missed_cost_usd": None, "requests_missed_priced": 0}, 0.5) == (
        " (on 3 of them, 15,000 tokens the cache had held were sent again)")
    assert _missed({**item, "missed_tokens": 0}, 0.5) == ""         # everything was read all the same: nothing to say
    assert _missed({**item, "missed_tokens": None}, 0.5) == ""
    # ... and what cost most comes first; with no dollars, what re-sent most
    led = Ledger()
    row = lambda cause, **kw: led.record(LedgerEntry(  # noqa: E731
        session_id="s", agent_id="a", provider="anthropic", model="claude-unknown-9", cause=cause, prompt_tokens=9000,
        cached_tokens=0, avoidable_tokens=0, avoidable_cost=None, legit=True, **kw))
    for _ in range(3):
        row("model_switch")
    row("first_write")
    row("ttl_expired", missed_tokens=500)
    assert [e["cause"] for e in findings(led)["expected"]] == ["ttl_expired", "model_switch", "first_write"]


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
    assert "2 x too long" in line and "(on 1 of them, 3,000 tokens the cache had held were sent again)" in line
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


def test_the_import_gives_no_figure_for_calls_that_never_cached(tmp_path):
    """Three small calls that read and wrote nothing, two hours apart: nothing was held, nothing expired."""
    log = Log()
    for i in range(3):
        log.user(i * 7200)
        log.reply(i * 7200 + 5, usage(fresh=900))
    log.write(tmp_path / "-proj" / "sess-1.jsonl")
    led = Ledger()
    import_claude_code(tmp_path, led)
    with led.reader() as c:
        rows = c.execute("SELECT cause, missed_tokens, missed_cost FROM entries ORDER BY recorded_at, id").fetchall()
    assert len(rows) == 3 and all(r[1] is None and r[2] is None for r in rows), rows


def test_no_figure_where_the_provider_does_not_report_what_it_cached():
    """OpenAI reports what was read, not what was written: what the cache held when it expired would
    be a guess, so the expiry is named and no figure is given."""
    from test_classifier_verdicts import gpt

    headers = {**S, "x-prefixwise-cache-ttl": "300"}
    body = lambda *more: {"model": "gpt-5.6-terra", "messages": [{"role": "system", "content": SYS}, U1, *more]}  # noqa: E731
    proxy = scripted([gpt(3000, 0), gpt(3300, 2944), gpt(3600, 0)])
    proxy.handle("openai", body(), headers)
    proxy.handle("openai", body(A1, GO), headers)
    idle(proxy, 600)
    entry = proxy.handle("openai", body(A1, GO, A1, GO), headers)[1]
    assert entry.cause == "ttl_expired", entry.evidence
    assert entry.missed_tokens is None and entry.missed_cost is None
