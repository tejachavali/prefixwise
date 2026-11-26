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

from test_classifier_verdicts import A1, GO, OPUS, S, SYS, U1, claude, idle, last_marked, marked, scripted
from test_import_claude_code import Log, build, ts, usage

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


# -- what a second review found: "unchanged" has to be said by the classifier, not inferred ------

def test_no_figure_for_a_new_conversation_with_no_session_id():
    """A single-message request after a longer conversation, with no session id, past the lifetime:
    an expiry of the shared system prompt is named, but the prompt is another conversation's."""
    from test_classifier_verdicts import shift

    proxy = scripted([claude(write=30000), claude(write=23000)])
    base = {**BASE, "system": marked()}
    proxy.handle("anthropic", {**base, "messages": [U1, A1, GO, A1, GO]}, {})
    for sid, _ in list(proxy.lane_threads):
        shift(proxy, sid, sent=600, done=600)
    entry = proxy.handle("anthropic", {**base, "messages": [{"role": "user", "content": "new task " + "paste " * 20000}]}, {})[1]
    assert entry.cause == "ttl_expired" and "new conversation" in entry.evidence
    assert entry.missed_tokens is None and entry.missed_cost is None


def test_what_was_left_is_what_the_previous_turn_read_plus_wrote_when_it_reports_writes():
    proxy = scripted([claude(write=3000), claude(read=3000, write=100), claude(write=3300)])
    base = {**BASE, "system": marked()}
    proxy.handle("anthropic", {**base, "messages": [U1]}, S)
    proxy.handle("anthropic", {**base, "messages": [U1, A1, GO]}, S)
    idle(proxy, 600)
    entry = proxy.handle("anthropic", {**base, "messages": [U1, A1, GO, A1, GO]}, S)[1]
    assert (entry.cause, entry.missed_tokens) == ("ttl_expired", 3100)        # 3,000 read + 100 written, not 100
    # a server shaped like Anthropic's that reports no write count: what it left is not known
    bare = lambda read: {"input_tokens": 3000 - read, "output_tokens": 5, "cache_read_input_tokens": read}  # noqa: E731
    proxy = scripted([bare(0), bare(2800), bare(0)])
    proxy.handle("anthropic", {**base, "messages": [U1]}, S)
    proxy.handle("anthropic", {**base, "messages": [U1, A1, GO]}, S)
    idle(proxy, 600)
    entry = proxy.handle("anthropic", {**base, "messages": [U1, A1, GO, A1, GO]}, S)[1]
    assert entry.missed_tokens is None and entry.missed_cost is None, (entry.cause, entry.evidence)


def test_the_front_is_priced_at_the_long_lifetime_it_was_written_at():
    """3,000 tokens written for an hour ahead of 30,000 written for five minutes: the expired front is
    the 1-hour part. Rates given as "input", and the long-context tier, are followed."""
    table = PriceTable.load("")
    u = {"input_tokens": 200, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 33000,
         "cache_creation": {"ephemeral_1h_input_tokens": 3000, "ephemeral_5m_input_tokens": 30000}}
    assert table.missed_cost(OPUS, 3000, 33200, "anthropic", u) == pytest.approx(
        3000 * (RATES["cache_write_1h"] - RATES["cache_read"]) / 1e6)
    assert table.missed_cost(OPUS, 5000, 33200, "anthropic", u) == pytest.approx(
        (3000 * (RATES["cache_write_1h"] - RATES["cache_read"]) + 2000 * (RATES["cache_write_5m"] - RATES["cache_read"])) / 1e6)
    custom = PriceTable(models={"m": {"input": 2.0, "output": 8.0, "cache_read": 0.5, "cache_write": "input"}},
                        as_of="2026-11-26", sources=["test"])
    gpt_like = {"prompt_tokens": 4000, "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3000}}
    assert custom.missed_cost("m", 3000, 4000, "openai", gpt_like) == pytest.approx(3000 * (2.0 - 0.5) / 1e6)
    tiered = PriceTable(models={"m": {"input": 2.0, "output": 8.0, "cache_read": 0.5, "cache_write": 3.0,
                                      "long_prompt": {"above_prompt_tokens": 1000, "input": 4.0, "cache_read": 1.0,
                                                      "cache_write": 6.0}}}, as_of="2026-11-26", sources=["test"])
    assert tiered.missed_cost("m", 3000, 4000, "openai", gpt_like) == pytest.approx(3000 * (6.0 - 1.0) / 1e6)
    assert tiered.missed_cost("m", 300, 900, "openai", {**gpt_like, "prompt_tokens": 900}) == pytest.approx(
        300 * (3.0 - 0.5) / 1e6)


def import_rows(tmp_path, log):
    log.write(tmp_path / "-proj" / "sess-1.jsonl")
    led = Ledger()
    import_claude_code(tmp_path, led)
    with led.reader() as c:
        return c.execute("SELECT cause, missed_tokens, missed_cost, evidence, model FROM entries "
                         "ORDER BY recorded_at, id").fetchall()


def test_the_import_gives_a_figure_only_when_nothing_else_changed_in_the_gap(tmp_path):
    day = 86400
    # the plain case: one model, one version, a day apart
    log = Log()
    log.user(1); log.reply(5, usage(w1h=100000))
    log.user(day); log.reply(day + 5, usage(w1h=100500))
    rows = import_rows(tmp_path / "plain", log)
    assert (rows[1][0], rows[1][1]) == ("ttl_expired", 100000) and rows[1][2] > 0
    # priced at what the front was written at, not a blend with the 5-minute tail and the uncached input
    log = Log()
    log.user(1); log.reply(5, usage(w1h=100000))
    log.user(day); log.reply(day + 5, usage(w1h=100000, w5=50000, fresh=20000))
    rows = import_rows(tmp_path / "mixed", log)
    assert rows[1][1] == 100000 and rows[1][4] == OPUS
    assert rows[1][2] == pytest.approx(100000 * (RATES["cache_write_1h"] - RATES["cache_read"]) / 1e6)
    # the logged system prompt changed in the gap
    log = Log()
    snap = lambda t, v: log.lines.append({"type": "attachment", "timestamp": ts(t),  # noqa: E731
                                          "attachment": {"type": "prompt_snapshot", "systemPrompt": [v]}})
    snap(0, "v1"); log.user(1); log.reply(5, usage(w1h=100000))
    snap(day - 1, "v2"); log.user(day); log.reply(day + 5, usage(w1h=100500))
    rows = import_rows(tmp_path / "system", log)
    assert rows[1][0] == "ttl_expired" and rows[1][1] is None and rows[1][2] is None
    # Claude Code was upgraded in the gap
    log = Log()
    log.user(1); log.reply(5, usage(w1h=100000), version="2.1.330")
    log.user(day); log.reply(day + 5, usage(w1h=100500), version="2.1.332")
    rows = import_rows(tmp_path / "upgrade", log)
    assert rows[1][0] == "ttl_expired" and "upgraded" in rows[1][3] and rows[1][1] is None
    # another model was used in between
    log = Log()
    log.user(1); log.reply(5, usage(w1h=100000))
    log.user(50); log.reply(55, usage(w5=100200), model="claude-haiku-5-5")
    log.user(day); log.reply(day + 5, usage(w1h=100500))
    rows = import_rows(tmp_path / "switch", log)
    back = rows[2]
    assert back[4] == "claude-opus-5-5" and back[1] is None, back
    # the previous request's usage is a sum of several model calls: what it left is not exact
    log = Log()
    summed = {**usage(w1h=100000), "iterations": [{"input_tokens": 1}, {"input_tokens": 1}]}
    log.user(1); log.reply(5, summed)
    log.user(day); log.reply(day + 5, usage(w1h=100500))
    rows = import_rows(tmp_path / "summed", log)
    assert rows[1][1] is None, rows[1]
    # the effort setting changed in the gap, on a model where effort is part of the prompt...
    for model, figure in (("claude-opus-5", None), ("claude-opus-5-5", 100000)):  # ...and where it is not
        log = Log()
        log.user(1); log.reply(5, usage(w1h=100000), model=model, effort="high")
        log.user(day); log.reply(day + 5, usage(w1h=100500), model=model, effort="low")
        rows = import_rows(tmp_path / f"effort-{model}", log)
        assert (rows[1][0], rows[1][1]) == ("ttl_expired", figure), (model, rows[1])
    # Anthropic's own diagnosis names a change; an answer that is no verdict does not
    for reason, figure in (("system_changed", None), ("tools_changed", None), ("messages_changed", None),
                           ("previous_message_not_found", 100000)):
        log = Log()
        log.user(1); log.reply(5, usage(w1h=100000))
        log.user(day); log.reply(day + 5, usage(w1h=100500), diagnostics={"cache_miss_reason": {"type": reason}})
        rows = import_rows(tmp_path / reason, log)
        assert (rows[1][0], rows[1][1]) == ("ttl_expired", figure), (reason, rows[1])


def test_no_more_is_counted_than_this_requests_own_markers_cover(tmp_path):
    """The previous turn marked the system prompt and the last message and left 30,000. This one marks
    the system prompt only and wrote 3,000: a read stops at a marker, so a live cache could have
    served 3,000 of it, not 30,000."""
    b = {"model": OPUS, "max_tokens": 9, "system": marked()}
    p = scripted([claude(write=30000), claude(write=3000, fresh=27300)])
    p.handle("anthropic", {**b, "messages": last_marked([U1, A1, GO])}, S)
    idle(p, 600)
    e = p.handle("anthropic", {**b, "messages": [U1, A1, GO, A1, GO]}, S)[1]
    assert (e.cause, e.missed_tokens) == ("ttl_expired", 3000), (e.cause, e.missed_tokens, e.evidence)
    assert e.missed_cost == pytest.approx(3000 * (RATES["cache_write_5m"] - RATES["cache_read"]) / 1e6)
    # the import: 100,000 left, 20,000 written again under this request's markers
    log = Log()
    log.user(1); log.reply(5, usage(w1h=100000))
    log.user(86400); log.reply(86405, usage(w1h=20000, fresh=80500))
    rows = import_rows(tmp_path, log)
    assert (rows[1][0], rows[1][1]) == ("ttl_expired", 20000), rows[1]
    assert missed_for_turn("ttl_expired", True, 30300, 1000, 30300, 30000, covered_tokens=3000) == 2000
    assert missed_for_turn("ttl_expired", True, 30300, 0, 30300, 30000, covered_tokens=None) == 30000


def test_a_price_file_with_a_read_rate_above_the_write_rate_gives_no_negative_figure():
    odd = PriceTable(models={"m": {"input": 1.0, "output": 4.0, "cache_read": 3.0, "cache_write": 2.0}},
                     as_of="2026-11-26", sources=["test"])
    u = {"input_tokens": 100, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1000}
    assert odd.missed_cost("m", 1500, 1100, "anthropic", u) == 0.0


def test_the_line_reads_right_with_one_priced_a_tiny_sum_and_large_counts():
    item = {"requests": 1500, "missed_tokens": 1200, "missed_cost_usd": 0.0542, "requests_missed": 3,
            "requests_missed_priced": 1}
    assert "on the 1 that is priced" in _missed(item, None)
    assert "$" not in _missed({**item, "missed_cost_usd": 0.00002}, 0.5)     # would print as $0.0000
    led = Ledger()
    for _ in range(1500):
        led.record(LedgerEntry(session_id="s", agent_id="a", provider="anthropic", model=OPUS, cause="model_switch",
                               prompt_tokens=10, cached_tokens=0, avoidable_tokens=0, avoidable_cost=None, legit=True))
    line = next(ln for ln in render(findings(led)).splitlines() if ln.startswith("Expected misses"))
    assert "1,500 x switched model" in line
