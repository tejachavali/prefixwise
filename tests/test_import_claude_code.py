"""prefixwise import claude-code: Claude Code session logs, read offline.

The fixtures copy the structure of real logs (entry types, field names, how one
response is split across entries) without any real content.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prefixwise.importers import import_claude_code
from prefixwise.ledger import Ledger

T0 = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)


def ts(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def usage(read=0, w5=0, w1h=0, fresh=50, out=20):
    return {"input_tokens": fresh, "output_tokens": out, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": w5 + w1h,
            "cache_creation": {"ephemeral_5m_input_tokens": w5, "ephemeral_1h_input_tokens": w1h}}


class Log:
    def __init__(self, ids="a"):
        self.lines: list[dict] = []
        self.n = 0
        self.ids = ids  # response ids are unique across logs

    def user(self, t):
        self.lines.append({"type": "user", "timestamp": ts(t), "message": {"role": "user", "content": "x"}})

    def attach(self, t, kind):
        self.lines.append({"type": "attachment", "timestamp": ts(t), "attachment": {"type": kind}})

    def compact(self, t):
        self.lines.append({"type": "system", "subtype": "compact_boundary", "timestamp": ts(t),
                           "compactMetadata": {"preTokens": 100000, "postTokens": 9000}})

    def reply(self, t, u, model="claude-opus-5-5", split=True, ttl="1h", **extra):
        self.n += 1
        msg = {"id": f"msg_{self.ids}{self.n}", "model": model, "usage": u, "content": [{"type": "text", "text": "y"}]}
        if "diagnostics" in extra:
            msg["diagnostics"] = extra.pop("diagnostics")
        base = {"type": "assistant", "timestamp": ts(t), "promptCacheRequested": ttl, "message": msg, **extra}
        if split:  # Claude Code logs one entry per content block; not every entry has requestId
            self.lines.append({**base, "requestId": f"req_{self.ids}{self.n}"})
            self.lines.append({**base, "timestamp": ts(t + 1), "message": {**msg, "usage": {**u, "output_tokens": 1}}})
        else:
            self.lines.append({**base, "requestId": f"req_{self.ids}{self.n}"})

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(x) for x in self.lines) + "\nnot json\n")


def build(root: Path):
    main = Log()
    main.user(0); main.reply(5, usage(w1h=20000))                     # first request
    main.user(30); main.reply(35, usage(read=20000, w1h=500))         # hit
    main.user(60); main.attach(60, "deferred_tools_delta")
    main.reply(65, usage(w1h=21000))                                   # tool list changed
    main.compact(90); main.user(95); main.reply(100, usage(w1h=9000))  # compaction
    main.user(130); main.reply(135, usage(w5=9300), model="claude-haiku-5-5")  # model switch: its own cache
    main.user(160); main.reply(165, usage(read=9300, w5=200), model="claude-haiku-5-5")  # hit, 5m write
    main.user(165 + 900); main.reply(1070, usage(w5=9600), model="claude-haiku-5-5")     # 15 min later
    main.user(1100); main.reply(1105, usage(fresh=9000), model="claude-haiku-5-5")       # unexplained
    main.write(root / "-proj" / "sess-1.jsonl")
    sub = Log(ids="s")
    sub.user(40); sub.reply(45, usage(w5=12000), split=False)
    sub.user(50); sub.reply(55, usage(read=12000, w5=100), split=False)
    path = root / "-proj" / "sess-1" / "subagents" / "agent-abc.jsonl"
    sub.write(path)
    path.with_name("agent-abc.meta.json").write_text(json.dumps({"agentType": "Explore", "spawnDepth": 1}))


def rows(led):
    with led.reader() as c:
        c.row_factory = None
        cur = c.execute("SELECT agent_id, cause, detail, prompt_tokens, cached_tokens, evidence, request_id, "
                        "cost, recorded_at FROM entries ORDER BY recorded_at, id")
        return cur.fetchall()


def test_causes_from_the_log(tmp_path):
    build(tmp_path)
    led = Ledger()
    stats = import_claude_code(tmp_path, led)
    assert stats["imported"] == 10 and stats["logs"] == 2
    main = [r for r in rows(led) if r[0] == "main"]
    # A deferred-tools load is appended after the cached prefix (Claude Code docs): no cause by itself.
    assert [r[1] for r in main] == ["first_write", "none", "unknown", "compaction_rewrite",
                                    "model_switch", "none", "ttl_expired", "unknown"]
    assert "deferred tools loaded" in main[2][5] and "appends after the cached prefix" in main[2][5]
    assert main[6][2] == "ttl_shortened"  # asked for 1h, the write was kept 5m, 15 minutes passed
    with led.reader() as c:
        ttls = [t for (t,) in c.execute("SELECT cache_ttl_seconds FROM entries WHERE cause = 'ttl_expired'")]
    assert ttls == [300.0]  # its own write was kept 5 minutes (it asked for 1 hour: ttl_shortened)
    with led.reader() as c:
        first = c.execute("SELECT cache_ttl_seconds FROM entries WHERE agent_id = 'main' "
                          "ORDER BY recorded_at, id LIMIT 1").fetchone()[0]
    assert first == 3600.0  # a stream's first request has no previous turn, yet its 1-hour write is known
    assert "a change Claude Code does not log" in main[7][5]
    assert all(r[5].startswith("from the log: ") for r in main)
    assert main[0][6] == "msg_a1" and len({r[6] for r in main}) == 8  # split responses counted once
    assert main[0][3] == 20050 and main[1][4] == 20000
    assert main[0][7] is not None and main[0][7] > 0  # priced (claude-opus-5-5)
    assert abs(main[0][8] - (T0.timestamp() + 6)) < 0.01  # dated by the log, not by the import


def test_subagents_are_labelled_by_type(tmp_path):
    build(tmp_path)
    led = Ledger()
    import_claude_code(tmp_path, led)
    sub = [r for r in rows(led) if r[0] != "main"]
    assert [r[0] for r in sub] == ["subagent:Explore"] * 2
    assert sub[0][1] == "first_write" and "subagent" in sub[0][5]
    with led.reader() as c:
        sessions = {s for (s,) in c.execute("SELECT DISTINCT session_id FROM entries")}
    assert sessions == {"sess-1", "sess-1/agent-abc"}


def test_reimport_adds_only_new_requests_and_since_filters(tmp_path):
    build(tmp_path)
    led = Ledger()
    import_claude_code(tmp_path, led)
    again = import_claude_code(tmp_path, led)
    assert again["imported"] == 0 and again["skipped_known"] == 10
    later = Ledger()
    s = import_claude_code(tmp_path, later, since=T0.timestamp() + 1000)
    assert s["imported"] == 2  # the 15-minute and the unexplained turn
    assert [r[1] for r in rows(later)] == ["ttl_expired", "unknown"]  # still judged against earlier turns


ENV = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "TZ": "UTC"}  # dates in the output: one zone everywhere


def test_cli_import_prints_the_report(tmp_path):
    build(tmp_path)
    db = tmp_path / "x.db"
    out = subprocess.run([sys.executable, "-m", "prefixwise", "import", "claude-code", "--path", str(tmp_path),
                          "--ledger", str(db)], capture_output=True, text=True, env=ENV)
    assert out.returncode == 0, out.stderr
    assert "Imported 10 requests from 2 Claude Code logs" in out.stdout
    assert "asked for a 1-hour cache but the entry was kept for 5 minutes" in out.stdout
    assert "subagent:Explore" in out.stdout and "API prices" in out.stdout
    # The shareable headline comes first (the totals once, then the details, then the caveats)
    o = out.stdout
    assert o.index("prefixwise (Claude Code, 2026-11-07)") < o.index("% from cache") < o.index("Details:") \
        < o.index("About these numbers")
    assert o.count("132,450 prompt tokens") == 1 and "pip install prefixwise" in o
    out = subprocess.run([sys.executable, "-m", "prefixwise", "import", "claude-code", "--path", str(tmp_path / "nope"),
                          "--ledger", str(db)], capture_output=True, text=True, env=ENV)
    assert out.returncode == 1 and "no Claude Code logs" in out.stderr



def one_log(tmp_path, build_log):
    log = Log()
    build_log(log)
    log.write(tmp_path / "-p" / "s.jsonl")
    led = Ledger()
    import_claude_code(tmp_path, led)
    return [r for r in rows(led)]


def test_review_findings(tmp_path):
    # Errors and interrupts are logged as <synthetic> responses: no API call, never a "model switch".
    def synth(log):
        log.user(0); log.reply(5, usage(w1h=20000))
        log.lines.append({"type": "assistant", "timestamp": ts(10), "isApiErrorMessage": True,
                          "message": {"id": "msg_err", "model": "<synthetic>", "usage": usage(fresh=0, out=0)}})
        log.user(20); log.reply(25, usage(read=20000, w1h=300))
    r = one_log(tmp_path / "a", synth)
    assert [x[1] for x in r] == ["first_write", "none"]

    # Switching back to a model whose cache is warm is a hit; a first request that reads is a hit.
    def warm(log):
        log.user(0); log.reply(5, usage(read=19000, w1h=500))
        log.user(10); log.reply(15, usage(w5=15000), model="claude-haiku-5-5")
        log.user(20); log.reply(25, usage(read=19500, w1h=400))
    assert [x[1] for x in one_log(tmp_path / "b", warm)] == ["none", "model_switch", "none"]

    # An upgrade, a changed system prompt snapshot, an effort change (on a model where it matters).
    def changes(log):
        log.user(0); log.reply(5, usage(w1h=20000), version="2.1.318", perTurnEffort="medium",
                               model="claude-opus-5")
        log.user(10); log.reply(15, usage(w1h=20100), version="2.1.319", perTurnEffort="medium",
                                model="claude-opus-5")
        log.lines.append({"type": "attachment", "timestamp": ts(20), "attachment": {
            "type": "prompt_snapshot", "systemPrompt": ["v1"]}})
        log.user(21); log.reply(25, usage(read=20100, w1h=10), version="2.1.319", perTurnEffort="medium",
                                model="claude-opus-5")
        log.lines.append({"type": "attachment", "timestamp": ts(30), "attachment": {
            "type": "prompt_snapshot", "systemPrompt": ["v2"]}})
        log.user(31); log.reply(35, usage(w1h=20200), version="2.1.319", perTurnEffort="medium",
                                model="claude-opus-5")
        log.user(40); log.reply(45, usage(w1h=20300), version="2.1.319", perTurnEffort="high",
                                model="claude-opus-5")
    r = one_log(tmp_path / "c", changes)
    assert [x[1] for x in r] == ["first_write", "system_edit", "none", "system_edit", "param_rendered"]
    assert "upgraded (2.1.318 -> 2.1.319)" in r[1][5]

    # Effort changes do not break the cache on 5.5 models.
    def safe_effort(log):
        log.user(0); log.reply(5, usage(w1h=20000), perTurnEffort="medium", model="claude-opus-5-5")
        log.user(10); log.reply(15, usage(w1h=20100), perTurnEffort="high", model="claude-opus-5-5")
    assert [x[1] for x in one_log(tmp_path / "d", safe_effort)] == ["first_write", "unknown"]

    # Anthropic's own diagnosis, when present, names the cause; "not found" is no verdict.
    def native(log):
        log.user(0); log.reply(5, usage(w1h=20000))
        log.user(10); log.reply(15, usage(w1h=20100), diagnostics={"cache_miss_reason": {"type": "tools_changed"}})
        log.user(20); log.reply(25, usage(w1h=20200),
                                diagnostics={"cache_miss_reason": {"type": "previous_message_not_found"}})
    r = one_log(tmp_path / "e", native)
    assert [x[1] for x in r] == ["first_write", "tool_catalog_change", "unknown"]
    assert "Anthropic's own diagnosis" in r[1][5]

    # Subagent entries inside an older main log are their own stream, not the main conversation's previous turn.
    def sidechain(log):
        log.user(0); log.reply(5, usage(w1h=20000))
        log.lines.append({"type": "assistant", "timestamp": ts(7), "isSidechain": True, "requestId": "req_side",
                          "message": {"id": "msg_side", "model": "claude-haiku-5-5", "usage": usage(w5=9000)}})
        log.user(10); log.reply(15, usage(read=20000, w1h=100))
    r = one_log(tmp_path / "f", sidechain)
    assert sorted((x[0], x[1]) for x in r) == [("main", "first_write"), ("main", "none"),
                                               ("subagent:unknown", "first_write")]


def test_unreadable_files_are_skipped_not_fatal(tmp_path):
    build(tmp_path)
    (tmp_path / "-proj" / "dir.jsonl").mkdir()  # a directory named like a log
    (tmp_path / "-proj" / "gone.jsonl").symlink_to(tmp_path / "nowhere")  # a dangling link
    led = Ledger()
    s = import_claude_code(tmp_path, led)
    assert s["imported"] == 10


def test_odd_field_types_do_not_crash(tmp_path):
    def odd(log):
        log.user(0); log.reply(5, {"input_tokens": "lots", "output_tokens": "x", "cache_read_input_tokens": None,
                                   "cache_creation_input_tokens": 100, "cache_creation": "weird"})
    assert len(one_log(tmp_path, odd)) == 1


def named(tmp_path, build_log):
    """one_log, with each row as a dict of the columns these tests read."""
    log = Log()
    build_log(log)
    log.write(tmp_path / "-p" / "s.jsonl")
    led = Ledger()
    import_claude_code(tmp_path, led)
    cols = ("cause", "legit", "evidence", "gap_seconds", "detail", "unread_write_tokens")
    with led.reader() as c:
        c.row_factory = None
        return [dict(zip(cols, r)) for r in
                c.execute(f"SELECT {', '.join(cols)} FROM entries ORDER BY recorded_at, id")]


WARM = 60_000  # the system prompt and tools at the front, found in the cache whatever the conversation did
OPUS, FABLE, HAIKU = "claude-opus-5-5", "claude-fable-5-2", "claude-haiku-5-5"
FULL = "read what the previous request"
HALF = "at least half"


def test_what_a_real_session_showed(tmp_path):
    """Wrong verdicts found on a real 10,517-call Claude Code session (2026-11-26), as fixtures
    (experiments/dogfood/2026-11-26-claude-code-logs.md). About 60,000 tokens at the front of its
    prompt were read on the first call and after 21 hours idle."""
    def compaction(log):
        # 900k was cached; the compacted prompt is about 100k and still reads the front: more than
        # half of the new prompt, and none of the conversation.
        log.user(0); log.reply(5, usage(read=890_000, w1h=10_000))
        log.compact(60); log.user(65); log.reply(70, usage(read=WARM, w1h=40_000))
        log.user(100); log.reply(105, usage(read=WARM + 40_000, w1h=500))
    got = named(tmp_path / "a", compaction)
    assert [(r["cause"], r["legit"]) for r in got] == [("none", None), ("compaction_rewrite", 1), ("none", None)]
    # the summary call that read the 10k write is not logged: it is not counted as left unread
    assert got[1]["unread_write_tokens"] is None

    def switch_to_a_model_used_long_before(log):
        log.user(0); log.reply(5, usage(read=WARM, w1h=440_000), model=FABLE)
        log.user(9000); log.reply(9005, usage(read=WARM, w1h=50_000))                # the conversation, on Opus
        log.user(9200); log.reply(9205, usage(read=WARM, w1h=51_000), model=FABLE)   # 55% of its prompt read
    got = named(tmp_path / "b", switch_to_a_model_used_long_before)
    assert got[2]["cause"] == "model_switch" and got[2]["legit"] == 1
    assert f"model {OPUS} -> {FABLE}" in got[2]["evidence"] and "own cache, last used 9200 s ago" in got[2]["evidence"]

    def upgrade_after_the_cache_expired(log):
        log.user(0); log.reply(5, usage(read=WARM, w1h=440_000), version="2.1.320")
        log.user(4000); log.reply(4005, usage(read=WARM, w1h=440_500), version="2.1.322")  # idle over an hour
    got = named(tmp_path / "c", upgrade_after_the_cache_expired)
    assert got[1]["cause"] == "ttl_expired" and got[1]["legit"] == 1
    assert "4000 s since the previous request" in got[1]["evidence"]
    assert "also in between: Claude Code was upgraded (2.1.320 -> 2.1.322)" in got[1]["evidence"]

    def upgrade_inside_the_lifetime_is_a_hit(log):  # three real upgrades were full hits
        log.user(0); log.reply(5, usage(read=WARM, w1h=440_000), version="2.1.320")
        log.user(600); log.reply(605, usage(read=500_000, w1h=500), version="2.1.322")
    got = named(tmp_path / "d", upgrade_inside_the_lifetime_is_a_hit)
    assert got[1]["cause"] == "none" and FULL in got[1]["evidence"]


def test_caches_are_per_model(tmp_path):
    def back_to_an_intact_cache(log):
        log.user(0); log.reply(5, usage(read=WARM, w1h=40_000))
        log.user(10); log.reply(15, usage(read=0, w1h=112_000), model=FABLE)
        log.user(20); log.reply(25, usage(read=100_000, w1h=13_000))     # all that Opus left
    got = named(tmp_path / "a", back_to_an_intact_cache)
    assert [r["cause"] for r in got] == ["none", "model_switch", "none"]
    assert FULL + " on this model left" in got[2]["evidence"]

    def side_call_then_the_real_cache_expired(log):
        log.user(0); log.reply(5, usage(read=WARM, w1h=440_000))
        log.user(10); log.reply(15, usage(w5=9_000), model=HAIKU, ttl="5m")
        log.user(4000); log.reply(4005, usage(read=WARM, w1h=441_000))   # Opus: 4000 s since its own last request
    got = named(tmp_path / "b", side_call_then_the_real_cache_expired)
    assert got[2]["cause"] == "ttl_expired", got[2]
    assert "4000 s since the previous request on this model" in got[2]["evidence"]
    assert f"also in between: model {HAIKU} -> {OPUS}" in got[2]["evidence"]
    assert got[2]["gap_seconds"] == 3994.0                               # since Opus's own previous turn ended

    def side_call_then_a_miss_inside_the_lifetime(log):
        log.user(0); log.reply(5, usage(read=WARM, w1h=440_000))
        log.user(10); log.reply(15, usage(w5=9_000), model=HAIKU, ttl="5m")
        log.user(20); log.reply(25, usage(read=WARM, w1h=441_000))       # nothing logged explains it
    got = named(tmp_path / "c", side_call_then_a_miss_inside_the_lifetime)
    assert got[2]["cause"] == "unknown" and "on this model" in got[2]["evidence"]   # the side call is not blamed

    def first_request_on_a_model(log):
        log.user(0); log.reply(5, usage(read=WARM, w1h=50_000))
        log.user(30); log.reply(35, usage(read=WARM, w1h=50_000), model=FABLE)     # 55% read: the rest is the switch's
        log.user(60); log.reply(65, usage(read=9_000, w5=100, fresh=10), model=HAIKU, ttl="5m")  # all of it read
    got = named(tmp_path / "d", first_request_on_a_model)
    assert [r["cause"] for r in got] == ["none", "model_switch", "none"]
    assert "read its whole prompt" in got[2]["evidence"]


def test_a_change_at_the_front_is_named_only_when_under_half_was_read(tmp_path):
    """The system prompt, the effort and the version sit at the front of the prompt. A read that went
    far past the front was not broken there, whatever changed."""
    def upgrade_and_most_was_read(log):
        log.user(0); log.reply(5, usage(read=WARM, w1h=440_000), version="2.1.320")
        log.user(600); log.reply(605, usage(read=440_000, w1h=61_000), version="2.1.321")  # 88% of what was left
    got = named(tmp_path / "a", upgrade_and_most_was_read)
    assert got[1]["cause"] == "none" and HALF in got[1]["evidence"] and got[1]["unread_write_tokens"] is None

    def anthropic_says_which(log):
        log.user(0); log.reply(5, usage(read=WARM, w1h=440_000), version="2.1.320")
        log.user(600); log.reply(605, usage(read=300_000, w1h=201_000), version="2.1.321",
                                 diagnostics={"cache_miss_reason": {"type": "messages_changed"}})
    got = named(tmp_path / "b", anthropic_says_which)
    assert got[1]["cause"] == "history_edit" and "Anthropic's own diagnosis" in got[1]["evidence"]

    def effort(log, read):
        log.user(0); log.reply(5, usage(read=WARM, w1h=440_000), perTurnEffort="medium", model="claude-opus-5")
        log.user(60); log.reply(65, usage(read=read, w1h=500_000 - read), perTurnEffort="high", model="claude-opus-5")
    assert named(tmp_path / "c", lambda log: effort(log, 440_000))[1]["cause"] == "none"
    assert named(tmp_path / "d", lambda log: effort(log, WARM))[1]["cause"] == "param_rendered"

    def snapshot_changed_but_the_cache_had_expired(log):
        log.lines.append({"type": "attachment", "timestamp": ts(0), "attachment": {
            "type": "prompt_snapshot", "systemPrompt": ["v1"]}})
        log.user(1); log.reply(5, usage(read=WARM, w1h=440_000))
        log.lines.append({"type": "attachment", "timestamp": ts(7200), "attachment": {
            "type": "prompt_snapshot", "systemPrompt": ["v2"]}})
        log.user(7201); log.reply(7205, usage(read=0, w1h=500_500))
    assert named(tmp_path / "e", snapshot_changed_but_the_cache_had_expired)[1]["cause"] == "ttl_expired"


def test_the_hit_test_is_against_what_was_left_in_the_cache(tmp_path):
    def uncached_tail(log):
        # The previous request left 100k cached and had a 60k tail it did not cache: reading the 100k
        # is everything, though it is under 95% of that request's 160k prompt.
        log.user(0); log.reply(5, usage(read=WARM, w1h=40_000, fresh=60_000))
        log.user(30); log.reply(35, usage(read=100_000, w1h=62_000))
    assert FULL in named(tmp_path / "a", uncached_tail)[1]["evidence"]

    def share(log, read):
        log.user(0); log.reply(5, usage(read=WARM, w1h=40_000))
        log.user(30); log.reply(35, usage(read=read, w1h=101_000 - read))
    assert FULL in named(tmp_path / "b", lambda log: share(log, 95_000))[1]["evidence"]   # 95%: a full reuse
    got = named(tmp_path / "c", lambda log: share(log, 94_000))[1]                         # under it, nothing logged
    assert got["cause"] == "none" and HALF in got["evidence"]
    got = named(tmp_path / "d", lambda log: share(log, 49_000))[1]                         # under half
    assert got["cause"] == "unknown"

    def a_call_that_cached_nothing_in_between(log):
        log.user(0); log.reply(5, usage(w1h=100_000))
        log.user(10); log.reply(15, usage(fresh=3_000))                   # read nothing, wrote nothing
        log.user(40); log.reply(45, usage(read=40_000, w1h=61_000))       # 40% of the 100k still held
    got = named(tmp_path / "e", a_call_that_cached_nothing_in_between)
    assert got[2]["cause"] == "unknown", got[2]         # measured against the 100k, not the call that left nothing

    def previous_usage_is_a_sum(log):
        # 4,000 s later, 60% of the prompt read. Against a sum of two calls nothing exact can be said:
        # it is not called an expiry.
        log.user(0); log.reply(5, {**usage(read=200_000, w1h=6_000), "iterations": [{}, {}]})
        log.user(4000); log.reply(4005, usage(read=66_000, w1h=44_000))
    got = named(tmp_path / "f", previous_usage_is_a_sum)
    assert got[1]["cause"] == "none" and HALF in got[1]["evidence"]


def test_expiry_without_the_write_split(tmp_path):
    """Older logs do not say how long a write is kept."""
    def bare(read, w, **kw):
        return {"input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": read,
                "cache_creation_input_tokens": w, **kw}

    def asked_for_five_minutes(log):
        log.user(0); log.reply(5, bare(0, 100_000), ttl="5m")
        log.user(400); log.reply(405, bare(0, 100_500), ttl="5m")
    got = named(tmp_path / "a", asked_for_five_minutes)
    assert got[1]["cause"] == "ttl_expired" and "kept 300 s" in got[1]["evidence"]

    def nothing_known(log):
        log.user(0); log.reply(5, bare(0, 100_000), ttl=None, version="2.1.100")
        log.user(86_400); log.reply(86_405, bare(0, 100_500), ttl=None, version="2.1.101")   # a day, and an upgrade
        log.user(86_500); log.reply(86_505, bare(0, 101_000), ttl=None, version="2.1.101")   # 100 s: not claimed
    got = named(tmp_path / "b", nothing_known)
    assert got[1]["cause"] == "ttl_expired" and "longer than any cache lifetime" in got[1]["evidence"]
    assert got[2]["cause"] == "unknown"


def test_send_time_is_the_last_input_before_the_response(tmp_path):
    """Claude Code writes a deferred_tools_record when the response arrives. Taking it for the send
    time hid two real 5-minute expiries (gaps of 318 s read as 153 s and 292 s)."""
    def slow_previous_response(log, record="deferred_tools_record"):
        log.user(0); log.attach(170, record)
        log.reply(170, usage(w5=110_000), ttl="5m", split=False)   # sent at 0, took 170 s to answer
        log.user(319); log.attach(325, record)
        log.reply(325, usage(w5=110_500), ttl="5m", split=False)   # sent 319 s after the first: expired
    got = named(tmp_path / "a", slow_previous_response)
    assert got[1]["cause"] == "ttl_expired" and "319 s since the previous request" in got[1]["evidence"]
    # a record type not known here, written with the response, is not an input either
    got = named(tmp_path / "b", lambda log: slow_previous_response(log, "some_new_record"))
    assert got[1]["cause"] == "ttl_expired" and "319 s since the previous request" in got[1]["evidence"]

    def an_entry_stamped_after_the_response(log):
        log.user(0); log.reply(5, usage(w5=110_000), ttl="5m", split=False)
        log.user(4000); log.attach(4005.03, "some_new_record")     # 30 ms after the response it precedes
        log.reply(4005, usage(w5=110_500), ttl="5m", split=False)
        log.user(4100); log.reply(4105, usage(read=0, w5=111_000), ttl="5m", split=False)
    got = named(tmp_path / "c", an_entry_stamped_after_the_response)
    assert got[1]["cause"] == "ttl_expired" and "4000 s" in got[1]["evidence"]     # the user entry still counts
    assert got[2]["cause"] == "unknown" and "100 s" in got[2]["evidence"]          # and the next is not "4095 s"

    def a_user_entry_stamped_after_the_response(log):              # a clock oddity: it cannot be this request's input
        log.user(0); log.reply(5, usage(w5=110_000), ttl="5m", split=False)
        log.user(200); log.user(400)
        log.reply(205, usage(read=110_000, w5=500), ttl="5m", split=False)
    got = named(tmp_path / "g", a_user_entry_stamped_after_the_response)
    assert got[1]["gap_seconds"] == 195.0

    def an_attachment_written_with_the_request(log):
        log.user(0); log.reply(5, usage(w5=110_000), ttl="5m", split=False)
        log.attach(310, "task_reminder")                           # no user entry: a reminder started the turn
        log.reply(313, usage(w5=110_500), ttl="5m", split=False)
    got = named(tmp_path / "d", an_attachment_written_with_the_request)
    assert got[1]["cause"] == "ttl_expired" and "310 s since the previous request" in got[1]["evidence"]

    def queued_while_streaming(log):
        log.user(0); log.reply(5, usage(w1h=20_000), split=False)
        log.attach(3, "queued_command")                            # typed while the first response streamed
        log.reply(8, usage(read=20_000, w1h=300), split=False)
        log.reply(12, usage(read=20_300, w1h=300), split=False)    # no input logged at all
    got = named(tmp_path / "e", queued_while_streaming)
    assert [r["cause"] for r in got] == ["first_write", "none", "none"]
    assert [r["gap_seconds"] for r in got[1:]] == [0.0, 0.0]       # sent when the previous response ended

    def a_late_block_of_the_previous_response(log):
        log.user(0); log.reply(5, usage(w1h=20_000), split=False)
        first = dict(log.lines[-1])
        log.user(6); log.reply(8, usage(read=20_000, w1h=300), split=False)
        log.lines.append({**first, "timestamp": ts(11)})           # a block of the first response, logged late
    got = named(tmp_path / "f", a_late_block_of_the_previous_response)
    assert sorted(r["cause"] for r in got) == ["first_write", "none"]
    assert [r["gap_seconds"] for r in got if r["cause"] == "none"] == [0.0]       # clamped, never negative


def test_rows_from_an_earlier_version_of_the_rules_are_kept_and_counted(tmp_path):
    log = Log()
    log.user(0); log.reply(5, usage(w1h=20_000))
    log.user(30); log.reply(35, usage(read=20_000, w1h=500))
    log.write(tmp_path / "-p" / "s.jsonl")
    led = Ledger(str(tmp_path / "l.db"))
    assert import_claude_code(tmp_path, led)["older_verdicts"] == 0
    led.conn.execute("UPDATE entries SET classifier_version = 'log-2', cause = 'unknown'")  # as an earlier release wrote them
    led.conn.commit()
    stats = import_claude_code(tmp_path, led)
    assert (stats["imported"], stats["skipped_known"], stats["older_verdicts"]) == (0, 2, 2)
    assert {r[1] for r in rows(led)} == {"unknown"}  # kept as they were
    out = subprocess.run([sys.executable, "-m", "prefixwise", "import", "claude-code", "--path", str(tmp_path),
                          "--ledger", str(tmp_path / "l.db")], capture_output=True, text=True)
    assert "2 requests" in out.stderr and "earlier version of the rules" in out.stderr and "new ledger" in out.stderr
