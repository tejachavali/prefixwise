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
    assert ttls == [300.0]  # the lifetime it was judged against: what the 5m/1h advice reads
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


ENV = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}


def test_cli_import_prints_the_report(tmp_path):
    build(tmp_path)
    db = tmp_path / "x.db"
    out = subprocess.run([sys.executable, "-m", "prefixwise", "import", "claude-code", "--path", str(tmp_path),
                          "--ledger", str(db)], capture_output=True, text=True, env=ENV)
    assert out.returncode == 0, out.stderr
    assert "Imported 10 requests from 2 Claude Code logs" in out.stdout
    assert "asked for a 1-hour cache but the entry was kept for 5 minutes" in out.stdout
    assert "subagent:Explore" in out.stdout and "API prices" in out.stdout
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
