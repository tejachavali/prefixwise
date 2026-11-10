"""prefixwise import codex: Codex rollout logs, read offline.

The fixtures follow the rollout format in openai/codex (codex-rs/history/src/rollout_payload.rs,
codex-rs/protocol/src/protocol.rs, Nov 2026): {"timestamp", "type", "payload"} lines.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prefixwise.importers import import_codex
from prefixwise.ledger import Ledger

T0 = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)


def ts(s):
    return (T0 + timedelta(seconds=s)).isoformat().replace("+00:00", "Z")


def line(t, kind, payload):
    return {"timestamp": ts(t), "type": kind, "payload": payload}


def usage(inp, cached, write=0, out=10):
    return {"input_tokens": inp, "cached_input_tokens": cached, "cache_write_input_tokens": write,
            "output_tokens": out, "reasoning_output_tokens": 0, "total_tokens": inp + out}


def record(t, rid, inp, cached, write=0):
    return line(t, "token_usage_record", {"thread_id": "th1", "turn_id": "t", "session_id": "s",
                                          "root_turn_id": "t", "response_id": rid, "usage": usage(inp, cached, write),
                                          "turn_token_usage": usage(inp, cached), "thread_token_usage": usage(inp, cached)})


def write_log(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n{broken\n")


def build(root: Path):
    main = [line(0, "session_meta", {"id": "th1", "timestamp": ts(0), "cwd": "/x"}),
            line(1, "turn_context", {"model": "gpt-5.6"}),
            record(5, "resp_1", 12000, 0, 12000),          # first request
            record(15, "resp_2", 12500, 11800),            # hit
            line(20, "dynamic_tools_updated", {"thread_id": "th1", "tools": [{"name": "new_tool"}]}),
            record(25, "resp_3", 13000, 0, 13000),          # tool catalog changed
            line(30, "compacted", {"message": "summary"}),
            record(35, "resp_4", 6000, 0, 6000),            # compaction
            line(40, "turn_context", {"model": "gpt-5.6-luna"}),
            record(45, "resp_5", 6500, 0, 6500),            # model switch
            record(55, "resp_6", 7000, 0)]                  # unexplained
    write_log(root / "sessions" / "2026" / "11" / "07" / "rollout-2026-11-07T09-00-00-th1.jsonl", main)
    sub = [line(0, "session_meta", {"id": "th2", "parent_thread_id": "th1"}),
           line(1, "turn_context", {"model": "gpt-5.6"}),
           record(6, "resp_s1", 9000, 0, 9000)]
    write_log(root / "sessions" / "2026" / "11" / "07" / "rollout-2026-11-07T09-00-01-th2.jsonl", sub)
    old = [line(0, "session_meta", {"id": "th3"}), line(1, "turn_context", {"model": "gpt-5.6"}),
           line(5, "event_msg", {"type": "token_count", "info": {"last_token_usage": usage(8000, 0),
                                                                  "total_token_usage": usage(8000, 0)}}),
           line(9, "event_msg", {"type": "token_count", "info": {"last_token_usage": usage(8400, 7900),
                                                                  "total_token_usage": usage(16400, 7900)}})]
    write_log(root / "archived_sessions" / "rollout-2026-11-01T09-00-00-th3.jsonl", old)


def rows(led):
    with led.reader() as c:
        return c.execute("SELECT session_id, agent_id, cause, prompt_tokens, cached_tokens, evidence, request_id, "
                         "cost FROM entries ORDER BY recorded_at, id").fetchall()


def test_codex_causes(tmp_path):
    build(tmp_path)
    led = Ledger()
    s = import_codex(tmp_path, led)
    assert s["imported"] == 9 and s["logs"] == 3
    main = [r for r in rows(led) if r[0] == "th1"]
    assert [r[2] for r in main] == ["first_write", "none", "tool_catalog_change", "compaction_rewrite",
                                    "model_switch", "unknown"]
    assert main[0][3] == 12000 and main[1][4] == 11800  # input includes cached tokens
    assert "OpenAI's cache lifetime is not fixed" in main[5][5]
    assert main[0][6] == "resp_1"


def test_subagent_threads_and_older_logs(tmp_path):
    build(tmp_path)
    led = Ledger()
    import_codex(tmp_path, led)
    by = {r[0]: r for r in rows(led)}
    assert by["th2"][1] == "subagent" and by["th2"][2] == "first_write"
    old = [r for r in rows(led) if r[0] == "th3"]
    assert [r[2] for r in old] == ["first_write", "none"]  # token_count events when no records


def test_reimport_and_cli(tmp_path):
    build(tmp_path)
    led = Ledger()
    import_codex(tmp_path, led)
    assert import_codex(tmp_path, led)["imported"] == 0
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    out = subprocess.run([sys.executable, "-m", "prefixwise", "import", "codex", "--path", str(tmp_path),
                          "--ledger", str(tmp_path / "c.db")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "Imported 9 requests from 3 Codex logs" in out.stdout


def test_reordered_tools_are_waste_and_a_changed_catalog_is_not(tmp_path):
    """Reddit (kantorcodes1): Codex rebuilds its dynamic tool list between runs. The same tools in another
    order are `tool_reserialize` (waste, priced); a tool added or a schema edited is `tool_catalog_change`."""
    a = {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
    b = {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
    b2 = {**b, "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "mode": {"type": "string"}}}}
    log = [line(0, "session_meta", {"id": "th9", "timestamp": ts(0)}), line(1, "turn_context", {"model": "gpt-5.6-luna"}),
           line(2, "dynamic_tools_updated", {"thread_id": "th9", "tools": [a, b]}),
           record(5, "r1", 12000, 0, 12000),
           line(8, "dynamic_tools_updated", {"thread_id": "th9", "tools": [b, a]}),   # same tools, reordered
           record(10, "r2", 12400, 0),
           line(12, "dynamic_tools_updated", {"thread_id": "th9", "tools": [a, b2]}),  # a schema edited
           record(15, "r3", 12800, 0),
           line(17, "dynamic_tools_updated", {"thread_id": "th9", "tools": [a, b2, {"name": "grep"}]}),  # one added
           record(20, "r4", 13200, 0)]
    write_log(tmp_path / "sessions" / "rollout-2026-11-09T09-00-00-th9.jsonl", log)
    led = Ledger()
    import_codex(tmp_path, ledger=led)
    with led.reader() as c:
        got = c.execute("SELECT request_id, cause, legit, avoidable_tokens, avoidable_cost FROM entries "
                        "ORDER BY recorded_at, id").fetchall()
    by = {r[0]: r for r in got}
    assert by["r1"][1] == "first_write"
    assert by["r2"][1] == "tool_reserialize" and by["r2"][2] == 0 and by["r2"][3] == 12000  # the previous prompt, unread
    assert by["r2"][4] is not None and by["r2"][4] > 0  # priced: gpt-5.6-luna is in the price table
    assert by["r3"][1] == "tool_catalog_change" and by["r3"][2] == 1 and by["r3"][3] == 0
    assert by["r4"][1] == "tool_catalog_change" and by["r4"][2] == 1 and by["r4"][3] == 0


def test_a_codex_tool_whose_required_list_is_reordered_is_waste_as_in_the_proxy(tmp_path):
    """The importer's tool-set hash treats JSON Schema's `required` as a set, as the proxy does."""
    def t(required):
        return {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {}, "n": {}},
                                                    "required": required}}
    log = [line(0, "session_meta", {"id": "th8", "timestamp": ts(0)}), line(1, "turn_context", {"model": "gpt-5.6-luna"}),
           line(2, "dynamic_tools_updated", {"thread_id": "th8", "tools": [t(["path", "n"])]}),
           record(5, "r1", 12000, 0, 12000),
           line(8, "dynamic_tools_updated", {"thread_id": "th8", "tools": [t(["n", "path"])]}),
           record(10, "r2", 12400, 0)]
    write_log(tmp_path / "sessions" / "rollout-2026-11-09T09-00-00-th8.jsonl", log)
    led = Ledger()
    import_codex(tmp_path, ledger=led)
    with led.reader() as c:
        by = dict(c.execute("SELECT request_id, cause FROM entries").fetchall())
    assert by["r2"] == "tool_reserialize"
