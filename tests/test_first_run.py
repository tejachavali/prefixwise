"""What a new user meets first: a server that reports no cached tokens (Ollama, llama.cpp), the bare
command, a report before any run."""

import json

from prefixwise.cli import main
from prefixwise.classifier import CACHE_NOT_REPORTED
from prefixwise.pricing import cache_reported
from prefixwise.proxy import PrefixwiseProxy

SYSTEM = {"role": "system", "content": "You are a helpful assistant. " * 300}
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}} for n in ("a", "b")]


def test_cache_reported_reads_whether_the_usage_has_a_count_at_all():
    assert not cache_reported("openai", {"prompt_tokens": 900, "completion_tokens": 2})  # Ollama, llama.cpp
    assert cache_reported("openai", {"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": 0}})
    assert cache_reported("openai", {"prompt_tokens": 900, "cached_tokens": 0})
    assert not cache_reported("openai", {"input_tokens": 900, "output_tokens": 2})  # Responses, no details
    assert cache_reported("openai", {"input_tokens": 900, "input_tokens_details": {"cached_tokens": 0}})
    assert cache_reported("anthropic", {"input_tokens": 900})
    assert cache_reported("openai", {})  # no usage at all is decided elsewhere


def _proxy(path, usage_of):
    p = PrefixwiseProxy(ledger_path=path, mock=True)
    p._forward = lambda provider, body, *a, **k: (200, {"id": "chatcmpl-x", "choices": [
        {"message": {"role": "assistant", "content": "ok"}}], "usage": usage_of(body)})
    return p


def _turns(n, tools_flip_at=None):
    msgs, out = [SYSTEM, {"role": "user", "content": "hi"}], []
    for i in range(n):
        tools = TOOLS[::-1] if i == tools_flip_at else TOOLS
        out.append({"model": "llama3.2:3b", "session_id": "s", "tools": tools, "messages": list(msgs)})
        msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": f"next {i}"}]
    return out


def test_a_server_without_cached_tokens_is_unknown_not_a_routing_miss_and_prompt_causes_are_still_named(tmp_path):
    p = _proxy(str(tmp_path / "l.db"), lambda b: {"prompt_tokens": len(json.dumps(b)) // 4, "completion_tokens": 2})
    causes = [p.handle("openai", t)[0]["cause"] for t in _turns(5, tools_flip_at=2)]
    # the reorder on turn 3, and the flip back on turn 4, are named from the prompt alone
    assert causes == ["first_write", "unknown", "tool_reserialize", "tool_reserialize", "unknown"]
    rows = p.ledger.turns("s")
    assert rows[1]["evidence"] == CACHE_NOT_REPORTED and rows[4]["evidence"] == CACHE_NOT_REPORTED


def test_a_server_that_reports_zero_cached_still_gets_the_timing_and_routing_rules(tmp_path):
    p = _proxy(str(tmp_path / "l.db"), lambda b: {"prompt_tokens": len(json.dumps(b)) // 4,
                                                  "prompt_tokens_details": {"cached_tokens": 0}})
    causes = [p.handle("openai", t)[0]["cause"] for t in _turns(3)]
    assert causes[0] == "first_write" and causes[1] == "routing_miss"  # a reported 0 is a real 0


def test_the_report_and_the_session_view_say_the_count_is_missing(tmp_path, capsys):
    path = str(tmp_path / "l.db")
    p = _proxy(path, lambda b: {"prompt_tokens": len(json.dumps(b)) // 4, "completion_tokens": 2})
    for t in _turns(3):
        p.handle("openai", t)
    assert main(["report", "--ledger", path]) == 0
    out = capsys.readouterr().out
    assert "This server does not report cached tokens (2 requests" in out and "not known, not zero" in out
    assert "Cause not determined" not in out
    assert main(["report", "--ledger", path, "--session", "latest"]) == 0
    out = capsys.readouterr().out
    assert "cache hit rate not reported by the server" in out.splitlines()[0]
    rows = [ln.split() for ln in out.splitlines() if ln[:1].isdigit()]
    assert rows[0][2:4] == ["0", "0.0%"] and rows[1][2:4] == ["?", "?"]  # turn 1 is a real first write
    assert "?: the server reports no cached-token count" in out


def test_the_bare_command_shows_how_to_start_and_a_missing_ledger_points_to_run(tmp_path, capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "prefixwise run -- <your agent>" in out and "prefixwise --help" in out
    try:
        main(["report", "--ledger", str(tmp_path / "none.db")])
    except SystemExit as exc:
        assert "prefixwise run -- <your agent>" in str(exc)
    else:
        raise AssertionError("expected an exit")


def test_a_null_details_object_is_no_count_and_kv_confirmed_evictions_still_win():
    from prefixwise.classifier import MissCause, RequestSnapshot, classify
    assert not cache_reported("openai", {"prompt_tokens": 10, "prompt_tokens_details": None})  # vLLM without the flag
    from prefixwise.fingerprint import Fingerprinter
    blocks = Fingerprinter().fingerprint_blocks([("system", "s" * 4000), ("messages", "q")])
    prev = RequestSnapshot(model="m", provider="openai", blocks=blocks, prompt_tokens=1200, cached_tokens=0)
    curr = RequestSnapshot(model="m", provider="openai", blocks=blocks, prompt_tokens=1200, cached_tokens=0,
                           gap_seconds=2.0)
    curr.cache_reported = False
    assert classify(prev, curr).cause == MissCause.UNKNOWN
    curr.engine, curr.evicted_blocks = True, 50
    assert classify(prev, curr).cause == MissCause.EVICTED  # the events saw the blocks go


def test_no_provider_minimum_is_applied_to_a_server_that_reports_no_count(tmp_path):
    p = _proxy(str(tmp_path / "l.db"), lambda b: {"prompt_tokens": 40, "completion_tokens": 2})
    short = [{"model": "llama3.2:3b", "session_id": "s", "messages": [{"role": "user", "content": "hi"}]},
             {"model": "llama3.2:3b", "session_id": "s", "messages": [{"role": "user", "content": "hi"},
                                                                      {"role": "assistant", "content": "ok"},
                                                                      {"role": "user", "content": "more"}]}]
    assert [p.handle("openai", t)[0]["cause"] for t in short] == ["first_write", "unknown"]  # not below_min_prefix


def test_explain_reads_the_count_the_same_way(tmp_path, capsys):
    a = {"request": {"model": "llama3.2:3b", "messages": [SYSTEM, {"role": "user", "content": "hi"}]},
         "usage": {"prompt_tokens": 900}}
    b = {"request": {"model": "llama3.2:3b", "messages": [SYSTEM, {"role": "user", "content": "hi"},
                                                         {"role": "assistant", "content": "ok"},
                                                         {"role": "user", "content": "more"}]},
         "usage": {"prompt_tokens": 920}}
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    main(["explain", str(f), "--json"])
    out = capsys.readouterr().out
    assert "unknown" in out.lower() and "routing_miss" not in out.lower()
