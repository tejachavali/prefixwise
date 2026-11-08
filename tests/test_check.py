"""prefixwise check: a CI gate for cache regressions (prefixwise.check, the mock's tool loop, the
pytest plugin, the baseline and the Action)."""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from prefixwise import check
from prefixwise.ledger import Ledger, LedgerEntry
from prefixwise.pricing import PriceTable
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")

# A scripted agent. Scenarios are picked by argv so one script serves every test:
#   shuffled: the tool list rotates each turn (tool_reserialize)
#   dated:    int(time.time()) in the system prompt (volatile_system, needs the first-turn delay)
#   good:     append-only turns with a fixed tool list and a breakpoint
#   edited:   the previous user message is rewritten on turn 2 (history_edit: --strict only)
#   loop:     OpenAI chat with tools: asserts the mock's tool call, answers it (two turns)
#   single:   one turn, no tools (nothing eligible)
#   static:   a fixed date in the system prompt, otherwise good
AGENT = r"""
import json, os, sys, time, urllib.request
A = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages"
O = os.environ["OPENAI_BASE_URL"] + "/chat/completions"
def call(url, body, session=None):
    h = {"content-type": "application/json", "x-api-key": "k", "authorization": "Bearer k"}
    if session: h["x-prefixwise-session-id"] = session
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=h)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())
system = "You are a careful assistant. " * 200
tools = [{"name": n, "description": "d " * 50, "input_schema": {"type": "object",
          "properties": {"path": {"type": "string"}}, "required": ["path"]}} for n in ("read", "write", "grep")]
def anth(sys_text, tool_list, msgs, session):
    return call(A, {"model": "claude-sonnet-5-5", "max_tokens": 5,
                    "system": [{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}],
                    "tools": tool_list, "messages": msgs}, session)
def follow(msgs, r):
    nxt = ([{"type": "tool_result", "tool_use_id": r["content"][0]["id"], "content": "ok"}]
           if r["stop_reason"] == "tool_use" else "next")
    return msgs + [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": nxt}]
for scenario in sys.argv[1:]:
    msgs = [{"role": "user", "content": "hi"}]
    if scenario == "shuffled":
        for i in range(3):
            msgs = follow(msgs, anth(system, tools[i:] + tools[:i], msgs, scenario))
    elif scenario == "dated":
        for i in range(2):
            msgs = follow(msgs, anth(f"now={int(time.time())} " + system, tools, msgs, scenario))
    elif scenario in ("good", "static"):
        text = ("Knowledge cutoff 2025-01-01. " if scenario == "static" else "") + system
        for i in range(3):
            msgs = follow(msgs, anth(text, tools, msgs, scenario))
    elif scenario == "edited":
        msgs = follow(msgs, anth(system, tools, msgs, scenario))
        msgs = follow(msgs, anth(system, tools, msgs, scenario))
        msgs[1] = {"role": "assistant", "content": "a rewritten earlier answer"}  # a block the previous turn sent; not the opening message (a new conversation)
        anth(system, tools, msgs, scenario)
    elif scenario == "loop":
        ot = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object",
               "properties": {"path": {"type": "string"}, "n": {"type": "integer"}}, "required": ["path", "n"]}}}]
        m = [{"role": "system", "content": system}, {"role": "user", "content": "hi"}]
        r = call(O, {"model": "gpt-5.6", "tools": ot, "messages": m}, scenario)
        msg = r["choices"][0]["message"]
        assert r["choices"][0]["finish_reason"] == "tool_calls" and msg["tool_calls"][0]["function"]["name"] == "read", r
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "prefixwise", "n": 1}, msg
        m += [msg, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "ok"}]
        r = call(O, {"model": "gpt-5.6", "tools": ot, "messages": m}, scenario)
        assert r["choices"][0]["finish_reason"] == "stop", r
    elif scenario == "single":
        anth(system, [], msgs, scenario)
    elif scenario == "crash":
        sys.exit(3)
"""


def run_check(tmp_path, *args, scenarios=(), env_extra=None):
    script = tmp_path / "agent.py"
    script.write_text(AGENT)
    env = {**os.environ, "PYTHONPATH": SRC, **(env_extra or {})}
    for k in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "PREFIXWISE_URL", "PREFIXWISE_CHECK"):
        env.pop(k, None)
    return subprocess.run([sys.executable, "-m", "prefixwise", "check", *args, "--",
                           sys.executable, str(script), *scenarios],
                          capture_output=True, text=True, env=env, timeout=120, cwd=tmp_path)


def test_check_fails_on_shuffled_tools_and_a_per_second_timestamp_and_passes_a_good_agent(tmp_path):
    r = run_check(tmp_path, scenarios=("shuffled", "dated", "good", "loop"))
    assert r.returncode == 2, r.stdout + r.stderr
    out = r.stdout
    assert "FAIL  tool_reserialize  scenario=shuffled  turn=2  first changed block: tools[0]" in out
    assert "FAIL  volatile_system  scenario=dated  turn=2  first changed block: system[3]" in out
    assert "scenario=good" not in out and "scenario=loop" not in out
    assert "fix: Send tools in a fixed order" in out and "[proven]" in out
    assert re.search(r"tool_reserialize on claude-sonnet-5-5: 2 of \d+ eligible turns", out)
    assert "no baseline: tokens only" in out
    assert out.rstrip().splitlines()[-1].startswith("FAIL: 3 findings in 4 scenarios")
    # the good scenarios alone pass, exit 0, and the mock's tool loop gave the loop scenario two turns
    r = run_check(tmp_path, "--first-turn-delay", "0", "--json", scenarios=("good", "loop"))
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert rep["findings"] == [] and rep["summary"]["eligible_turns_by_model"] == {"claude-sonnet-5-5": 2, "gpt-5.6": 1}
    assert rep["summary"]["requests"] == 5 and rep["warnings"] == []


def test_allow_list_keys_on_scenario_cause_and_block(tmp_path):
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps([{"scenario": "shuffled", "cause": "tool_reserialize", "divergence_role": "tools"}]))
    r = run_check(tmp_path, "--first-turn-delay", "0", "--allow", str(allow), scenarios=("shuffled",))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALLOW tool_reserialize  scenario=shuffled" in r.stdout and "PASS: 2 findings (2 allowed)" in r.stdout
    allow.write_text(json.dumps([{"scenario": "other", "cause": "tool_reserialize"}]))
    assert run_check(tmp_path, "--first-turn-delay", "0", "--allow", str(allow), scenarios=("shuffled",)).returncode == 2
    allow.write_text(json.dumps([{"cause": "*"}]))  # any cause, any scenario
    assert run_check(tmp_path, "--first-turn-delay", "0", "--allow", str(allow), scenarios=("shuffled",)).returncode == 0


def test_strict_adds_the_edit_causes_and_fail_on_names_any_cause(tmp_path):
    assert run_check(tmp_path, "--first-turn-delay", "0", scenarios=("edited",)).returncode == 0
    r = run_check(tmp_path, "--first-turn-delay", "0", "--strict", scenarios=("edited",))
    assert r.returncode == 2 and "FAIL  history_edit  scenario=edited  turn=3" in r.stdout, r.stdout + r.stderr
    r = run_check(tmp_path, "--first-turn-delay", "0", "--fail-on", "history_edit", scenarios=("edited",))
    assert r.returncode == 2
    r = run_check(tmp_path, "--fail-on", "not_a_cause", scenarios=("good",))
    assert r.returncode == 2 and "unknown cause" in r.stderr


def test_a_failing_command_is_exit_1_never_a_pass_and_findings_still_win(tmp_path):
    r = run_check(tmp_path, "--first-turn-delay", "0", scenarios=("good", "crash"))
    assert r.returncode == 1 and "RUN FAILED" in r.stdout and "the command exited 3" in r.stdout, r.stdout
    r = run_check(tmp_path, "--first-turn-delay", "0", scenarios=("shuffled", "crash"))
    assert r.returncode == 2 and "the command exited 3" in r.stdout
    env = {**os.environ, "PYTHONPATH": SRC}
    r = subprocess.run([sys.executable, "-m", "prefixwise", "check", "--", "definitely-not-a-command"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and "command not found" in r.stderr
    r = subprocess.run([sys.executable, "-m", "prefixwise", "check"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 2 and "usage:" in r.stderr


def test_warns_on_a_static_date_in_the_system_prompt_and_on_no_multi_turn_sequence(tmp_path):
    r = run_check(tmp_path, "--first-turn-delay", "0", scenarios=("static",))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARN scenario=static: a date is in the system prompt" in r.stdout
    assert "2025" not in r.stdout  # the text itself never appears
    r = run_check(tmp_path, "--first-turn-delay", "0", scenarios=("single",))
    assert r.returncode == 0 and "WARN 0 cache-eligible multi-turn sequences observed" in r.stdout
    # a changing date is a finding, not a warning: no double report; nor a warning when it is not gated
    r = run_check(tmp_path, scenarios=("dated",))
    assert r.returncode == 2 and "WARN" not in r.stdout
    r = run_check(tmp_path, "--fail-on", "tool_reserialize", scenarios=("dated",))
    assert r.returncode == 0 and "WARN" not in r.stdout


def test_bodies_mode_judges_saved_trajectories_offline(tmp_path):
    env = {**os.environ, "PYTHONPATH": SRC}
    r = subprocess.run([sys.executable, "-m", "prefixwise", "check", "--bodies", str(ROOT / "examples" / "session.jsonl")],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "FAIL  tool_reserialize  scenario=session.jsonl  turn=4  first changed block: tools[0]" in r.stdout
    assert "FAIL  tool_result_mutation  scenario=session.jsonl  turn=3" in r.stdout
    assert "mock usage" not in r.stdout  # real usage was given
    r = subprocess.run([sys.executable, "-m", "prefixwise", "check", "--fail-on", "volatile_system", "--bodies",
                        str(ROOT / "examples" / "session.jsonl")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("PASS")


# -- the mock's tool loop, in every API and in streams -------------------------------------

@pytest.fixture
def mock_server():
    proxy = PrefixwiseProxy(ledger_path=":memory:", mock=True, upstreams=None)
    proxy.versioned = {"openai"}
    proxy.api_key = None
    proxy.mock_tool_calls = True
    proxy.check_mode = True
    server = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield proxy, base
    finally:
        server.shutdown()
        server.server_close()
        proxy.close()


def post(base, path, body, headers=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read()


def sse_json(raw: bytes) -> list[dict]:
    return [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


ANTH_TOOLS = [{"type": "web_search_20250305", "name": "web_search"},  # built in: no schema, skipped
              {"name": "lookup", "description": "d", "input_schema": {"type": "object", "properties": {
                  "q": {"type": "string"}, "k": {"type": "integer"}, "on": {"type": "boolean"},
                  "mode": {"type": "string", "enum": ["fast", "slow"]}, "tags": {"type": "array"},
                  "opt": {"type": ["null", "number"]}}, "required": ["q", "k", "on", "mode", "tags", "opt"]}}]


def test_mock_answers_a_user_turn_that_offers_tools_with_a_typed_tool_call_then_stops(mock_server):
    _, base = mock_server
    body = {"model": "claude-haiku-5-5", "max_tokens": 5, "tools": ANTH_TOOLS, "messages": [{"role": "user", "content": "hi"}]}
    _, raw = post(base, "/v1/messages", body)
    r = json.loads(raw)
    assert r["stop_reason"] == "tool_use" and r["content"][0]["name"] == "lookup"
    assert r["content"][0]["input"] == {"q": "prefixwise", "k": 1, "on": False, "mode": "fast", "tags": [], "opt": 1.0}
    body["messages"] += [{"role": "assistant", "content": r["content"]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": r["content"][0]["id"], "content": "x"}]}]
    r2 = json.loads(post(base, "/v1/messages", body)[1])
    assert r2["stop_reason"] == "end_turn" and r2["content"][0]["type"] == "text"
    # a new user message after the answer gets one more tool call: one round per user turn, never a loop
    body["messages"] += [{"role": "assistant", "content": r2["content"]}, {"role": "user", "content": "more"}]
    assert json.loads(post(base, "/v1/messages", body)[1])["stop_reason"] == "tool_use"
    # no tools: text, as before
    assert json.loads(post(base, "/v1/messages", {**body, "tools": []})[1])["stop_reason"] == "end_turn"


def test_mock_tool_calls_alternate_and_honour_tool_choice(mock_server):
    _, base = mock_server
    # An agent that renders tool output as a plain user message: tool call, text, tool call, text...
    body = {"model": "claude-haiku-5-5", "max_tokens": 5, "tools": ANTH_TOOLS, "messages": [{"role": "user", "content": "hi"}]}
    seen = []
    for i in range(4):
        r = json.loads(post(base, "/v1/messages", body)[1])
        seen.append(r["stop_reason"])
        body["messages"] += [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": f"Tool output {i}"}]
    assert seen == ["tool_use", "end_turn", "tool_use", "end_turn"]
    # tool_choice none: never a call; a forced tool: that one
    fresh = {"model": "claude-haiku-5-5", "max_tokens": 5, "tools": ANTH_TOOLS, "messages": [{"role": "user", "content": "hi"}]}
    assert json.loads(post(base, "/v1/messages", {**fresh, "tool_choice": {"type": "none"}}, {"x-prefixwise-session-id": "a"})[1])["stop_reason"] == "end_turn"
    two = ANTH_TOOLS + [{"name": "second", "input_schema": {"type": "object", "properties": {}}}]
    r = json.loads(post(base, "/v1/messages", {**fresh, "tools": two, "tool_choice": {"type": "tool", "name": "second"}},
                        {"x-prefixwise-session-id": "b"})[1])
    assert r["content"][0]["name"] == "second"
    chat_tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}, {"type": "function", "function": {"name": "g", "parameters": {}}}]
    r = json.loads(post(base, "/v1/chat/completions", {"model": "gpt-5.6", "tools": chat_tools, "tool_choice": "none",
                                                       "messages": [{"role": "user", "content": "hi"}]}, {"x-prefixwise-session-id": "c"})[1])
    assert r["choices"][0]["finish_reason"] == "stop"
    r = json.loads(post(base, "/v1/chat/completions", {"model": "gpt-5.6", "tools": chat_tools,
                                                       "tool_choice": {"type": "function", "function": {"name": "g"}},
                                                       "messages": [{"role": "user", "content": "hi"}]}, {"x-prefixwise-session-id": "d"})[1])
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "g"


def test_mock_tool_calls_in_openai_chat_and_responses_and_in_streams(mock_server):
    _, base = mock_server
    chat_tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}}]
    r = json.loads(post(base, "/v1/chat/completions", {"model": "gpt-5.6", "tools": chat_tools, "messages": [{"role": "user", "content": "hi"}]})[1])
    c = r["choices"][0]
    assert c["finish_reason"] == "tool_calls" and c["message"]["content"] is None
    assert c["message"]["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a": "prefixwise"}'}
    after = [{"role": "user", "content": "hi"}, c["message"], {"role": "tool", "tool_call_id": c["message"]["tool_calls"][0]["id"], "content": "ok"}]
    assert json.loads(post(base, "/v1/chat/completions", {"model": "gpt-5.6", "tools": chat_tools, "messages": after})[1])["choices"][0]["finish_reason"] == "stop"
    # chat stream: the tool call rides a delta, finish_reason tool_calls
    _, raw = post(base, "/v1/chat/completions", {"model": "gpt-5.6", "tools": chat_tools, "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    chunks = sse_json(raw)
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "f"
    assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
    # responses: a function_call item; function_call_output ends the round
    resp_tools = [{"type": "function", "name": "g", "parameters": {"type": "object", "properties": {}}}]
    rs = {"x-prefixwise-session-id": "resp"}  # its own conversation: tool calls alternate per conversation
    r = json.loads(post(base, "/v1/responses", {"model": "gpt-5.6", "tools": resp_tools, "input": "hi"}, rs)[1])
    item = r["output"][0]
    assert item["type"] == "function_call" and item["name"] == "g" and item["arguments"] == "{}"
    r2 = json.loads(post(base, "/v1/responses", {"model": "gpt-5.6", "tools": resp_tools, "input": [
        {"role": "user", "content": "hi"}, item, {"type": "function_call_output", "call_id": item["call_id"], "output": "ok"}]}, rs)[1])
    assert r2["output"][0]["type"] == "message"
    _, raw = post(base, "/v1/responses", {"model": "gpt-5.6", "tools": resp_tools, "input": "hi", "stream": True}, rs)
    names = [e["type"] for e in sse_json(raw)]
    assert "response.function_call_arguments.done" in names and names[-1] == "response.completed"
    assert sse_json(raw)[-1]["response"]["output"][0]["type"] == "function_call"
    # anthropic stream: a tool_use block with input_json_delta, stop_reason tool_use
    _, raw = post(base, "/v1/messages", {"model": "claude-haiku-5-5", "max_tokens": 5, "stream": True, "tools": ANTH_TOOLS,
                                         "messages": [{"role": "user", "content": "hi"}]})
    events = sse_json(raw)
    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"]["type"] == "tool_use" and start["content_block"]["input"] == {}
    delta = next(e for e in events if e["type"] == "content_block_delta")
    assert json.loads(delta["delta"]["partial_json"])["q"] == "prefixwise"
    assert next(e for e in events if e["type"] == "message_delta")["delta"]["stop_reason"] == "tool_use"


def test_the_mock_without_check_never_calls_tools_and_the_scenario_route_is_check_only():
    proxy = PrefixwiseProxy(ledger_path=":memory:", mock=True, upstreams=None)
    proxy.versioned = {"openai"}
    server = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        r = json.loads(post(base, "/v1/messages", {"model": "claude-haiku-5-5", "max_tokens": 5, "tools": ANTH_TOOLS,
                                                   "messages": [{"role": "user", "content": "hi"}]})[1])
        assert r["stop_reason"] == "end_turn"
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(base, "/api/v1/check/scenario", {"name": "x"})
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        proxy.close()


def test_first_turn_delay_holds_only_a_conversations_first_response(mock_server):
    proxy, base = mock_server
    proxy.mock_first_turn_delay = 0.6
    body = {"model": "claude-haiku-5-5", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
    t = time.monotonic()
    first = json.loads(post(base, "/v1/messages", body)[1])
    first_took = time.monotonic() - t
    assert first_took >= 0.6
    body["messages"] += [{"role": "assistant", "content": first["content"]}, {"role": "user", "content": "next"}]
    t = time.monotonic()
    post(base, "/v1/messages", body)
    assert time.monotonic() - t < first_took - 0.3  # the second response is not held


# -- the pytest plugin ----------------------------------------------------------------------

def test_pytest_plugin_names_each_test_as_the_scenario(tmp_path):
    (tmp_path / "test_agent.py").write_text(r"""
import json, os, urllib.request
def call():
    url = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages"
    system = "x " * 300
    msgs = [{"role": "user", "content": "hi"}]
    for text in ("a", "b"):
        body = {"model": "claude-haiku-5-5", "max_tokens": 5, "system": system + text, "messages": msgs}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", "x-api-key": "k"})
        urllib.request.urlopen(req, timeout=10).read()
def test_one():
    call()
def test_two():
    call()
""")
    env = {**os.environ, "PYTHONPATH": SRC}
    for k in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "PREFIXWISE_URL", "PREFIXWISE_CHECK"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, "-m", "prefixwise", "check", "--first-turn-delay", "0", "--strict", "--json", "--",
                        sys.executable, "-m", "pytest", "-q", "-p", "no:prefixwise", "-p", "prefixwise.pytest_plugin", "-p", "no:cacheprovider",  # the same module whether or not the package (and its pytest11 entry point) is installed
                        str(tmp_path / "test_agent.py")], capture_output=True, text=True, env=env, timeout=120, cwd=tmp_path)
    assert r.returncode == 2, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    scenarios = sorted({f["scenario"] for f in rep["findings"]})
    assert scenarios == ["test_agent.py::test_one", "test_agent.py::test_two"], rep
    assert all(f["cause"] == "system_edit" and f["turn"] == 2 for f in rep["findings"])
    assert rep["summary"]["scenarios"] == 2 and rep["summary"]["agent_exit"] == 0


def test_plugin_is_inert_outside_check(monkeypatch):
    from prefixwise import pytest_plugin

    monkeypatch.delenv("PREFIXWISE_CHECK", raising=False)
    monkeypatch.setenv("PREFIXWISE_URL", "http://127.0.0.1:9")  # nothing listens; must not be called
    called = []
    monkeypatch.setattr(pytest_plugin.urllib.request, "urlopen", lambda *a, **k: called.append(a))
    pytest_plugin._tell("x")
    assert called == []


# -- the baseline and the projection -----------------------------------------------------

def entry(session, cause, model="claude-sonnet-5-5", cached=0, avoidable=0, legit=None, provider="anthropic"):
    return LedgerEntry(session_id=session, agent_id="a", provider=provider, model=model, cause=cause, prompt_tokens=10_000,
                       cached_tokens=cached, avoidable_tokens=avoidable, avoidable_cost=None, legit=legit)


def test_baseline_from_a_ledger_scales_to_a_month_and_holds_counts_only(tmp_path):
    led = Ledger()
    day = 86400.0
    t0 = 1_700_000_000.0
    for i in range(10):  # 10 days, one first turn + two eligible turns a day on sonnet, a haiku side call
        led.record(entry(f"s{i}", "first_write"), at=t0 + i * day)
        led.record(entry(f"s{i}", "none", cached=8_000), at=t0 + i * day + 60)
        led.record(entry(f"s{i}", "history_edit", avoidable=6_000, legit=False), at=t0 + i * day + 120)
        led.record(entry(f"s{i}", "first_write", model="claude-haiku-5-5"), at=t0 + i * day + 130)
    led.record(LedgerEntry(session_id="s0", agent_id="a", provider="anthropic", model="claude-sonnet-5-5", cause=None,
                           prompt_tokens=0, cached_tokens=0, avoidable_tokens=0, avoidable_cost=None, status=500),
               at=t0 + 9 * day + 200)  # a failure: not counted
    base = check.baseline_from_ledger(led, now=t0 + 10 * day)
    assert base["version"] == 1 and base["window_days"] == pytest.approx(9.0, abs=0.01)
    son = base["models"]["claude-sonnet-5-5"]
    scale = 30 / 9
    assert son["turns_per_month"] == pytest.approx(30 * scale, rel=0.01)
    assert son["eligible_turns_per_month"] == pytest.approx(20 * scale, rel=0.01)
    assert son["mean_reusable_tokens"] == 7_000.0  # (8,000 read + 6,000 not read) / 2
    hai = base["models"]["claude-haiku-5-5"]
    assert hai["eligible_turns_per_month"] == 0 and hai["mean_reusable_tokens"] == 0.0
    assert json.dumps(base).count("s0") == 0  # no ids
    path = tmp_path / check.BASELINE_FILE
    path.write_text(json.dumps(base))
    assert check.load_baseline(path)["models"].keys() == {"claude-sonnet-5-5", "claude-haiku-5-5"}
    path.write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError):
        check.load_baseline(path)
    path.write_text(json.dumps({"version": 1, "models": {"m": {"turns_per_month": "lots"}}}))
    with pytest.raises(ValueError):
        check.load_baseline(path)


def test_projection_uses_the_eligible_denominator_and_the_write_minus_read_rate():
    rows = [{"session_id": "s", "model": "claude-sonnet-5-5", "provider": "anthropic", "cause": c, "status": 200,
             "prompt_tokens": 2000, "cached_tokens": 0, "avoidable_tokens": 1000 if c == "tool_reserialize" else 0,
             "divergence_role": "tools" if c == "tool_reserialize" else None, "divergence_index": 0,
             "evidence": None, "detail": None, "request_id": None}
            for c in ("first_write", "tool_reserialize", "none", "none", "tool_reserialize")]
    rep = check.judge(rows, check.DEFAULT_FAIL_ON)
    assert rep.requests == 5 and rep.eligible == 4 and len(rep.findings) == 2  # the first turn is not eligible
    switched = check.judge(rows + [{**rows[2], "model": "claude-haiku-5-5", "cause": "model_switch"}], check.DEFAULT_FAIL_ON)
    assert switched.eligible == 4 and "claude-haiku-5-5" not in switched.eligible_by_model  # nor a model switch
    prices = PriceTable.load()
    base = {"version": 1, "models": {"claude-sonnet-5-5": {"provider": "anthropic", "turns_per_month": 500_000,
                                                            "eligible_turns_per_month": 400_000, "mean_reusable_tokens": 10_000}}}
    check.project(rep, base, prices)
    p = rep.projections[0]
    assert (p.cause, p.affected, p.eligible, p.share, p.tokens_in_run) == ("tool_reserialize", 2, 4, 0.5, 2000)
    assert p.tokens_per_month == 0.5 * 400_000 * 10_000
    rate = prices.miss_rate("claude-sonnet-5-5", "anthropic", 10_000)
    r = prices.rates_for("claude-sonnet-5-5")
    assert rate == pytest.approx((r["cache_write_5m"] - r["cache_read"]) / 1e6)  # written again, minus the read it replaces
    assert p.usd_per_month == pytest.approx(p.tokens_per_month * rate)
    text = check.render(rep)
    assert "projected: ~$" in text and "an estimate" in text and "50%" in text
    # OpenAI: input minus read; an unknown model: tokens only
    assert prices.miss_rate("gpt-5.6-terra", "openai", 1000) == pytest.approx(
        (prices.rates_for("gpt-5.6-terra")["input"] - prices.rates_for("gpt-5.6-terra")["cache_read"]) / 1e6)
    rep2 = check.judge([{**rows[1], "model": "mystery-9"}, {**rows[2], "model": "mystery-9"}], check.DEFAULT_FAIL_ON)
    check.project(rep2, base, prices)
    assert rep2.projections[0].usd_per_month is None and "no turns on mystery-9" in rep2.projections[0].note
    check.project(rep2 := check.judge(rows, check.DEFAULT_FAIL_ON), None, prices)
    assert rep2.projections[0].note.startswith("no baseline")


def test_export_baseline_writes_the_file_and_check_picks_it_up_from_the_working_directory(tmp_path):
    led = Ledger(str(tmp_path / "prod.db"))
    for i in range(3):
        led.record(entry(f"s{i}", "first_write"), at=1_700_000_000 + i * 3600)
        led.record(entry(f"s{i}", "none", cached=5000), at=1_700_000_000 + i * 3600 + 60)
    env = {**os.environ, "PYTHONPATH": SRC}
    r = subprocess.run([sys.executable, "-m", "prefixwise", "export", "--baseline", "--ledger", str(tmp_path / "prod.db")],
                       capture_output=True, text=True, env=env, timeout=60, cwd=tmp_path)
    assert r.returncode == 0 and (tmp_path / check.BASELINE_FILE).exists(), r.stderr
    assert "eligible turns/month" in r.stderr
    base = json.loads((tmp_path / check.BASELINE_FILE).read_text())
    assert base["models"]["claude-sonnet-5-5"]["mean_reusable_tokens"] == 5000.0
    r = run_check(tmp_path, "--first-turn-delay", "0", "--json", scenarios=("shuffled",))  # cwd holds the baseline
    rep = json.loads(r.stdout)
    assert rep["projections"][0]["eligible_turns_per_month"] == base["models"]["claude-sonnet-5-5"]["eligible_turns_per_month"]
    assert rep["projections"][0]["usd_per_month"] > 0


# -- packaging -----------------------------------------------------------------------------

def test_the_action_runs_prefixwise_check_and_the_plugin_is_an_entry_point():
    action = (ROOT / "action.yml").read_text()
    assert "using: composite" in action and "prefixwise check" in action and "PIPESTATUS" in action
    assert "fail-on" in action and "strict" in action and "allow" in action and "baseline" in action
    assert 'prefixwise = "prefixwise.pytest_plugin"' in (ROOT / "pyproject.toml").read_text()
    assert "**`prefixwise check`" in (ROOT / "CHANGELOG.md").read_text()  # the gate is in the changelog
