"""prefixwise run -- <agent>: zero-setup, and a report on that run."""

import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")

AGENT = r"""
import json, os, sys, urllib.request
def call(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    return urllib.request.urlopen(req, timeout=10).status
system = "You are a careful assistant. " * 200
msgs = [{"role": "user", "content": "hi"}]
a = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages"
for i in range(2):
    call(a, {"model": "claude-sonnet-5-5", "max_tokens": 5, "system": system, "messages": msgs},
         {"x-api-key": "the-agents-own-key"})
o = os.environ["OPENAI_BASE_URL"] + "/chat/completions"
call(o, {"model": "gpt-5.6-terra", "messages": [{"role": "system", "content": system}] + msgs},
     {"authorization": "Bearer sk-own"})
sys.exit(int(sys.argv[1]))
"""


def run(tmp_path, *args):
    script = tmp_path / "agent.py"
    script.write_text(AGENT)
    env = {**os.environ, "PYTHONPATH": SRC}
    env.pop("ANTHROPIC_BASE_URL", None)
    env.pop("OPENAI_BASE_URL", None)
    return subprocess.run([sys.executable, "-m", "prefixwise", "run", "--mock", "--ledger",
                           str(tmp_path / "l.db"), *args], capture_output=True, text=True, env=env, timeout=60)


def test_run_points_the_agent_at_the_proxy_and_reports_that_run(tmp_path):
    r = run(tmp_path, "--", sys.executable, str(tmp_path / "agent.py"), "3")
    assert r.returncode == 3  # the agent's own exit code
    assert "this run" in r.stderr and "3 requests" in r.stderr, r.stderr
    assert "All runs: prefixwise report" in r.stderr
    r2 = run(tmp_path, "--", sys.executable, str(tmp_path / "agent.py"), "0")
    assert r2.returncode == 0 and "3 requests" in r2.stderr  # only this run, not the first one too


def test_run_explains_when_nothing_arrived_and_handles_bad_commands(tmp_path):
    r = run(tmp_path, "--", sys.executable, "-c", "pass")
    assert r.returncode == 0 and "No requests reached Prefixwise" in r.stderr
    assert run(tmp_path, "--", "definitely-not-a-command").returncode == 127
    assert run(tmp_path).returncode == 2


def test_run_routes_each_provider_to_its_own_upstream_with_the_agents_keys(tmp_path):
    from test_proxy import OPENAI_OK, FakeUpstream

    anthropic_ok = {"id": "msg", "type": "message", "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 1}}
    a, o = FakeUpstream(payload=anthropic_ok), FakeUpstream(payload=OPENAI_OK)
    script = tmp_path / "agent.py"
    script.write_text(AGENT)
    env = {**os.environ, "PYTHONPATH": SRC, "ANTHROPIC_BASE_URL": a.url, "OPENAI_BASE_URL": o.url + "/v1"}
    try:
        r = subprocess.run([sys.executable, "-m", "prefixwise", "run", "--ledger", str(tmp_path / "l.db"), "--",
                            sys.executable, str(script), "0"], capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stderr
        assert [x["path"] for x in a.requests] == ["/v1/messages", "/v1/messages"]
        assert [x["path"] for x in o.requests] == ["/v1/chat/completions"]
        assert a.requests[0]["headers"]["x-api-key"] == "the-agents-own-key"
        assert o.requests[0]["headers"]["authorization"] == "Bearer sk-own"
    finally:
        a.server.shutdown()
        o.server.shutdown()
