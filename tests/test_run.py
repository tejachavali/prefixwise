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


# -- review findings -------------------------------------------------------------------

PATIENT = r"""
import signal, sys, time
got = []
signal.signal(signal.SIGINT, lambda *a: got.append("int"))  # an agent cancelling its turn
print("ready", flush=True)
for _ in range(100):
    time.sleep(0.05)
    if got:
        break
print("child saw", got, flush=True)
sys.exit(0)
"""


def test_ctrl_c_reaches_the_agent_and_prefixwise_survives(tmp_path):
    import signal
    import time

    script = tmp_path / "patient.py"
    script.write_text(PATIENT)
    env = {**os.environ, "PYTHONPATH": SRC}
    p = subprocess.Popen([sys.executable, "-m", "prefixwise", "run", "--mock", "--ledger", str(tmp_path / "l.db"),
                          "--", sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, start_new_session=True)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(0.2)
    os.killpg(p.pid, signal.SIGINT)  # what the terminal does on Ctrl-C: the whole foreground group
    out, err = p.communicate(timeout=30)
    assert "child saw ['int']" in out  # the agent handled it (it was not SIGKILLed)
    assert p.returncode == 0 and "this run" in err  # Prefixwise survived and reported


def test_sigterm_is_passed_on_to_the_agent(tmp_path):
    import signal
    import time

    env = {**os.environ, "PYTHONPATH": SRC}
    p = subprocess.Popen([sys.executable, "-m", "prefixwise", "run", "--mock", "--ledger", str(tmp_path / "l.db"),
                          "--", sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(0.2)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=30)
    assert p.returncode == 128 + signal.SIGTERM  # the agent's own end, by shell convention


def test_gateway_base_urls_are_kept_as_given_and_keys_are_never_replaced(tmp_path):
    from test_proxy import OPENAI_OK, FakeUpstream

    a = FakeUpstream(payload={"id": "m", "type": "message", "content": [], "usage": {"input_tokens": 1}})
    o = FakeUpstream(payload=OPENAI_OK)
    script = tmp_path / "agent.py"
    script.write_text(AGENT)
    env = {**os.environ, "PYTHONPATH": SRC, "ANTHROPIC_BASE_URL": a.url,
           "OPENAI_BASE_URL": o.url + "/gateway/acct/openai",  # e.g. a Cloudflare / Azure style gateway
           "PREFIXWISE_API_KEY": "sk-must-not-be-used"}
    try:
        r = subprocess.run([sys.executable, "-m", "prefixwise", "run", "--ledger", str(tmp_path / "l.db"), "--",
                            sys.executable, str(script), "0"], capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stderr
        assert [x["path"] for x in o.requests] == ["/gateway/acct/openai/chat/completions"]
        assert o.requests[0]["headers"]["authorization"] == "Bearer sk-own"
        assert a.requests[0]["headers"]["x-api-key"] == "the-agents-own-key"
        assert "acct" not in r.stderr  # only scheme and host are printed, never the gateway path
    finally:
        a.server.shutdown()
        o.server.shutdown()


def test_refuses_to_chain_onto_another_prefixwise_proxy(tmp_path):
    from test_proxy import run_proxy

    _, server, port = run_proxy()
    try:
        env = {**os.environ, "PYTHONPATH": SRC, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}
        r = subprocess.run([sys.executable, "-m", "prefixwise", "run", "--ledger", str(tmp_path / "l.db"), "--",
                            sys.executable, "-c", "pass"], capture_output=True, text=True, env=env, timeout=30)
        assert r.returncode == 2 and "already points at a Prefixwise proxy" in r.stderr
    finally:
        server.shutdown()


def test_report_wording_follows_from_the_numbers():
    from prefixwise.cli import _show_url
    from prefixwise.findings import render

    base = {"requests": 3, "requests_failed": 0, "prompt_tokens": 3000, "cached_tokens": 0,
            "cache_hit_rate": 0.0, "avoidable_tokens": 0, "avoidable_share": 0.0, "cost_usd": None,
            "avoidable_cost_usd": None, "requests_priced": 0, "tasks_successful": 0,
            "tokens_per_successful_task": None}
    unknown = {"cause": "unknown", "what": "the cause could not be determined", "requests": 2,
               "avoidable_tokens": 0, "avoidable_cost_usd": None}
    text = render({"summary": base, "waste": [], "expected": [], "unexplained": [unknown], "failures": []})
    assert "Cause not determined for 2 requests" in text and "Expected misses" not in text
    assert "as well as it can be" not in text
    only_failed = {**base, "requests": 0, "requests_failed": 4, "prompt_tokens": 0, "cache_hit_rate": None}
    text = render({"summary": only_failed, "waste": [], "expected": [], "unexplained": [],
                   "failures": [{"error": "http_401", "requests": 4}], "rejected_credentials": 4})
    assert text.startswith("No request got a response with usage: 4 failed.") and "subscription" in text
    waste = {"cause": "tool_reserialize", "what": "tools reordered", "requests": 1, "avoidable_tokens": 900,
             "avoidable_cost_usd": 0.01, "fix": "sort them", "proxy_fix": "canonical_tools", "fix_status": "proven"}
    partial = {**base, "avoidable_tokens": 900, "avoidable_share": 0.3, "requests_priced": 2, "cost_usd": 0.05,
               "avoidable_cost_usd": 0.01}
    text = render({"summary": partial, "waste": [waste], "expected": [], "unexplained": [], "failures": []},
                  command="run")
    assert "2 of 3 requests priced" in text and "priced requests only" in text
    assert "prefixwise run --fix canonical_tools" in text
    assert _show_url("https://user:secret@gw.example.com:8443/k/abc?key=1") == "https://gw.example.com:8443"
