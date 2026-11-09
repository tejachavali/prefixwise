"""A dollar line that is always there, and a result one can paste.

A Claude id the table does not list is priced at the nearest version of its family and the report
says so; a model no rule prices is named as unpriced instead of a silent dash. `report --share`
prints a short block of counts and dollars, with no prompt text to leak."""

import json
import subprocess
import sys
from pathlib import Path

from prefixwise.findings import findings, render, render_share
from prefixwise.pricing import PriceTable
from test_details import a_usage, body, proxy


def test_a_claude_point_release_is_priced_at_the_nearest_known_version():
    t = PriceTable.load(None)
    assert t.resolve("claude-opus-4-8") == ("claude-opus-4-8", t.models["claude-opus-4-8"], False)
    key, rates, approx = t.resolve("claude-opus-4-9")  # not listed: the newest opus at or below 4.9
    assert (key, approx) == ("claude-opus-4-8", True) and rates["input"] == 5.0
    assert t.resolve("claude-sonnet-5-2")[0] == "claude-sonnet-5"
    assert t.resolve("claude-haiku-6")[0] == "claude-haiku-5-5"  # nothing at or below: the nearest above
    assert t.resolve("claude-haiku-4-5")[2] is False  # listed exactly now
    assert t.resolve("us.anthropic.claude-opus-4-8-v1:0") == ("claude-opus-4-8", t.models["claude-opus-4-8"], False)
    assert t.resolve("gpt-5.6") == (None, None, False)  # no family rule for OpenAI ids: unpriced, said so
    assert t.resolve("some-llama")[1] is None
    # a dated snapshot on a major-only id is not a minor version: Opus 4 (20250514) is the $15 generation
    key, rates, approx = t.resolve("claude-opus-4-20250514")
    assert (key, approx) == ("claude-opus-4-1", True) and rates["input"] == 15.0
    assert t.resolve("claude-opus-4-9-fast")[1] is None
    # every current Claude model on the pricing page is in the table
    for m in ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5",
              "claude-sonnet-5", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-fable-5", "claude-mythos-5-1"):
        assert m in t.models, m


def test_the_report_names_approximate_and_unpriced_models():
    p = proxy([a_usage(write5=8_000), a_usage(write5=8_000)])
    p.handle("anthropic", body([{"role": "user", "content": "hi"}], model="claude-opus-4-9"), {"x-prefixwise-session-id": "a"})
    p.handle("anthropic", body([{"role": "user", "content": "hi"}], model="claude-newthing-7"), {"x-prefixwise-session-id": "b"})
    f = findings(p.ledger)
    assert f["approximate_prices"] == [("claude-opus-4-9", "claude-opus-4-8")]
    assert f["unpriced_models"] == ["claude-newthing-7"]
    text = render(f)
    assert "Priced at a related model's rate: claude-opus-4-9 as claude-opus-4-8" in text
    assert "Not priced: claude-newthing-7" in text and "1 of 2 requests priced" in text
    share = render_share(f)
    assert "pricing: 1 of 2 requests priced; priced at a related model's rate: claude-opus-4-9 as claude-opus-4-8; not priced: claude-newthing-7" in share


def test_share_block_has_the_numbers_and_no_text():
    p = proxy([a_usage(write5=8_000), a_usage(fresh=8_000), a_usage(fresh=8_000)])
    secret = "zqx-never-shown"
    msgs = [{"role": "user", "content": f"please {secret}"}]
    p.handle("anthropic", body(msgs, model="claude-opus-4-8"), {"x-prefixwise-session-id": "s"})
    for i in range(2):  # tool order drifts: waste
        b = body(msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": f"t{i}"}], model="claude-opus-4-8")
        b["tools"] = list(reversed(b["tools"])) if i == 0 else b["tools"]
        p.handle("anthropic", b, {"x-prefixwise-session-id": "s"})
    out = render_share(findings(p.ledger), "7d")
    assert out.startswith("prefixwise (7d)") and "3 requests" in out and "$" in out
    assert "top cause:" in out and "fix:" in out and "pip install prefixwise" in out
    assert secret not in out and "please" not in out
    assert len(out.splitlines()) <= 8


def test_cli_report_share(tmp_path):
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PATH": "/usr/bin:/bin"}
    ledger = tmp_path / "l.db"
    agent = tmp_path / "agent.py"
    agent.write_text("import json, os, urllib.request\n"
                     "base = os.environ['OPENAI_BASE_URL']\n"
                     "for i in range(2):\n"
                     "    body = {'model': 'gpt-5.6-luna', 'messages': [{'role': 'system', 'content': 'x ' * 50}, {'role': 'user', 'content': 'hi'}]}\n"
                     "    req = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer u'})\n"
                     "    urllib.request.urlopen(req).read()\n")
    run = subprocess.run([sys.executable, "-m", "prefixwise", "run", "--mock", "--ledger", str(ledger), "--", sys.executable, str(agent)],
                         capture_output=True, text=True, env=env, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    share = subprocess.run([sys.executable, "-m", "prefixwise", "report", "--ledger", str(ledger), "--share"],
                           capture_output=True, text=True, env=env, cwd=tmp_path)
    assert share.returncode == 0, share.stderr
    assert share.stdout.startswith("prefixwise\n2 requests") and "github.com/tejachavali/prefixwise" in share.stdout
