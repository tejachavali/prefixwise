"""What a first-time user meets: `prefixwise demo`, a gate that cannot pass on nothing, and one-line
errors where there were tracebacks. Each test here failed before its change."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prefixwise import demo
from prefixwise.cli import COMMAND_ORDER, build_parser, main

ROOT = Path(__file__).resolve().parent.parent
# No setting of the developer's shell reaches the commands under test.
ENV = {k: v for k, v in os.environ.items()
       if not k.startswith("PREFIXWISE_") and k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL",
                                                         "OPENAI_BASE_URL", "PORT")}
ENV.update(PYTHONPATH=str(ROOT / "src"), NO_COLOR="1")
POST = ("import json,os,sys,urllib.request as u,urllib.error as e\n"
        "b=json.dumps({'model':'gpt-5.6-luna','messages':[{'role':'user','content':'hi'}]}).encode()\n"
        "try: u.urlopen(u.Request(os.environ['OPENAI_BASE_URL']+sys.argv[1],b,{'content-type':'application/json'})).read()\n"
        "except e.HTTPError as x: print('status', x.code)\n")


def cli(*args, cwd=None, env=None):
    return subprocess.run([sys.executable, "-m", "prefixwise", *args], capture_output=True, text=True,
                          cwd=cwd, env=env or ENV, timeout=120)


# -- prefixwise demo ---------------------------------------------------------------------

def test_demo_names_three_causes_and_the_fixed_run_has_none(tmp_path):
    out = cli("demo", "--pause", "0", "--keep", str(tmp_path / "keep"))
    assert out.returncode == 0, out.stderr
    text = out.stdout
    before, after = text.split("act 5: the fix")
    for cause in ("tool_reserialize", "volatile_system", "history_edit"):
        assert cause in before
        assert cause not in after
    assert before.count("first_write") == 1 and after.count("first_write") == 1
    assert "\033[" not in text                      # no colour codes into a pipe
    assert "illustrative" in text                   # mock numbers are labelled as such
    assert "prefixwise import claude-code" in text  # and it says where to go next
    # the ledgers it kept say the same: nine turns each, avoidable tokens only before
    rows = {}
    for name in ("before", "after"):
        rep = cli("report", "--ledger", str(tmp_path / "keep" / f"{name}.db"), "--json")
        rows[name] = json.loads(rep.stdout)["summary"]
    assert rows["before"]["requests"] == rows["after"]["requests"] == 9
    assert rows["before"]["avoidable_tokens"] > 0 and rows["after"]["avoidable_tokens"] == 0


def test_demo_runs_in_process_and_leaves_nothing_behind(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert main(["demo", "--pause", "0", "--provider", "openai"]) == 0
    assert "tool_reserialize" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []           # no ledger in the user's directory


def test_demo_takes_nothing_from_the_shell(tmp_path):
    """A collector, fixes or an engine set in the shell must not reach it: "no network" has to be true,
    and the broken run has to break."""
    env = {**ENV, "PREFIXWISE_COLLECTOR": "http://127.0.0.1:9", "PREFIXWISE_FIXES": "optimize",
           "PREFIXWISE_HMAC_KEY": "k", "PREFIXWISE_ENGINE": "vllm", "PREFIXWISE_LEDGER_PATH": str(tmp_path / "x.db")}
    out = cli("demo", "--pause", "0", env=env)
    assert out.returncode == 0, out.stderr
    before = out.stdout.split("act 5: the fix")[0]
    assert before.count("tool_reserialize") >= 2 and "volatile_system" in before and "history_edit" in before
    assert not (tmp_path / "x.db").exists()


def test_demo_keep_never_overwrites_a_ledger(tmp_path):
    mine = tmp_path / "before.db"
    mine.write_bytes(b"my own ledger")
    out = cli("demo", "--pause", "0", "--keep", str(tmp_path))
    assert out.returncode == 1 and "already holds before.db" in out.stderr and "Traceback" not in out.stderr
    assert mine.read_bytes() == b"my own ledger" and not (tmp_path / "after.db").exists()
    blocked = tmp_path / "a-file"
    blocked.write_text("x")
    out = cli("demo", "--pause", "0", "--keep", str(blocked / "sub"))
    assert out.returncode == 1 and "cannot keep the ledgers" in out.stderr and "Traceback" not in out.stderr


def test_demo_failures_are_one_line_and_leave_nothing(tmp_path, monkeypatch, capsys):
    made = []
    real = demo.tempfile.mkdtemp
    monkeypatch.setattr(demo.tempfile, "mkdtemp", lambda **kw: made.append(real(dir=str(tmp_path), **kw)) or made[-1])
    for boom, code in ((demo.DemoError("the provider answered 500"), 1), (KeyboardInterrupt(), 130)):
        def fail(*a, _boom=boom, **kw):
            raise _boom
        monkeypatch.setattr(demo, "run_session", fail)
        assert demo.run(["--pause", "0", "--no-color"]) == code
        err = capsys.readouterr().err
        assert "Traceback" not in err and err.strip().startswith("prefixwise demo:")
    assert len(made) == 2 and not any(os.path.exists(d) for d in made)
    for bad in ("-1", "nan"):
        with pytest.raises(SystemExit) as exc:
            demo.run(["--pause", bad])
        assert exc.value.code == 2


def test_demo_into_a_closed_pipe_or_without_a_terminal():
    piped = subprocess.run(["bash", "-c", f"{sys.executable} -m prefixwise demo --pause 0 | head -2"],
                           capture_output=True, text=True, env=ENV, timeout=120)
    assert piped.stdout.startswith("Prefixwise demo") and "Traceback" not in piped.stderr
    assert "BrokenPipe" not in piped.stderr
    closed = subprocess.run([sys.executable, "-m", "prefixwise", "demo", "--step", "--pause", "0"],
                            capture_output=True, text=True, env=ENV, timeout=120, preexec_fn=lambda: os.close(0))
    assert closed.returncode == 0 and "before and after" in closed.stdout and "Traceback" not in closed.stderr


def test_demo_colours_only_a_terminal(monkeypatch, capsys):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    assert demo.run(["--pause", "0"]) == 0
    assert "\033[" not in capsys.readouterr().out          # captured output is not a terminal
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert demo.run(["--pause", "0"]) == 0
    assert "\033[" in capsys.readouterr().out
    monkeypatch.setenv("NO_COLOR", "1")
    assert demo.run(["--pause", "0"]) == 0
    assert "\033[" not in capsys.readouterr().out


def test_demo_live_without_a_key_is_one_line(monkeypatch, capsys):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert demo.run(["--live"]) == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "Traceback" not in err


def test_hallway_script_still_runs():
    out = subprocess.run([sys.executable, str(ROOT / "experiments/demo/hallway.py"), "--auto", "--pause", "0"],
                         capture_output=True, text=True, env=ENV, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "before and after" in out.stdout


def test_help_lists_demo_first_and_every_command():
    sub = next(a for a in build_parser()._actions if getattr(a, "choices", None))
    listed = [row.dest for row in sub._choices_actions]
    assert listed[0] == "demo"
    assert sorted(listed) == sorted(sub.choices) == sorted(COMMAND_ORDER)
    assert listed == list(COMMAND_ORDER)
    assert cli("help").stdout == cli("--help").stdout
    assert cli("help", "check").stdout == cli("check", "--help").stdout
    assert cli("help", "demo").stdout == cli("demo", "--help").stdout
    assert "prefixwise demo" in cli().stdout                 # the no-argument text starts a new user there


# -- prefixwise check: nothing checked is not a pass -------------------------------------

def test_check_with_no_requests_is_not_a_pass():
    out = cli("check", "--", sys.executable, "-c", "print('ran')")
    assert out.returncode == 1
    assert "NOTHING CHECKED" in out.stdout and "no request reached the mock" in out.stdout
    assert "OPENAI_BASE_URL" in out.stdout
    js = json.loads(cli("check", "--json", "--", sys.executable, "-c", "pass").stdout)
    assert js["exit_code"] == 1 and js["summary"]["nothing_checked"] is True


def test_check_allow_empty_passes_and_says_so():
    out = cli("check", "--allow-empty", "--", sys.executable, "-c", "pass")
    assert out.returncode == 0
    assert out.stdout.strip().splitlines()[-1].startswith("PASS")
    assert "--allow-empty" in out.stdout


def test_check_failed_command_keeps_its_own_verdict():
    out = cli("check", "--", sys.executable, "-c", "raise SystemExit(3)")
    assert out.returncode == 1 and "RUN FAILED" in out.stdout


def test_example_scenario_fails_the_gate_and_passes_when_sorted():
    pytest.importorskip("openai")
    cmd = ("check", "--json", "--", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           str(ROOT / "examples" / "agent_scenarios"))
    bad = json.loads(cli(*cmd).stdout)
    assert bad["exit_code"] == 2
    assert {f["cause"] for f in bad["findings"]} == {"tool_reserialize"}
    assert bad["findings"][0]["scenario"].endswith("test_refactor.py::test_refactor")
    good = json.loads(cli(*cmd, env={**ENV, "SORT_TOOLS": "1"}).stdout)
    assert good["exit_code"] == 0 and good["summary"]["requests"] == 3 and not good["findings"]


# -- prefixwise run: why nothing arrived -------------------------------------------------

def test_run_says_why_no_request_arrived(tmp_path):
    led = str(tmp_path / "l.db")
    crashed = cli("run", "--mock", "--ledger", led, "--", sys.executable, "-c", "raise SystemExit(4)")
    assert crashed.returncode == 4
    assert "exited 4" in crashed.stderr and "its own output says why" in crashed.stderr
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:" in crashed.stderr   # it may also have run and gone elsewhere
    missing = cli("run", "--mock", "--ledger", led, "--", "no-such-command-prefixwise")
    assert missing.returncode == 127 and "did not start" in missing.stderr
    quiet = cli("run", "--mock", "--ledger", led, "--", sys.executable, "-c", "pass")
    assert quiet.returncode == 0
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:" in quiet.stderr and "/v1" in quiet.stderr
    assert "see the README" not in quiet.stderr


# -- one line, not a traceback -----------------------------------------------------------

def no_traceback(out, code, *words):
    assert out.returncode == code, (out.returncode, out.stderr)
    assert "Traceback" not in out.stderr
    for w in words:
        assert w in out.stderr, out.stderr


def test_a_file_that_is_not_a_ledger(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hello\n")
    for command in (("report",), ("fixes",), ("export",), ("request", "abc")):
        no_traceback(cli(*command, "--ledger", str(text)), 1, "cannot open", "as a ledger")
    no_traceback(cli("report", "--ledger", str(tmp_path)), 1, "cannot open", "as a ledger")
    assert text.read_text() == "hello\n"


def test_task_and_prune_do_not_create_a_ledger(tmp_path):
    no_traceback(cli("task", "t1", "--success", cwd=tmp_path), 1, "no ledger at")
    no_traceback(cli("prune", "--older-than", "30d", cwd=tmp_path), 1, "no ledger at")
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "l.db")
    out = cli("run", "--mock", "--ledger", path, "--", sys.executable, "-c", POST, "/chat/completions")
    assert out.returncode == 0, out.stderr
    return path


def test_export_errors_are_one_line(ledger, tmp_path):
    no_traceback(cli("export", "--ledger", ledger, "-o", str(tmp_path / "no" / "dir" / "x.jsonl")), 2, "cannot write")
    no_traceback(cli("export", "--ledger", ledger, "--where", "banana"), 2, "FIELD=VALUE", "banana")
    no_traceback(cli("export", "--ledger", ledger, "--baseline", "-o", str(tmp_path / "no" / "b.json")), 2,
                 "cannot write")
    ok = cli("export", "--ledger", ledger, "--where", "model=gpt-5.6-luna")
    assert ok.returncode == 0 and len(ok.stdout.splitlines()) == 1


def test_a_filter_that_empties_the_report_says_so(ledger):
    assert cli("report", "--ledger", ledger).stdout.startswith("1 request")
    out = cli("report", "--ledger", ledger, "--since", "0s")
    assert out.returncode == 0
    assert "No requests since 0s" in out.stdout and "1 older request:" in out.stdout
    assert "Point your agent" not in out.stdout


def test_bad_price_and_allow_files_name_the_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    no_traceback(cli("prices", "--prices", str(bad)), 1, "bad.json", "not JSON")
    no_traceback(cli("prices", "--prices", str(tmp_path / "nope.json")), 1, "nope.json", "cannot read")
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2]")
    no_traceback(cli("prices", "--prices", str(lst)), 1, "list.json")
    no_traceback(cli("check", "--allow", str(tmp_path / "nope.json"), "--", "true"), 2, "nope.json", "allow-list")
    no_traceback(cli("check", "--allow", str(bad), "--", "true"), 2, "bad.json", "not JSON")


def test_import_gateway_tells_a_missing_path_from_no_path(tmp_path):
    no_traceback(cli("import", "gateway", "--path", "nope.jsonl", cwd=tmp_path), 1, "no file or folder at nope.jsonl")
    no_traceback(cli("import", "gateway", cwd=tmp_path), 1, "needs --path")


def test_explain_refuses_what_is_not_a_request(tmp_path):
    junk = tmp_path / "junk.json"
    junk.write_text('{"a": 1}')
    out = cli("explain", str(junk), str(junk))
    no_traceback(out, 1, "not a request", "junk.json")
    assert "NONE" not in out.stdout                 # it used to answer "prefix reused"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    no_traceback(cli("explain", str(empty)), 1, "holds no request")
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
    no_traceback(cli("explain", str(empty), str(real)), 1, "empty.jsonl holds no request")
    arr = tmp_path / "arr.json"
    arr.write_text("[]")
    no_traceback(cli("explain", str(arr)), 1, "holds no request")
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text('{"model": "m", "messages": [{"role": "user", "content": "hi"}]}\n{"b": 2}\n')
    no_traceback(cli("explain", str(mixed)), 1, "record 2", "not a request")
    # every request the block extractor reads still explains: a prompt can be only a system prompt and tools
    tools = [{"name": "t", "input_schema": {"type": "object"}}]
    for body in ({"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                 {"model": "m", "input": "hi"},
                 {"model": "m", "system": "You are a bot.", "tools": tools},
                 {"model": "m", "instructions": "You are a bot.", "tools": tools},
                 {"request": {"model": "m", "system": "s", "tools": tools}},
                 {"request": {"model": "m", "messages": []}, "usage": {"prompt_tokens": 5}}):
        ok = tmp_path / "ok.json"
        ok.write_text(json.dumps(body))
        assert cli("explain", str(ok)).returncode == 0, body
    # an API whose prompt is not read is refused, where two different requests compared as unchanged
    for a, b in (({"model": "g", "contents": [{"parts": [{"text": "one"}]}]},
                  {"model": "g", "contents": [{"parts": [{"text": "something else entirely"}]}]}),
                 ({"model": "c", "prompt": "one"}, {"model": "c", "prompt": "another"})):
        fa, fb = tmp_path / "a.json", tmp_path / "b.json"
        fa.write_text(json.dumps(a))
        fb.write_text(json.dumps(b))
        out = cli("explain", str(fa), str(fb))
        no_traceback(out, 1, "not a request Prefixwise can read")
        assert "NONE" not in out.stdout


def test_check_bodies_with_no_request_is_nothing_checked_not_an_error(tmp_path):
    empty, real = tmp_path / "empty.jsonl", tmp_path / "real.jsonl"
    empty.write_text("")
    real.write_text(json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}) + "\n")
    out = cli("check", "--bodies", str(empty))
    assert out.returncode == 1 and "the files hold no request" in out.stdout and "NOTHING CHECKED" in out.stdout
    assert cli("check", "--bodies", str(empty), "--allow-empty").returncode == 0      # as before this change
    assert cli("check", "--bodies", str(empty), str(real)).returncode == 0            # one empty file among others


def test_check_tells_refused_requests_from_none(tmp_path):
    """A command whose requests reach the mock and are refused (a path it does not serve) is not told
    that nothing reached it."""
    out = cli("check", "--", sys.executable, "-c", POST, "/embeddings")
    assert out.returncode == 1 and "NOTHING CHECKED" in out.stdout
    assert "1 request reached Prefixwise and none got a verdict" in out.stdout
    assert "no request reached the mock" not in out.stdout


def test_the_action_can_allow_an_empty_run():
    action = (ROOT / "action.yml").read_text()
    assert "allow-empty:" in action and 'args+=(--allow-empty)' in action


def test_examples_are_priced():
    out = cli("explain", str(ROOT / "examples" / "session.jsonl"), "--json")
    assert out.returncode == 0
    from prefixwise.pricing import PriceTable
    table = PriceTable.load()
    for name in ("turn1.json", "turn2.json"):
        model = json.loads((ROOT / "examples" / name).read_text())["model"]
        assert table.resolve(model)[1] is not None, model
    for line in (ROOT / "examples" / "session.jsonl").read_text().splitlines():
        assert table.resolve(json.loads(line)["request"]["model"])[1] is not None


# -- serve: how to point a client at it --------------------------------------------------

def test_api_index_lists_endpoints_that_answer(tmp_path):
    from prefixwise import api
    from prefixwise.proxy import PrefixwiseProxy

    proxy = PrefixwiseProxy(ledger_path=str(tmp_path / "l.db"), mock=True)
    try:
        status, body = api.get(proxy, "/api/v1", "")
        assert status == 200 and body["endpoints"]
        for url in body["endpoints"]:
            if "<id>" in url or "/sessions/" in url:
                continue                              # need an id or a recorded session
            path, _, query = url.partition("?")
            assert api.get(proxy, path, query)[0] == 200, url
    finally:
        proxy.close()


def test_client_url_is_one_a_client_can_use():
    from prefixwise.proxy import _client_url

    assert _client_url("0.0.0.0", 8080) == "http://127.0.0.1:8080"
    assert _client_url("localhost", 8080) == "http://127.0.0.1:8080"
    assert _client_url("127.0.0.1", 9000) == "http://127.0.0.1:9000"
    assert _client_url("10.0.0.5", 8080) == "http://10.0.0.5:8080"


def test_serve_prints_what_a_client_sets(tmp_path):
    proc = subprocess.Popen([sys.executable, "-m", "prefixwise", "serve", "--mock", "--port", "0", "--ledger",
                             str(tmp_path / "l.db")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    try:
        seen = ""
        for line in proc.stdout:
            seen += line
            if "clients:" in line:
                break
        assert "OPENAI_BASE_URL=http://127.0.0.1:" in seen and "/v1  ANTHROPIC_BASE_URL=http://127.0.0.1:" in seen
        assert "GET /api/v1" in seen
    finally:
        proc.terminate()
        proc.wait(10)


def test_a_failed_upstream_is_explained_in_the_report(tmp_path):
    out = cli("run", "--ledger", str(tmp_path / "l.db"), "--", sys.executable, "-c", POST, "/chat/completions",
              env={**ENV, "OPENAI_BASE_URL": "http://127.0.0.1:9/v1", "ANTHROPIC_BASE_URL": "http://127.0.0.1:9"})
    assert "prefixwise_proxy_error" in out.stderr
    assert "no usable answer from the upstream" in out.stderr


def test_an_import_says_where_its_ledger_is(tmp_path):
    log = tmp_path / "requests.jsonl"
    log.write_text(json.dumps({"start_time": 1795000000, "end_time": 1795000002,
                               "request": {"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "hi"}]},
                               "response": {"id": "r1", "usage": {"prompt_tokens": 9, "completion_tokens": 1}}}) + "\n")
    out = cli("import", "gateway", "--path", str(log), cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "from a gateway log" in out.stdout and "keeps its own ledger" in out.stdout
    assert "prefixwise report --ledger prefixwise-gateway.db --detail" in out.stdout
    assert cli("report", cwd=tmp_path).returncode == 1          # plain `report` reads another file, as the import said
