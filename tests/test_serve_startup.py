"""`prefixwise serve` as a user starts it: the real command, a real request.

With no upstream the proxy is a mock (its help, the Dockerfile and the guide all say so). For a long
time it was not: serve() turned "not asked for" into "no", and every request got a 500 while /health
stayed green. Nothing started the command end to end, so nothing caught it."""
import json
import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

BODY = json.dumps({"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "hi"}]}).encode()


def _env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREFIXWISE_")}
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env.update(PYTHONPATH=src, PYTHONUNBUFFERED="1", HOME=str(tmp_path), **extra)
    return env


def _started(args, tmp_path, **extra_env):
    """Start the command; return (process, base URL) once its banner names the port."""
    proc = subprocess.Popen(args, cwd=tmp_path, env=_env(tmp_path, **extra_env), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for _ in range(40):
        line = proc.stdout.readline()
        if not line:
            break
        found = re.search(r"Prefixwise proxy on (http://127\.0\.0\.1:\d+)", line)
        if found:
            return proc, found.group(1)
    proc.kill()
    raise AssertionError("no banner from " + " ".join(args))


def _post(base):
    req = urllib.request.Request(base + "/v1/chat/completions", data=BODY, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


SERVE = [sys.executable, "-m", "prefixwise", "serve", "--port", "0"]
ENTRY_POINT = [sys.executable, "-c", "from prefixwise.proxy import serve; serve(port=0)"]  # prefixwise-proxy


@pytest.mark.parametrize("args, env", [
    (SERVE, {}),                                                            # the Docker image's default
    (ENTRY_POINT, {}),
    (SERVE + ["--upstream", "http://127.0.0.1:9"], {"PREFIXWISE_MOCK_UPSTREAM": "1"}),
])
def test_with_no_upstream_serve_is_a_mock_that_answers(tmp_path, args, env):
    proc, base = _started(args, tmp_path, **env)
    try:
        status, payload = _post(base)
        assert status == 200, payload
        assert payload["usage"]["prompt_tokens"] > 0
        health = json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())
        assert health["mock"] is True
    finally:
        proc.kill()
        proc.wait()


def _fails(args, tmp_path):
    out = subprocess.run(args, cwd=tmp_path, env=_env(tmp_path), text=True, capture_output=True, timeout=30)
    assert out.returncode != 0
    assert "Traceback" not in out.stderr, out.stderr
    return out.stderr


@pytest.mark.parametrize("upstream", ["notaurl", "ftp://example.com", "http://"])
def test_a_bad_upstream_is_refused_at_startup(tmp_path, upstream):
    err = _fails(SERVE + ["--upstream", upstream], tmp_path)
    assert "upstream must be an http:// or https:// URL" in err and upstream in err


def test_a_port_that_cannot_be_bound_says_what_to_do(tmp_path):
    err = _fails([sys.executable, "-m", "prefixwise", "serve", "--mock", "--port", "99999"], tmp_path)
    assert "--port must be between 0 and 65535" in err
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = str(busy.getsockname()[1])
        for cmd in (["serve", "--mock", "--port", port], ["run", "--mock", "--port", port, "--", "true"]):
            err = _fails([sys.executable, "-m", "prefixwise", *cmd], tmp_path)
            assert f"port {port} on 127.0.0.1 is already in use" in err and "--port" in err
