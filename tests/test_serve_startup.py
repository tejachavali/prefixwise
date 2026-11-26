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
import threading
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
    watchdog = threading.Timer(20, proc.kill)  # a server that prints no banner fails the test, never hangs it
    watchdog.start()
    seen = []
    try:
        for line in proc.stdout:
            seen.append(line)
            found = re.search(r"Prefixwise proxy on (http://127\.0\.0\.1:\d+)", line)
            if found:
                return proc, found.group(1), seen
    finally:
        watchdog.cancel()
    proc.kill()
    raise AssertionError("no banner from " + " ".join(args) + ":\n" + "".join(seen))


def _post(base):
    req = urllib.request.Request(base + "/v1/chat/completions", data=BODY, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


SERVE = [sys.executable, "-m", "prefixwise", "serve", "--port", "0"]
ENTRY_POINT = [sys.executable, "-c", "from prefixwise.proxy import serve; serve(port=0)"]  # prefixwise-proxy


def _listener():
    """A local port that records whether anything connected to it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    sock.settimeout(0.5)
    return sock, f"http://127.0.0.1:{sock.getsockname()[1]}"


def _was_contacted(sock) -> bool:
    try:
        sock.accept()[0].close()
        return True
    except OSError:
        return False


@pytest.mark.parametrize("args, env", [
    (SERVE, {}),                                                            # the Docker image's default
    (ENTRY_POINT, {}),
])
def test_with_no_upstream_serve_is_a_mock_that_answers(tmp_path, args, env):
    proc, base, _ = _started(args, tmp_path, **env)
    try:
        status, payload = _post(base)
        assert status == 200, payload
        assert payload["usage"]["prompt_tokens"] > 0
        health = json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())
        assert health["mock"] is True
    finally:
        proc.kill()
        proc.wait()


def test_mock_by_env_never_contacts_the_upstream_and_says_so(tmp_path):
    """PREFIXWISE_MOCK_UPSTREAM=1 with an upstream set: no request, and no engine probe (which would
    carry the API key), reaches the upstream. Before, serve forwarded to it."""
    sock, url = _listener()
    proc, base, banner = _started(SERVE + ["--upstream", url], tmp_path, PREFIXWISE_MOCK_UPSTREAM="1",
                                  PREFIXWISE_API_KEY="sk-test-not-a-key")
    try:
        assert _post(base)[0] == 200
        for _ in range(5):  # the banner's remaining lines
            banner.append(proc.stdout.readline())
        assert not _was_contacted(sock)
        text = "".join(banner)
        assert "MOCK upstream" in text and f"{url} is not used" in text
        assert "sk-test-not-a-key" not in text
    finally:
        proc.kill()
        proc.wait()
        sock.close()


def test_a_key_with_no_upstream_is_a_mock_with_a_warning(tmp_path):
    proc, base, banner = _started(SERVE, tmp_path, PREFIXWISE_API_KEY="sk-test-not-a-key")
    try:
        for _ in range(3):
            banner.append(proc.stdout.readline())
        assert "PREFIXWISE_API_KEY is set but no upstream is" in "".join(banner)
        assert _post(base)[0] == 200
    finally:
        proc.kill()
        proc.wait()


def _fails(args, tmp_path, **extra_env):
    out = subprocess.run(args, cwd=tmp_path, env=_env(tmp_path, **extra_env), text=True, capture_output=True,
                         timeout=30)
    assert out.returncode != 0
    assert "Traceback" not in out.stderr, out.stderr
    return out.stderr


@pytest.mark.parametrize("upstream", ["notaurl", "ftp://127.0.0.1:9", "http://", "http://localhost:80a0",
                                      "https://user:pw@gw.example/v1", "http://[::1"])
def test_a_bad_upstream_is_refused_at_startup(tmp_path, upstream):
    err = _fails(SERVE + ["--upstream", upstream], tmp_path)
    assert "--upstream must" in err, err
    assert "pw" not in err                              # a password in the URL is not echoed


def test_a_bad_upstream_is_refused_before_anything_is_sent_to_it(tmp_path):
    sock, url = _listener()
    try:
        err = _fails(SERVE + ["--upstream", url.replace("http://", "ftp://")], tmp_path)
        assert "--upstream must be an http:// or https:// URL" in err
        assert not _was_contacted(sock)
    finally:
        sock.close()


def test_an_upstream_with_stray_whitespace_is_trimmed(tmp_path):
    sock, url = _listener()
    proc, base, banner = _started(SERVE + ["--engine", "none"], tmp_path, PREFIXWISE_UPSTREAM_URL=url + " \r")
    try:
        assert f"upstream: {url}\n" in "".join(banner + [proc.stdout.readline()])
    finally:
        proc.kill()
        proc.wait()
        sock.close()


def test_run_names_the_variable_that_holds_a_bad_base_url(tmp_path):
    cmd = [sys.executable, "-m", "prefixwise", "run", "--", sys.executable, "-c", "print('agent ran')"]
    err = _fails(cmd, tmp_path, OPENAI_BASE_URL="localhost:11434/v1")
    assert "OPENAI_BASE_URL must be an http:// or https:// URL" in err and "Unset OPENAI_BASE_URL" in err
    # A general upstream that `run` never uses does not stop it.
    out = subprocess.run(cmd, cwd=tmp_path, env=_env(tmp_path, PREFIXWISE_UPSTREAM_URL="notaurl"), text=True,
                         capture_output=True, timeout=60)
    assert "agent ran" in out.stdout, out.stderr


def test_a_port_that_cannot_be_bound_says_what_to_do(tmp_path):
    for cmd in (["serve", "--mock"], ["collect"]):
        err = _fails([sys.executable, "-m", "prefixwise", *cmd, "--port", "99999"], tmp_path)
        assert "--port must be between 0 and 65535" in err
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = str(busy.getsockname()[1])
        for cmd in (["serve", "--mock", "--port", port], ["collect", "--port", port],
                    ["run", "--mock", "--port", port, "--", "true"], ["check", "--port", port, "--", "true"]):
            err = _fails([sys.executable, "-m", "prefixwise", *cmd], tmp_path)
            assert f"port {port} on 127.0.0.1 is already in use" in err and "--port" in err
