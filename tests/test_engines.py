"""SGLang / LMCache support in the open-engine experiments."""

import subprocess
import sys
from pathlib import Path

from prefixwise.proxy import PrefixwiseProxy

ROOT = Path(__file__).resolve().parent.parent


def _forwarded_body(engine) -> dict:
    proxy = PrefixwiseProxy(upstream_url="http://unused", api_key="k", engine=engine)
    seen = {}

    def capture(provider, body, request_id, session_id, client_headers, api="chat"):
        seen.update(body)
        return 200, {"id": "x", "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}}

    proxy._forward = capture
    proxy.handle("vllm", {"model": "m", "session_id": "s1", "messages": [{"role": "user", "content": "hi"}]})
    return seen


def test_session_id_reaches_vllm_but_not_sglang():
    # vLLM stamps a body session_id onto KV events; SGLang gets it only with --kv-events
    # (test_sglang_events.py).
    assert _forwarded_body("vllm")["session_id"] == "s1"
    assert "session_id" not in _forwarded_body("sglang")
    assert "session_id" not in _forwarded_body(None)  # hosted APIs reject unknown fields


def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True, text=True)


def test_e1_open_mock_on_each_engine(tmp_path):
    for engine in ("sglang", "vllm-lmcache"):
        r = run("experiments/e1_open/e1_open.py", "--mock", "--engine", engine, "--output-dir", str(tmp_path))
        assert r.returncode == 0 and "Pass: YES" in r.stdout, r.stdout + r.stderr


def test_e_evict_mock_offload_recovers_gpu_only_is_evicted(tmp_path):
    gpu = run("experiments/e_evict/e_evict.py", "--mock", "--engine", "vllm", "--repeats", "2",
              "--output-dir", str(tmp_path))
    assert gpu.returncode == 0, gpu.stdout + gpu.stderr
    assert "T3 0/" in gpu.stdout and "`evicted`" in gpu.stdout  # one engine, prefix gone: inferred eviction
    off = run("experiments/e_evict/e_evict.py", "--mock", "--engine", "vllm-lmcache", "--repeats", "2",
              "--output-dir", str(tmp_path))
    assert off.returncode == 0 and "`none` x2 (2/2 correct)" in off.stdout, off.stdout
