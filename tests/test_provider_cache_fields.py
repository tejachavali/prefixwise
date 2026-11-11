"""OpenAI-compatible APIs that are not OpenAI: their cache counts are read where they report them, and
OpenAI's own rules (its 1,024-token minimum) are not applied to them.

DeepSeek's API reports cache reads as usage.prompt_cache_hit_tokens (and prompt_cache_miss_tokens),
the field LiteLLM maps to cached tokens (litellm/types/utils.py, DEEPSEEK MAPPING). Tencent's TokenHub
goes through LiteLLM's plain OpenAI adapter, so its usage is read as OpenAI's."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prefixwise.classifier import MissCause, RequestSnapshot, _min_prefix, classify, openai_model
from prefixwise.fingerprint import Fingerprinter
from prefixwise.pricing import breakdown, cache_reported, normalize_usage
from prefixwise.proxy import PrefixwiseProxy, extract_blocks

DEEPSEEK_USAGE = {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240,
                  "prompt_cache_hit_tokens": 1152, "prompt_cache_miss_tokens": 48}


def test_deepseek_cache_hits_are_read():
    assert normalize_usage("openai", DEEPSEEK_USAGE) == (1200, 1152)
    assert cache_reported("openai", DEEPSEEK_USAGE)
    b = breakdown("openai", DEEPSEEK_USAGE)
    assert (b.cache_read, b.uncached_input, b.output) == (1152, 48, 40)
    assert cache_reported("openai", {**DEEPSEEK_USAGE, "prompt_cache_hit_tokens": 0})  # a reported 0 is a miss


def test_prompt_tokens_missing_is_hit_plus_miss_and_standard_fields_win():
    usage = {"prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 36, "completion_tokens": 1}
    assert normalize_usage("openai", usage) == (100, 64)
    both = {**DEEPSEEK_USAGE, "prompt_tokens_details": {"cached_tokens": 1100}}
    assert normalize_usage("openai", both) == (1200, 1100)


def test_ollama_style_usage_is_still_not_reported():
    assert not cache_reported("openai", {"prompt_tokens": 500, "completion_tokens": 3})
    assert not cache_reported("openai", {"prompt_tokens": 500, "prompt_cache_hit_tokens": True})  # not a count


def test_openai_models_are_recognised():
    for m in ("gpt-5", "gpt-4o-mini", "o3", "o4-mini", "chatgpt-4o-latest", "codex-mini-latest", "openai/gpt-5"):
        assert openai_model(m), m
    for m in ("deepseek-chat", "deepseek-reasoner", "hy4", "hunyuan-turbos", "Qwen/Qwen3-4B", "kimi-k2", "m", ""):
        assert not openai_model(m), m


def snap(model, prompt, cached, engine=False):
    return RequestSnapshot(model=model, provider="openai", blocks=[], prompt_tokens=prompt, cached_tokens=cached,
                           engine=engine)


def test_openais_minimum_applies_to_openai_models_only():
    assert _min_prefix(snap("gpt-5", 500, 0)) == 1024
    assert _min_prefix(snap("deepseek-chat", 500, 0)) is None
    assert _min_prefix(snap("Qwen/Qwen3-4B", 500, 0, engine=True)) is None  # vLLM caches from one block
    assert _min_prefix(snap("gpt-oss-20b", 500, 0, engine=True)) is None  # an OpenAI-named model on an engine


def test_a_short_deepseek_prompt_that_missed_is_not_below_the_minimum():
    fp = Fingerprinter()
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]

    def s(model, prompt, cached):
        body = {"model": model, "messages": msgs}
        return RequestSnapshot(model=model, provider="openai", blocks=fp.fingerprint_blocks(extract_blocks(body, "openai")),
                               prompt_tokens=prompt, cached_tokens=cached, gap_seconds=5.0)
    assert classify(s("gpt-5", 500, 0), s("gpt-5", 520, 0)).cause is MissCause.BELOW_MIN_PREFIX
    assert classify(s("deepseek-chat", 500, 0), s("deepseek-chat", 520, 0)).cause is not MissCause.BELOW_MIN_PREFIX


class Upstream:
    def __init__(self, usages):
        self.usages = list(usages)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                reply = {"id": "x", "model": "deepseek-chat",
                         "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                         "usage": outer.usages.pop(0)}
                b = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def test_through_the_proxy_a_deepseek_hit_is_a_hit():
    first = {"prompt_tokens": 3000, "completion_tokens": 5, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 3000}
    second = {"prompt_tokens": 3040, "completion_tokens": 5, "prompt_cache_hit_tokens": 2944,
              "prompt_cache_miss_tokens": 96}
    up = Upstream([first, second])
    p = PrefixwiseProxy(upstream_url=up.url, mock=False)
    msgs = [{"role": "system", "content": "You are a careful assistant. " * 300}, {"role": "user", "content": "hi"}]
    try:
        p.handle("openai", {"model": "deepseek-chat", "messages": msgs}, {"x-prefixwise-session-id": "s"})
        msgs2 = msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
        _, entry = p.handle("openai", {"model": "deepseek-chat", "messages": msgs2}, {"x-prefixwise-session-id": "s"})
        assert entry.cached_tokens == 2944 and entry.cause == "none"  # before: 0 cached, an unknown miss
    finally:
        up.server.shutdown()
