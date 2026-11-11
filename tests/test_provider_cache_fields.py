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
    for m in ("gpt-5", "gpt-4o-mini", "o3", "o4-mini", "chatgpt-4o-latest", "codex-mini-latest", "openai/gpt-5",
              "openrouter/openai/gpt-4o", "ft:gpt-4o-mini:org::abc123", "computer-use-preview"):
        assert openai_model(m), m
    for m in ("deepseek-chat", "deepseek-flash", "hy4", "hunyuan-turbos", "Qwen/Qwen3-4B", "kimi-k2", "m", "",
              "gpt-oss-120b", "openai/gpt-oss-20b"):
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

    def s(model, prompt, cached, turn=1):
        body = {"model": model, "messages": msgs if turn == 1 else
                msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "and then?"}]}
        return RequestSnapshot(model=model, provider="openai", blocks=fp.fingerprint_blocks(extract_blocks(body, "openai")),
                               prompt_tokens=prompt, cached_tokens=cached, gap_seconds=5.0)

    def turn2(model, prompt):  # an append-only second turn, 5 s later, that read nothing
        return classify(s(model, prompt, 0), s(model, prompt + 20, 0, turn=2))
    assert turn2("gpt-5", 500).cause is MissCause.BELOW_MIN_PREFIX
    short = turn2("deepseek-flash", 500)
    # its minimum is not known here: neither below_min_prefix nor a routing miss
    assert short.cause is MissCause.UNKNOWN and "minimum cacheable length is not known" in short.evidence
    # an identical repeat seconds later: not a retry (waste) either. Live, DeepSeek cached nothing of a
    # 239-token prompt sent three times, and the third was called retry_replay with 239 avoidable tokens
    same = classify(s("deepseek-flash", 239, 0, turn=2), s("deepseek-flash", 239, 0, turn=2))
    assert same.cause is MissCause.UNKNOWN and not same.legit
    assert classify(s("gpt-5", 239, 0, turn=2), s("gpt-5", 239, 0, turn=2)).cause is MissCause.BELOW_MIN_PREFIX
    assert classify(s("deepseek-flash", 3000, 0, turn=2), s("deepseek-flash", 3000, 0, turn=2)).cause \
        is MissCause.RETRY_REPLAY  # a long identical retry is still one
    # a long prompt that read nothing within the shortest TTL is still a routing miss on any provider
    assert turn2("deepseek-flash", 3000).cause is MissCause.ROUTING_MISS
    assert turn2("gpt-5", 3000).cause is MissCause.ROUTING_MISS


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


def test_an_upstream_that_ends_in_v1_is_not_given_a_second_v1():
    """Live (Muse, DeepSeek): `serve --upstream https://api.deepseek.com/v1` sent /v1/v1/chat/completions
    and every request came back 404. Most OpenAI-compatible APIs document a base URL ending in /v1."""
    for base in ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/",
                 "https://tokenhub-intl.tencentcloudmaas.com/v1"):
        p = PrefixwiseProxy(upstream_url=base, mock=False)
        host = base.rstrip("/").removesuffix("/v1")
        assert p.target("openai", "/v1/chat/completions") == host + "/v1/chat/completions"
        assert p.target("openai", "/v1/models") == host + "/v1/models"
    p = PrefixwiseProxy(upstream_url="https://api.deepseek.com", mock=False)
    assert p.target("openai", "/v1/chat/completions") == "https://api.deepseek.com/v1/chat/completions"
    p = PrefixwiseProxy(upstream_url="https://api.anthropic.com", mock=False)
    assert p.target("anthropic", "/v1/messages") == "https://api.anthropic.com/v1/messages"
