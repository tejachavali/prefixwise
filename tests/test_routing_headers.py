"""Routing keys reach the router. Prefixwise forwarded a fixed list of headers, which left out the
keys SGLang Model Gateway, sgl-router, vLLM and SGLang route by (x-smg-routing-key,
x-smg-target-worker, x-sgl-routing-key, x-data-parallel-rank) and what the gateway hashes when there
is no key (x-forwarded-for, cookie). A session then landed on any worker, and with each worker's KV
events joined its misses were confirmed as routing_miss: Prefixwise caused the misses it blamed on
routing. Found reading the routers' source (sgl-model-gateway policies/consistent_hashing.rs,
sgl-router config, vLLM's serving.py)."""

from prefixwise.proxy import PrefixwiseProxy

CLIENT = {"x-smg-routing-key": "conv-7", "x-smg-target-worker": "w2", "x-sgl-routing-key": "conv-7",
          "x-data-parallel-rank": "1", "x-forwarded-for": "10.0.0.5", "cookie": "route=w2",
          "x-dynamo-session-id": "conv-7",
          "x-session-id": "conv-7", "x-my-router-session": "abc", "x-unrelated": "no"}


def _engine(url="http://router:30000"):
    p = PrefixwiseProxy(upstream_url=url, mock=False)
    p.engine = "sglang"
    return p


def _sent(p, headers):
    req = p._upstream_request("openai", {"model": "m", "messages": []}, "rid", "conv-7", headers)
    return {k.lower(): v for k, v in req.header_items()}


def test_routing_keys_pass_through_and_nothing_else_new(monkeypatch):
    monkeypatch.delenv("PREFIXWISE_PASS_HEADERS", raising=False)
    monkeypatch.delenv("PREFIXWISE_API_KEY", raising=False)
    sent = _sent(_engine(), CLIENT)
    for k in ("x-smg-routing-key", "x-smg-target-worker", "x-sgl-routing-key", "x-data-parallel-rank",
              "x-forwarded-for", "cookie", "x-session-id", "x-dynamo-session-id"):
        assert sent.get(k) == CLIENT[k], k
    assert "x-my-router-session" not in sent and "x-unrelated" not in sent


def test_a_custom_session_header_passes_when_named(monkeypatch):
    monkeypatch.setenv("PREFIXWISE_PASS_HEADERS", "X-My-Router-Session, ")
    sent = _sent(PrefixwiseProxy(upstream_url="http://router:30000", mock=False), CLIENT)
    assert sent.get("x-my-router-session") == "abc" and "x-unrelated" not in sent


def test_the_cli_flag_adds_to_the_env(monkeypatch):
    from argparse import Namespace

    from prefixwise.cli import _pass_headers
    monkeypatch.setenv("PREFIXWISE_PASS_HEADERS", "a")
    _pass_headers(Namespace(pass_header=["b", "c"]))
    import os
    assert os.environ["PREFIXWISE_PASS_HEADERS"] == "a,b,c"


def test_cookie_and_forwarded_for_go_to_an_engine_only(monkeypatch):
    """To a hosted API they could carry a login cookie or an internal address meant for the ingress
    in front of Prefixwise."""
    monkeypatch.delenv("PREFIXWISE_PASS_HEADERS", raising=False)
    hosted = _sent(PrefixwiseProxy(upstream_url="https://api.openai.com", mock=False), CLIENT)
    assert "cookie" not in hosted and "x-forwarded-for" not in hosted
    assert hosted.get("x-session-id") == "conv-7"  # routing keys still pass
    assert "cookie" not in PrefixwiseProxy(upstream_url="https://api.openai.com", mock=False)._passed(CLIENT, "openai")


def test_prefixwise_and_body_headers_are_never_passed_even_when_named(monkeypatch):
    monkeypatch.setenv("PREFIXWISE_PASS_HEADERS", "x-prefixwise-token,Content-Length,host,x-ok")
    p = PrefixwiseProxy(upstream_url="http://router:30000", mock=False)
    assert p.pass_headers == {"x-ok"}
