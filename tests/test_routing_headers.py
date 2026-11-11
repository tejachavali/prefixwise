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


def _sent(p, headers):
    req = p._upstream_request("openai", {"model": "m", "messages": []}, "rid", "conv-7", headers)
    return {k.lower(): v for k, v in req.header_items()}


def test_routing_keys_pass_through_and_nothing_else_new(monkeypatch):
    monkeypatch.delenv("PREFIXWISE_PASS_HEADERS", raising=False)
    monkeypatch.delenv("PREFIXWISE_API_KEY", raising=False)
    sent = _sent(PrefixwiseProxy(upstream_url="http://router:30000", mock=False), CLIENT)
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


def test_a_replaced_authorization_in_front_of_an_engine_is_warned_once(monkeypatch, capsys):
    p = PrefixwiseProxy(upstream_url="http://router:30000", api_key="server-key", mock=False)
    p.engine = "sglang"
    for _ in range(2):
        sent = _sent(p, {"authorization": "Bearer client-a"})
        assert sent["authorization"] == "Bearer server-key"
    assert capsys.readouterr().err.count("replaces each client's authorization") == 1


def test_no_warning_for_a_hosted_api_or_a_client_with_a_routing_key(monkeypatch, capsys):
    p = PrefixwiseProxy(upstream_url="https://api.openai.com", api_key="server-key", mock=False)
    _sent(p, {"authorization": "Bearer client-a"})  # no engine: a hosted API
    p2 = PrefixwiseProxy(upstream_url="http://router:30000", api_key="server-key", mock=False)
    p2.engine = "sglang"
    _sent(p2, {"authorization": "Bearer client-a", "x-smg-routing-key": "conv-1"})
    assert "replaces each client's authorization" not in capsys.readouterr().err
