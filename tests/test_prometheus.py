"""GET /metrics: Prometheus text format, parsed with the official client's parser."""

import json
import threading
import urllib.error
import urllib.request

import pytest

parser = pytest.importorskip("prometheus_client.parser")

from prefixwise.ledger import LedgerEntry
from prefixwise.prom import Metrics
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler


def families(text: str) -> dict:
    return {f.name: f for f in parser.text_string_to_metric_families(text)}


def value(fams, name, **labels):
    for fam in fams.values():
        for s in fam.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return None


def entry(**kw):
    base = dict(session_id="s", agent_id="a", provider="anthropic", model="claude-haiku-5-5", cause="none",
                prompt_tokens=10_000, cached_tokens=9_000, avoidable_tokens=0, avoidable_cost=None)
    return LedgerEntry(**{**base, **kw})


def test_counters_histograms_and_failures_parse():
    m = Metrics()
    m.observe(entry(cost=0.01, latency_ms=1200, overhead_ms=3))
    m.observe(entry(cause="volatile_system", cached_tokens=0, avoidable_tokens=9_000, avoidable_cost=0.009,
                    cost=0.02, cache_write_tokens=10_000, latency_ms=900, overhead_ms=4))
    m.observe(entry(cause=None, prompt_tokens=0, cached_tokens=0, status=529, error="overloaded_error"))
    m.observe(entry(model='we"ird\\model\nx', cost=None))  # label escaping
    f = families(m.render())
    assert value(f, "prefixwise_requests_total", cause="none", model="claude-haiku-5-5") == 1
    assert value(f, "prefixwise_avoidable_tokens_total", cause="volatile_system") == 9_000
    assert value(f, "prefixwise_avoidable_cost_usd_total", cause="volatile_system") == pytest.approx(0.009)
    assert value(f, "prefixwise_cost_usd_total", cause="none", model="claude-haiku-5-5") == pytest.approx(0.01)
    assert value(f, "prefixwise_requests_failed_total", error="overloaded_error") == 1
    assert value(f, "prefixwise_requests_total", model='we"ird\\model\nx') == 1
    assert f["prefixwise_upstream_latency_seconds"].type == "histogram"
    assert value(f, "prefixwise_upstream_latency_seconds_bucket", provider="anthropic", le="1.0") == 1
    assert value(f, "prefixwise_upstream_latency_seconds_count", provider="anthropic") == 2
    assert value(f, "prefixwise_overhead_seconds_bucket", le="0.005") == 2
    # Hit rate is PromQL over the counters: cached / prompt.
    assert (value(f, "prefixwise_cached_tokens_total", cause="none", model="claude-haiku-5-5")
            + value(f, "prefixwise_cached_tokens_total", cause="volatile_system")) == 9_000
    assert value(f, "prefixwise_info", version=__import__("prefixwise").__version__, engine="none") == 1


def test_kv_event_health_gauges():
    f = families(Metrics().render("vllm", {"connected": True, "batches": 7, "dropped_messages": 2}))
    assert value(f, "prefixwise_kv_events_connected") == 1
    assert value(f, "prefixwise_kv_events_dropped_messages_total") == 2
    assert value(f, "prefixwise_info", engine="vllm") == 1


@pytest.fixture
def server():
    started = []

    def start(**kw):
        proxy = PrefixwiseProxy(mock=True)
        proxy.auth_token = kw.get("token")
        srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}", proxy
    yield start
    for s in started:
        s.shutdown()


def get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode()


def test_metrics_endpoint_counts_proxied_traffic(server):
    url, _ = server()
    body = json.dumps({"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]}).encode()
    for _ in range(2):
        req = urllib.request.Request(url + "/v1/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "X-Prefixwise-Session-Id": "s"})
        urllib.request.urlopen(req, timeout=5).read()
    status, ctype, text = get(url + "/metrics")
    assert status == 200 and ctype.startswith("text/plain; version=0.0.4")
    f = families(text)
    assert value(f, "prefixwise_requests_total", cause="first_write") == 1
    assert value(f, "prefixwise_requests_total", cause="none") == 1


def test_metrics_endpoint_honours_the_token_including_bearer(server):
    url, _ = server(token="t0ken")
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(url + "/metrics")
    assert exc.value.code == 401
    assert get(url + "/metrics", {"X-Prefixwise-Token": "t0ken"})[0] == 200
    assert get(url + "/metrics", {"Authorization": "Bearer t0ken"})[0] == 200  # Prometheus' own way
    with pytest.raises(urllib.error.HTTPError):
        get(url + "/metrics", {"Authorization": "Bearer wrong"})


def test_dashboard_and_alert_rules_use_only_exported_metrics():
    """A renamed metric must not silently empty a panel or disarm an alert."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    m = Metrics()
    m.observe(entry(cost=0.0, avoidable_cost=0.0, cache_write_tokens=0, latency_ms=1, overhead_ms=1))
    m.observe(entry(cause=None, status=500, error="x"))
    exported = {line.split("{")[0].split(" ")[0] for line in m.render("vllm", {"connected": True}).splitlines()
                if line and not line.startswith("#")}
    dashboard = (root / "deploy/grafana/prefixwise.json").read_text()
    rules = (root / "deploy/prometheus/prefixwise-rules.yml").read_text()
    json.loads(dashboard)  # valid JSON
    used = set(re.findall(r"\bprefixwise_[a-z_]+(?:_total|_bucket|_connected|_info)\b", dashboard + rules))
    assert used and used <= exported, sorted(used - exported)
