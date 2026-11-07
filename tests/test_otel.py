"""OpenTelemetry export: spans per OpenTelemetry's GenAI semantic conventions, plus prefixwise.*."""

import json
import threading
import time
import urllib.request

import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from prefixwise.otel import mode_from
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler
from prefixwise.sdk import Recorder

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def _provider():
    # The global provider can be set once per process: the application's, here an in-memory one.
    if isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
        trace.set_tracer_provider(provider)
    else:
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(EXPORTER))


@pytest.fixture
def spans():
    EXPORTER.clear()
    return EXPORTER.get_finished_spans


SYSTEM = "You are careful. " * 300
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"


def post(base, path, body, headers):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


@pytest.fixture
def proxy_base():
    proxy = PrefixwiseProxy(mock=True, otel="1")
    srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_proxy_spans_follow_genai_conventions_and_join_the_callers_trace(proxy_base, spans):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    h = {"traceparent": f"00-{TRACE_ID}-00f067aa0ba902b7-01", "x-prefixwise-session-id": "s1"}
    post(proxy_base, "/v1/chat/completions", {"model": "gpt-5.6", "messages": msgs}, h)
    post(proxy_base, "/v1/chat/completions", {"model": "gpt-5.6", "messages": msgs + [
        {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]}, h)
    first, second = spans()
    assert first.name == "chat gpt-5.6" and first.kind == SpanKind.CLIENT
    assert format(first.context.trace_id, "032x") == TRACE_ID  # in the caller's trace
    assert format(first.parent.span_id, "016x") == "00f067aa0ba902b7"
    a = dict(second.attributes)
    assert a["gen_ai.operation.name"] == "chat" and a["gen_ai.provider.name"] == "openai"
    assert a["openai.api.type"] == "chat_completions" and a["gen_ai.request.model"] == "gpt-5.6"
    assert a["gen_ai.usage.input_tokens"] == 2000 and a["gen_ai.usage.cache_read.input_tokens"] == 1000
    assert "gen_ai.usage.cache_write.input_tokens" not in a  # only when non-zero
    assert a["gen_ai.conversation.id"] == "s1" and a["gen_ai.response.id"]
    assert dict(first.attributes)["prefixwise.cache.cause"] == "first_write"
    assert a["prefixwise.cache.cause"] == "none" and a["prefixwise.cache.avoidable_tokens"] == 0
    assert "gen_ai.usage.cache_creation.input_tokens" not in a  # the removed name


def test_no_made_up_conversation_id(proxy_base, spans):
    post(proxy_base, "/v1/chat/completions", {"model": "gpt-5.6", "messages": [{"role": "user", "content": "x"}]}, {})
    a = dict(spans()[0].attributes)
    assert "gen_ai.conversation.id" not in a and a["prefixwise.session_id"] == "default"


def test_anthropic_input_tokens_include_cache_reads_and_writes(spans):
    rec = Recorder(":memory:")
    rec.proxy.spans = __import__("prefixwise.otel", fromlist=["Spans"]).Spans("genai")
    body = {"model": "claude-haiku-5-5", "max_tokens": 8,
            "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "hi"}]}
    usage = {"input_tokens": 100, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 500,
             "output_tokens": 7}
    before = time.time_ns()
    rec.record("anthropic", body, {"id": "msg_1", "model": "claude-haiku-5-5", "usage": usage},
               session_id="a1", latency_ms=1500)
    span = spans()[0]
    a = dict(span.attributes)
    assert a["gen_ai.provider.name"] == "anthropic" and "openai.api.type" not in a
    assert a["gen_ai.usage.input_tokens"] == 1600  # 100 + 1000 + 500, as the spec computes it for Anthropic
    assert a["gen_ai.usage.cache_read.input_tokens"] == 1000
    assert a["gen_ai.usage.cache_write.input_tokens"] == 500 and a["gen_ai.usage.output_tokens"] == 7
    assert abs((before - span.start_time) / 1e6 - 1500) < 200  # starts when the call was sent


def test_in_process_spans_nest_under_the_current_span(spans):
    from prefixwise.otel import Spans
    rec = Recorder(":memory:")
    rec.proxy.spans = Spans("genai")
    with trace.get_tracer("app").start_as_current_span("handle-request") as outer:
        rec.record("openai", {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]},
                   {"id": "c1", "model": "gpt-5.6", "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
    ours = [s for s in spans() if s.name == "chat gpt-5.6"][0]
    assert ours.parent.span_id == outer.get_span_context().span_id


def test_cause_mode_adds_no_usage_so_nothing_is_counted_twice(spans):
    from prefixwise.otel import Spans
    rec = Recorder(":memory:")
    rec.proxy.spans = Spans("cause")
    rec.record("openai", {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]},
               {"id": "c1", "model": "gpt-5.6", "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
    span = spans()[0]
    assert span.name == "prefixwise.cache" and span.kind == SpanKind.INTERNAL
    assert not [k for k in span.attributes if k.startswith("gen_ai.")]
    assert span.attributes["prefixwise.cache.cause"] == "first_write"


def test_failed_requests_are_error_spans(spans):
    from prefixwise.otel import Spans
    proxy = PrefixwiseProxy(mock=True)
    proxy.spans = Spans("genai")
    ctx = proxy.prepare("openai", {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]}, {})
    proxy.record_failure(ctx, 429, {"error": {"type": "rate_limit_error"}})
    span = spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "rate_limit_error"
    assert "gen_ai.usage.input_tokens" not in span.attributes


def test_export_can_never_break_a_request(spans, capsys):
    from prefixwise.otel import Spans
    proxy = PrefixwiseProxy(mock=True)
    proxy.spans = Spans("genai")
    proxy.spans.tracer = None  # every export now fails
    for _ in range(2):
        ctx = proxy.prepare("openai", {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]}, {})
        result, entry = proxy.finalize(ctx, 200, {"id": "c", "model": "gpt-5.6",
                                                  "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
        assert entry is not None and result["cause"]
    assert capsys.readouterr().err.count("OpenTelemetry export failed") == 1


def test_off_by_default_and_settings(monkeypatch):
    monkeypatch.delenv("PREFIXWISE_OTEL", raising=False)
    assert PrefixwiseProxy(mock=True).spans is None
    monkeypatch.setenv("PREFIXWISE_OTEL", "cause")
    assert PrefixwiseProxy(mock=True).spans.mode == "cause"
    assert [mode_from(v) for v in (None, "0", "off", "1", "true", "genai", "cause", True)] == [
        None, None, None, "genai", "genai", "genai", "cause", "genai"]
    with pytest.raises(ValueError):
        mode_from("yes please")


# --- Review findings: each is a scenario that must hold. ---

import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def run_py(code, env=None, timeout=60):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "PYTHONPATH": SRC, **(env or {})})


def test_in_an_app_prefixwise_never_takes_over_the_tracer_provider(tmp_path):
    """Apps wrap clients at import time and configure OpenTelemetry later: their provider must win."""
    code = """
import prefixwise
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
rec = prefixwise.Recorder()                       # before the app configures OpenTelemetry
exp = InMemorySpanExporter(); app = TracerProvider(); app.add_span_processor(SimpleSpanProcessor(exp))
trace.set_tracer_provider(app)
print(trace.get_tracer_provider() is app)
rec.record("openai", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
           {"id": "c", "model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
print([s.name for s in exp.get_finished_spans()])
"""
    out = run_py(code, {"PREFIXWISE_OTEL": "1", "PREFIXWISE_LEDGER_PATH": str(tmp_path / "l.db")})
    assert out.stdout.split() == ["True", "['chat", "m']"], out.stderr


def test_serve_exports_otlp_and_honours_the_standard_variables(tmp_path):
    """`prefixwise serve --otel` end to end, against a stand-in OTLP/HTTP collector."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    code = """
import http.server, json, os, socket, subprocess, sys, threading, time, urllib.request
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
got = []
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        r = ExportTraceServiceRequest(); r.ParseFromString(self.rfile.read(int(self.headers["content-length"])))
        got.append(r); self.send_response(200); self.end_headers()
    def log_message(self, *a): pass
col = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=col.serve_forever, daemon=True).start()
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
env = {**os.environ, "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{col.server_address[1]}",
       "OTEL_RESOURCE_ATTRIBUTES": "service.name=my-gateway", "OTEL_BSP_SCHEDULE_DELAY": "100"}
p = subprocess.Popen([sys.executable, "-m", "prefixwise", "serve", "--mock", "--otel", "--port", str(port)],
                     env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
for _ in range(100):
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health"); break
    except Exception: time.sleep(0.1)
body = json.dumps({"model": "claude-haiku-5-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages", data=body.encode(),
    headers={"content-type": "application/json", "x-prefixwise-session-id": "e2e"}, method="POST")).read()
for _ in range(50):
    if got: break
    time.sleep(0.1)
p.terminate(); out = p.communicate(timeout=10)[0]
sp = got[0].resource_spans[0]
print({a.key: a.value.string_value for a in sp.resource.attributes}["service.name"])
span = sp.scope_spans[0].spans[0]
attrs = {a.key: a.value for a in span.attributes}
print(span.name, span.kind, attrs["prefixwise.cache.cause"].string_value, attrs["gen_ai.conversation.id"].string_value)
print("otel: " in out)
"""
    out = run_py(code, {"PREFIXWISE_LEDGER_PATH": str(tmp_path / "l.db")}, timeout=90)
    assert out.stdout.split() == ["my-gateway", "chat", "claude-haiku-5-5", "3", "first_write", "e2e", "True"], \
        out.stderr[-2000:]


def test_a_bad_setting_never_breaks_an_app(monkeypatch, capsys):
    monkeypatch.setenv("PREFIXWISE_OTEL", "otlp")
    rec = Recorder(":memory:")
    assert rec.proxy.spans is None
    assert "unknown PREFIXWISE_OTEL value" in capsys.readouterr().err


def test_a_call_prefixwise_could_not_classify_is_not_an_error_span(spans):
    from prefixwise.otel import Spans
    proxy = PrefixwiseProxy(mock=True)
    proxy.spans = Spans("genai")
    proxy._finalize_locked = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug"))
    ctx = proxy.prepare("openai", {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]}, {})
    proxy.finalize_safely(ctx, 200, {"id": "c", "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
    span = spans()[0]
    assert span.status.status_code != StatusCode.ERROR and "error.type" not in span.attributes
    assert span.attributes["prefixwise.recorded"] is False


def test_a_user_id_is_never_a_conversation_id(proxy_base, spans):
    post(proxy_base, "/v1/messages", {"model": "claude-haiku-5-5", "max_tokens": 8,
                                      "metadata": {"user_id": "alice@example.com"},
                                      "messages": [{"role": "user", "content": "hi"}]}, {})
    a = dict(spans()[0].attributes)
    assert "gen_ai.conversation.id" not in a and a["prefixwise.session_id"] == "alice@example.com"


def test_no_usage_reported_means_no_usage_attributes(spans):
    from prefixwise.otel import Spans
    proxy = PrefixwiseProxy(mock=True)
    proxy.spans = Spans("genai")
    ctx = proxy.prepare("openai", {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]}, {})
    proxy.finalize(ctx, 200, {"id": "c", "model": "gpt-5.6"})
    a = dict(spans()[0].attributes)
    assert not [k for k in a if k.startswith("gen_ai.usage.")]
    assert "gen_ai.request.stream" not in a  # set only for streams


def test_the_actual_provider_is_named(spans):
    from prefixwise.otel import Spans, provider_name
    rec = Recorder(":memory:")
    rec.proxy.spans = Spans("genai")
    rec.record("anthropic", {"model": "anthropic.claude-haiku-5-5", "max_tokens": 64,
                             "messages": [{"role": "user", "content": "hi"}]},
               {"id": "m", "usage": {"input_tokens": 5, "output_tokens": 1}, "stop_reason": "end_turn"},
               genai_provider="aws.bedrock")
    a = dict(spans()[0].attributes)
    assert a["gen_ai.provider.name"] == "aws.bedrock" and not [k for k in a if k.startswith("openai.")]
    assert a["gen_ai.request.max_tokens"] == 64 and a["gen_ai.response.finish_reasons"] == ("end_turn",)
    assert [provider_name(upstream=u) for u in ("https://api.groq.com/openai", "https://x.openai.azure.com",
                                                "http://127.0.0.1:8000", "https://api.openai.com")] == [
        "groq", "azure.ai.openai", None, "openai"]
    assert provider_name("xai") == "x_ai" and provider_name("vllm_custom") == "vllm_custom"


def test_trace_context_reaches_the_upstream_and_the_span_names_it(spans):
    import http.server
    seen = {}

    class Upstream(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            seen.update({k.lower(): v for k, v in self.headers.items()})
            out = json.dumps({"id": "c", "model": "gpt-5.6", "service_tier": "default",
                              "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": "x"}}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    up = http.server.HTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    proxy = PrefixwiseProxy(upstream_url=f"http://127.0.0.1:{up.server_address[1]}", mock=False, otel="1")
    srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        tp = f"00-{TRACE_ID}-00f067aa0ba902b7-01"
        post(f"http://127.0.0.1:{srv.server_address[1]}", "/v1/chat/completions",
             {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hi"}]},
             {"traceparent": tp, "authorization": "Bearer sk-test"})
    finally:
        srv.shutdown()
        up.shutdown()
    assert seen.get("traceparent") == tp
    a = dict(spans()[0].attributes)
    assert a["server.address"] == "127.0.0.1" and a["server.port"] == up.server_address[1]
    assert a["openai.response.service_tier"] == "default" and a["gen_ai.response.finish_reasons"] == ("stop",)
