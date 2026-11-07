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
