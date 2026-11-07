"""OpenTelemetry export: one span per recorded request, with why its cache missed.

Off unless PREFIXWISE_OTEL is set (or `prefixwise serve --otel`). Applies to the proxy,
`prefixwise run`, `prefixwise.wrap` and the LiteLLM callback alike: every row the ledger
records becomes a span.

    PREFIXWISE_OTEL=1 (or genai)  a GenAI client span, per OpenTelemetry's GenAI semantic
                                  conventions (Anthropic and OpenAI pages, Nov 2026): name
                                  `chat {model}`, kind CLIENT, gen_ai.* usage including
                                  gen_ai.usage.cache_read/cache_write.input_tokens, plus
                                  prefixwise.* attributes.
    PREFIXWISE_OTEL=cause         only the prefixwise.* attributes, on an INTERNAL span named
                                  `prefixwise.cache`: for apps whose own instrumentation already
                                  emits the GenAI span, so token totals are not counted twice.

Spans go to the application's tracer provider if it has one; otherwise Prefixwise sets one up
that exports OTLP over HTTP, configured by the standard OTEL_EXPORTER_OTLP_* and
OTEL_SERVICE_NAME variables (needs: pip install 'prefixwise[otel]'). Through the proxy, a
request carrying a W3C `traceparent` header gets its span in the caller's trace. Spans hold
counts, ids and Prefixwise's verdicts, never prompt or completion text.
"""

from __future__ import annotations

import os
import sys
import time

from . import __version__

MODES = {"1": "genai", "true": "genai", "yes": "genai", "on": "genai", "genai": "genai", "cause": "cause"}


def mode_from(value) -> str | None:
    """True / "1" / "genai" -> "genai"; "cause" -> "cause"; unset, "0", "false", "off" -> None."""
    if value is None or value is False:
        return None
    if value is True:
        return "genai"
    v = str(value).strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return None
    if v not in MODES:
        raise ValueError(f"unknown PREFIXWISE_OTEL value {value!r}: expected 1, genai or cause")
    return MODES[v]


def _ensure_provider() -> None:
    """Use the application's tracer provider; with none configured, set up OTLP over HTTP."""
    from opentelemetry import trace

    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        return  # the application configured OpenTelemetry: its exporters, its sampling
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        print("prefixwise: PREFIXWISE_OTEL is set but no OpenTelemetry exporter is configured or installed; "
              "spans go nowhere. pip install 'prefixwise[otel]', or configure a tracer provider in your app",
              file=sys.stderr)
        return
    attrs = {} if os.environ.get("OTEL_SERVICE_NAME") else {"service.name": "prefixwise"}
    provider = TracerProvider(resource=Resource.create(attrs))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)  # flushed at exit by the SDK's own atexit hook


class Spans:
    """Turns ledger rows into spans. export() never raises."""

    def __init__(self, mode: str):
        from opentelemetry import trace

        _ensure_provider()
        self.mode = mode
        self.tracer = trace.get_tracer("prefixwise", __version__)
        self._warned = False

    def export(self, ctx: dict, entry, payload: dict | None = None, *, default_session: str | None = None) -> None:
        try:
            self._export(ctx, entry, payload or {}, default_session)
        except Exception as exc:  # noqa: BLE001 - telemetry must never affect a request
            if not self._warned:
                self._warned = True
                print(f"prefixwise: OpenTelemetry export failed ({type(exc).__name__}: {exc}); "
                      "requests are unaffected", file=sys.stderr)

    def _export(self, ctx: dict, entry, payload: dict, default_session: str | None) -> None:
        from opentelemetry import propagate, trace
        from opentelemetry.trace import SpanKind, Status, StatusCode

        headers = ctx.get("headers") or {}
        parent = propagate.extract(headers) if "traceparent" in headers else None  # else: current context
        end = time.time_ns()
        start = ctx.get("span_start", ctx.get("sent_at"))
        start = min(int(start * 1e9), end) if start else end
        failed = entry.cause is None

        attrs: dict = {}
        if self.mode == "genai":
            name, kind = f"chat {entry.model}", SpanKind.CLIENT
            attrs["gen_ai.operation.name"] = "chat"
            attrs["gen_ai.provider.name"] = entry.provider  # "anthropic" or "openai" (OpenAI-compatible)
            attrs["gen_ai.request.model"] = entry.model
            if entry.provider == "openai" and entry.api in ("chat", "responses"):
                attrs["openai.api.type"] = "responses" if entry.api == "responses" else "chat_completions"
            if entry.stream is not None:
                attrs["gen_ai.request.stream"] = bool(entry.stream)
            if payload.get("model"):
                attrs["gen_ai.response.model"] = str(payload["model"])
            if payload.get("id"):
                attrs["gen_ai.response.id"] = str(payload["id"])
            if not failed:
                # input_tokens includes cache reads and writes (the spec's rule for Anthropic, whose
                # own input_tokens excludes them); the cache counts only when non-zero.
                attrs["gen_ai.usage.input_tokens"] = int(entry.prompt_tokens)
                attrs["gen_ai.usage.output_tokens"] = int(entry.completion_tokens or 0)
                if entry.cached_tokens:
                    attrs["gen_ai.usage.cache_read.input_tokens"] = int(entry.cached_tokens)
                if entry.cache_write_tokens:
                    attrs["gen_ai.usage.cache_write.input_tokens"] = int(entry.cache_write_tokens)
            # Only an id the client gave: never one Prefixwise made up (the spec forbids fallbacks).
            if entry.session_id and entry.session_id != default_session:
                attrs["gen_ai.conversation.id"] = entry.session_id
        else:
            name, kind = "prefixwise.cache", SpanKind.INTERNAL
            attrs["prefixwise.model"] = entry.model
            attrs["prefixwise.provider"] = entry.provider

        attrs["prefixwise.request_id"] = entry.request_id or ""
        attrs["prefixwise.session_id"] = entry.session_id
        attrs["prefixwise.agent_id"] = entry.agent_id
        if entry.task_id:
            attrs["prefixwise.task_id"] = entry.task_id
        if failed:
            attrs["error.type"] = entry.error or f"http_{entry.status}"
        else:
            attrs["prefixwise.cache.cause"] = entry.cause
            if entry.legit is not None:
                attrs["prefixwise.cache.legit"] = bool(entry.legit)  # True: an expected miss, not waste
            if entry.detail:
                attrs["prefixwise.cache.detail"] = entry.detail
            if entry.secondary_causes:
                attrs["prefixwise.cache.secondary_causes"] = entry.secondary_causes.split(",")
            if entry.native_reason:
                attrs["prefixwise.cache.provider_reason"] = entry.native_reason
            if entry.evidence:
                attrs["prefixwise.cache.evidence"] = entry.evidence
            attrs["prefixwise.cache.avoidable_tokens"] = int(entry.avoidable_tokens or 0)
            if entry.avoidable_cost is not None:
                attrs["prefixwise.cache.avoidable_cost_usd"] = float(entry.avoidable_cost)
            if entry.cost is not None:
                attrs["prefixwise.cost_usd"] = float(entry.cost)
            if entry.fixes:
                attrs["prefixwise.fixes"] = entry.fixes.split(",")

        span = self.tracer.start_span(name, context=parent, kind=kind, attributes=attrs, start_time=start)
        if failed:
            span.set_status(Status(StatusCode.ERROR))
        span.end(end_time=end)
