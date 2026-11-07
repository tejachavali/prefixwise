"""OpenTelemetry export: one span per recorded request, with why its cache missed.

Off unless PREFIXWISE_OTEL is set (or `prefixwise serve --otel` / `run --otel`). Applies to the
proxy, `prefixwise run`, `prefixwise.wrap` and the LiteLLM callback alike: every row the ledger
records becomes a span.

    PREFIXWISE_OTEL=1 (or genai)  a GenAI client span, per OpenTelemetry's GenAI semantic
                                  conventions (Anthropic and OpenAI pages, Nov 2026): name
                                  `chat {model}`, kind CLIENT, gen_ai.* usage including
                                  gen_ai.usage.cache_read/cache_write.input_tokens, plus
                                  prefixwise.* attributes.
    PREFIXWISE_OTEL=cause         a separate INTERNAL span `prefixwise.cache` with only the
                                  prefixwise.* attributes: for apps whose own instrumentation
                                  already emits the GenAI span, so tokens are not counted twice.

Where spans go: inside your application (wrap, LiteLLM), to the tracer provider your app sets
up; Prefixwise never installs one there, so it cannot take over your app's telemetry. As its
own process (`serve`, `run`), Prefixwise is the application: it installs an OTLP exporter
configured by the standard OTEL_* variables (needs: pip install 'prefixwise[otel]'). Spans hold
counts, ids and Prefixwise's verdicts, never prompt or completion text.
"""

from __future__ import annotations

import os
import sys
import time
from urllib.parse import urlsplit

from . import __version__

MODES = {"1": "genai", "true": "genai", "yes": "genai", "on": "genai", "genai": "genai", "cause": "cause"}

# gen_ai.provider.name well-known values (the spec: if one applies, it MUST be used).
_LITELLM_PROVIDERS = {
    "openai": "openai", "anthropic": "anthropic", "azure": "azure.ai.openai", "azure_ai": "azure.ai.inference",
    "bedrock": "aws.bedrock", "vertex_ai": "gcp.vertex_ai", "vertex_ai_beta": "gcp.vertex_ai",
    "gemini": "gcp.gemini", "cohere": "cohere", "cohere_chat": "cohere", "watsonx": "ibm.watsonx.ai",
    "perplexity": "perplexity", "xai": "x_ai", "deepseek": "deepseek", "groq": "groq", "mistral": "mistral",
    "moonshot": "moonshot_ai", "ollama": "ollama", "ollama_chat": "ollama",
}
_HOSTS = {
    "api.openai.com": "openai", "api.anthropic.com": "anthropic", "api.groq.com": "groq",
    "api.deepseek.com": "deepseek", "api.x.ai": "x_ai", "api.mistral.ai": "mistral",
    "api.perplexity.ai": "perplexity", "api.moonshot.ai": "moonshot_ai", "api.cohere.com": "cohere",
}


def provider_name(litellm_provider: str | None = None, upstream: str | None = None) -> str | None:
    """The well-known gen_ai.provider.name for a LiteLLM provider or an upstream URL, if any."""
    if litellm_provider:
        return _LITELLM_PROVIDERS.get(litellm_provider, litellm_provider)
    host = (urlsplit(upstream).hostname or "") if upstream else ""
    if host.endswith(".openai.azure.com"):
        return "azure.ai.openai"
    return _HOSTS.get(host)


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


def install_provider() -> str:
    """For Prefixwise's own processes only (serve, run): export over OTLP, configured by the
    standard variables. Returns where spans go, for the startup banner."""
    from opentelemetry import trace

    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        return "the configured tracer provider"
    choice = (os.environ.get("OTEL_TRACES_EXPORTER") or "otlp").split(",")[0].strip().lower()
    if choice == "none":
        return "nowhere (OTEL_TRACES_EXPORTER=none)"
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        if choice == "console":
            exporter, where = ConsoleSpanExporter(), "stdout"
        elif (os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
              or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf") == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter()
            where = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT") or "localhost:4317"
            where = f"OTLP gRPC {where}"
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter()
            where = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4318"
            where = f"OTLP {where}"
    except ImportError as exc:
        print(f"prefixwise: OpenTelemetry export is on but its exporter is not installed ({exc.name}); "
              "pip install 'prefixwise[otel]' (gRPC: opentelemetry-exporter-otlp-proto-grpc)", file=sys.stderr)
        return "nowhere (exporter not installed)"
    named = os.environ.get("OTEL_SERVICE_NAME") or "service.name=" in os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    provider = TracerProvider(resource=Resource.create({} if named else {"service.name": "prefixwise"}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)  # flushed at exit by the SDK's own atexit hook
    return where


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class Spans:
    """Turns ledger rows into spans. export() never raises."""

    def __init__(self, mode: str):
        from opentelemetry import trace

        self.mode = mode
        # A proxy tracer: it follows whatever provider is (or is later) set globally.
        self.tracer = trace.get_tracer("prefixwise", __version__)
        self._warned = False

    def export(self, ctx: dict, entry, payload: dict | None = None, *, default_session: str | None = None,
               upstream: str | None = None) -> None:
        try:
            self._export(ctx, entry, payload if isinstance(payload, dict) else {}, default_session, upstream)
        except Exception as exc:  # noqa: BLE001 - telemetry must never affect a request
            if not self._warned:
                self._warned = True
                print(f"prefixwise: OpenTelemetry export failed ({type(exc).__name__}: {exc}); "
                      "requests are unaffected", file=sys.stderr)

    def _export(self, ctx: dict, entry, payload: dict, default_session: str | None, upstream: str | None) -> None:
        from opentelemetry import propagate
        from opentelemetry.trace import SpanKind, Status, StatusCode

        headers = ctx.get("headers") or {}
        parent = propagate.extract(headers) if "traceparent" in headers else None  # else: current context
        end = time.time_ns()
        start = ctx.get("span_start", ctx.get("sent_at"))
        start = min(int(start * 1e9), end) if start else end
        body = ctx.get("body") if isinstance(ctx.get("body"), dict) else {}
        failed = entry.status >= 400 or entry.error not in (None, "not_recorded")
        verdict = entry.cause is not None

        attrs: dict = {}
        if self.mode == "genai":
            name, kind = f"chat {entry.model}", SpanKind.CLIENT
            gp = ctx.get("genai_provider") or provider_name(upstream=upstream) or entry.provider
            attrs["gen_ai.operation.name"] = "chat"
            attrs["gen_ai.provider.name"] = gp
            attrs["gen_ai.request.model"] = entry.model
            if upstream:
                parts = urlsplit(upstream)
                if parts.hostname:
                    attrs["server.address"] = parts.hostname
                    attrs["server.port"] = parts.port or (443 if parts.scheme == "https" else 80)
            if gp == "openai":
                attrs["openai.api.type"] = "responses" if entry.api == "responses" else "chat_completions"
                tier = payload.get("service_tier")
                if tier:
                    attrs["openai.response.service_tier"] = str(tier)
                if body.get("service_tier"):
                    attrs["openai.request.service_tier"] = str(body["service_tier"])
            if entry.stream:
                attrs["gen_ai.request.stream"] = True  # set only for streams (unset means not streamed)
            mt = _int(body.get("max_tokens") or body.get("max_completion_tokens") or body.get("max_output_tokens"))
            if mt is not None:
                attrs["gen_ai.request.max_tokens"] = mt
            n = _int(body.get("n"))
            if n is not None and n != 1:
                attrs["gen_ai.request.choice.count"] = n
            seed = _int(body.get("seed"))
            if seed is not None:
                attrs["gen_ai.request.seed"] = seed
            if body.get("previous_response_id"):
                attrs["gen_ai.request.previous_response.id"] = str(body["previous_response_id"])
            if ctx.get("compaction_event"):
                attrs["gen_ai.conversation.compacted"] = True
            if payload.get("model"):
                attrs["gen_ai.response.model"] = str(payload["model"])
            if payload.get("id"):
                attrs["gen_ai.response.id"] = str(payload["id"])
            reasons = [str(c.get("finish_reason")) for c in payload.get("choices") or []
                       if isinstance(c, dict) and c.get("finish_reason")]
            if payload.get("stop_reason"):
                reasons = [str(payload["stop_reason"])]
            if reasons:
                attrs["gen_ai.response.finish_reasons"] = reasons
            if verdict and payload.get("usage"):  # counts the provider reported; none reported, none set
                # input_tokens includes cache reads and writes (the spec's rule for Anthropic, whose
                # own input_tokens excludes them); the cache counts only when non-zero.
                attrs["gen_ai.usage.input_tokens"] = int(entry.prompt_tokens)
                attrs["gen_ai.usage.output_tokens"] = int(entry.completion_tokens or 0)
                if entry.cached_tokens:
                    attrs["gen_ai.usage.cache_read.input_tokens"] = int(entry.cached_tokens)
                if entry.cache_write_tokens:
                    attrs["gen_ai.usage.cache_write.input_tokens"] = int(entry.cache_write_tokens)
            # Only a conversation id the client gave: never one made up, never a user id.
            if entry.session_id and entry.session_id != default_session and not ctx.get("session_is_user"):
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
        elif not verdict:
            attrs["prefixwise.recorded"] = False  # the call succeeded; Prefixwise could not classify it
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
