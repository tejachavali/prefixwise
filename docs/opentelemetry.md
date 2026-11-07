# OpenTelemetry

Prefixwise can export one span per request it records, so the reason a request missed the
prompt cache sits next to the rest of your traces (Jaeger, Tempo, Honeycomb, Datadog, Langfuse,
Phoenix, or anything that takes OTLP).

## Turning it on

| Where | How |
|---|---|
| Proxy | `prefixwise serve --otel` (or `PREFIXWISE_OTEL=1`) |
| `prefixwise run` | `prefixwise run --otel -- <agent>` |
| `prefixwise.wrap`, LiteLLM callback | `PREFIXWISE_OTEL=1` in the environment |

Spans go to your application's tracer provider if it has configured one. Otherwise Prefixwise
sets one up that exports OTLP over HTTP, configured by the standard variables:

```bash
pip install 'prefixwise[otel]'
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 prefixwise serve --upstream https://api.anthropic.com --otel
```

`service.name` is `prefixwise` unless `OTEL_SERVICE_NAME` says otherwise.

## Two modes

- **`PREFIXWISE_OTEL=1`** (`genai`): a GenAI client span per OpenTelemetry's GenAI semantic
  conventions (the Anthropic and OpenAI pages, as of Nov 2026), plus Prefixwise's attributes.
  Use it when nothing else traces these LLM calls, e.g. behind the proxy.
- **`PREFIXWISE_OTEL=cause`**: an INTERNAL span `prefixwise.cache` with only Prefixwise's
  attributes, no `gen_ai.*`. Use it when your app's own instrumentation already emits the GenAI
  span for the call, so dashboards that sum `gen_ai.usage.*` do not count tokens twice.

## Attributes

| Attribute | Value |
|---|---|
| span name, kind | `chat {model}`, CLIENT (`prefixwise.cache`, INTERNAL in `cause` mode) |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | `anthropic` or `openai` (also OpenAI-compatible servers: vLLM, SGLang, ...) |
| `openai.api.type` | `chat_completions` or `responses` |
| `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.request.stream` | from the request and response |
| `gen_ai.usage.input_tokens` | all prompt tokens, cache reads and writes included (for Anthropic, `input_tokens + cache_read + cache_creation`, as the spec computes it) |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` | when non-zero |
| `gen_ai.usage.output_tokens` | |
| `gen_ai.conversation.id` | the session id, only when the client gave one (never one Prefixwise made up) |
| `prefixwise.cache.cause` | the miss cause (`none` for a hit; see the README's cause table) |
| `prefixwise.cache.legit` | `true` for an expected miss (first request, model switch), `false` for waste |
| `prefixwise.cache.detail` | the sub-reason, e.g. `ttl_shortened`, `lookback_exceeded` |
| `prefixwise.cache.avoidable_tokens`, `prefixwise.cache.avoidable_cost_usd`, `prefixwise.cost_usd` | |
| `prefixwise.cache.provider_reason`, `prefixwise.cache.evidence`, `prefixwise.cache.secondary_causes`, `prefixwise.fixes` | when present |
| `prefixwise.request_id`, `prefixwise.session_id`, `prefixwise.agent_id`, `prefixwise.task_id` | join keys to the ledger and `/api/v1` |
| `error.type`, span status ERROR | failed requests (no usage) |

Spans never carry prompt or completion text.

## Traces

Through the proxy, a request carrying a W3C `traceparent` header gets its span in the caller's
trace (OpenTelemetry-instrumented HTTP clients send it). With `prefixwise.wrap`, the span is a
child of whatever span is current when the call returns. The LiteLLM callback runs on LiteLLM's
logging thread, where your span is usually not current, so its spans may start new traces.

The span covers the call: from when the proxy received the request (or, in process, when the
SDK call started) to when its response was recorded. Exporting can never fail a request: an
error is reported once on stderr and ignored.
