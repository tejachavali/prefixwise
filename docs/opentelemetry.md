# OpenTelemetry

Prefixwise can export one span per request it records, so the reason a request missed the
prompt cache sits next to the rest of your traces (Jaeger, Tempo, Honeycomb, Datadog, Langfuse,
Phoenix, or anything that takes OTLP). Attribute names follow OpenTelemetry's GenAI semantic
conventions as of November 2026 (the Anthropic and OpenAI pages of
[semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai));
those conventions are still in development and may change.

## Turning it on

| Where | How | Where spans go |
|---|---|---|
| Proxy | `prefixwise serve --otel` (or `PREFIXWISE_OTEL=1`) | OTLP, set up by Prefixwise (below) |
| `prefixwise run` | `prefixwise run --otel -- <agent>` | OTLP, set up by Prefixwise |
| `prefixwise.wrap`, LiteLLM callback | `PREFIXWISE_OTEL=1` in the environment | your app's tracer provider |

**Inside your app** (wrap, LiteLLM), Prefixwise never installs a tracer provider: spans go to
the one your app configures, whenever it configures it, with its exporters and sampling. If the
app configures none, spans go nowhere.

**As its own process** (`serve`, `run`), Prefixwise is the application, so it sets up the
exporter, from the standard variables:

```bash
pip install 'prefixwise[otel]'
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 prefixwise serve --upstream https://api.anthropic.com --otel
```

- `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, headers, timeouts: as
  the OpenTelemetry SDK reads them. Default `http://localhost:4318`.
- `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`: OTLP over gRPC (needs `pip install
  opentelemetry-exporter-otlp-proto-grpc`; default HTTP/protobuf).
- `OTEL_TRACES_EXPORTER=console` prints spans; `none` turns export off.
- `service.name` is `prefixwise` unless `OTEL_SERVICE_NAME` or `OTEL_RESOURCE_ATTRIBUTES` sets it.

The startup banner says where spans go. A bad `PREFIXWISE_OTEL` value turns export off with a
notice; it never stops the proxy or your app.

## Two modes

- **`PREFIXWISE_OTEL=1`** (`genai`): a GenAI client span per request, plus Prefixwise's
  attributes. Use it when nothing else traces these LLM calls, e.g. behind the proxy.
- **`PREFIXWISE_OTEL=cause`**: a separate INTERNAL span `prefixwise.cache` with only
  Prefixwise's attributes, no `gen_ai.*`. Use it when your app's own instrumentation already
  emits the GenAI span for the call, so dashboards that sum `gen_ai.usage.*` do not count
  tokens twice.

## Attributes

| Attribute | Value |
|---|---|
| span name, kind | `chat {model}`, CLIENT (`prefixwise.cache`, INTERNAL in `cause` mode) |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | the well-known value for the actual provider when Prefixwise knows it: `anthropic`, `openai`, `aws.bedrock`, `gcp.vertex_ai`, `azure.ai.openai`, `groq`, `deepseek`, `mistral`, `x_ai`, ... (from LiteLLM's provider, or the proxy's upstream host); otherwise `openai` for OpenAI-compatible servers (vLLM, SGLang) and `anthropic` for Anthropic's format |
| `openai.api.type`, `openai.request/response.service_tier` | OpenAI only: `chat_completions` or `responses`; the tier when sent or returned |
| `server.address`, `server.port` | the proxy's upstream (not set in process, where Prefixwise does not see the URL) |
| `gen_ai.request.model`, `.max_tokens`, `.seed`, `.choice.count` (when not 1), `.previous_response.id`, `.stream` (only when streaming) | from the request |
| `gen_ai.response.model`, `.id`, `.finish_reasons` | from the response |
| `gen_ai.usage.input_tokens` | all prompt tokens, cache reads and writes included (for Anthropic, `input_tokens + cache_read + cache_creation`, as the spec computes it). Set only when the provider reported usage |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` | when non-zero |
| `gen_ai.usage.output_tokens` | |
| `gen_ai.conversation.id` | the session id, only when the client gave one: never one Prefixwise made up, and never a user id used as a session (Anthropic `metadata.user_id`, LiteLLM's `user`) |
| `gen_ai.conversation.compacted` | `true` when the client signalled a compaction |
| `prefixwise.cache.cause` | the miss cause (`none` for a hit; see the README's cause table) |
| `prefixwise.cache.legit` | `true` for an expected miss (first request, model switch), `false` for waste |
| `prefixwise.cache.detail` | the sub-reason, e.g. `ttl_shortened`, `lookback_exceeded` |
| `prefixwise.cache.avoidable_tokens`, `prefixwise.cache.avoidable_cost_usd`, `prefixwise.cost_usd` | |
| `prefixwise.cache.provider_reason`, `prefixwise.cache.evidence`, `prefixwise.cache.secondary_causes`, `prefixwise.fixes` | when present |
| `prefixwise.request_id`, `prefixwise.session_id`, `prefixwise.agent_id`, `prefixwise.task_id` | join keys to the ledger and `/api/v1` |
| `error.type`, span status ERROR | failed requests (no usage) |
| `prefixwise.recorded = false` | the call succeeded but Prefixwise could not classify it (not an error) |

Spans never carry prompt or completion text.

## Traces

Through the proxy, a request carrying a W3C `traceparent` header gets its span in the caller's
trace (OpenTelemetry-instrumented HTTP clients send it), and `traceparent` / `tracestate` are
passed on to the upstream, so a traced engine or gateway joins the same trace. With
`prefixwise.wrap`, the span is a child of whatever span is current when the call returns. The
LiteLLM callback runs on LiteLLM's logging thread, where your span is usually not current, so
its spans may start new traces.

The span covers the call: from when the proxy received the request (or, in process, when the
SDK call started) to when its response was recorded; a resend without a field an upstream
rejected stays within the one span. Exporting can never fail a request: an error is reported
once on stderr and ignored.
