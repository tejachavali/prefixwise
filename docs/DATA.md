# Data first: what Prefixwise records, how metrics are defined, how they get out

Design note, 2026-11-07.

## Principles

1. **Record facts; derive metrics on read.** The ledger stores what was observed for each
   request (token counts by kind, timings, status, the verdict and its evidence). Every
   metric is computed from those facts in one place, so a better definition, or a new
   price table, applies to the whole history. Stored derived values (cost, avoidable
   tokens) are kept only as "as computed then", with the price-table date.
2. **Record every request, failures included.** A ledger of successful requests only
   biases every rate (survivorship). Failed requests get a row with their status and no
   verdict.
3. **One definition per metric.** Each metric has a name, formula, unit, valid groupings
   and caveats, in one module (`prefixwise.metrics`). The CLI, the HTTP API, exports and
   any future UI call it; none computes its own. The catalog is served by the API, so the
   data documents itself.
4. **Ratios are ratios of sums.** A cache hit rate over a group is
   `sum(cached) / sum(prompt)`, never a mean of per-request rates (which overweights
   small requests and can reverse across groups).
5. **Get the data out in standard shapes.** Raw rows as JSONL/CSV; operational metrics in
   Prometheus format; per-request telemetry as OpenTelemetry GenAI conventions, so it lands
   in the tools teams already run. Prefixwise is not a second observability platform; it
   is the part that knows *why* a cache missed.
6. **Never prompt text.** Rows hold counts, causes, hashes and the caller's own labels.
   Exports never include block hashes.

## Facts per request (schema 3)

Already recorded: session, agent, task, provider, model, cause, legit, prompt/cached/
completion tokens, avoidable tokens and cost, cost and price date, native reason, fixes
applied, classifier version, request id, time.

Added (observed today, previously dropped):

| field | why |
|---|---|
| `status`, `error` | failed requests are rows too: error rates, unbiased denominators |
| `api`, `stream` | chat / responses / messages; streamed or not |
| `uncached_tokens`, `cache_write_tokens` | the token breakdown behind cost and hit rate |
| `latency_ms`, `overhead_ms` | upstream time; Prefixwise's own processing time |
| `gap_seconds` | time since the session's previous turn (TTL questions) |
| `evidence`, `secondary_causes` | why the verdict was given, in Prefixwise's own words |
| `divergence_role`, `divergence_index` | where the prompt first changed (system, tools, messages, ...) |
| `schema_version` | rows say which schema wrote them |

## Metrics (v1 catalog)

Defined once in `prefixwise.metrics`, groupable by cause, session, agent, model, provider,
task, day/hour:

- requests, failed requests, error rate
- prompt, cached, uncached, cache-write, completion tokens
- cache hit rate = cached / prompt
- avoidable tokens, avoidable share = avoidable / prompt
- cost, avoidable cost, priced requests (dollars cover priced requests only, and say so)
- waste requests (verdict says the miss was avoidable)
- tokens per successful task, cost per successful task
- latency p50/p95, Prefixwise overhead p50/p95
- native agreement (where a provider gives its own reason)

## Interfaces

- **Python:** `prefixwise.metrics.query(ledger, metrics, by=..., where=..., bucket=...)`.
- **HTTP (`/api/v1`, same token auth as the proxy):** `summary`, `breakdown?by=`,
  `timeseries?bucket=`, `requests` (filter + paginate), `requests/{id}`, `fixes`, `prices`,
  `metrics` (the catalog), `POST explain`, `POST tasks`. Existing paths keep working.
- **CLI:** `report`, `fixes`, `prices`, `request` become renderers of the same calls;
  `prefixwise export --format jsonl|csv` for raw rows.
- **Prometheus / OpenTelemetry:** second step. Findings (checked 2026-11-07 against the
  spec repositories): every GenAI convention is still *Development* (experimental), and they
  moved to `open-telemetry/semantic-conventions-genai`, which has no release yet. The last
  released set (core semconv v1.41.1) defines spans named `{gen_ai.operation.name}
  {gen_ai.request.model}` with `gen_ai.provider.name` (not the deprecated `gen_ai.system`),
  `gen_ai.conversation.id`, and usage `gen_ai.usage.input_tokens` (which **includes** cached
  tokens), `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`
  (renamed `cache_write` on the unreleased main), and the histogram `gen_ai.client.token.usage`.
  There is no convention for cost or for a cache-miss cause: those are Prefixwise's own
  (`prefixwise.*`). Plan: emit the released v1.41.1 names; map to Prometheus by the OTel
  compatibility rules (dots to underscores, unit suffixes, `_total` on counters).

## What this is not

No dashboards beyond a report, no prompt storage, no streaming bus. Data leaves through
documented, versioned shapes.
