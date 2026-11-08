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
   verdict: an upstream error, an unreachable upstream, a stream cut off by the upstream, or
   by the client (status 499). A response that arrived without usage is not a failure: it is a
   verdict row (`unknown`), streamed or not.
3. **One definition per metric.** Each metric has a name, formula, unit, valid groupings
   and caveats, in one module (`prefixwise.metrics`). The CLI, the HTTP API, exports and
   any future UI call it; none computes its own. The catalog is served by the API, so the
   data documents itself.
4. **Never a silently partial number.** A sum over a group where some rows lack the field
   (rows written before schema 3) is null, not a partial total. Dollars say how many requests
   were priced.
5. **Ratios are ratios of sums.** A cache hit rate over a group is
   `sum(cached) / sum(prompt)`, never a mean of per-request rates (which overweights
   small requests and can reverse across groups).
6. **Get the data out in standard shapes.** Raw rows as JSONL/CSV; operational metrics in
   Prometheus format; per-request telemetry as OpenTelemetry GenAI conventions, so it lands
   in the tools teams already run. Prefixwise is not a second observability platform; it
   is the part that knows *why* a cache missed.
7. **Never prompt text.** Rows hold counts, causes, hashes and the caller's own labels.
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
| `cache_ttl_seconds` (schema 5) | the request's own cache lifetime: what its write was granted, or asked for (Anthropic 300 or 3,600; the `X-Prefixwise-Cache-Ttl` header; empty when unknown): what the 5-minute vs 1-hour advice reads |
| `unread_write_tokens`, `unread_write_cost` (schema 6) | a miss of the same conversation that replaced the prefix (or let it expire) and read nothing right after the previous turn wrote it: the tokens it wrote and never had read, and the surcharge paid to write them above sending them uncached (`prefixwise report`: "cache writes never read"; not part of avoidable cost) |
| `detail` (schema 4) | the why behind the cause, where the evidence allows: `ttl_shortened`, `lookback_exceeded`, `markers_ignored`, `sent_before_cache_written`, `shared_prefix_not_reused`, `thinking_stripped`, `cache_key_changed` (`prefixwise.details`); a dimension and filter in `/api/v1` |

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
- latency p50/p95, Prefixwise overhead p50/p95 (requests with a verdict; exact, computed in
  SQLite, so a query never loads a whole column)
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

## Reads never slow the proxy

Metrics, the API and exports read through their own read-only SQLite connection (WAL lets
readers and the writer run together), so a large query never holds the lock the proxy
records requests under.

An independent review of the first version found the task metrics defined twice (so the
report, the API and `/ledger` could disagree), percentiles that loaded every value under the
writer's lock, silently partial sums, and a response without usage counted two ways. All are
fixed above, each with a test that replays the case.

## Sinks (copies of the ledger)

`prefixwise collect --sink URL` (or `serve --sink`, or `PREFIXWISE_SINKS`) sends every committed row to a
sink as well: `file:///path/rows.jsonl` (one JSON object per line) or
`clickhouse://user:password@host:8123/db.table` (`clickhouses://` for TLS). The SQLite ledger stays the
store the report and the API read; a sink never delays a row and a failing sink is reported once. Rows
hold what the ledger holds, plus `recorded_at` (unix seconds). The ClickHouse table, created by you:

```sql
CREATE TABLE prefixwise.entries (
    session_id String, agent_id String, provider String, model String, cause Nullable(String),
    prompt_tokens Int64, cached_tokens Int64, avoidable_tokens Int64, avoidable_cost Nullable(Float64),
    task_success Nullable(UInt8), legit Nullable(UInt8), native_reason Nullable(String),
    classifier_version String, task_id Nullable(String), request_id Nullable(String),
    completion_tokens Int64, fixes Nullable(String), cost Nullable(Float64), price_as_of Nullable(String),
    status Int64, error Nullable(String), api Nullable(String), stream Nullable(UInt8),
    uncached_tokens Nullable(Int64), cache_write_tokens Nullable(Int64), latency_ms Nullable(Float64),
    overhead_ms Nullable(Float64), gap_seconds Nullable(Float64), evidence Nullable(String),
    secondary_causes Nullable(String), divergence_role Nullable(String), divergence_index Nullable(Int64),
    detail Nullable(String), cache_ttl_seconds Nullable(Float64), unread_write_tokens Nullable(Int64),
    unread_write_cost Nullable(Float64), schema_version Int64, recorded_at Float64
) ENGINE = MergeTree ORDER BY (model, recorded_at);
```

`cause` is null on a failure row (`status` >= 400 or `error` set); booleans arrive as 0/1. A sink receives rows
from the moment it is configured; earlier rows are in the ledger (`prefixwise export`). A ClickHouse error that
names a permanent cause (a schema mismatch) drops the batch and is counted, while a busy server is retried.

## Baseline (`prefixwise export --baseline`)

`.prefixwise-baseline.json`, for `prefixwise check`'s monthly projection: per model, `turns_per_month`,
`eligible_turns_per_month` (turns with a previous turn to reuse: first turns and model switches are not eligible) and
`mean_reusable_tokens` (what an eligible turn read from the cache, or could have), scaled from the
ledger's span (`window_days`, at least one day) to 30 days. Counts and means only: no ids, no text. The
projection is `affected eligible turns / eligible turns in the run x eligible_turns_per_month x
mean_reusable_tokens x (the rate a miss pays above a hit per token)`, and is labelled an estimate.
