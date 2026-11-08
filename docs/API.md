# Prefixwise API (v1)

`prefixwise serve` serves the API next to the proxy, under `/api/v1`. Every number comes from
`prefixwise.metrics`, the same definitions `prefixwise report` uses, so the CLI and the API
cannot disagree. Why it is built this way: [DATA.md](DATA.md).

- **Auth:** with `PREFIXWISE_AUTH_TOKEN` set, send `X-Prefixwise-Token`. Without it, the API
  is open to whoever can reach the port (localhost by default).
- **Errors:** a 4xx with `{"error": {"message": "..."}}`. Unknown metrics, dimensions and
  filters are 400s that name the valid ones.
- **Privacy:** no endpoint returns prompt text or block hashes; rows hold counts, causes,
  timings and your own labels.
- **Stability:** `/api/v1` paths and field names only grow. Metrics may be added; existing
  ones keep their meaning. The older `/ledger`, `/requests/{id}` and `/tasks` keep working.

## Common parameters

| parameter | meaning |
|---|---|
| `metrics=a,b` | which metrics (names from `/metrics`); default: a standard set |
| `since`, `until` | unix seconds, or a duration back from now: `30m`, `24h`, `7d`, `2w` |
| `cause`, `session`, `agent`, `model`, `provider`, `task`, `api`, `error` | equality filters |

## Endpoints

### `GET /api/v1/metrics`
The catalog: every metric's `name`, `unit`, `kind` (sum, ratio, percentile, task) and
`description`, plus the `dimensions` you can group by and the `filters` you can apply.

### `GET /api/v1/summary`
Every metric (or `metrics=`) over the whole ledger or a window.
```json
{"summary": {"requests": 120, "requests_failed": 2, "error_rate": 0.016, "prompt_tokens": 1840000,
             "cached_tokens": 1702000, "cache_hit_rate": 0.925, "avoidable_tokens": 38000,
             "cost_usd": 4.12, "tokens_per_successful_task": 61000.0, "...": "..."}}
```

### `GET /api/v1/breakdown?by=cause`
Metrics per group. `by`: cause, session, agent, model, provider, task, api, status, error,
divergence, day, hour. `order=-avoidable_tokens` sorts (a leading `-` is descending);
`limit=` caps the rows.
```json
{"by": "cause", "rows": [{"cause": "volatile_system", "requests": 14, "avoidable_tokens": 31000, "...": "..."}]}
```

### `GET /api/v1/timeseries?bucket=day`
Like breakdown, by `day` or `hour` (UTC), oldest first.

### `GET /api/v1/requests`
Request rows, newest first: verdict, evidence, token breakdown, timings, status. `limit`
(default 100, max 1000); page back with `before=<next_before>`.
```json
{"requests": [{"id": 812, "request_id": "...", "session_id": "s1", "cause": "none", "evidence": "...",
               "prompt_tokens": 15400, "cached_tokens": 15104, "latency_ms": 1180.4, "status": 200, "...": "..."}],
 "next_before": 812}
```

### `GET /api/v1/requests/{request_id}`
One row, by the `X-Prefixwise-Request-Id` the proxy returned.

### `GET /api/v1/fixes`
Fixes ranked by the avoidable tokens (and dollars) this ledger attributes to their cause, with
whether the proxy can apply them and the evidence behind them.

### `GET /api/v1/findings`
What `prefixwise report` says, as data: a summary, the avoidable-waste causes with a plain
explanation and the fix for each, expected misses, and failures. `since`/`until` narrow it.

### `GET /api/v1/prices`
The dated price table behind the dollar metrics, and the files it came from.

### `POST /api/v1/tasks`
`{"task_id": "fix-bug-1", "success": true}`: record a task outcome, for the per-task metrics.

### `POST /api/v1/ingest`

Turn records from an edge (`prefixwise.transport`; see `docs/specs/hyperscale.md`): `{"records": [<TurnRecord.to_dict()>, ...]}`, at most 1,000 per call. Answers `{"accepted": n, "duplicates": d, "rejected": [{"index": i, "error": "..."}]}`: a request id already ingested is a duplicate (a retried batch adds no rows while the collector runs; the last 200,000 ids are kept in memory), a malformed record is named by index and skipped, and a record the collector fails on becomes a `not_recorded` row. Served only by a collector (`prefixwise collect`, or a proxy with its own ledger); an edge answers 404 with its collector's URL. Needs the collector's `X-Prefixwise-Token` when one is set.

### `POST /api/v1/explain`
`{"previous": <request body>, "current": <request body>, "current_usage": {...}?}`: why would
(or did) the cache miss between two requests? The same classifier as the proxy, offline.

### `POST /api/v1/check/scenario`
`{"name": "<scenario>"}`: under `prefixwise check` only (404 otherwise), names the scenario that
requests without a session id of their own are recorded under from now on. The bundled pytest plugin
posts each test's node id here.

## Raw data

`prefixwise export [--format jsonl|csv] [--since 7d] [--where session=s1] [-o FILE]` writes
every row, oldest first, for spreadsheets, notebooks or a warehouse.
