# Prometheus and Grafana

`prefixwise serve` exposes `GET /metrics` in the Prometheus text format: every request it
records, by provider, model and miss cause. Point Prometheus at it, import the dashboard, load
the alert rules, and the cache's health sits next to the rest of your services.

## Scrape it

```yaml
# prometheus.yml
scrape_configs:
  - job_name: prefixwise
    static_configs:
      - targets: ["127.0.0.1:8080"]      # where prefixwise serve listens
    # With PREFIXWISE_AUTH_TOKEN set, /metrics needs it too:
    # authorization:
    #   credentials: <your token>        # sent as Authorization: Bearer
rule_files:
  - prefixwise-rules.yml                 # deploy/prometheus/prefixwise-rules.yml
```

Several Prefixwise processes (one per replica or team) are scraped as separate targets; the
queries below sum across them.

## What it exports

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `prefixwise_requests_total` | counter | provider, model, cause | Requests with a verdict; `cause="none"` is a cache hit |
| `prefixwise_prompt_tokens_total` | counter | provider, model, cause | Prompt tokens |
| `prefixwise_cached_tokens_total` | counter | provider, model, cause | Prompt tokens read from the cache |
| `prefixwise_cache_write_tokens_total` | counter | provider, model, cause | Prompt tokens written to the cache, where the provider reports it |
| `prefixwise_avoidable_tokens_total` | counter | provider, model, cause | Tokens that could have been cache reads (waste causes only) |
| `prefixwise_completion_tokens_total` | counter | provider, model, cause | Output tokens |
| `prefixwise_cost_usd_total` | counter | provider, model, cause | Dollars, priced models only (`prefixwise prices`) |
| `prefixwise_avoidable_cost_usd_total` | counter | provider, model, cause | What the avoidable tokens cost as uncached input rather than cache reads |
| `prefixwise_requests_failed_total` | counter | provider, model, error | Upstream errors, cut-off streams, client disconnects (no verdict) |
| `prefixwise_upstream_latency_seconds` | histogram | provider | Upstream time (streams: to the last byte) |
| `prefixwise_overhead_seconds` | histogram | | Time Prefixwise itself adds |
| `prefixwise_kv_events_connected`, `prefixwise_kv_events_*_total` | gauge, counters | | With `--kv-events`: the engine's event stream, lost messages, decode errors, restarts, overflows |
| `prefixwise_info` | gauge | version, engine | Always 1 |

Counters cover the process's lifetime; Prometheus keeps the history and `rate()` handles
restarts. Labels are bounded on purpose: session, agent and task ids would create a series per
conversation, so they stay in the ledger and `/api/v1` ([API.md](API.md)). The causes are the
19 in the README's "What it tells you".

## Queries

```promql
# Share of prompt tokens served from cache
sum(rate(prefixwise_cached_tokens_total[5m])) / sum(rate(prefixwise_prompt_tokens_total[5m]))

# Avoidable share, and where it comes from
sum(rate(prefixwise_avoidable_tokens_total[5m])) / sum(rate(prefixwise_prompt_tokens_total[5m]))
sum by (cause) (rate(prefixwise_avoidable_tokens_total[1h]))

# Dollars per hour lost to avoidable misses, by cause
sum by (cause) (rate(prefixwise_avoidable_cost_usd_total[1h])) * 3600

# Prefixwise's own overhead, p95
histogram_quantile(0.95, sum by (le) (rate(prefixwise_overhead_seconds_bucket[5m])))
```

## Dashboard and alerts

- **Grafana:** Dashboards → New → Import → upload [`deploy/grafana/prefixwise.json`](../deploy/grafana/prefixwise.json)
  and pick your Prometheus data source. Panels: served from cache (overall and per model),
  avoidable share and dollars per hour, avoidable tokens and dollars by cause, misses by cause,
  spend by model, failures by error, Prefixwise's overhead, upstream latency, and the KV-event
  stream. A `model` variable filters everything.
- **Alerts:** [`deploy/prometheus/prefixwise-rules.yml`](../deploy/prometheus/prefixwise-rules.yml):
  hit rate below 80% for 15 minutes, avoidable share above 10% for 30 minutes, failures above
  5%, and a lost or disconnected KV-event stream. The thresholds are starting points; agent
  workloads usually serve over 90% of prompt tokens from cache.

When an alert fires, `prefixwise report --since 1h` (or `GET /api/v1/findings?since=1h`) says
which causes grew and what to change.
