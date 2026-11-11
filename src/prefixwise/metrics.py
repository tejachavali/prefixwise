"""Every Prefixwise metric, defined once (docs/DATA.md).

The ledger stores facts per request; this module turns them into metrics. The CLI,
the HTTP API and exports all call `query`; none computes its own numbers. The
catalog is data, so it can be served and read by people and tools alike.

Rules:
- Token, cost and cause metrics count requests that got a verdict (`error IS NULL`).
  Failed requests (upstream error, cut-off stream) count only in requests_failed and
  error_rate.
- A ratio over a group is a ratio of sums, never a mean of per-request ratios.
- Dollars cover priced requests only; `requests_priced` says how many.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

OK = "error IS NULL"


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str
    description: str
    sql: str | None = None  # an aggregate over the group's rows
    ratio: tuple[str, str] | None = None  # numerator, denominator: other metrics' names
    percentile: tuple[str, float] | None = None  # column, q
    task: bool = False  # computed over the group's requests that belong to resolved tasks


CATALOG: tuple[Metric, ...] = (
    Metric("requests", "{request}", "Requests that got a verdict.", f"SUM(CASE WHEN {OK} THEN 1 ELSE 0 END)"),
    Metric("requests_failed", "{request}", "Requests with no verdict: upstream error, cut-off stream, no usage.",
           f"SUM(CASE WHEN NOT ({OK}) THEN 1 ELSE 0 END)"),
    Metric("requests_all", "{request}", "Every request seen.", "COUNT(*)"),
    Metric("error_rate", "1", "requests_failed / requests_all.", ratio=("requests_failed", "requests_all")),
    Metric("prompt_tokens", "{token}", "Prompt (input) tokens, cached ones included.",
           f"COALESCE(SUM(CASE WHEN {OK} THEN prompt_tokens END), 0)"),
    Metric("cached_tokens", "{token}", "Prompt tokens read from the cache.",
           f"COALESCE(SUM(CASE WHEN {OK} THEN cached_tokens END), 0)"),
    Metric("uncached_tokens", "{token}", "Prompt tokens neither read from nor written to the cache. "
           "null when any row in the group predates schema 3 (a partial sum would mislead).",
           f"CASE WHEN SUM(CASE WHEN {OK} AND uncached_tokens IS NULL THEN 1 ELSE 0 END) > 0 THEN NULL "
           f"ELSE COALESCE(SUM(CASE WHEN {OK} THEN uncached_tokens END), 0) END"),
    Metric("cache_write_tokens", "{token}", "Prompt tokens written to the cache. null when any row in "
           "the group predates schema 3.",
           f"CASE WHEN SUM(CASE WHEN {OK} AND cache_write_tokens IS NULL THEN 1 ELSE 0 END) > 0 THEN NULL "
           f"ELSE COALESCE(SUM(CASE WHEN {OK} THEN cache_write_tokens END), 0) END"),
    Metric("completion_tokens", "{token}", "Output tokens.",
           f"COALESCE(SUM(CASE WHEN {OK} THEN completion_tokens END), 0)"),
    Metric("cache_hit_rate", "1", "cached_tokens / prompt_tokens.", ratio=("cached_tokens", "prompt_tokens")),
    Metric("avoidable_tokens", "{token}", "Prompt tokens that missed the cache for a reason that was waste: "
           "what a fix would have saved.", f"COALESCE(SUM(CASE WHEN {OK} THEN avoidable_tokens END), 0)"),
    Metric("avoidable_share", "1", "avoidable_tokens / prompt_tokens.", ratio=("avoidable_tokens", "prompt_tokens")),
    Metric("requests_wasteful", "{request}", "Requests whose miss was waste (legit = false).",
           f"SUM(CASE WHEN {OK} AND legit = 0 THEN 1 ELSE 0 END)"),
    Metric("requests_priced", "{request}", "Requests with a known price; the dollar metrics cover these only.",
           f"SUM(CASE WHEN {OK} AND cost IS NOT NULL THEN 1 ELSE 0 END)"),
    Metric("cost_usd", "USD", "Dollars for priced requests.", f"SUM(CASE WHEN {OK} THEN cost END)"),
    Metric("cost_cache_read_usd", "USD", "Dollars for re-reading prompt tokens already in the cache, at the "
           "cache-read rate: the floor a long, well-cached session pays every turn (requests with a cost split).",
           f"SUM(CASE WHEN {OK} THEN cost_cache_read END)"),
    Metric("cost_input_usd", "USD", "Dollars for prompt tokens neither read from nor written to the cache "
           "(requests with a cost split).", f"SUM(CASE WHEN {OK} THEN cost_input END)"),
    Metric("cost_cache_write_usd", "USD", "Dollars for prompt tokens written to the cache "
           "(requests with a cost split).", f"SUM(CASE WHEN {OK} THEN cost_cache_write END)"),
    Metric("cost_output_usd", "USD", "Dollars for output tokens (requests with a cost split).",
           f"SUM(CASE WHEN {OK} THEN cost_output END)"),
    Metric("requests_cost_split", "{request}", "Priced requests whose cost is split by part (recorded "
           "by 0.11.6 or later).", f"SUM(CASE WHEN {OK} AND cost_cache_read IS NOT NULL THEN 1 ELSE 0 END)"),
    Metric("avoidable_cost_usd", "USD", "What the avoidable tokens cost as uncached input rather than "
           "cache reads (priced requests).", f"SUM(CASE WHEN {OK} THEN avoidable_cost END)"),
    Metric("latency_ms_p50", "ms", "Median upstream latency of requests with a verdict (streams: to the "
           "last byte delivered).", percentile=("latency_ms", 0.50)),
    Metric("latency_ms_p95", "ms", "95th percentile upstream latency of requests with a verdict.",
           percentile=("latency_ms", 0.95)),
    Metric("overhead_ms_p50", "ms", "Median time Prefixwise itself added.", percentile=("overhead_ms", 0.50)),
    Metric("overhead_ms_p95", "ms", "95th percentile time Prefixwise itself added.",
           percentile=("overhead_ms", 0.95)),
    Metric("native_compared", "{request}", "Requests where the provider named its own miss reason "
           "(not counting unavailable or baseline-not-found answers, which are no verdict).",
           f"SUM(CASE WHEN {OK} AND native_reason IS NOT NULL AND native_reason NOT IN "
           "('unavailable', 'previous_message_not_found', 'comparison_response_not_found') THEN 1 ELSE 0 END)"),
    Metric("tasks_resolved", "{task}", "Tasks with a recorded outcome that have requests in the group.", task=True),
    Metric("tasks_successful", "{task}", "Of those, the successful ones.", task=True),
    Metric("tokens_per_successful_task", "{token}", "Prompt + completion tokens on resolved tasks (failed tasks "
           "included) / successful tasks. The number that matters.", task=True),
    Metric("cost_per_successful_task_usd", "USD", "Dollars on resolved tasks / successful tasks; only when every "
           "such request is priced (a partial sum would understate it).", task=True),
)

BY_NAME = {m.name: m for m in CATALOG}
DEFAULT = ("requests", "requests_failed", "prompt_tokens", "cached_tokens", "cache_hit_rate",
           "avoidable_tokens", "avoidable_share", "requests_wasteful", "requests_priced", "cost_usd",
           "avoidable_cost_usd")

# Groupings. Time buckets use the row's recorded time (UTC); rows from before
# 0.3.1 have none and fall in a null bucket.
DIMENSIONS = {
    "cause": "cause", "session": "session_id", "agent": "agent_id", "model": "model",
    "provider": "provider", "task": "task_id", "api": "api", "status": "COALESCE(status, 200)", "error": "error",
    "divergence": "divergence_role", "detail": "detail",
    "day": "strftime('%Y-%m-%d', recorded_at, 'unixepoch')",
    "hour": "strftime('%Y-%m-%dT%H:00Z', recorded_at, 'unixepoch')",
}
FILTERS = {"cause", "session", "agent", "model", "provider", "task", "api", "error", "status", "detail"}


def catalog() -> list[dict]:
    out = []
    for m in CATALOG:
        d = {k: v for k, v in asdict(m).items() if k in ("name", "unit", "description")}
        d["kind"] = "ratio" if m.ratio else "percentile" if m.percentile else "task" if m.task else "sum"
        out.append(d)
    return out


def _where(filters: dict | None, since: float | None, until: float | None) -> tuple[str, list]:
    clauses, args = [], []
    for k, v in (filters or {}).items():
        if k not in FILTERS:
            raise ValueError(f"cannot filter on {k!r}; filters: {sorted(FILTERS)}")
        if k == "status":
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ValueError(f"status must be an integer, not {v!r}") from None
        clauses.append(f"{DIMENSIONS[k]} = ?")
        args.append(v)
    if since is not None:
        clauses.append("recorded_at >= ?")
        args.append(since)
    if until is not None:
        clauses.append("recorded_at < ?")
        args.append(until)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", args


def _interpolate(got: dict | None, q: float) -> float | None:
    """The q-quantile from the two order statistics around it (rank (n-1)q)."""
    if not got:
        return None
    pos = (got["n"] - 1) * q
    lo = int(pos) + 1
    a = got.get(lo)
    b = got.get(min(lo + 1, got["n"]), a)
    return None if a is None else a + (b - a) * (pos - int(pos))


def query(ledger, metrics=None, by: str | None = None, filters: dict | None = None,
          since: float | None = None, until: float | None = None, order: str | None = None,
          limit: int | None = None) -> list[dict]:
    """Metrics over the ledger, optionally grouped by one dimension.

    Returns one row per group (one row in total when `by` is None), each with the
    group's key and the requested metrics. Unknown names raise ValueError."""
    names = list(metrics or DEFAULT)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; see the catalog")
    if by is not None and by not in DIMENSIONS:
        raise ValueError(f"cannot group by {by!r}; dimensions: {sorted(DIMENSIONS)}")
    # Ratios need their parts; compute those, report what was asked.
    needed: list[str] = []
    for n in names:
        m = BY_NAME[n]
        for part in (m.ratio or ()):
            if part not in needed:
                needed.append(part)
        if n not in needed:
            needed.append(n)
    sums = [n for n in needed if BY_NAME[n].sql]
    where, args = _where(filters, since, until)
    key = DIMENSIONS[by] if by else "NULL"
    sql = (f"SELECT {key} AS k, COUNT(*)" + "".join(", " + BY_NAME[n].sql for n in sums)
           + f" FROM entries{where} GROUP BY k")
    with ledger.reader() as conn:
        rows = {r[0]: dict(zip(sums, r[2:])) for r in conn.execute(sql, args).fetchall()}
        task_rows: dict = {}
        if any(BY_NAME[n].task for n in needed):
            # A task belongs to a group if it has any request there; its tokens and
            # cost come from the requests that got a verdict.
            for k, tok, ok, resolved, n_req, priced, cost in conn.execute(
                    f"""SELECT e.k, SUM(CASE WHEN e.ok THEN e.prompt_tokens + COALESCE(e.completion_tokens, 0) END),
                               COUNT(DISTINCT CASE WHEN t.success = 1 THEN t.task_id END),
                               COUNT(DISTINCT t.task_id), SUM(e.ok),
                               SUM(CASE WHEN e.ok AND e.cost IS NOT NULL THEN 1 ELSE 0 END),
                               SUM(CASE WHEN e.ok THEN e.cost END)
                        FROM (SELECT {key} AS k, prompt_tokens, completion_tokens, cost, task_id,
                                     ({OK}) AS ok FROM entries{where}) e
                        JOIN tasks t ON e.task_id = t.task_id GROUP BY e.k""", args):
                task_rows[k] = {"tasks_resolved": resolved, "tasks_successful": ok,
                                "tokens_per_successful_task": ((tok or 0) / ok) if ok else None,
                                "cost_per_successful_task_usd": ((cost or 0) / ok) if ok and priced == n_req
                                else None}
        samples: dict = {}
        for col, q in {BY_NAME[n].percentile for n in needed if BY_NAME[n].percentile}:
            # Exact, interpolated percentiles computed inside SQLite (window functions):
            # two rows per group come back, never the whole column.
            cond = f"{' AND' if where else ' WHERE'} {OK} AND {col} IS NOT NULL"
            for k, rn, n, v in conn.execute(
                    f"""SELECT k, rn, n, v FROM (
                          SELECT {key} AS k, {col} AS v,
                                 ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY {col}) AS rn,
                                 COUNT(*) OVER (PARTITION BY {key}) AS n
                          FROM entries{where}{cond})
                        WHERE rn = CAST((n - 1) * ? AS INTEGER) + 1
                           OR rn = MIN(CAST((n - 1) * ? AS INTEGER) + 2, n)""", (*args, q, q)):
                samples.setdefault((k, col, q), {"n": n})[rn] = v
    out = []
    for k, vals in rows.items():
        row = {by: k} if by else {}
        for n in names:
            m = BY_NAME[n]
            if m.ratio:
                num, den = vals.get(m.ratio[0]), vals.get(m.ratio[1])
                row[n] = (num / den) if num is not None and den else None
            elif m.percentile:
                row[n] = _interpolate(samples.get((k, *m.percentile)), m.percentile[1])
            elif m.task:
                row[n] = task_rows.get(k, {}).get(n, 0 if n.startswith("tasks_") else None)
            else:
                row[n] = vals.get(n)
        out.append(row)
    if order:
        desc, field = order.startswith("-"), order.lstrip("-")
        present = [r for r in out if r.get(field) is not None]
        present.sort(key=lambda r: r[field], reverse=desc)
        out = present + [r for r in out if r.get(field) is None]  # missing values last
    elif by in ("day", "hour"):
        out.sort(key=lambda r: (r[by] is None, r[by] or ""))
    return out[:limit] if limit else out


def summary(ledger, **kw) -> dict:
    """Every metric in the catalog for the whole ledger (or a filter / time window)."""
    rows = query(ledger, [m.name for m in CATALOG], **kw)
    return rows[0] if rows else {m.name: None for m in CATALOG}
