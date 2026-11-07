"""The Prefixwise HTTP API, version 1 (docs/API.md).

Every number comes from prefixwise.metrics, the same definitions the CLI uses.
Served by `prefixwise serve` under /api/v1, behind the same X-Prefixwise-Token as
the proxy. Responses are JSON; errors are {"error": {"message": ...}} with a 4xx.

    GET  /api/v1/metrics                       the catalog: metrics, dimensions, filters
    GET  /api/v1/summary                       every metric, whole ledger or a window
    GET  /api/v1/breakdown?by=cause            metrics per group
    GET  /api/v1/timeseries?bucket=day         metrics per day or hour
    GET  /api/v1/requests                      request rows, newest first, paginated
    GET  /api/v1/requests/{request_id}         one request
    GET  /api/v1/fixes                         fixes ranked by what they would save
    GET  /api/v1/prices                        the price table behind the dollars
    POST /api/v1/tasks                         {"task_id", "success"}: record an outcome
    POST /api/v1/explain                       {"previous", "current"}: why would it miss?

Common query parameters: `metrics=a,b` (default: a standard set), `since`/`until`
(unix seconds, or a duration back from now such as 24h or 7d), and equality
filters on cause, session, agent, model, provider, task, api, error.
"""

from __future__ import annotations

import time
from urllib.parse import parse_qs

from . import metrics
from .fixes import recommend

VERSION = "v1"
PREFIX = "/api/v1"
MAX_PAGE = 1000


class ApiError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _time(value: str | None, name: str) -> float | None:
    """Unix seconds, or a duration back from now (30m, 24h, 7d, 2w)."""
    if not value:
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    v = value.strip().lower()
    try:
        if v[-1] in units:
            return time.time() - float(v[:-1]) * units[v[-1]]
        return float(v)
    except ValueError:
        raise ApiError(400, f"{name} must be unix seconds or a duration like 24h, not {value!r}") from None


def _params(query: str) -> dict[str, str]:
    return {k: v[-1] for k, v in parse_qs(query, keep_blank_values=True).items()}


def _common(p: dict) -> dict:
    names = [n for n in p.get("metrics", "").split(",") if n] or None
    filters = {k: p[k] for k in metrics.FILTERS if k in p}
    return {"metrics": names, "filters": filters, "since": _time(p.get("since"), "since"),
            "until": _time(p.get("until"), "until")}


def _limit(p: dict, default: int) -> int:
    try:
        n = int(p.get("limit", default))
    except ValueError:
        raise ApiError(400, "limit must be an integer") from None
    return max(1, min(n, MAX_PAGE))


def get(proxy, path: str, query: str) -> tuple[int, dict]:
    """Answer a GET under /api/v1. Returns (status, JSON payload)."""
    led = proxy.ledger
    p = _params(query)
    sub = path[len(PREFIX):].rstrip("/") or "/"
    try:
        if sub == "/metrics":
            return 200, {"version": VERSION, "metrics": metrics.catalog(),
                         "dimensions": sorted(metrics.DIMENSIONS), "filters": sorted(metrics.FILTERS)}
        if sub == "/summary":
            c = _common(p)
            names = c.pop("metrics") or [m.name for m in metrics.CATALOG]
            rows = metrics.query(led, names, **c)
            return 200, {"summary": rows[0] if rows else {n: None for n in names}}
        if sub in ("/breakdown", "/timeseries"):
            by = p.get("by", "cause") if sub == "/breakdown" else p.get("bucket", "day")
            if sub == "/timeseries" and by not in ("day", "hour"):
                raise ApiError(400, "bucket must be day or hour")
            c = _common(p)
            rows = metrics.query(led, c.pop("metrics"), by=by, order=p.get("order"),
                                 limit=_limit(p, MAX_PAGE), **c)
            return 200, {"by": by, "rows": rows}
        if sub == "/requests":
            return 200, _requests(led, p)
        if sub.startswith("/requests/"):
            row = led.get_request(sub[len("/requests/"):])
            if row is None:
                return 404, _err("unknown request id (or not recorded yet)")
            return 200, {"request": row}
        if sub == "/fixes":
            return 200, recommend(led)
        if sub == "/prices":
            return 200, {"as_of": proxy.prices.as_of, "sources": proxy.prices.sources,
                         "models": proxy.prices.models}
        return 404, _err(f"no such endpoint: {path}")
    except ApiError as exc:
        return exc.status, _err(str(exc))
    except ValueError as exc:  # unknown metric / dimension / filter
        return 400, _err(str(exc))


def _requests(led, p: dict) -> dict:
    """Request rows, newest first. `before` (a row id from next_before) pages back."""
    filters = {k: p[k] for k in metrics.FILTERS if k in p}
    where, args = metrics._where(filters, _time(p.get("since"), "since"), _time(p.get("until"), "until"))
    if "before" in p:
        try:
            before = int(p["before"])
        except ValueError:
            raise ApiError(400, "before must be an integer row id") from None
        where += (" AND" if where else " WHERE") + " id < ?"
        args.append(before)
    limit = _limit(p, 100)
    with led._lock:
        cur = led.conn.execute(f"SELECT * FROM entries{where} ORDER BY id DESC LIMIT ?", (*args, limit + 1))
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
    more = len(rows) > limit
    rows = rows[:limit]
    return {"requests": rows, "next_before": rows[-1]["id"] if more and rows else None}


def post(proxy, path: str, body) -> tuple[int, dict]:
    """Answer a POST under /api/v1."""
    sub = path[len(PREFIX):].rstrip("/")
    if not isinstance(body, dict):
        return 400, _err("body must be a JSON object")
    if sub == "/tasks":
        task_id, success = body.get("task_id"), body.get("success")
        if not isinstance(task_id, str) or not task_id or not isinstance(success, bool):
            return 400, _err('expected {"task_id": "<id>", "success": true|false}')
        proxy.ledger.set_task_outcome(task_id, success)
        return 200, {"ok": True, "task_id": task_id, "success": success}
    if sub == "/explain":
        from .explain import Turn, explain_pair

        prev, curr = body.get("previous"), body.get("current")
        if not isinstance(curr, dict) or (prev is not None and not isinstance(prev, dict)):
            return 400, _err('expected {"previous": <request body>, "current": <request body>}')
        usage = body.get("current_usage")
        result = explain_pair(Turn(body=prev) if prev else None,
                              Turn(body=curr, usage=usage if isinstance(usage, dict) else None),
                              provider=body.get("provider"))
        return 200, {"explanation": result.to_dict()}
    return 404, _err(f"no such endpoint: {path}")


def _err(message: str) -> dict:
    return {"error": {"message": message}}
