"""Import a gateway's request log: the same verdicts the live proxy would have given, from logs.

A gateway, an inference provider or a LiteLLM deployment usually logs each request with its body,
the response (with usage) and when it started and finished. Putting a proxy in production to try
Prefixwise is a change to make carefully; replaying a day of those logs is not. This replays every
logged request through the proxy's own pipeline (fingerprinting, conversation lanes, the
classifier, details, pricing), on the log's own clock, with no network:

- each entry becomes a start event (the request is prepared, as when it reached the proxy) and an
  end event (the logged response is recorded, as when it came back), processed in time order, so
  requests that overlapped in the log overlap in the replay and the in-flight checks (a request
  sent before the previous turn's response, a parallel request sent before the cache was written)
  see what they saw live;
- inside the replay the proxy's and the ledger's clock is the log's time, so gaps between turns,
  TTL checks and the ledger's timestamps are the logged ones.

The log is JSON Lines, one request per line. Field names are matched case-insensitively from the
lists below, so most gateway logs import as they are; nothing is required beyond a request body and
a response with usage.

  request body   request, request_body, body, req
  response       response, response_body, resp, output
  start time     start_time, startTime, started_at, timestamp, ts, created_at, time   (unix s/ms or ISO 8601)
  end time       end_time, endTime, completed_at, finished_at   (or start + latency_ms / duration_ms / duration)
  headers        request_headers, headers   (session ids: X-Prefixwise-Session-Id, x-session-id, ...)
  first byte     ttft_ms, time_to_first_token_ms, completion_start_time   (optional; otherwise the end)
  status         status, status_code   (a number; "success"/"ok" is 200, any other text a failure; default 200)
  provider       provider   (otherwise inferred: Anthropic's message shape, else OpenAI's)

Prompt text is read from the log to fingerprint it and is never stored: the ledger holds the same
hashes, counts and verdicts as the proxy's.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import time as _time
from pathlib import Path

REQUEST_KEYS = ("request", "request_body", "body", "req")
RESPONSE_KEYS = ("response", "response_body", "resp", "output")
START_KEYS = ("start_time", "starttime", "started_at", "timestamp", "ts", "created_at", "time")
END_KEYS = ("end_time", "endtime", "completed_at", "finished_at")
LATENCY_MS_KEYS = ("latency_ms", "duration_ms")
LATENCY_S_KEYS = ("duration", "latency")
HEADER_KEYS = ("request_headers", "headers")
TTFT_MS_KEYS = ("ttft_ms", "time_to_first_token_ms")
FIRST_BYTE_KEYS = ("completion_start_time", "completionstarttime", "first_token_time")
STATUS_KEYS = ("status", "status_code")


class _Clock:
    """Stands in for the `time` module inside the proxy and the ledger during a replay: time() is the
    log's current time; everything else (perf_counter, sleep, strftime) is the real module's."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(_time, name)


@contextlib.contextmanager
def _replay_clock(clock: _Clock):
    from . import ledger as ledger_mod
    from . import proxy as proxy_mod
    saved = proxy_mod.time, ledger_mod.time
    proxy_mod.time = ledger_mod.time = clock
    try:
        yield
    finally:
        proxy_mod.time, ledger_mod.time = saved


def _get(entry: dict, keys: tuple):
    lowered = {str(k).lower(): v for k, v in entry.items()}
    for k in keys:
        if k in lowered and lowered[k] is not None:
            return lowered[k]
    return None


def _json(value):
    """A body may be logged as an object or as a JSON string."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def parse_time(value) -> float | None:
    """Unix seconds, unix milliseconds, or ISO 8601 (a trailing Z, or no zone, is UTC)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1e9 if v > 1e17 else v / 1e6 if v > 1e14 else v / 1e3 if v > 1e11 else v  # ns, us, ms, s
    if isinstance(value, str):
        text = value.strip()
        try:
            return parse_time(float(text))
        except ValueError:
            pass
        try:
            when = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        return when.timestamp()
    return None


def _status(value) -> int:
    """A logged status: a number (or numeric text) as is; "success"/"ok" is 200; other text
    ("failure", "error") is a failure; absent is 200."""
    if value is None:
        return 200
    if isinstance(value, bool):
        return 200 if value else 500
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    return 200 if text in ("success", "succeeded", "ok", "completed") else 500


def _provider(entry: dict, body: dict, response: dict) -> str:
    named = str(_get(entry, ("provider",)) or "").lower()
    if named in ("anthropic", "openai"):
        return named
    usage = response.get("usage") or {}
    if response.get("type") == "message" or "input_tokens" in usage and "cache_read_input_tokens" in usage \
            or ("system" in body and "max_tokens" in body and "messages" in body and "input" not in body):
        return "anthropic"
    return "openai"


def read_log(path: str | os.PathLike) -> tuple[list[dict], int]:
    """(entries, lines that could not be used). Each entry: start, end, provider, body, response,
    headers, status, line. A directory reads every *.jsonl / *.json file in it, in name order."""
    p = Path(path)
    files = sorted(f for f in p.iterdir() if f.suffix in (".jsonl", ".json")) if p.is_dir() else [p]
    entries, bad = [], 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                body = _json(_get(raw, REQUEST_KEYS)) if isinstance(raw, dict) else None
                response = _json(_get(raw, RESPONSE_KEYS)) if isinstance(raw, dict) else None
                start = parse_time(_get(raw, START_KEYS)) if isinstance(raw, dict) else None
                if not isinstance(body, dict) or not isinstance(response, dict) or start is None:
                    bad += 1
                    continue
                end = parse_time(_get(raw, END_KEYS))
                if end is None:
                    ms = _get(raw, LATENCY_MS_KEYS)
                    s = _get(raw, LATENCY_S_KEYS)
                    end = start + (float(ms) / 1000.0 if isinstance(ms, (int, float)) else
                                   float(s) if isinstance(s, (int, float)) else 0.0)
                headers = _get(raw, HEADER_KEYS)
                status = _status(_get(raw, STATUS_KEYS))
                ttft = _get(raw, TTFT_MS_KEYS)
                first = parse_time(_get(raw, FIRST_BYTE_KEYS))
                if first is None and isinstance(ttft, (int, float)) and not isinstance(ttft, bool):
                    first = start + float(ttft) / 1000.0
                timed = _get(raw, END_KEYS + LATENCY_MS_KEYS + LATENCY_S_KEYS) is not None
                entries.append({"start": start, "end": max(start, end), "body": body, "response": response,
                                "provider": _provider(raw, body, response),
                                "headers": {str(k).lower(): str(v) for k, v in headers.items()}
                                if isinstance(headers, dict) else {},
                                "status": status,
                                # When the answer began: the cache entry existed from then on.
                                "first": min(max(first, start), max(start, end)) if first is not None else max(start, end),
                                # A log without durations says nothing about overlap: its requests are
                                # replayed one after another, never as in flight together.
                                "timed": timed,
                                "line": f"{f.name}:{n}"})
    return entries, bad


def ended(e: dict) -> float:
    """When the entry's request is recorded: its end, or its start when the log gave no duration
    (such requests are replayed start to end at once)."""
    return e["end"] if e.get("timed", True) else e["start"]


def replay(entries: list[dict], ledger_path: str, *, engine: str | None = None, hmac_key: bytes | None = None,
           prices: str | None = None, count_since: float | None = None) -> dict:
    """Run the entries through the proxy pipeline on the log's clock. Returns import counts, of the
    requests that ended at or after count_since when it is given (all are replayed either way)."""
    from .proxy import PrefixwiseProxy, is_responses_body

    clock = _Clock()
    stats = {"imported": 0, "failed": 0, "first": None, "last": None}
    events = []
    for i, e in enumerate(entries):
        if not e.get("timed", True):  # no durations: start, first byte, end at once, one after another
            events += [(e["start"], 1, i, 0), (e["start"], 1, i, 1), (e["start"], 1, i, 2)]
            continue
        # At equal times a response lands before the next request starts, except a request's own,
        # which comes after its start.
        events.append((e["start"], 1, i, 0))
        events.append((e.get("first", e["end"]), 0 if e.get("first", e["end"]) > e["start"] else 1, i, 1))
        events.append((e["end"], 0 if e["end"] > e["start"] else 1, i, 2))
    events.sort()
    with _replay_clock(clock):
        # Every setting explicit (empty, not None), so no PREFIXWISE_* variable of a live deployment
        # (a remote collector, an engine, KV events, fixes) leaks into the replay.
        proxy = PrefixwiseProxy(ledger_path=ledger_path, mock=False, upstream_url="http://gateway-log.invalid",
                                hmac_key=hmac_key, engine=engine or "", kv_events="", fixes=(), otel="",
                                collector_url="", prices=prices)
        ctxs: dict[int, dict] = {}
        try:
            _run(events, entries, proxy, clock, ctxs, stats, is_responses_body, count_since)
        finally:
            proxy.close()
    return stats


def _run(events, entries, proxy, clock, ctxs, stats, is_responses_body, count_since=None) -> None:
    def counted(e: dict) -> int:
        return int(count_since is None or ended(e) >= count_since)

    for t, _order, i, step in events:
        clock.now = t
        e = entries[i]
        if step == 1:  # the answer began: the request's cache entry exists
            if i in ctxs:
                proxy.response_started(ctxs[i])
            continue
        if step == 0:  # the request reaches the proxy
            body = {k: v for k, v in e["body"].items() if k not in ("stream", "stream_options")}
            api = "responses" if e["provider"] == "openai" and is_responses_body(body) else "chat"
            try:
                ctx = proxy.prepare(e["provider"], body, e["headers"], api=api)
            except Exception:  # noqa: BLE001 - a malformed logged body: skip it, like the proxy's 400
                stats["failed"] += counted(e)
                continue
            ctx["span_start"] = t
            ctxs[i] = ctx
            continue
        ctx = ctxs.pop(i, None)
        if ctx is None:
            continue
        ctx["span_end"] = t
        if e["status"] >= 400:
            proxy.record_failure(ctx, e["status"], e["response"])
            stats["failed"] += counted(e)
            continue
        _, entry = proxy.finalize_safely(ctx, e["status"], e["response"])
        if entry is None:
            stats["failed"] += counted(e)
            continue
        if not counted(e):
            continue
        stats["imported"] += 1
        stats["first"] = e["start"] if stats["first"] is None else min(stats["first"], e["start"])
        stats["last"] = t if stats["last"] is None else max(stats["last"], t)
