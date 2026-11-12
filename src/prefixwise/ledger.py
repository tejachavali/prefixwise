"""Efficiency ledger (SQLite).

Headline metric: tokens per successful task, defined as all tokens
(prompt + completion) spent on resolved tasks, failed ones included,
divided by the number of successful tasks. A request belongs to a task
through task_id; a task is resolved once its outcome is recorded
(set_task_outcome). Requests with no task_id, and tasks still in
progress, are left out rather than guessed. Cost is a secondary column.
Avoidable tokens are priced from versioned provider rate tables; if a
rate is unknown, cost is None, never invented.

Predicted vs realised savings: a fix toggle ships as "recommended" on
predicted savings and graduates to "proven" only after realised savings
are measured with task success held flat (realised >= 50% of predicted
over 50 requests, success flat).

Reuse-adjusted saving is a *model*, not a verified fact:
k = write multiplier / (1 - read multiplier), e.g. Anthropic 5-min
1.25/0.9 ~= 1.4 reads, OpenAI 12h 2.5/0.9 ~= 2.8. The read multiplier
is model-dependent, so 0.9 is not universal; label it as a model.
"""

from __future__ import annotations

import time

import json
import sqlite3
import sys
from dataclasses import astuple, dataclass, fields

SCHEMA_VERSION = 7


@dataclass
class LedgerEntry:
    session_id: str
    agent_id: str
    provider: str
    model: str
    cause: str
    prompt_tokens: int
    cached_tokens: int
    avoidable_tokens: int
    avoidable_cost: float | None
    task_success: bool | None = None  # if set with task_id, records the task outcome
    legit: bool | None = None
    native_reason: str | None = None
    classifier_version: str = "2.0"
    task_id: str | None = None
    request_id: str | None = None
    completion_tokens: int = 0
    fixes: str | None = None  # comma-separated proxy fixes that rewrote this request
    cost: float | None = None  # USD for the whole request; None = a rate was unknown
    price_as_of: str | None = None  # date of the price table that priced it
    # Schema 3 (docs/DATA.md): facts observed for every request, failures included.
    status: int = 200  # upstream HTTP status; >= 400 rows have no verdict and no tokens
    error: str | None = None  # short error class for failed requests
    api: str | None = None  # chat | responses | messages
    stream: bool | None = None
    uncached_tokens: int | None = None  # prompt tokens neither read from nor written to the cache
    cache_write_tokens: int | None = None  # prompt tokens written to the cache (all TTLs)
    latency_ms: float | None = None  # upstream time (to the last byte for streams)
    overhead_ms: float | None = None  # Prefixwise's own processing time
    gap_seconds: float | None = None  # since this session's previous turn on this model
    evidence: str | None = None  # why the verdict was given (Prefixwise's words, never prompt text)
    secondary_causes: str | None = None  # comma-separated
    divergence_role: str | None = None  # where the prompt first changed: system, tools, messages, ...
    divergence_index: int | None = None  # block index of that first change
    # Schema 4: the why behind the cause, where the evidence allows (prefixwise.details):
    # ttl_shortened, lookback_exceeded, markers_ignored, sent_before_cache_written,
    # shared_prefix_not_reused, thinking_stripped, cache_key_changed, host_restored,
    # output_rerendered, volatile_prefix_at_start, session_header_dropped.
    detail: str | None = None
    # Schema 5: this request's own cache lifetime: what its write was granted or what it asked for
    # (Anthropic 300 or 3600; an X-Prefixwise-Cache-Ttl header; None when unknown, as on OpenAI).
    cache_ttl_seconds: float | None = None
    # Schema 6: a miss that read nothing, right after the previous turn wrote the prefix to the
    # cache: those tokens were written (at the write rate) and never read. The surcharge paid
    # for writing them above sending them uncached; None when unpriced or not applicable.
    unread_write_tokens: int | None = None
    unread_write_cost: float | None = None
    # Schema 7: where `cost` went (USD, summing to it; None on unpriced and older rows). cache_read is the
    # re-read of what was already cached: on a well-cached long session, most of the bill.
    cost_cache_read: float | None = None
    cost_input: float | None = None
    cost_cache_write: float | None = None
    cost_output: float | None = None
    schema_version: int = SCHEMA_VERSION


# Causes that replaced the cached prefix (or let it expire): the previous turn's write is of no
# further use. A retry, a routing miss or an eviction leave the entry as it was, so they are not
# here: a later turn may still read it.
_WRITE_REPLACED = frozenset({"volatile_system", "system_edit", "tool_reserialize", "tool_catalog_change",
                             "history_edit", "tool_result_mutation", "compaction_rewrite", "ttl_expired",
                             "param_rendered"})


def unread_write(cause: str, cached_tokens: int, prev_write_5m: int, prev_write_1h: int,
                 prev_cache_write: int, prev_cached_tokens: int = 0) -> tuple[int, int, int]:
    """(5-minute, 1-hour, single-rate) tokens the previous turn wrote that this turn, a miss that
    replaced the prefix, left unread; all zero when not applicable.

    The previous turn wrote after what it read: its new entry spans [prev_cached_tokens,
    prev_cached_tokens + write). This turn's read, from the front of the prompt, covers the first
    `cached_tokens` of it; whatever it reads beyond `prev_cached_tokens` came from that write, and
    the rest of the write was paid for and never read. A read that stops inside the stable
    system+tools prefix (an agent that trims history every turn while its system prompt still
    hits) leaves the whole write unread: before, any read at all zeroed this (Reddit,
    Specialist-Can2889: 535k tokens written, 334k read, in one session that trims history).
    Where the two cache lifetimes both wrote, the 1-hour part comes first in the prompt (Anthropic
    requires 1-hour breakpoints before 5-minute ones), so a partial read consumes it first."""
    if cause not in _WRITE_REPLACED:
        return 0, 0, 0
    total = (prev_write_5m + prev_write_1h) or prev_cache_write
    if total <= 0:
        return 0, 0, 0
    read_from_write = max(0, cached_tokens - max(0, prev_cached_tokens))
    unread = max(0, total - read_from_write)
    if unread == 0:
        return 0, 0, 0
    if prev_write_5m or prev_write_1h:
        read_1h = min(prev_write_1h, read_from_write)
        read_5m = min(prev_write_5m, read_from_write - read_1h)
        return prev_write_5m - read_5m, prev_write_1h - read_1h, 0
    return 0, 0, unread


def avoidable_tokens(prompt_tokens: int, cached_tokens: int, *,
                     first_write_tokens: int = 0, legitimate_change_tokens: int = 0) -> int:
    return max(0, prompt_tokens - cached_tokens - first_write_tokens - legitimate_change_tokens)


def avoidable_for_turn(legit: bool | None, prompt_tokens: int, cached_tokens: int,
                       prev_prompt_tokens: int | None, prev_held_tokens: int | None = None) -> int:
    """Avoidable tokens for one classified turn.

    Only waste counts (legit is False). Legitimate changes (model
    switch, catalog change, compaction, first write) still cost tokens
    but are not waste; undetermined legitimacy (None: NONE, routing,
    unknown, ...) is not counted either, never a guess.

    For waste, avoidable is the prior prefix that could have been
    reused but was not: min(prev, curr) prompt tokens minus cache_read.
    Tokens appended after the prior prefix are new content, never
    avoidable. When what the previous turn left in the cache is known
    (`prev_held_tokens`: what it read plus what it wrote, which on
    Anthropic ends at its last breakpoint), no more than that could
    have been reused: the uncached tail after the breakpoint is not
    this turn's waste.
    """
    if legit is not False or not prev_prompt_tokens:
        # Without the previous turn's usage, how much of it was reusable is not known: claiming this
        # whole prompt would count appended content as waste.
        return 0
    reusable = min(prev_prompt_tokens, prompt_tokens)
    if prev_held_tokens:
        reusable = min(reusable, prev_held_tokens)
    return max(0, reusable - cached_tokens)


def reuse_adjusted_reads(write_multiplier: float, read_multiplier: float) -> float:
    """Model (not a verified fact): reads needed for a fix to pay.

    k = write_multiplier / (1 - read_multiplier).
    """
    if read_multiplier >= 1:
        raise ValueError("read_multiplier must be < 1")
    return write_multiplier / (1 - read_multiplier)


# Columns added after the first schema; added in place to older ledgers.
_MIGRATIONS = {"task_id": "TEXT", "request_id": "TEXT", "completion_tokens": "INTEGER",
               "fixes": "TEXT", "cost": "REAL", "price_as_of": "TEXT",
               "recorded_at": "REAL",  # unix time; NULL on rows from before 0.3.1
               # schema 3; NULL on older rows (status NULL means it succeeded)
               "status": "INTEGER", "error": "TEXT", "api": "TEXT", "stream": "INTEGER",
               "uncached_tokens": "INTEGER", "cache_write_tokens": "INTEGER",
               "latency_ms": "REAL", "overhead_ms": "REAL", "gap_seconds": "REAL",
               "evidence": "TEXT", "secondary_causes": "TEXT", "divergence_role": "TEXT",
               "divergence_index": "INTEGER", "schema_version": "INTEGER",
               "detail": "TEXT",  # schema 4
               "cache_ttl_seconds": "REAL",  # schema 5
               "unread_write_tokens": "INTEGER", "unread_write_cost": "REAL",  # schema 6
               "cost_cache_read": "REAL", "cost_input": "REAL", "cost_cache_write": "REAL",
               "cost_output": "REAL"}  # schema 7

_COLUMNS = [f.name for f in fields(LedgerEntry)]


class Ledger:
    def __init__(self, path: str = ":memory:"):
        # check_same_thread=False: the proxy serves on threads; every
        # read and write goes through self._lock instead.
        import threading

        self._lock = threading.Lock()
        # The ledger holds no prompt text, but its ids and counts are still
        # the user's business: create it owner-only (0600).
        if path != ":memory:":
            import os

            if not os.path.exists(path):
                os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        # timeout: the CLI and a running proxy may share one ledger file.
        self.path = path
        self.sinks: list = []  # copies of every row (prefixwise.sinks); never on the write's critical path
        self._sink_warned: set = set()
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        with self._lock:
            if path != ":memory:":
                # WAL lets `prefixwise report` read while the proxy writes.
                self.conn.execute("PRAGMA journal_mode=WAL")
                # WAL with NORMAL: a commit no longer waits for the disk; a power loss can lose
                # the last few rows, never corrupt the ledger (SQLite's documented guarantee).
                self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT, agent_id TEXT, provider TEXT, model TEXT,
                    cause TEXT, legit INTEGER, prompt_tokens INTEGER,
                    cached_tokens INTEGER, avoidable_tokens INTEGER,
                    avoidable_cost REAL, task_success INTEGER,
                    native_reason TEXT, classifier_version TEXT
                )"""
            )
            cols = {row[1] for row in self.conn.execute("PRAGMA table_info(entries)")}
            for col, typ in _MIGRATIONS.items():
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE entries ADD COLUMN {col} {typ}")
            self.conn.execute("CREATE INDEX IF NOT EXISTS entries_task ON entries(task_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS entries_request ON entries(request_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS entries_session ON entries(session_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS entries_time ON entries(recorded_at)")
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, success INTEGER NOT NULL, recorded_at REAL
                )"""
            )
            self.conn.commit()

    def add_sink(self, sink) -> None:
        """A LedgerSink (record(entry, at), close(timeout)) that gets a copy of every row once it is
        committed here. A sink that raises is reported once and skipped; the ledger is unaffected."""
        self.sinks.append(sink)

    def close_sinks(self, timeout: float = 5.0) -> None:
        for sink in self.sinks:
            try:
                sink.close(timeout)
            except Exception:  # noqa: BLE001
                pass

    def record(self, entry: LedgerEntry, at: float | None = None) -> int:
        """Store one row. `at` is when it happened (default now); imports pass the log's time."""
        values = [int(v) if isinstance(v, bool) else v for v in astuple(entry)]
        at = time.time() if at is None else at
        with self._lock:
            cur = self.conn.execute(
                f"INSERT INTO entries ({', '.join(_COLUMNS)}, recorded_at) "
                f"VALUES ({', '.join('?' * (len(_COLUMNS) + 1))})", (*values, at))
            if entry.task_id and entry.task_success is not None:
                self._set_task_outcome_locked(entry.task_id, entry.task_success)
            self.conn.commit()
            rowid = cur.lastrowid
        for sink in self.sinks:  # after the commit, outside the lock
            try:
                sink.record(entry, at)
            except Exception as exc:  # noqa: BLE001 - a warehouse must never cost a row here
                key = getattr(sink, "name", type(sink).__name__)
                if key not in self._sink_warned:
                    self._sink_warned.add(key)
                    print(f"prefixwise: sink {key} failed ({type(exc).__name__}: {exc}); the ledger is unaffected",
                          file=sys.stderr, flush=True)
        return rowid

    def prune(self, older_than_seconds: float, undated: bool = False) -> int:
        """Delete entries recorded more than older_than_seconds ago (and, with
        undated, rows from before timestamps were recorded). Tasks no remaining
        entry refers to go too. Returns the number of entries deleted."""
        with self._lock:
            cutoff = time.time() - older_than_seconds
            where = "recorded_at < ?" + (" OR recorded_at IS NULL" if undated else "")
            n = self.conn.execute(f"DELETE FROM entries WHERE {where}", (cutoff,)).rowcount
            self.conn.execute("DELETE FROM tasks WHERE task_id NOT IN "
                              "(SELECT task_id FROM entries WHERE task_id IS NOT NULL)")
            self.conn.commit()
            if n:
                self.conn.execute("VACUUM")
            return n

    def _set_task_outcome_locked(self, task_id: str, success: bool) -> None:
        import time

        self.conn.execute(
            """INSERT INTO tasks (task_id, success, recorded_at) VALUES (?,?,?)
               ON CONFLICT(task_id) DO UPDATE SET success=excluded.success,
                                                 recorded_at=excluded.recorded_at""",
            (task_id, int(success), time.time()))

    def set_task_outcome(self, task_id: str, success: bool) -> None:
        """Resolve a task. Re-recording an outcome overwrites it."""
        with self._lock:
            self._set_task_outcome_locked(task_id, success)
            self.conn.commit()

    def task_counts(self) -> dict:
        """Resolved and successful tasks. Defined in prefixwise.metrics."""
        from . import metrics

        m = metrics.query(self, ["tasks_resolved", "tasks_successful"])
        row = m[0] if m else {}
        return {"resolved": row.get("tasks_resolved") or 0, "successful": row.get("tasks_successful") or 0}

    def tokens_per_successful_task(self, agent_id: str | None = None) -> float | None:
        """Tokens on resolved tasks (failed tasks included) / successful tasks; None until
        a task succeeds. Defined in prefixwise.metrics."""
        from . import metrics

        m = metrics.query(self, ["tokens_per_successful_task"],
                          filters={"agent": agent_id} if agent_id else None)
        return m[0]["tokens_per_successful_task"] if m else None

    def reader(self):
        """A connection for long reads (metrics, the API, exports). A file ledger
        gets its own read-only connection, so a big query never blocks the proxy
        recording requests (WAL lets readers and the writer run together)."""
        import contextlib

        @contextlib.contextmanager
        def _cm():
            if self.path == ":memory:":
                with self._lock:
                    yield self.conn
                return
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=10)
            try:
                yield conn
            finally:
                conn.close()
        return _cm()

    def recent_request_ids(self, limit: int = 200_000) -> list[str]:
        """The newest request ids, for a collector's idempotency set after a restart."""
        with self._lock:
            return [r[0] for r in self.conn.execute(
                "SELECT request_id FROM entries WHERE request_id IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,))]

    def request_ids(self) -> set[str]:
        """Every request id in the ledger (imports skip what is already there)."""
        with self._lock:
            return {r[0] for r in self.conn.execute("SELECT request_id FROM entries WHERE request_id IS NOT NULL")}

    def get_request(self, request_id: str) -> dict | None:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM entries WHERE request_id = ?", (request_id,))
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        return dict(zip(names, row)) if row else None

    def latest_session(self) -> str | None:
        """The session of the most recent row, for `report --session latest`."""
        with self._lock:
            row = self.conn.execute(  # by time where known (imported logs arrive out of order), else by row
                "SELECT session_id FROM entries WHERE session_id IS NOT NULL "
                "ORDER BY recorded_at IS NULL, recorded_at DESC, id DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def turns(self, session_id: str, since: float | None = None) -> list[dict]:
        """One session's requests in time order (imported logs are recorded out of order; rows
        without a time, from before 0.3.1, come last), numbered from 1 over the whole session even
        when `since` cuts the list. Hashes, counts and verdicts only. `first_changed` names the
        block where the prompt first diverged from the turn it was compared with (the session's
        previous request on the same model and conversation) as role[position], the position
        counted from 0 over the whole prompt in Prefixwise's prefix order: tools, then system,
        then the conversation (`messages[n]`, or `tool_results[n]` for a tool result). One
        position per tool, per system prompt and per message, not per content block: `tools[0]`
        is the first tool; with six tools and one system prompt `messages[7]` is the first
        message. `a -> b[n]` means the block at n changed kind. On an append-only hit it is the
        first block of the new turn. None when nothing diverged, there was no verdict, or the
        turn is a first write or a model switch (what it was compared with, if anything, is
        another conversation's or another model's request)."""
        with self._lock:
            cur = self.conn.execute("SELECT * FROM entries WHERE session_id = ? "
                                    "ORDER BY recorded_at IS NULL, recorded_at, id", (session_id,))
            names = [d[0] for d in cur.description]
            rows = [dict(zip(names, r)) for r in cur.fetchall()]
        for n, r in enumerate(rows, 1):
            r["turn"] = n
            role, idx = r.get("divergence_role"), r.get("divergence_index")
            r["first_changed"] = (None if r.get("cause") in ("model_switch", "first_write") else
                                  f"{role}[{idx}]" if role and idx is not None else (role or None))
            p = r.get("prompt_tokens") or 0
            r["hit_rate"] = (r.get("cached_tokens") or 0) / p if p else None
        if since is not None:
            rows = [r for r in rows if r.get("recorded_at") is not None and r["recorded_at"] >= since]
        return rows

    def summary(self, since: float | None = None) -> dict:
        """Totals across the whole ledger (or since a time), for reports. Defined in prefixwise.metrics."""
        from . import metrics

        m = metrics.summary(self, since=since)
        priced = m["requests_priced"] or 0
        return {"requests": m["requests"] or 0, "failed_requests": m["requests_failed"] or 0,
                "prompt_tokens": m["prompt_tokens"] or 0, "cached_tokens": m["cached_tokens"] or 0,
                "completion_tokens": m["completion_tokens"] or 0, "avoidable_tokens": m["avoidable_tokens"] or 0,
                "cache_hit_rate": m["cache_hit_rate"], "avoidable_share": m["avoidable_share"],
                "tasks": {"resolved": m["tasks_resolved"] or 0, "successful": m["tasks_successful"] or 0},
                "tokens_per_successful_task": m["tokens_per_successful_task"],
                # Dollars cover only priced requests; priced_requests says how many.
                "priced_requests": priced,
                "cost": m["cost_usd"] if priced else None,
                "avoidable_cost": (m["avoidable_cost_usd"] or 0) if priced else None,
                "cost_per_successful_task": m["cost_per_successful_task_usd"]}

    ROLLUP_KEYS = ("cause", "session_id", "agent_id", "model", "provider", "task_id")

    def rollup(self, by: str = "cause", since: float | None = None) -> list[dict]:
        """Requests and tokens grouped by one column, most avoidable first.
        Defined in prefixwise.metrics; groups with no verdict (failures only) are left out."""
        from . import metrics

        if by not in self.ROLLUP_KEYS:
            raise ValueError(f"rollup key must be one of {self.ROLLUP_KEYS}")
        dim = {"session_id": "session", "agent_id": "agent", "task_id": "task"}.get(by, by)
        rows = metrics.query(self, ["requests", "prompt_tokens", "cached_tokens", "avoidable_tokens",
                                    "requests_wasteful", "requests_priced", "cost_usd", "avoidable_cost_usd",
                                    "tasks_successful", "tokens_per_successful_task"], by=dim, since=since)
        out = [{by: r[dim], "requests": r["requests"], "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"], "avoidable_tokens": r["avoidable_tokens"],
                "waste_requests": r["requests_wasteful"] or 0, "priced_requests": r["requests_priced"] or 0,
                "cost": r["cost_usd"], "avoidable_cost": r["avoidable_cost_usd"],
                "successful_tasks": r["tasks_successful"],
                "tokens_per_successful_task": r["tokens_per_successful_task"]} for r in rows if r["requests"]]
        out.sort(key=lambda r: (-r["avoidable_tokens"], -r["requests"]))
        return out

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM entries ORDER BY id DESC LIMIT ?", (limit,))
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]

    def rollup_by_cause(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT cause, COUNT(*), SUM(avoidable_tokens), SUM(avoidable_cost)
                   FROM entries WHERE error IS NULL GROUP BY cause ORDER BY SUM(avoidable_tokens) DESC"""
            ).fetchall()
        return [{"cause": c, "requests": n, "avoidable_tokens": t or 0,
                 "avoidable_cost": cost} for c, n, t, cost in rows]

    def to_json(self) -> str:
        s = self.summary()
        return json.dumps({"tokens_per_successful_task": s["tokens_per_successful_task"],
                           "tasks": s["tasks"],
                           "by_cause": self.rollup_by_cause()}, indent=2)
