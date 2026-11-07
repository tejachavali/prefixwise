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
from dataclasses import dataclass


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


def avoidable_tokens(prompt_tokens: int, cached_tokens: int, *,
                     first_write_tokens: int = 0, legitimate_change_tokens: int = 0) -> int:
    return max(0, prompt_tokens - cached_tokens - first_write_tokens - legitimate_change_tokens)


def avoidable_for_turn(legit: bool | None, prompt_tokens: int, cached_tokens: int,
                       prev_prompt_tokens: int | None) -> int:
    """Avoidable tokens for one classified turn.

    Only waste counts (legit is False). Legitimate changes (model
    switch, catalog change, compaction, first write) still cost tokens
    but are not waste; undetermined legitimacy (None: NONE, routing,
    unknown, ...) is not counted either, never a guess.

    For waste, avoidable is the prior prefix that could have been
    reused but was not: min(prev, curr) prompt tokens minus cache_read.
    Tokens appended after the prior prefix are new content, never
    avoidable.
    """
    if legit is not False:
        return 0
    reusable = min(prev_prompt_tokens, prompt_tokens) if prev_prompt_tokens else prompt_tokens
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
               "recorded_at": "REAL"}  # unix time; NULL on rows from before 0.3.1


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
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        with self._lock:
            if path != ":memory:":
                # WAL lets `prefixwise report` read while the proxy writes.
                self.conn.execute("PRAGMA journal_mode=WAL")
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
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, success INTEGER NOT NULL, recorded_at REAL
                )"""
            )
            self.conn.commit()

    def record(self, entry: LedgerEntry) -> int:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO entries (session_id, agent_id, provider, model, cause,
                    legit, prompt_tokens, cached_tokens, avoidable_tokens,
                    avoidable_cost, task_success, native_reason, classifier_version,
                    task_id, request_id, completion_tokens, fixes, cost, price_as_of, recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (entry.session_id, entry.agent_id, entry.provider, entry.model,
                 entry.cause, None if entry.legit is None else int(entry.legit),
                 entry.prompt_tokens, entry.cached_tokens, entry.avoidable_tokens,
                 entry.avoidable_cost,
                 None if entry.task_success is None else int(entry.task_success),
                 entry.native_reason, entry.classifier_version,
                 entry.task_id, entry.request_id, entry.completion_tokens, entry.fixes,
                 entry.cost, entry.price_as_of, time.time()),
            )
            if entry.task_id and entry.task_success is not None:
                self._set_task_outcome_locked(entry.task_id, entry.task_success)
            self.conn.commit()
            return cur.lastrowid

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

    def task_counts(self, agent_id: str | None = None) -> dict:
        q = """SELECT COUNT(*), COALESCE(SUM(t.success), 0) FROM tasks t
               WHERE EXISTS (SELECT 1 FROM entries e WHERE e.task_id = t.task_id
                             {agent})"""
        args: list = []
        agent = ""
        if agent_id:
            agent, args = "AND e.agent_id = ?", [agent_id]
        with self._lock:
            resolved, successful = self.conn.execute(q.format(agent=agent), args).fetchone()
        return {"resolved": resolved, "successful": successful}

    def tokens_per_successful_task(self, agent_id: str | None = None) -> float | None:
        """All tokens on resolved tasks (failures included) / successful tasks.

        None when no task has succeeded yet: the ratio is undefined, not zero.
        """
        q = """SELECT COALESCE(SUM(e.prompt_tokens + COALESCE(e.completion_tokens, 0)), 0)
               FROM entries e JOIN tasks t ON e.task_id = t.task_id"""
        args: list = []
        if agent_id:
            q += " WHERE e.agent_id = ?"
            args.append(agent_id)
        with self._lock:
            total = self.conn.execute(q, args).fetchone()[0]
        successful = self.task_counts(agent_id)["successful"]
        if not successful:
            return None
        return total / successful

    def get_request(self, request_id: str) -> dict | None:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM entries WHERE request_id = ?", (request_id,))
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        return dict(zip(names, row)) if row else None

    def summary(self) -> dict:
        """Totals across the whole ledger, for reports."""
        with self._lock:
            n, prompt, cached, completion, avoidable = self.conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(cached_tokens), 0),
                          COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(avoidable_tokens), 0)
                   FROM entries""").fetchone()
            priced, cost, avoidable_cost = self.conn.execute(
                """SELECT COUNT(cost), COALESCE(SUM(cost), 0), COALESCE(SUM(avoidable_cost), 0)
                   FROM entries""").fetchone()
        return {"requests": n, "prompt_tokens": prompt, "cached_tokens": cached,
                "completion_tokens": completion, "avoidable_tokens": avoidable,
                "cache_hit_rate": (cached / prompt) if prompt else None,
                "avoidable_share": (avoidable / prompt) if prompt else None,
                "tasks": self.task_counts(),
                "tokens_per_successful_task": self.tokens_per_successful_task(),
                # Dollars cover only priced requests; priced_requests says how many.
                "priced_requests": priced,
                "cost": cost if priced else None,
                "avoidable_cost": avoidable_cost if priced else None,
                "cost_per_successful_task": self.cost_per_successful_task()}

    def cost_per_successful_task(self) -> float | None:
        """Dollars on resolved tasks (failures included) / successful tasks.
        None unless every request on those tasks was priced: a partial sum
        would understate it."""
        with self._lock:
            n, priced, total = self.conn.execute(
                """SELECT COUNT(*), COUNT(e.cost), COALESCE(SUM(e.cost), 0)
                   FROM entries e JOIN tasks t ON e.task_id = t.task_id""").fetchone()
        successful = self.task_counts()["successful"]
        if not successful or not n or priced != n:
            return None
        return total / successful

    ROLLUP_KEYS = ("cause", "session_id", "agent_id", "model", "provider", "task_id")

    def rollup(self, by: str = "cause") -> list[dict]:
        """Requests and tokens grouped by one column, most avoidable first."""
        if by not in self.ROLLUP_KEYS:
            raise ValueError(f"rollup key must be one of {self.ROLLUP_KEYS}")
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT {by}, COUNT(*), COALESCE(SUM(prompt_tokens), 0),
                           COALESCE(SUM(cached_tokens), 0), COALESCE(SUM(avoidable_tokens), 0),
                           SUM(CASE WHEN legit = 0 THEN 1 ELSE 0 END),
                           COUNT(cost), SUM(cost), SUM(avoidable_cost)
                    FROM entries GROUP BY {by}
                    ORDER BY SUM(avoidable_tokens) DESC, COUNT(*) DESC""").fetchall()
            # Tokens per successful task within each group: tokens on resolved
            # tasks (failures included) / distinct successful tasks.
            per_task = {k: (tok, ok) for k, tok, ok in self.conn.execute(
                f"""SELECT e.{by}, SUM(e.prompt_tokens + COALESCE(e.completion_tokens, 0)),
                           COUNT(DISTINCT CASE WHEN t.success = 1 THEN t.task_id END)
                    FROM entries e JOIN tasks t ON e.task_id = t.task_id
                    GROUP BY e.{by}""").fetchall()}
        out = []
        for k, n, pt, ct, at, w, priced, cost, acost in rows:
            tok, ok = per_task.get(k, (0, 0))
            out.append({by: k, "requests": n, "prompt_tokens": pt, "cached_tokens": ct,
                        "avoidable_tokens": at, "waste_requests": w or 0,
                        "priced_requests": priced, "cost": cost, "avoidable_cost": acost,
                        "successful_tasks": ok,
                        "tokens_per_successful_task": (tok / ok) if ok else None})
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
                   FROM entries GROUP BY cause ORDER BY SUM(avoidable_tokens) DESC"""
            ).fetchall()
        return [{"cause": c, "requests": n, "avoidable_tokens": t or 0,
                 "avoidable_cost": cost} for c, n, t, cost in rows]

    def to_json(self) -> str:
        return json.dumps({"tokens_per_successful_task": self.tokens_per_successful_task(),
                           "tasks": self.task_counts(),
                           "by_cause": self.rollup_by_cause()}, indent=2)
