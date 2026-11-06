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
_MIGRATIONS = {"task_id": "TEXT", "request_id": "TEXT", "completion_tokens": "INTEGER"}


class Ledger:
    def __init__(self, path: str = ":memory:"):
        # check_same_thread=False: the proxy serves on threads; every
        # read and write goes through self._lock instead.
        import threading

        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
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
                    task_id, request_id, completion_tokens)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (entry.session_id, entry.agent_id, entry.provider, entry.model,
                 entry.cause, None if entry.legit is None else int(entry.legit),
                 entry.prompt_tokens, entry.cached_tokens, entry.avoidable_tokens,
                 entry.avoidable_cost,
                 None if entry.task_success is None else int(entry.task_success),
                 entry.native_reason, entry.classifier_version,
                 entry.task_id, entry.request_id, entry.completion_tokens),
            )
            if entry.task_id and entry.task_success is not None:
                self._set_task_outcome_locked(entry.task_id, entry.task_success)
            self.conn.commit()
            return cur.lastrowid

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
