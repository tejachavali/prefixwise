"""Efficiency ledger (SQLite).

Headline metric: tokens per successful task. Cost is a secondary column.
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
    task_success: bool | None = None
    legit: bool | None = None
    native_reason: str | None = None
    classifier_version: str = "2.0"


def avoidable_tokens(prompt_tokens: int, cached_tokens: int, *,
                     first_write_tokens: int = 0, legitimate_change_tokens: int = 0) -> int:
    return max(0, prompt_tokens - cached_tokens - first_write_tokens - legitimate_change_tokens)


def reuse_adjusted_reads(write_multiplier: float, read_multiplier: float) -> float:
    """Model (not a verified fact): reads needed for a fix to pay.

    k = write_multiplier / (1 - read_multiplier).
    """
    if read_multiplier >= 1:
        raise ValueError("read_multiplier must be < 1")
    return write_multiplier / (1 - read_multiplier)


class Ledger:
    def __init__(self, path: str = ":memory:"):
        # check_same_thread=False: the proxy serves on threads; guard
        # with a lock instead (see record()).
        import threading

        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
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
        self.conn.commit()

    def record(self, entry: LedgerEntry) -> int:
        with self._lock:
            return self._record_locked(entry)

    def _record_locked(self, entry: LedgerEntry) -> int:
        cur = self.conn.execute(
            """INSERT INTO entries (session_id, agent_id, provider, model, cause,
                legit, prompt_tokens, cached_tokens, avoidable_tokens,
                avoidable_cost, task_success, native_reason, classifier_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (entry.session_id, entry.agent_id, entry.provider, entry.model,
             entry.cause, None if entry.legit is None else int(entry.legit),
             entry.prompt_tokens, entry.cached_tokens, entry.avoidable_tokens,
             entry.avoidable_cost,
             None if entry.task_success is None else int(entry.task_success),
             entry.native_reason, entry.classifier_version),
        )
        self.conn.commit()
        return cur.lastrowid

    def tokens_per_successful_task(self, agent_id: str | None = None) -> float | None:
        q = "SELECT SUM(prompt_tokens), COUNT(*) FROM entries WHERE task_success = 1"
        args: list = []
        if agent_id:
            q += " AND agent_id = ?"
            args.append(agent_id)
        total, count = self.conn.execute(q, args).fetchone()
        if not count:
            return None
        return total / count

    def rollup_by_cause(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT cause, COUNT(*), SUM(avoidable_tokens), SUM(avoidable_cost)
               FROM entries GROUP BY cause ORDER BY SUM(avoidable_tokens) DESC"""
        ).fetchall()
        return [{"cause": c, "requests": n, "avoidable_tokens": t or 0,
                 "avoidable_cost": cost} for c, n, t, cost in rows]

    def to_json(self) -> str:
        return json.dumps({"tokens_per_successful_task": self.tokens_per_successful_task(),
                           "by_cause": self.rollup_by_cause()}, indent=2)
