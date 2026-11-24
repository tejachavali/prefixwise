"""How long a provider's prompt cache actually lasts, learned from the turns Prefixwise has seen.

Anthropic states its cache lifetime (5 minutes, or 1 hour on request). OpenAI's default retention,
DeepSeek's and Gemini's implicit caches do not give one fixed number, and in practice it varies. On
these, a turn whose unchanged prefix came back from the cache after an idle gap shows the cache
was alive at least that long, and a turn that read nothing (or under half) after a gap may have
found it expired. Per provider and model, this keeps both:

- alive_up_to: the longest idle gap after which an unchanged prefix was still read (needs
  MIN_EVIDENCE such hits after a gap of more than a minute, so a few lucky turns prove nothing);
- cold_from: the shortest gap, longer than every hit, after which an unchanged prefix read less than
  half, when at least MIN_EVIDENCE turns read so little at that gap or longer, and none was read.

The classifier uses them only when no TTL is known for the request (classifier.classify): a
short read within alive_up_to is a routing miss (the cache would still have held it), one at
cold_from or later is an expiry, both "inferred" and naming the evidence. Between the two, and
without enough evidence, it stays unknown. Nothing here is a promise about the provider: it is
what this ledger saw.

The evidence comes from the classifier's own split of turns: hits (`none`) and the low reads on an
unchanged prefix (Classification.cold_unchanged), whatever cause they were then given, so what is
learned does not depend on what it is later used to decide.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass

MIN_EVIDENCE = 3        # turns on each side before a window is claimed
MIN_IDLE_SECONDS = 60   # a hit after a shorter gap says nothing about lifetime
KEEP = 2000             # most recent gaps kept per (provider, model) and kind


@dataclass(frozen=True)
class Observed:
    """What this ledger saw for one provider and model. Either side may be None (not enough evidence)."""
    alive_up_to: float | None
    cold_from: float | None
    hits: int    # unchanged prefix read after an idle gap of more than MIN_IDLE_SECONDS
    colds: int   # unchanged prefix read less than half, at cold_from or later

    def describe(self) -> str:
        parts = []
        if self.alive_up_to is not None:
            parts.append(f"read after idle gaps up to {_dur(self.alive_up_to)} ({self.hits} turns)")
        if self.cold_from is not None:
            parts.append(f"nothing (or under half) read after {_dur(self.cold_from)} or more ({self.colds} turns)")
        return "; ".join(parts)


def _dur(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def estimate(hit_gaps, cold_gaps) -> Observed | None:
    """The window from the gaps of hits and of cold reads on an unchanged prefix."""
    idle_hits = [g for g in hit_gaps if g is not None and g > MIN_IDLE_SECONDS]
    alive = max(idle_hits) if len(idle_hits) >= MIN_EVIDENCE else None
    longest_hit = max((g for g in hit_gaps if g is not None), default=0.0)
    beyond = sorted(g for g in cold_gaps if g is not None and g > longest_hit)
    cold = beyond[0] if len(beyond) >= MIN_EVIDENCE else None
    if alive is None and cold is None:
        return None
    return Observed(alive, cold, len(idle_hits), len(beyond) if cold is not None else 0)


class LifetimeTracker:
    """Per (provider, model), the idle gaps of hits and of cold reads on an unchanged prefix, in
    memory, bounded. Thread-safe; seeded from the ledger when a collector starts."""

    def __init__(self):
        self._hits: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=KEEP))
        self._colds: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=KEEP))
        self._cache: dict[tuple, Observed | None] = {}
        self._lock = threading.Lock()

    def observe(self, provider: str, model: str, gap: float | None, hit: bool) -> None:
        if gap is None:
            return
        key = (provider, model)
        with self._lock:
            (self._hits if hit else self._colds)[key].append(float(gap))
            self._cache.pop(key, None)

    def estimate(self, provider: str, model: str) -> Observed | None:
        key = (provider, model)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = estimate(self._hits.get(key, ()), self._colds.get(key, ()))
            return self._cache[key]

    def seed(self, rows) -> None:
        """Rows of (provider, model, gap_seconds, hit) from the ledger, oldest first."""
        for provider, model, gap, hit in rows:
            self.observe(provider, model, gap, bool(hit))


# The ledger's own record of the same split: a hit is cause `none`; a cold read on an unchanged
# prefix is one the classifier's evidence says so of (its wording since #171: "nothing read from
# cache" or "-token unchanged prefix"). Rows whose TTL was known are left out: their lifetime is
# stated, not learned.
LEDGER_SQL = (
    "SELECT provider, model, gap_seconds, CASE WHEN cause = 'none' THEN 1 ELSE 0 END FROM entries "
    "WHERE error IS NULL AND gap_seconds IS NOT NULL AND cache_ttl_seconds IS NULL AND prompt_tokens > 0 "
    "AND (cause = 'none' OR (cause IN ('routing_miss', 'ttl_expired', 'unknown') "
    "AND (evidence LIKE '%nothing read from cache%' OR evidence LIKE '%-token unchanged prefix%') "
    "AND evidence NOT LIKE '%KV events%')) ")


def from_ledger(ledger, since: float | None = None, until: float | None = None) -> dict[tuple, Observed]:
    """Per (provider, model), the observed window from the ledger (only those with enough evidence)."""
    sql, args = LEDGER_SQL, []
    for col, op, val in (("recorded_at", ">=", since), ("recorded_at", "<", until)):
        if val is not None:
            sql += f"AND {col} {op} ? "
            args.append(val)
    with ledger.reader() as conn:
        rows = conn.execute(sql + "ORDER BY id", args).fetchall()
    hits: dict[tuple, list] = defaultdict(list)
    colds: dict[tuple, list] = defaultdict(list)
    for provider, model, gap, hit in rows:
        (hits if hit else colds)[(provider, model)].append(gap)
    out = {}
    for key in set(hits) | set(colds):
        obs = estimate(hits.get(key, []), colds.get(key, []))
        if obs is not None:
            out[key] = obs
    return out
