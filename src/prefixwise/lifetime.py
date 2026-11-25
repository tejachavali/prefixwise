"""How long a provider's prompt cache actually lasts, learned from the turns Prefixwise has seen.

Anthropic states its cache lifetime (5 minutes, or 1 hour on request). OpenAI's default retention and
DeepSeek's implicit cache do not give one fixed number, and in practice it varies. On
these, a turn whose unchanged prefix came back from the cache after an idle gap shows the cache
was alive at least that long, and a turn that read nothing (or under half) after a gap may have
found it expired. Per provider and model, this keeps both:

- alive_up_to: an idle gap after which an unchanged prefix was still read at least MIN_EVIDENCE times
  (the MIN_EVIDENCE-th longest such gap, so one lucky off-peak hit does not stretch it), counting only
  hits after more than MIN_IDLE_SECONDS;
- cold_from: the shortest gap, longer than every hit and than the shortest TTL any provider keeps,
  after which at least MIN_EVIDENCE unchanged prefixes read less than half, and only once at least
  MIN_EVIDENCE idle hits show the cache works at all: a cache that never hits is not "expiring".

The classifier uses them only when no TTL is known for the request (classifier.classify): a
short read within alive_up_to is a routing miss (the cache would still have held it), one at
cold_from or later is an expiry, both "inferred" and naming the evidence. Between the two, and
without enough evidence, it stays unknown. Nothing here is a promise about the provider: it is
what this ledger saw.

The evidence comes from the classifier's own split of turns: hits (`none`) and the low reads on an
unchanged prefix (Classification.cold_unchanged), whatever cause they were then given, so what is
learned does not depend on what it is later used to decide. A low read a changed cache key explains
is left out. The collector marks each such row in the ledger (`lifetime_evidence`), so the report and
the next collector read exactly what was used.

Limit: the gap is idle time per conversation, not per cache entry. Where many conversations share a
long prefix (one system prompt across a fleet), others can keep it warm, so a hit after a long gap
can overstate the lifetime; the MIN_EVIDENCE-th longest hit, not the longest, limits that.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass

MIN_EVIDENCE = 5        # turns on each side before a window is claimed
MIN_IDLE_SECONDS = 60   # a hit after a shorter gap says nothing about lifetime
MIN_PROVIDER_TTL = 300  # classifier.MIN_PROVIDER_TTL_SECONDS: no cache expires sooner
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
    idle_hits = sorted((g for g in hit_gaps if g is not None and g > MIN_IDLE_SECONDS), reverse=True)
    if len(idle_hits) < MIN_EVIDENCE:
        return None  # no evidence the cache holds anything across an idle gap: claim nothing
    alive = idle_hits[MIN_EVIDENCE - 1]
    floor = max(max(g for g in hit_gaps if g is not None), MIN_PROVIDER_TTL)
    beyond = sorted(g for g in cold_gaps if g is not None and g > floor)
    cold = beyond[0] if len(beyond) >= MIN_EVIDENCE else None
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


# The rows the collector marked as evidence (ledger column lifetime_evidence, schema 8): 1 a hit
# after an idle gap, 0 a cold read on an unchanged prefix.
LEDGER_SQL = ("SELECT provider, model, gap_seconds, lifetime_evidence FROM entries "
              "WHERE lifetime_evidence IS NOT NULL AND gap_seconds IS NOT NULL ")


def from_ledger(ledger, since: float | None = None, until: float | None = None) -> dict[tuple, Observed]:
    """Per (provider, model), the observed window from the ledger (only those with enough evidence)."""
    sql, args = LEDGER_SQL, []
    for col, op, val in (("recorded_at", ">=", since), ("recorded_at", "<", until)):
        if val is not None:
            sql += f"AND {col} {op} ? "
            args.append(val)
    with ledger.reader() as conn:
        rows = conn.execute(sql + "ORDER BY id", args).fetchall()
    # The most recent KEEP per model and kind, as a collector would hold them.
    hits: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=KEEP))
    colds: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=KEEP))
    for provider, model, gap, hit in rows:
        (hits if hit else colds)[(provider, model)].append(gap)
    out = {}
    for key in set(hits) | set(colds):
        obs = estimate(hits.get(key, []), colds.get(key, []))
        if obs is not None:
            out[key] = obs
    return out
