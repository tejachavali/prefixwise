"""The collector: Prefixwise's control plane.

It receives one `TurnRecord` per completed request (`prefixwise.records`) from an edge (the proxy,
`prefixwise.wrap`, the LiteLLM callback, or over the wire), compares it with the previous turn of
its own conversation, names the cause (`prefixwise.classifier`, `prefixwise.details`), prices it
and writes the ledger. All conversation state lives here: the lanes, the snapshots the Responses
API stitches to, the per-session facts. The edge keeps only what a request needs on its way out.

One collector serves one ledger; `ingest` is serialized by one lock, which is the ordering
guarantee a conversation needs (turn N is recorded before turn N+1 is compared with it). It never
sees prompt text: a record is hashes, positions and counts.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict

from .classifier import (UNSEEN_PARENT, Classification, MissCause, RequestSnapshot, classify, first_divergence,
                         served_all)
from .details import Facts, detail_for, last_block_marked, last_marker_shared, marked_tokens, shared_breakpoint
from .diagnostics import AgreementScorer
from .ledger import Ledger, LedgerEntry, avoidable_for_turn, held_for_turn, missed_for_turn, unread_write
from .lifetime import LEDGER_SQL, LifetimeTracker
from .pricing import PriceTable, breakdown, cache_reported, completion_tokens_of, normalize_usage, write_reported
from .prom import Metrics
from .records import RecordError, TurnRecord

SEEN_REQUESTS = 200_000  # request ids remembered for idempotent ingest (a retried batch adds no rows)
# Seconds a turn may appear to have been sent before the compared turn ended and still count as
# sequential: the two timestamps may come from different edges' clocks.
CLOCK_SKEW = float(os.environ.get("PREFIXWISE_CLOCK_SKEW", "1.0"))
MAX_BATCH = 1000

# Sessions remembered at once (snapshot, facts); the least recently seen are forgotten. A
# proxy serving many short sessions stays bounded in memory.
MAX_SESSIONS = int(os.environ.get("PREFIXWISE_MAX_SESSIONS", "10000"))
# Conversations remembered per (session, model) lane: parallel subagents sharing one session
# are each compared with their own previous turn. Each remembered conversation holds its latest
# turn's fingerprints and facts, about 0.75 kB per tool, system block and message (measured: 37 kB
# for a 20-message turn with 20 tools, 780 kB for a 1,000-message one). A conversation is known
# by its opening message, so every request with an opening not seen before adds one: the 65th
# drops the least recently used.
LANE_THREADS = max(1, int(os.environ.get("PREFIXWISE_LANE_THREADS", "64")))
# ... and a lane's memory is bounded: its remembered turns hold at most this many blocks between them
# (about 12 MB at 0.75 kB each), so 64 very long conversations, or one agent that trims a long
# history every turn, cannot grow a lane to 50 MB. Past the budget the least recently used go, but
# never below the 8 conversations a lane kept before it remembered 64, whatever their size.
LANE_BLOCKS = max(0, int(os.environ.get("PREFIXWISE_LANE_BLOCKS", "16000")))
LANE_THREADS_FLOOR = 8
_CONVERSATION = ("messages", "tool_results", "user")


class _SessionMap(OrderedDict):
    """LRU dict: reads and writes refresh a key; oldest keys are evicted."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.limit:
            self.popitem(last=False)


class _Lane:
    """One (session, model) lane: each conversation's latest turn, by opening message (LRU)."""

    __slots__ = ("latest", "threads")

    def __init__(self):
        self.latest = None
        self.threads: OrderedDict = OrderedDict()


def _identity(blocks) -> tuple:
    """(opening message hash or None, masked system hashes): what names a conversation, and
    what tells a subagent of another type (its system prompt beyond dates and ids)."""
    opening = next((b.raw for b in blocks if b.role in _CONVERSATION), None)
    return opening, tuple(b.masked for b in blocks if b.role in ("system", "system_dynamic"))


def _best_parent(lane: _Lane, identity: tuple, compaction: bool = False) -> tuple[tuple, bool, bool]:
    """(the turn to compare with, whether another conversation came between, whether this
    request starts a new one).

    A conversation is known by its opening message. Its previous turn is the lane's latest turn
    when that is the same conversation (the usual case), else this conversation's own latest turn
    if remembered (interleaved subagents). If neither, it is a new conversation only when its
    system prompt differs beyond dates and ids from the latest turn's (a subagent of another
    type); otherwise it is compared with the latest turn as before, so a rewritten first message
    (an edit, a compaction) never passes for a new conversation."""
    opening, system = identity
    latest = lane.threads[lane.latest]
    if opening is None or lane.latest is None or opening == lane.latest:
        return latest, False, False
    mine = lane.threads.get(opening)
    if mine is not None:
        return mine, True, False
    theirs = _identity(latest[0].blocks)[1]
    return latest, False, bool(not compaction and system and system != theirs)


def _age(started: float, parent, parent_done: float | None) -> float | None:
    """Seconds since the compared turn (a RequestSnapshot) was sent: the age of what it left in the
    cache. Providers count a cache entry's lifetime from the request that wrote or read it, so that
    turn's own generation time is part of the age. Its done time stands in when its send time is not
    known; None when neither is. (A snapshot's send time is never after its done time: `_ingest_locked`
    bounds it where it reads the record.)"""
    sent = getattr(parent, "sent_at", None) or parent_done
    return max(0.0, started - sent) if sent else None


_WALKED = ("messages", "tool_results")  # the conversation, in the order it was sent


def _earlier_point(parent_blocks, blocks) -> tuple[int, int] | None:
    """(blocks here, blocks in the compared turn) when this request's conversation (messages and
    tool results, in order) is a proper prefix of the compared turn's: an earlier point of the same
    conversation, with nothing in it changed. None otherwise."""
    mine = [b for b in blocks if b.role in _WALKED]
    theirs = [b for b in parent_blocks if b.role in _WALKED]
    if not mine or len(mine) >= len(theirs):
        return None
    if any(a.role != b.role or a.raw != b.raw for a, b in zip(mine, theirs)):
        return None
    return len(mine), len(theirs)


SEED_ROWS = 20_000  # recent ledger turns a collector learns cache lifetimes from at start


class Collector:
    def __init__(self, ledger: Ledger, prices: PriceTable, metrics: Metrics | None = None,
                 scorer: AgreementScorer | None = None):
        self.ledger = ledger
        self.prices = prices
        self.metrics = metrics or Metrics()
        self.scorer = scorer or AgreementScorer()
        self._lock = threading.RLock()
        # Conversation state (docs/ARCHITECTURE.md): per (session, model) lane, each
        # conversation's latest turn; per session, the latest snapshot on any model (for
        # model_switch), when it was seen, its last response id and its request facts; per
        # Responses id, the snapshot a stitched turn continues.
        self.lane_threads: dict[tuple, _Lane] = _SessionMap(MAX_SESSIONS)
        self.last_snapshot: dict[str, RequestSnapshot] = _SessionMap(MAX_SESSIONS)
        self.last_seen: dict[str, float] = _SessionMap(MAX_SESSIONS)
        self.last_response_id: dict[str, str] = _SessionMap(MAX_SESSIONS)
        self.snapshot_by_response: dict[str, RequestSnapshot] = _SessionMap(MAX_SESSIONS)
        self.session_facts: dict[str, Facts] = _SessionMap(MAX_SESSIONS)
        self._seen: OrderedDict = OrderedDict()  # request ids ingested, for idempotent batches
        self.stats = {"records": 0, "failures": 0, "duplicates": 0, "rejected": 0, "errors": 0, "concurrent": 0}
        # How long caches with no stated TTL have been seen to last (prefixwise.lifetime), seeded
        # from the ledger's recent turns so a restart does not forget it.
        self.lifetimes = LifetimeTracker()
        try:
            with ledger.reader() as conn:
                rows = conn.execute(LEDGER_SQL + "ORDER BY id DESC LIMIT ?", (SEED_ROWS,)).fetchall()
            self.lifetimes.seed(reversed(rows))
        except Exception as exc:  # noqa: BLE001 - start with nothing learned, and say so
            print(f"prefixwise: cache lifetimes not loaded from the ledger ({type(exc).__name__}: {exc})",
                  file=sys.stderr)

    # -- what the edge asks before a request goes out -----------------------------------------

    def stitched_opening(self, previous_response_id: str):
        """The opening of the conversation a stitched Responses turn continues, if its parent is
        known (None otherwise)."""
        with self._lock:
            prior = self.snapshot_by_response.get(str(previous_response_id))
        return _identity(prior.blocks)[0] if prior is not None else None

    def chain_parent(self, session_id: str, lane: tuple, identity: tuple, compaction: bool,
                     previous_response_id: str | None) -> str | None:
        """The response id of the previous turn of this request's own conversation: what the
        provider's diagnostics are chained to, so its verdict compares the pair the classifier
        compares. None for a new conversation."""
        with self._lock:
            if previous_response_id:
                return str(previous_response_id)  # the turn it names
            state = self.lane_threads.get(lane)
            if state is not None:
                parent, _, new_thread = _best_parent(state, identity, compaction)
                return None if new_thread else parent[3]
            return self.last_response_id.get(session_id)

    # -- one record in, one row out -----------------------------------------------------------

    def ingest(self, rec: TurnRecord) -> tuple[dict, LedgerEntry]:
        """Classify and record one completed turn. Returns (the verdict as the API reports it,
        the ledger row). Raises on a broken record or ledger; the edge decides what that means
        for the request (the proxy fails open)."""
        with self._lock:
            return self._ingest_locked(rec)

    def seed_seen(self, request_ids) -> None:
        """Request ids already in the ledger (after a restart): their redelivery is a duplicate."""
        ids = [rid for rid in request_ids if rid]
        with self._lock:
            for rid in reversed(ids):  # given newest first: insert oldest first, so the newest survive eviction
                self._seen[rid] = None
            while len(self._seen) > SEEN_REQUESTS:
                self._seen.popitem(last=False)

    def ingest_many(self, items: list, replay: bool = False) -> dict:
        """A batch from an edge over the wire (POST /api/v1/ingest, or the record log). Each item
        is a record's dict form; malformed ones are reported back by index and skipped, a request
        id already ingested is a duplicate (a retried batch) and skipped, and a record the
        collector itself fails on becomes a failure row (`not_recorded`) rather than a lost
        request. Never raises on bad input; raises on a broken ledger.

        replay: records this collector (or its predecessor) already turned into rows, read again
        after a restart to rebuild the conversation state: no row, no metric, no agreement score,
        and a duplicate check that only seeds the set."""
        if not isinstance(items, list):
            raise RecordError("records must be a list")
        if len(items) > MAX_BATCH:
            raise RecordError(f"at most {MAX_BATCH} records per batch")
        accepted, duplicates, rejected = 0, 0, []
        replayed: set = set()
        for i, item in enumerate(items):
            try:
                rec = TurnRecord.from_dict(item)
            except RecordError as exc:
                rejected.append({"index": i, "error": str(exc)})
                continue
            if replay:
                # State only. The duplicate set is seeded from the ledger, not from here: a record that
                # was delivered but never written must still be written when it is delivered again.
                if rec.request_id in replayed:
                    continue
                replayed.add(rec.request_id)
                if not rec.failed:
                    try:
                        with self._lock:
                            self._ingest_locked(rec, write=False)
                    except Exception as exc:  # noqa: BLE001 - a record that broke once must not break every start
                        with self._lock:
                            self.stats["errors"] += 1
                        print(f"prefixwise: record {rec.request_id} skipped in replay ({type(exc).__name__}: {exc})",
                              file=sys.stderr, flush=True)
                accepted += 1
                continue
            with self._lock:
                if rec.request_id in self._seen:
                    duplicates += 1
                    continue
                self._seen[rec.request_id] = None
                while len(self._seen) > SEEN_REQUESTS:
                    self._seen.popitem(last=False)
            try:
                if rec.failed:
                    self.ingest_failure(rec)
                else:
                    self.ingest(rec)
            except Exception as exc:  # noqa: BLE001 - one record must not sink the batch
                with self._lock:
                    self.stats["errors"] += 1
                print(f"prefixwise: record {rec.request_id} not classified ({type(exc).__name__}: {exc})",
                      file=sys.stderr, flush=True)
                rec.error = "not_recorded"
                try:
                    self.ingest_failure(rec)
                except Exception:
                    with self._lock:  # nothing written: a retry of this record must not read as a duplicate
                        self._seen.pop(rec.request_id, None)
                    raise
            accepted += 1
        with self._lock:
            self.stats["duplicates"] += duplicates
            self.stats["rejected"] += len(rejected)
        return {"accepted": accepted, "duplicates": duplicates, "rejected": rejected}

    def ingest_failure(self, rec: TurnRecord) -> LedgerEntry:
        """A request that produced no verdict (upstream error, cut-off stream, a recording
        failure) is a row too, so error rates exist and every other rate has an honest
        denominator (docs/DATA.md). The ledger write may raise; metrics count it first."""
        entry = LedgerEntry(
            session_id=rec.session_id, agent_id=rec.agent_id, provider=rec.provider,
            model=rec.model, cause=None, prompt_tokens=0, cached_tokens=0, avoidable_tokens=0,
            avoidable_cost=None, task_id=rec.task_id, request_id=rec.request_id,
            status=rec.status, error=str(rec.error or f"http_{rec.status}")[:80], api=rec.api,
            stream=rec.stream, latency_ms=rec.latency_ms, overhead_ms=rec.overhead_ms)
        with self._lock:
            self.ledger.record(entry)  # the row first, as for verdicts: a redelivery is then counted once
            self.stats["failures"] += 1
        self.metrics.observe(entry)
        return entry

    def _stitch(self, previous_response_id: str, fps: list) -> tuple[list | None, RequestSnapshot | None]:
        """Responses API with previous_response_id: the input holds only new items. The cached
        prefix is the prior turn's blocks plus these, so the comparable request is prior message
        blocks + new message blocks, under this turn's own tools and instructions (both are
        resent). Returns (blocks, prior snapshot); (None, None) if the prior turn was never seen.
        Caller holds the lock."""
        prior = self.snapshot_by_response.get(previous_response_id)
        if prior is None:
            return None, None
        head = [b for b in fps if b.role not in ("messages", "tool_results")]
        new_msgs = [b for b in fps if b.role in ("messages", "tool_results")]
        old_msgs = [b for b in prior.blocks if b.role in ("messages", "tool_results")]
        # The prior turn's system and developer items that opened its input are in the stored
        # response too, after its instructions; a continuing turn resends only the instructions.
        prior_system = [b for b in prior.blocks if b.role == "system"]
        carried = prior_system[sum(1 for b in head if b.role == "system"):]
        return head + carried + old_msgs + new_msgs, prior

    def _ingest_locked(self, rec: TurnRecord, write: bool = True) -> tuple[dict, LedgerEntry]:
        """Classify against the conversation's previous turn, write the row, then commit the new
        state. The row comes first: if the ledger refuses it, nothing has changed here, so the
        record can be delivered again and compared with the same predecessor."""
        provider, session_id, usage = rec.provider, rec.session_id, rec.usage
        prompt_tokens, cached = normalize_usage(provider, usage)
        native_reason, response_id, fps = rec.native_reason, rec.response_id, list(rec.blocks)
        now = rec.done_at or time.time()  # when the response completed (a record may arrive later)
        # The gap a cache entry had to survive: from the previous response to this request's send
        # (not to its response: this request's own generation time is not idle time). A request is
        # never sent after it ended: an edge that records a call after the fact may stamp it so.
        started = min(rec.sent_at or now, now)
        stitched = rec.api == "responses" and bool(rec.previous_response_id)
        lane = (session_id, rec.model)
        state = self.lane_threads.get(lane)
        identity = _identity(fps)
        new_thread, interleaved, parent_facts = False, False, None
        prior_latest = state.latest if state is not None else None
        prior_threads = len(state.threads) if state is not None else 0  # before this turn is added
        if state is not None:
            # Compare with the previous turn of this request's own conversation: the lane's latest
            # turn, unless another conversation interleaved (parallel subagents on one model).
            if stitched:
                parent = state.threads[state.latest]  # (its parent is named: stitched below)
            else:
                parent, interleaved, new_thread = _best_parent(state, identity, rec.compaction_event)
            prev, prev_done, parent_facts, _ = parent
            gap_seconds = max(0.0, started - prev_done)
            cache_age = _age(started, prev, prev_done)
            concurrent = started < prev_done - CLOCK_SKEW  # sent while the compared turn was still in flight
            # An earlier point of its own conversation: every message and tool result it carries
            # opens the compared turn, which goes on after them. Told by the blocks alone, never by
            # the clock: it is the record of an earlier turn delivered late (an edge's batch, a
            # retried delivery) when it was also sent before that turn was, and otherwise a rewind
            # to, or a replay of, an earlier turn.
            earlier = None if stitched else _earlier_point(prev.blocks, fps)
            parent_sent = getattr(prev, "sent_at", None)
            late = earlier is not None and bool(parent_sent) and started < parent_sent - CLOCK_SKEW
        else:
            # First turn on this model: FIRST_WRITE for a new session, else
            # MODEL_SWITCH (the classifier compares models first).
            prev = self.last_snapshot.get(session_id)
            gap_seconds = max(0.0, started - self.last_seen[session_id]) if session_id in self.last_seen else None
            cache_age = None  # another model's turn: no cache entry of this model to age
            concurrent = late = False
            earlier, parent_sent = None, None
        unknown_parent = concurrent_verdict = False
        lifetime_evidence = None  # set below when this turn shows how long the cache held (prefixwise.lifetime)
        if stitched:
            stitched_fps, prior = self._stitch(str(rec.previous_response_id), fps)
            if stitched_fps is None:
                unknown_parent = True  # continues a turn this collector never saw
            else:
                fps, prev = stitched_fps, prior
                # Its conversation is its parent's: file it there, and time the gap from it.
                identity = _identity(fps)
                mine = state.threads.get(identity[0]) if state is not None else None
                if mine is not None:
                    gap_seconds, parent_facts = max(0.0, started - mine[1]), mine[2]
                # The age of what it could read is counted from the turn it continues, which need
                # not be the conversation's latest (two turns can continue one response).
                cache_age = _age(started, prior, mine[1] if mine is not None and mine[0] is prior else None)
        curr = RequestSnapshot(model=rec.model, provider=provider, blocks=fps,
                               prompt_tokens=prompt_tokens, cached_tokens=cached,
                               gap_seconds=gap_seconds, cache_age_seconds=cache_age,
                               compaction_event=rec.compaction_event,
                               params=dict(rec.params), has_breakpoint=rec.has_breakpoint,
                               ttl_seconds=rec.ttl_seconds, fork_parent_session_id=rec.fork_of,
                               session_id=session_id)
        curr.engine = rec.engine is not None
        curr.engine_caches = rec.engine_caches
        curr.block_size = rec.engine_block
        curr.cache_reported = cache_reported(provider, usage)
        parts = breakdown(provider, usage)
        curr.cache_written = parts.cache_write_5m + parts.cache_write_1h + parts.cache_write
        curr.write_reported = write_reported(provider, usage)
        curr.sent_at = started
        curr.new_thread, curr.interleaved = new_thread, interleaved
        curr.shared_session = rec.session_shared or rec.session_is_user
        facts = rec.facts
        curr.last_block_marked = last_block_marked(facts)
        curr.marked_tokens = marked_tokens(facts, rec.blocks, prompt_tokens)
        curr.shared_breakpoint = not stitched and state is not None and shared_breakpoint(parent_facts, facts)
        if rec.engine_verdict is not None:
            verdict = rec.engine_verdict
            curr.evicted_blocks, curr.resident_blocks = verdict[0], verdict[1]
            curr.routed = tuple(verdict[2]) if len(verdict) > 2 and verdict[2] is not None else None
        # Only a verdict the join made on a hit counts as restored (engines.Restored, sent as a 4th
        # element): a miss's removed blocks on a turn the classifier calls a hit were never restored.
        restored = bool(rec.engine_verdict and len(rec.engine_verdict) > 3 and rec.engine_verdict[3] is True)
        early = None
        if earlier is not None and not rec.compaction_event and not unknown_parent:
            # (With a compaction event it is compared as always: a compaction is what it is.)
            # Its predecessor is the turn before this point, not the one at hand, which is a later
            # turn of the conversation: nothing in its conversation changed, and no change is
            # claimed there. The rules that compare nothing still apply (a compared turn too short
            # to cache), and so does the comparison of tools and system prompt, which stands against
            # any turn of the lane.
            point = f"an earlier point of this conversation (its {earlier[0]} blocks open the compared turn's {earlier[1]})"
            if late:
                why = (f"{point}: its record arrived late, after a later turn's; it was sent "
                       f"{parent_sent - started:.1f} s before that turn")
            elif parent_sent and started >= parent_sent:
                why = (f"{point}, sent {started - parent_sent:.1f} s after that turn: the conversation was rewound "
                       "to an earlier turn, or an earlier turn was sent again")
            elif parent_sent:
                why = f"{point}, sent within the clock allowance of that turn: a rewind, a replay or a late record"
            else:
                why = f"{point}: a rewind, a replay or a late record (when that turn was sent is not known)"
            early = classify(prev, curr, native_reason=native_reason, earlier_point=why)
        if unknown_parent:
            result = Classification(MissCause.UNKNOWN, None, UNSEEN_PARENT, native_reason=native_reason)
        elif early is not None and early.own_read:
            # Tools and system as they were: judged by its own read. (Nothing was compared, so the
            # row says nothing of how long the cache lasts: no lifetime evidence.)
            result = early
        elif (concurrent and not stitched and not new_thread
              and curr.cached_tokens < 0.5 * (min(prev.prompt_tokens, prompt_tokens) if prev.prompt_tokens
                                             else prompt_tokens)):
            # (A new conversation was compared with another conversation's turn only to find its
            # lane: it has no previous turn of its own to have overlapped.)
            # It could not have read what the compared turn wrote (that was still being written), and
            # the record of an earlier turn may be the one that arrived late: no cause is claimed.
            concurrent_verdict = True
            result = Classification(MissCause.UNKNOWN, None,
                                    f"sent {prev_done - started:.1f} s before the previous turn of this conversation "
                                    "had finished: a parallel request, or its record arrived late; it could not have "
                                    "read that turn's cache write", native_reason=native_reason)
        else:
            learn = rec.ttl_seconds is None and rec.engine is None and provider != "anthropic"
            if learn:
                curr.observed_lifetime = self.lifetimes.estimate(provider, rec.model)
            # (An earlier point whose tools or system prompt changed: that change, weighed as always.)
            result = early if early is not None else classify(prev, curr, native_reason=native_reason)
            if learn and prev is not None and gap_seconds is not None and prompt_tokens > 0 and curr.cache_reported:
                lifetime_evidence = (1 if result.cause is MissCause.NONE else 0 if result.cold_unchanged else None)
            if interleaved:
                result.evidence = (f"{result.evidence}; compared with this conversation's last turn "
                                   "(another conversation in this session came between)")
        # The why behind the cause, where the evidence allows (prefixwise.details).
        prev_facts = parent_facts if state is not None else self.session_facts.get(session_id)
        if prev is None or stitched:
            prev_facts = None  # a stitched branch: the lane's latest request may be another branch
        # Whether the classifier judged an unchanged prompt that came back short (it says so itself:
        # a new conversation, a changed block, a request with no markers are not that). Read before
        # a detail can replace the verdict below.
        unchanged_prompt = result.cold_unchanged and not result.own_read
        # No detail on a turn left unjudged (in flight with its previous turn, or continuing one never
        # seen) or judged by its own read: its timing is not a gap, the facts at hand are another
        # turn's, and a detail could promote it to a cause.
        undetailed = concurrent_verdict or unknown_parent or result.own_read
        new_cause, detail, why = (None, None, "") if undetailed else detail_for(result.cause.value, prev_facts, facts,
                                            prompt_tokens=prompt_tokens, cached_tokens=cached,
                                            evicted_blocks=curr.evicted_blocks if rec.engine_caches == 1 and restored else None,
                                            engine=rec.engine is not None,
                                            prompt_served=prev is not None and served_all(prev, curr))
        if new_cause:
            cause_name, legit = new_cause
            result = Classification(MissCause(cause_name), legit,
                                    f"{why} (classifier: {result.cause.value}, {result.evidence})",
                                    result.secondary + [result.cause], native_reason)
        elif why:
            result.evidence = f"{result.evidence}; {why}"
        if facts.no_prefix_cache and result.cause is not MissCause.NONE:
            # Nothing could have been read back: not an eviction, not routing, not the client's edit,
            # and no detail's advice (move a timestamp, stagger requests) would change that.
            detail = None
            result = Classification(MissCause.UNKNOWN, True,
                                    "the engine keeps no prefix cache (SGLang --disable-radix-cache), so every "
                                    f"turn recomputes its prompt; no cause applies (classifier: {result.cause.value})",
                                    [], native_reason)

        # (No "first changed" block on a row judged by its own read: nothing was compared.)
        div = (first_divergence(prev.blocks, curr.blocks, engine=curr.engine)
               if prev is not None and not result.own_read else None)
        # Priced as the model that answered, never as the one that was asked for: behind a gateway's
        # fallback routing they can be different models at different prices.
        priced_model = str(rec.response_model or rec.model)
        held = held_for_turn(prev, result.cause.value, cached)
        entry = LedgerEntry(
            session_id=session_id, agent_id=rec.agent_id,
            provider=provider, model=rec.model, cause=result.cause.value,
            prompt_tokens=prompt_tokens, cached_tokens=cached,
            avoidable_tokens=avoidable_for_turn(result.legit, prompt_tokens, cached,
                                                prev.prompt_tokens if prev else None, held),
            avoidable_cost=None, legit=result.legit, native_reason=native_reason,
            **self.prices.cost_fields(priced_model, provider, usage),
            task_id=rec.task_id, request_id=rec.request_id,
            completion_tokens=completion_tokens_of(provider, usage),
            fixes=",".join(rec.fixes_applied) or None,
            status=rec.status, api=rec.api, stream=rec.stream,
            uncached_tokens=parts.uncached_input,
            cache_write_tokens=parts.cache_write_5m + parts.cache_write_1h + parts.cache_write,
            latency_ms=rec.latency_ms, gap_seconds=gap_seconds,
            evidence=result.evidence,
            secondary_causes=",".join(c.value for c in result.secondary) or None,
            divergence_role=div["role"] if div else None, divergence_index=div["index"] if div else None,
            detail=detail, cache_ttl_seconds=rec.ttl_seconds, overhead_ms=rec.overhead_ms,
            cache_age_seconds=cache_age,
        )
        entry.avoidable_cost = self.prices.avoidable_cost(priced_model, entry.avoidable_tokens, prompt_tokens,
                                                          provider, usage)
        # What an expiry sent again: only for a prompt the classifier judged unchanged, a previous
        # turn whose usage reports what it wrote (so what it left is known, zero included), and a
        # request whose own usage does too (so what its markers cover is known).
        left = prev.cached_tokens + prev.cache_written if prev is not None and prev.write_reported else None
        entry.missed_tokens = missed_for_turn(entry.cause, result.legit, prompt_tokens, cached,
                                              prev.prompt_tokens if prev else None, left,
                                              unchanged=unchanged_prompt,
                                              covered_tokens=cached + entry.cache_write_tokens
                                              if write_reported(provider, usage) else None,
                                              marker_shared=last_marker_shared(prev_facts, rec.facts))
        if entry.missed_tokens is not None:
            entry.missed_cost = self.prices.missed_cost(priced_model, entry.missed_tokens, prompt_tokens,
                                                        provider, usage)
        if entry.cost is None:  # no rates, or no usage: nothing about this request has a dollar figure
            entry.avoidable_cost = entry.missed_cost = None
        # The previous turn's cache write, left unread: only when that turn is this conversation's
        # own: the same opening message, its own interleaved turn, or a compaction (which rewrites
        # the opening) while no other conversation shares the lane. Not another conversation this
        # request fell back to (a same-model side call), nor an edited opening message, which
        # identity alone cannot tell from one.
        own = state is not None and not new_thread and (
            interleaved or identity[0] == prior_latest
            or (rec.compaction_event and prior_threads <= 1))
        if own and prev_facts is not None and prev is not None and prompt_tokens > 0:
            w5, w1, cw = unread_write(result.cause.value, cached, prev_facts.write_5m, prev_facts.write_1h,
                                      prev_facts.cache_write, prev.cached_tokens)
            if w5 or w1 or cw:
                entry.unread_write_tokens = w5 + w1 + cw
                entry.unread_write_cost = self.prices.write_surcharge(priced_model, prev.prompt_tokens, w5, w1, cw)
        if entry.cost is not None:
            entry.price_as_of = self.prices.as_of
        if lifetime_evidence == 0 and detail == "cache_key_changed":
            lifetime_evidence = None  # a changed cache key explains the cold read, not the cache's lifetime
        entry.lifetime_evidence = lifetime_evidence

        if write:  # the row first: a refused row leaves the state untouched for a redelivery
            self.ledger.record(entry)
            if lifetime_evidence is not None:  # only once written: a replay or a redelivery is not counted twice
                self.lifetimes.observe(provider, rec.model, gap_seconds, hit=bool(lifetime_evidence))
            self.metrics.observe(entry)
            self.stats["records"] += 1
            if concurrent_verdict:
                self.stats["concurrent"] += 1
            if result.score:  # a verdict overriding the compared pair is not an (dis)agreement
                self.scorer.record(result.cause, native_reason, provider)
        # Then the conversation state this turn leaves behind. (A record that arrived late is an
        # older turn: it must not become the conversation's latest, which the later one already is.
        # A later turn that continues its response must still find it.)
        if late:
            if response_id and rec.api == "responses":
                self.snapshot_by_response[response_id] = curr
        else:
            self.last_seen[session_id] = now
            self.session_facts[session_id] = facts
            self.last_snapshot[session_id] = curr
            if state is None:
                state = _Lane()
            state.threads[identity[0]] = (curr, now, facts, response_id)
            state.threads.move_to_end(identity[0])
            while len(state.threads) > LANE_THREADS or (
                    len(state.threads) > min(LANE_THREADS_FLOOR, LANE_THREADS)
                    and sum(len(t[0].blocks) for t in state.threads.values()) > LANE_BLOCKS):
                state.threads.popitem(last=False)
            state.latest = identity[0]
            self.lane_threads[lane] = state  # (re)assigning refreshes the lane in the LRU
            if response_id:
                self.last_response_id[session_id] = response_id
                if rec.api == "responses":
                    self.snapshot_by_response[response_id] = curr
        return {"request_id": rec.request_id, "status": rec.status, "cause": result.cause.value,
                "legit": result.legit, "native_reason": native_reason, "usage": usage,
                "prompt_tokens": prompt_tokens, "cached_tokens": cached,
                "avoidable_tokens": entry.avoidable_tokens, "fixes_applied": list(rec.fixes_applied),
                "cost": entry.cost, "avoidable_cost": entry.avoidable_cost, "evidence": result.evidence}, entry
