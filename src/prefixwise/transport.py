"""Records over the wire: an edge sends its turn records to a collector elsewhere.

    prefixwise collect --port 8081                              # the collector: one ledger for the fleet
    PREFIXWISE_COLLECTOR=http://collector:8081 prefixwise serve  # an edge: hashes here, verdicts there
    PREFIXWISE_COLLECTOR=... python app.py                      # prefixwise.wrap / the LiteLLM callback too

`Sender` is one background thread per edge with a bounded queue: `send` never blocks a request and
never raises. Records go in batches to POST /api/v1/ingest, in the order they were queued (one
sender, so a conversation's turns arrive in order: turn N's record is queued at N's response end,
before turn N+1 is even sent). A collector that is down is retried with backoff; when the queue is
full, or a batch has failed its last attempt, records are dropped and counted (`stats()`), never
held: the request path pays nothing for a slow collector. A batch the collector rejects (401, 400)
is dropped and counted too; the first such error is printed once.

`RemoteCollector` is what the edge's proxy talks to in place of `prefixwise.collector.Collector`:
same methods, but `ingest` queues the record and answers without a verdict (the verdict is made at
the collector, so the response carries `X-Prefixwise-Queued` instead of `X-Prefixwise-Cause`), and
the provider's diagnostics are still chained to the previous turn's response id, which the edge
remembers per conversation.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict

from .collector import LANE_THREADS, MAX_SESSIONS
from .ledger import LedgerEntry
from .pricing import PriceTable, normalize_usage
from .prom import Metrics
from .records import TurnRecord

INGEST_PATH = "/api/v1/ingest"
QUEUE_SIZE = 10_000  # records waiting to be sent; beyond this the newest are dropped and counted
BATCH_SIZE = 200
FLUSH_SECONDS = 0.25  # a partial batch waits at most this long
MAX_ATTEMPTS = 6  # per batch: 0.5, 1, 2, 4, 8 s between attempts, then dropped and counted
TIMEOUT_SECONDS = 10.0
_RETRY_STATUSES = (408, 429, 500, 502, 503, 504)


class Sender:
    def __init__(self, url: str, token: str | None = None, *, queue_size: int = QUEUE_SIZE,
                 batch_size: int = BATCH_SIZE, flush_seconds: float = FLUSH_SECONDS,
                 max_attempts: int = MAX_ATTEMPTS, timeout: float = TIMEOUT_SECONDS):
        self.url = url.rstrip("/")
        self.token = token
        self.batch_size, self.flush_seconds = max(1, batch_size), flush_seconds
        self.max_attempts, self.timeout = max(1, max_attempts), timeout
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stats = {"queued": 0, "sent": 0, "dropped": 0, "rejected": 0, "duplicates": 0,
                       "retries": 0, "failed_batches": 0, "last_error": None}
        self._in_flight = 0  # records taken from the queue, not yet delivered or dropped
        self._abandoned = False  # close() gave up waiting: what was left is already counted dropped
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._warned: set[str] = set()  # each kind of trouble is printed once
        self._opener = urllib.request.build_opener(_NoRedirect())  # a redirect would turn the POST into a GET
        self._thread = threading.Thread(target=self._run, name="prefixwise-sender", daemon=True)
        self._thread.start()

    # -- the edge's side -------------------------------------------------------------------

    def send(self, record: dict) -> bool:
        """Queue one record; False (and a count) if the queue is full. Never blocks, never raises."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._lock:
                self._stats["dropped"] += 1
            self._warn("full", "the send queue is full: records are being dropped (is the collector reachable?)")
            return False
        with self._lock:
            self._stats["queued"] += 1
        return True

    def stats(self) -> dict:
        with self._lock:
            return {**self._stats, "pending": self._queue.qsize(), "url": self.url}

    def close(self, timeout: float = 5.0) -> None:
        """Send what is queued (up to `timeout` seconds), then stop. What could not be sent in
        time is counted as dropped."""
        self._stop.set()
        self._thread.join(timeout)
        with self._lock:
            left = self._queue.qsize() + (self._in_flight if self._thread.is_alive() else 0)
            self._abandoned = self._thread.is_alive()  # whatever the thread does now is not counted twice
        if left:
            with self._lock:
                self._stats["dropped"] += left
            self._warn("close", f"{left} record(s) still queued at shutdown were not delivered")

    # -- the thread -------------------------------------------------------------------------

    def _run(self) -> None:
        """Runs for the life of the process: whatever a batch does, the next one is still sent."""
        while True:
            try:
                batch = self._take()
                if batch:
                    try:
                        self._post(batch)
                    finally:
                        with self._lock:
                            self._in_flight = 0
                elif self._stop.is_set():
                    return
            except Exception as exc:  # noqa: BLE001 - the thread must outlive any surprise
                with self._lock:
                    self._stats["last_error"] = f"{type(exc).__name__}: {exc}"
                self._warn("internal", f"sender error ({type(exc).__name__}: {exc}); records keep flowing")
                time.sleep(0.1)

    def _take(self) -> list[dict]:
        """Up to batch_size records: the first waits (briefly when stopping), the rest until the
        batch is full or flush_seconds have passed since the first."""
        batch: list[dict] = []
        try:
            batch.append(self._queue.get(timeout=0.05 if self._stop.is_set() else 0.5))
        except queue.Empty:
            return batch
        with self._lock:
            self._in_flight = 1
        deadline = time.monotonic() + self.flush_seconds
        while len(batch) < self.batch_size and not self._stop.is_set():  # stopping: send what is in hand
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
            with self._lock:
                self._in_flight = len(batch)
        return batch

    def _post(self, batch: list[dict]) -> None:
        data = json.dumps({"records": batch}).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "prefixwise-edge"}
        if self.token:
            headers["X-Prefixwise-Token"] = self.token
        error = "no attempt made"
        for attempt in range(self.max_attempts):
            try:
                req = urllib.request.Request(self.url + INGEST_PATH, data=data, headers=headers, method="POST")
                with self._opener.open(req, timeout=self.timeout) as r:
                    raw = r.read()
                # Any 2xx is delivered; the reply's counts are read as far as they can be.
                try:
                    reply = json.loads(raw or b"{}")
                except ValueError:
                    reply = {}
                reply = reply if isinstance(reply, dict) else {}
                rejected = reply.get("rejected") if isinstance(reply.get("rejected"), list) else []
                with self._lock:
                    if self._abandoned:
                        return  # counted as dropped at close; delivered after all, but not twice in the stats
                    self._stats["sent"] += len(batch) - len(rejected)
                    self._stats["rejected"] += len(rejected)
                    self._stats["duplicates"] += int(reply.get("duplicates") or 0)
                if rejected:
                    first = rejected[0].get("error") if isinstance(rejected[0], dict) else rejected[0]
                    self._warn("rejected", f"the collector rejected {len(rejected)} record(s): {first}")
                return
            except urllib.error.HTTPError as exc:
                if 300 <= exc.code < 400:
                    self._drop(batch, f"HTTP {exc.code} redirect from {self.url}{INGEST_PATH} to "
                                      f"{exc.headers.get('Location')}: point PREFIXWISE_COLLECTOR at the final URL")
                    return
                if exc.code not in _RETRY_STATUSES:  # our fault (401, 400, 413): retrying cannot help
                    self._drop(batch, f"HTTP {exc.code} from {self.url}{INGEST_PATH}")
                    return
                error = f"HTTP {exc.code}"
            except Exception as exc:  # noqa: BLE001 - URLError, socket errors, IncompleteRead, BadStatusLine, ...
                error = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
            with self._lock:
                self._stats["retries"] += 1
                self._stats["last_error"] = error
            if attempt + 1 < self.max_attempts:
                if self._stop.is_set():
                    break  # shutting down: do not spend the drain budget on backoff
                time.sleep(min(8.0, 0.5 * 2 ** attempt))
        self._drop(batch, f"gave up after {attempt + 1} attempt(s) ({error})", failed=True)

    def _drop(self, batch: list[dict], why: str, failed: bool = False) -> None:
        with self._lock:
            if self._abandoned:
                return  # already counted at close
            self._stats["dropped"] += len(batch)
            self._stats["last_error"] = why
            if failed:
                self._stats["failed_batches"] += 1
        self._warn("drop:" + why.split(" ")[0], f"{len(batch)} record(s) not delivered: {why}")

    def _warn(self, kind: str, msg: str) -> None:
        """One line per kind of trouble (queue full, a rejection, a drop, an internal error)."""
        with self._lock:
            if kind in self._warned:
                return
            self._warned.add(kind)
        print(f"prefixwise: {msg}", file=sys.stderr, flush=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


class _Chain(OrderedDict):
    """LRU of (session, model) -> OrderedDict(opening -> (response id, system identity))."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def touch(self, key) -> OrderedDict:
        lane = self.get(key)
        if lane is None:
            lane = self[key] = OrderedDict()
        self.move_to_end(key)
        while len(self) > self.limit:
            self.popitem(last=False)
        return lane


class RemoteCollector:
    """The edge's collector when the real one is elsewhere (see the module docstring)."""

    remote = True

    def __init__(self, sender: Sender, prices: PriceTable | None = None, *, lane_threads: int = LANE_THREADS,
                 max_sessions: int = MAX_SESSIONS):
        self.sender = sender
        self.prices = prices or PriceTable.load()
        self.metrics = Metrics()  # failures are counted here; verdict counters live at the collector
        self.ledger = None  # no ledger on an edge: the collector has it
        # What Collector.chain_parent reads, kept here the same way: per lane, each conversation's
        # latest response id (None when the response carried none); per session, the last one.
        self._chain = _Chain(max_sessions)
        self._last_response_id: OrderedDict = OrderedDict()
        self._lane_threads = lane_threads
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return self.sender.url

    def stats(self) -> dict:
        return self.sender.stats()

    def close(self, timeout: float = 5.0) -> None:
        self.sender.close(timeout)

    def stitched_opening(self, previous_response_id: str):
        return None  # a stitched turn waits for every pending recording of its lane (proxy)

    def chain_parent(self, session_id: str, lane: tuple, identity: tuple, compaction: bool,
                     previous_response_id: str | None) -> str | None:
        if previous_response_id:
            return str(previous_response_id)
        opening, system = identity
        with self._lock:
            threads = self._chain.get(lane)
            if not threads:  # first turn on this model: the session's last response, as the collector does
                return self._last_response_id.get(session_id)
            mine = threads.get(opening)
            if mine is not None:
                return mine[0]
            latest_id, latest_system = next(reversed(threads.values()))
            # Same system prompt (beyond dates and ids): the same conversation with a rewritten
            # opening, or a compaction; a different one is a new conversation (no parent).
            return latest_id if compaction or not system or system == latest_system else None

    def ingest(self, rec: TurnRecord) -> tuple[dict, LedgerEntry]:
        from .collector import _identity

        opening, system = _identity(rec.blocks)
        with self._lock:
            threads = self._chain.touch((rec.session_id, rec.model))
            threads[opening] = (rec.response_id, system)  # None when the response carried no id
            threads.move_to_end(opening)
            while len(threads) > self._lane_threads:
                threads.popitem(last=False)
            if rec.response_id:
                self._last_response_id[rec.session_id] = rec.response_id
                self._last_response_id.move_to_end(rec.session_id)
                while len(self._last_response_id) > self._chain.limit:
                    self._last_response_id.popitem(last=False)
        queued = self.sender.send(rec.to_dict())
        prompt_tokens, cached = normalize_usage(rec.provider, rec.usage)
        entry = LedgerEntry(session_id=rec.session_id, agent_id=rec.agent_id, provider=rec.provider,
                            model=rec.model, cause=None, prompt_tokens=prompt_tokens, cached_tokens=cached,
                            avoidable_tokens=0, avoidable_cost=None, task_id=rec.task_id,
                            request_id=rec.request_id, status=rec.status, api=rec.api, stream=rec.stream,
                            latency_ms=rec.latency_ms, overhead_ms=rec.overhead_ms)
        return {"request_id": rec.request_id, "status": rec.status, "cause": None, "legit": None,
                "native_reason": rec.native_reason, "usage": rec.usage, "prompt_tokens": prompt_tokens,
                "cached_tokens": cached, "avoidable_tokens": None, "fixes_applied": list(rec.fixes_applied),
                "cost": None, "avoidable_cost": None, "evidence": None,
                "queued": queued, "collector": self.sender.url}, entry

    def ingest_failure(self, rec: TurnRecord) -> LedgerEntry:
        entry = LedgerEntry(session_id=rec.session_id, agent_id=rec.agent_id, provider=rec.provider,
                            model=rec.model, cause=None, prompt_tokens=0, cached_tokens=0, avoidable_tokens=0,
                            avoidable_cost=None, task_id=rec.task_id, request_id=rec.request_id,
                            status=rec.status, error=str(rec.error or f"http_{rec.status}")[:80], api=rec.api,
                            stream=rec.stream, latency_ms=rec.latency_ms, overhead_ms=rec.overhead_ms)
        self.metrics.observe(entry)
        self.sender.send(rec.to_dict())
        return entry
