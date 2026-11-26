"""Live gateway events: what POST /api/v1/observe hands to the pipeline.

A gateway plugin posts each request after it finished: the body it sent the provider, the response
(or a stream's final usage), when it started and ended, the headers. That is a line of a gateway
log (prefixwise.gateway_import), so it is parsed the same way and run through the same steps the
import replays: the request is sent, the answer began, the response is recorded. The difference is
that the events arrive live, from many threads, and not quite in order:

- every step of every event waits in one heap, ordered by the event's own time, as the import
  sorts a log. A step runs only once its event has been here HOLD seconds, so an event posted a
  little after one that came later in the conversation (the plugin's threads raced, a retry) is
  still ahead of it when they run. Turns are recorded in turn order; requests that overlapped
  overlap, as in the import;
- one worker thread runs the steps, so the pipeline sees one event at a time, off the request
  thread. It runs on the events' own clock (prepare(at=), span_end), never by swapping the
  module clock as the import does, so live traffic through the same proxy is unaffected;
- an event that arrives after its turn was overtaken (later than HOLD) is still recorded, in the
  order it arrived, and counted as late.

Prompt text is read to fingerprint it and is never stored. Standard library only.
"""

from __future__ import annotations

import heapq
import itertools
import os
import threading
import time

from .gateway_import import EntryError, apply_step, parse_entry, steps

# How long an event waits for an earlier turn of its conversation (the LiteLLM callback's wait).
HOLD_SECONDS = float(os.environ.get("PREFIXWISE_OBSERVE_HOLD_SECONDS", "2.0"))
# Events accepted but not yet recorded; beyond this a post is refused (503), never blocking.
MAX_PENDING = int(os.environ.get("PREFIXWISE_OBSERVE_MAX_PENDING", "10000"))
# Events in one post.
MAX_BATCH = 1000


class Observer:
    """Takes parsed gateway events from any thread and records them through `proxy` in time order,
    on one worker thread (started on the first event)."""

    def __init__(self, proxy, hold: float | None = None, max_pending: int | None = None):
        from .proxy import is_responses_body

        self.proxy = proxy
        self.hold = HOLD_SECONDS if hold is None else float(hold)
        self.max_pending = MAX_PENDING if max_pending is None else int(max_pending)
        self._is_responses_body = is_responses_body
        self._cond = threading.Condition()
        self._heap: list[tuple] = []  # (time, order, seq, step): steps() of every pending event
        self._events: dict[int, tuple[dict, float]] = {}  # seq -> (entry, when it arrived)
        self._ctxs: dict[int, dict] = {}  # seq -> the request in flight between steps (worker only)
        self._seq = itertools.count()
        self._last = None  # the time of the last step run: an event earlier than it is late
        self._busy = False  # the worker is running a step (outside the lock)
        self._worker: threading.Thread | None = None
        self._closing = False
        self.stats = {"accepted": 0, "recorded": 0, "failed": 0, "rejected": 0, "late": 0, "refused": 0}

    # -- the request thread ----------------------------------------------------------------

    def post(self, payload) -> tuple[int, dict]:
        """One event, or {"events": [...]}. Returns (status, reply): 202 when every event was
        queued; 400 with each bad event's index and reason (the good ones are still queued);
        503 when the queue is full (nothing queued)."""
        if isinstance(payload, dict) and "events" in payload:
            raw = payload["events"]
            if not isinstance(raw, list):
                return 400, _err('"events" must be a list of events')
        else:
            raw = [payload]
        if not raw:
            return 400, _err('"events" is empty')
        if len(raw) > MAX_BATCH:
            return 400, _err(f"at most {MAX_BATCH} events per post, not {len(raw)}")
        entries, rejected = [], []
        for i, item in enumerate(raw):
            try:
                entries.append(parse_entry(item, f"event {i}"))
            except EntryError as exc:
                rejected.append({"index": i, "reason": str(exc)})
        if not self.submit(entries):
            self.stats["refused"] += len(entries)
            return 503, _err(f"too many events waiting to be recorded ({self.max_pending}); retry shortly")
        with self._cond:
            self.stats["rejected"] += len(rejected)
        reply = {"accepted": len(entries)}
        if rejected:
            reply["rejected"] = rejected
            reply["error"] = {"message": f"{len(rejected)} of {len(raw)} events could not be used: "
                                         + rejected[0]["reason"] + (" (and others)" if len(rejected) > 1 else "")}
            return 400, reply
        return 202, reply

    def submit(self, entries: list[dict]) -> bool:
        """Queue parsed entries (gateway_import.parse_entry). False, with nothing queued, when
        they would take the queue past max_pending or the observer is closing."""
        if not entries:
            return True
        now = time.monotonic()
        with self._cond:
            if self._closing or len(self._events) + len(entries) > self.max_pending:
                return False
            for e in entries:
                seq = next(self._seq)
                self._events[seq] = (e, now)
                if self._last is not None and e["start"] < self._last:
                    self.stats["late"] += 1
                for step in steps(e, seq):
                    heapq.heappush(self._heap, step)
            self.stats["accepted"] += len(entries)
            if self._worker is None:
                self._worker = threading.Thread(target=self._loop, name="prefixwise-observe", daemon=True)
                self._worker.start()
            self._cond.notify_all()
        return True

    def pending(self) -> int:
        with self._cond:
            return len(self._events)

    def snapshot(self) -> dict:
        with self._cond:
            return {**self.stats, "pending": len(self._events)}

    # -- the worker ------------------------------------------------------------------------

    def _next(self):
        """The next step to run, waiting until its event has been here `hold` seconds (at once
        when closing). None when closing and nothing is left. Caller holds _cond."""
        while True:
            if not self._heap:
                if self._closing:
                    return None
                self._cond.wait()
                continue
            t, _order, seq, step = self._heap[0]
            wait = self._events[seq][1] + self.hold - time.monotonic()
            if wait > 0 and not self._closing:
                self._cond.wait(wait)  # an earlier event may arrive meanwhile and take the head
                continue
            heapq.heappop(self._heap)
            return t, seq, step

    def _loop(self) -> None:
        while True:
            with self._cond:
                nxt = self._next()
                if nxt is None:
                    return
                t, seq, step = nxt
                entry = self._events[seq][0]
                self._last = t if self._last is None else max(self._last, t)
                self._busy = True
            crashed = False
            try:
                outcome = apply_step(self.proxy, entry, seq, step, t, self._ctxs, self._is_responses_body)
            except Exception:  # noqa: BLE001 - the pipeline fails open; never let the worker die
                outcome, crashed = "failed", True
                self._ctxs.pop(seq, None)
            with self._cond:
                self._busy = False
                if outcome is not None:
                    self.stats[outcome] += 1
                # Done with this event: recorded, refused at prepare, or broken mid-way. Its other
                # steps, if any are left, have nothing to do.
                if step == 2 or crashed or (step == 0 and outcome == "failed"):
                    self._events.pop(seq, None)
                    if step != 2:
                        self._heap = [s for s in self._heap if s[2] != seq]
                        heapq.heapify(self._heap)
                self._cond.notify_all()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until everything queued so far is recorded (holds still apply). True if it was."""
        end = time.monotonic() + timeout
        with self._cond:
            while self._events or self._busy:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
        return True

    def close(self, timeout: float = 5.0) -> None:
        """Record what is still waiting at once (no more holds), then stop. Later posts are refused."""
        with self._cond:
            self._closing = True
            self._cond.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout)


def _err(message: str) -> dict:
    return {"error": {"message": message}}
