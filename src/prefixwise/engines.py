"""Engine join for vLLM: its KV-cache events tell an eviction from everything else.

Behind one self-hosted engine, a request whose prefix is unchanged but was not
reused usually means the engine evicted it (GPU memory full). Two look-alikes
exist: the token prefix differs although the request does not (a chat template
that renders history differently), and data-parallel routing. vLLM's KV events
separate them (docs/ENGINES.md):

1. With `--engine vllm` and `--kv-events`, Prefixwise sends its session id in the
   request body and `vllm_xargs: {"kv_cache_report_mode": "full"}`, so vLLM
   reports every block a request uses, reused or new, tagged with the session
   (vllm/v1/core/kv_cache_manager.py, block_pool.py, 2026-11-06). A block's
   position comes from its parent (`parent_block_hash`), not from arrival order.
2. When a session's next request is sent, the blocks of its previous request are
   frozen as the chain to judge. Removals are recorded from then until the
   response, so evictions while the request waits in vLLM's queue count.
3. At the response, `cached_tokens` says where the cache read stopped: position
   cached_tokens // block_size. Only that block decides. Removed: a confirmed
   eviction. Still on the GPU: the token prefix changed there, so not an
   eviction. That negative claim is made only when nothing can make it wrong: the
   event stream caught up with the response, lost no messages, the model has
   full attention only, and the session had one request in flight.

Every other case returns nothing, and the classifier keeps its inference.
Offload tiers (LMCache) are not tracked; a prefix they restore is a hit and never
examined.

Wire format (vllm/distributed/kv_events.py): ZMQ PUB, multipart [topic, seq
(8 bytes, big-endian), msgpack [ts, events, dp_rank?]], each event a map tagged
by "type". Needs `pip install "prefixwise[engines]"` (pyzmq, msgspec).
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict

GPU = (None, "GPU")
FULL_KINDS = (None, "full_attention", "mla_attention", "sink_full_attention")
NOTHING = (None, None)


class _Window:
    """Blocks one session's requests reported since its last send, by position."""

    __slots__ = ("at", "block_size", "mixed")

    def __init__(self, mixed: bool = False):
        self.at: dict = {}  # position -> (group, hash); a later report of a position wins
        self.block_size: int | None = None
        self.mixed = mixed  # concurrent requests reported into it


class EngineEvents:
    def __init__(self, endpoint: str | None = None, max_sessions: int = 1000, max_chain: int = 4096,
                 max_alive: int = 1_000_000, settle_seconds: float = 0.05):
        self.endpoint = endpoint
        self.max_sessions, self.max_chain, self.max_alive = max_sessions, max_chain, max_alive
        self.settle_seconds = settle_seconds
        self._lock = threading.Lock()
        self._alive: dict = {}  # (group, hash) on the GPU -> its position, full-attention groups only
        self._windows: OrderedDict = OrderedDict()  # session -> _Window
        self._checks: OrderedDict = OrderedDict()  # token -> in-flight check
        self._tokens = itertools.count(1)
        self.hybrid = False  # the model has non-full-attention groups: no negative claims
        self.stats = {"batches": 0, "stored": 0, "removed": 0, "cleared": 0, "engine_restarts": 0,
                      "dropped_messages": 0, "decode_errors": 0, "overflows": 0}
        self._last_seq: int | None = None
        self._latest_ts = 0.0  # engine timestamp of the last batch read
        self._clock_offset: float | None = None  # min(receive time - engine ts): skew + least latency
        self._drained_at = 0.0  # last time the subscriber found the socket empty
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _errors(self) -> int:
        s = self.stats
        return s["dropped_messages"] + s["decode_errors"] + s["overflows"]

    # -- events ----------------------------------------------------------------------

    def handle(self, batch, seq: int | None = None) -> None:
        """One decoded payload: [ts, events, dp_rank?]."""
        with self._lock:
            if seq is not None and self._last_seq is not None:
                if seq > self._last_seq + 1:  # lost messages: a removal may be among them
                    self.stats["dropped_messages"] += seq - self._last_seq - 1
                elif seq <= self._last_seq:  # sequence restarted: the engine restarted, its cache is empty
                    self.stats["engine_restarts"] += 1
                    self._clear()
            if seq is not None:
                self._last_seq = seq
            self.stats["batches"] += 1
            for ev in batch[1]:
                kind = ev.get("type")
                if kind == "BlockStored" and ev.get("medium") in GPU:
                    if ev.get("kv_cache_spec_kind") in FULL_KINDS:
                        self._store(ev)
                    else:
                        self.hybrid = True
                elif kind == "BlockRemoved" and ev.get("medium") in GPU:
                    group = ev.get("group_idx") or 0
                    for h in ev.get("block_hashes") or []:
                        self._remove((group, h))
                elif kind == "AllBlocksCleared":
                    self.stats["cleared"] += 1
                    self._clear()
            if isinstance(batch[0], (int, float)):
                ts = float(batch[0])
                self._latest_ts = max(self._latest_ts, ts)
                offset = time.time() - ts  # the engine may run on another host with another clock
                self._clock_offset = offset if self._clock_offset is None else min(self._clock_offset, offset)

    def _store(self, ev: dict) -> None:
        group, hashes, session = ev.get("group_idx") or 0, ev.get("block_hashes") or [], ev.get("session_id")
        self.stats["stored"] += len(hashes)
        parent = ev.get("parent_block_hash")
        parent_pos = self._alive.get((group, parent)) if parent is not None else -1
        pos = None if parent_pos is None else parent_pos + 1  # unknown parent: unplaced
        w = self._window(session) if session else None
        if w is not None:
            w.block_size = w.block_size or ev.get("block_size")
        for h in hashes:
            key = (group, h)
            self._alive[key] = pos
            if w is not None and pos is not None and pos < self.max_chain:
                w.at[pos] = key
            pos = None if pos is None else pos + 1
        if len(self._alive) > self.max_alive:  # removals were missed: residency is unknown
            self._alive.clear()
            self._windows.clear()  # their blocks would all look removed: judge nothing from them
            self.stats["overflows"] += 1

    def _remove(self, key) -> None:
        self.stats["removed"] += 1
        self._alive.pop(key, None)
        for check in self._checks.values():
            if key in check["keys"]:
                check["lost"].add(key)

    def _clear(self) -> None:
        for check in self._checks.values():
            check["lost"].update(check["keys"])
        self._alive.clear()

    def _window(self, session: str) -> _Window:
        w = self._windows.get(session)
        if w is None:
            w = self._windows[session] = _Window()
            while len(self._windows) > self.max_sessions:
                self._windows.popitem(last=False)
        self._windows.move_to_end(session)
        return w

    # -- per request -----------------------------------------------------------------

    def request_sent(self, session_id: str) -> int:
        """Call when a session's request is about to be forwarded. Freezes the
        blocks of its previous request as the chain to judge, starts a new window,
        and returns a token for request_done."""
        with self._lock:
            concurrent = False
            for check in self._checks.values():
                if check["session"] == session_id:  # another request of this session in flight
                    check["concurrent"] = concurrent = True
            prev = self._windows.pop(session_id, None)
            self._windows[session_id] = _Window(mixed=concurrent)
            token = next(self._tokens)
            at = dict(prev.at) if prev else {}
            keys = set(at.values())
            self._checks[token] = {"session": session_id, "at": at, "keys": keys,
                                   "lost": {k for k in keys if k not in self._alive},
                                   "block_size": prev.block_size if prev else None,
                                   "concurrent": concurrent or bool(prev and prev.mixed),
                                   "errors": self._errors()}
            while len(self._checks) > self.max_sessions:  # requests that never completed
                self._checks.popitem(last=False)
            return token

    def request_done(self, token: int, cached_tokens: int | None,
                     prompt_tokens: int | None = None) -> tuple[int | None, int | None]:
        """At the response: (evicted_blocks, resident_blocks) for the classifier.

        (n > 0, 0): the block where the cache read stopped was removed: a confirmed
        eviction (n: blocks of the previous request removed). (0, n > 0): that block
        is still on the GPU: not an eviction. (None, None): nothing to judge, or the
        negative claim is not safe; the classifier keeps its inference."""
        with self._lock:
            check = self._checks.get(token)
            if check is None or cached_tokens is None or not check["at"] or not check["block_size"]:
                self._checks.pop(token, None)
                return NOTHING
            if prompt_tokens and cached_tokens >= prompt_tokens * 0.5:  # a hit: never examined
                self._checks.pop(token, None)
                return NOTHING
            key = check["at"].get(max(0, cached_tokens) // check["block_size"])
        if key is None:
            self._checks.pop(token, None)
            return NOTHING
        settled = key in check["lost"] or self._settle()  # removals published before the response
        with self._lock:
            self._checks.pop(token, None)
            if key in check["lost"]:
                return len(check["lost"]), 0
            if not settled or self.hybrid or check["concurrent"] or self._errors() != check["errors"]:
                return NOTHING
            return 0, len(check["keys"] - check["lost"])

    def discard(self, token: int | None) -> None:
        """The request never completed (upstream error, cut-off stream)."""
        if token is not None:
            with self._lock:
                self._checks.pop(token, None)

    def _settle(self) -> bool:
        """Wait until the stream has caught up with now: a batch stamped later
        than now was read (busy engine), or the socket was found empty (idle
        engine). False if that did not happen within settle_seconds."""
        if self._thread is None:
            return True
        asked = time.time()
        end = asked + self.settle_seconds
        while time.time() < end:
            offset = self._clock_offset or 0.0
            if self._drained_at >= asked or self._latest_ts + offset >= asked:
                return True
            time.sleep(0.002)
        return False

    def snapshot(self) -> dict:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {"endpoint": self.endpoint, "connected": alive, **self.stats, "hybrid_model": self.hybrid,
                    "gpu_blocks_tracked": len(self._alive), "sessions_tracked": len(self._windows),
                    "requests_in_flight": len(self._checks)}

    # -- subscriber ------------------------------------------------------------------

    def start(self) -> "EngineEvents":
        try:
            import msgspec  # noqa: F401
            import zmq  # noqa: F401
        except ImportError as exc:
            raise ValueError("--kv-events needs pyzmq and msgspec: pip install 'prefixwise[engines]'") from exc
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), name="prefixwise-kv-events", daemon=True)
        self._thread.start()
        ready.wait(5)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2)

    def _run(self, ready: threading.Event) -> None:
        import msgspec
        import zmq

        decode = msgspec.msgpack.Decoder().decode
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(self.endpoint)
        ready.set()
        try:
            while not self._stop.is_set():
                if not sock.poll(10):
                    self._drained_at = time.time()
                    continue
                frames = sock.recv_multipart()
                seq = int.from_bytes(frames[-2], "big", signed=True) if len(frames) >= 3 else None
                try:
                    self.handle(decode(frames[-1]), seq)
                except Exception:  # a malformed batch must never stop the subscriber
                    with self._lock:
                        self.stats["decode_errors"] += 1
        finally:
            sock.close(0)
