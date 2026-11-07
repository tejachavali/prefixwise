"""Engine join for vLLM: its KV-cache events tell an eviction from everything else.

Behind one self-hosted engine, a request whose prefix is unchanged but was not
reused usually means the engine evicted it (GPU memory full). Two look-alikes
exist: the token prefix differs although the request does not (a chat template
that renders history differently), and data-parallel routing. vLLM's KV events
separate them (docs/ENGINES.md):

1. With `--engine vllm` and `--kv-events`, Prefixwise sends its session id in the
   request body and `vllm_xargs: {"kv_cache_report_mode": "full"}`, so vLLM
   reports every block a request uses, reused or new, in order, tagged with the
   session (vllm/v1/core/kv_cache_manager.py, block_pool.py, 2026-11-06).
2. When a session's next request is sent, the blocks of its previous request are
   frozen as the chain to judge. Removals of those blocks are recorded from then
   until the response, so evictions while the request waits in vLLM's queue count.
3. At the response, `cached_tokens` says where the cache read stopped: block
   cached_tokens // block_size of the chain. If that block was removed, the miss
   is a confirmed eviction. If it is still on the GPU, the cache stopped at a block
   it still holds, so the token prefix changed there: not an eviction. Only that
   block matters, so output blocks a template never reuses, or older chains, cannot
   confirm anything.

Only full-attention block groups are tracked: hybrid models free sliding-window
blocks routinely. Offload tiers (LMCache) are not tracked; a prefix they restore
is a hit and never examined. Known limit: concurrent requests in one session
share a window, so a chain can mix branches (docs/ENGINES.md).

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


class _Window:
    """The blocks one session's requests reported since its last send, in order."""

    __slots__ = ("keys", "seen", "block_size")

    def __init__(self):
        self.keys: list = []
        self.seen: set = set()
        self.block_size: int | None = None


class EngineEvents:
    def __init__(self, endpoint: str | None = None, max_sessions: int = 1000, max_chain: int = 4096,
                 max_alive: int = 1_000_000, settle_seconds: float = 0.05):
        self.endpoint = endpoint
        self.max_sessions, self.max_chain, self.max_alive = max_sessions, max_chain, max_alive
        self.settle_seconds = settle_seconds
        self._lock = threading.Lock()
        self._alive: set = set()  # (group, block hash) on the GPU, full-attention groups only
        self._windows: OrderedDict = OrderedDict()  # session -> _Window
        self._checks: OrderedDict = OrderedDict()  # token -> in-flight check
        self._tokens = itertools.count(1)
        self.stats = {"batches": 0, "stored": 0, "removed": 0, "cleared": 0, "engine_restarts": 0,
                      "dropped_messages": 0, "decode_errors": 0, "overflows": 0}
        self._last_seq: int | None = None
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
                if kind == "BlockStored" and ev.get("medium") in GPU and ev.get("kv_cache_spec_kind") in FULL_KINDS:
                    self._store(ev)
                elif kind == "BlockRemoved" and ev.get("medium") in GPU:
                    group = ev.get("group_idx") or 0
                    for h in ev.get("block_hashes") or []:
                        self._remove((group, h))
                elif kind == "AllBlocksCleared":
                    self.stats["cleared"] += 1
                    self._clear()

    def _store(self, ev: dict) -> None:
        group, hashes, session = ev.get("group_idx") or 0, ev.get("block_hashes") or [], ev.get("session_id")
        self.stats["stored"] += len(hashes)
        keys = [(group, h) for h in hashes]
        self._alive.update(keys)
        if len(self._alive) > self.max_alive:  # removals were missed: stop trusting residency
            self._alive.clear()
            self.stats["overflows"] += 1
        if session:
            w = self._window(session)
            w.block_size = w.block_size or ev.get("block_size")
            for k in keys:  # in order; a request re-reported while queued adds nothing new
                if k not in w.seen and len(w.keys) < self.max_chain:
                    w.seen.add(k)
                    w.keys.append(k)

    def _remove(self, key) -> None:
        self.stats["removed"] += 1
        self._alive.discard(key)
        for check in self._checks.values():
            if key in check["chain_set"]:
                check["lost"].add(key)

    def _clear(self) -> None:
        for key in list(self._alive):
            self._remove(key)

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
            prev = self._windows.pop(session_id, None)
            self._window(session_id)
            token = next(self._tokens)
            chain = prev.keys if prev else []
            self._checks[token] = {"chain": chain, "chain_set": set(chain),
                                   "lost": {k for k in chain if k not in self._alive},
                                   "block_size": prev.block_size if prev else None,
                                   "errors": self._errors(), "sent": time.time()}
            while len(self._checks) > self.max_sessions:  # requests that never completed
                self._checks.popitem(last=False)
            return token

    def request_done(self, token: int, cached_tokens: int) -> tuple[int | None, int | None]:
        """At the response: (evicted_blocks, resident_blocks) for the classifier.

        (n > 0, 0): the block where the cache read stopped was removed: a confirmed
        eviction (n: previous-request blocks removed). (0, n > 0): that block is still
        on the GPU: not an eviction. (None, None): nothing to judge, or the event
        stream lost messages meanwhile; the classifier keeps its inference."""
        self._settle()
        with self._lock:
            check = self._checks.pop(token, None)
            if not check or not check["chain"] or not check["block_size"]:
                return None, None
            idx = max(0, cached_tokens) // check["block_size"]
            if idx >= len(check["chain"]):
                return None, None
            if check["chain"][idx] in check["lost"]:
                return len(check["lost"]), 0
            if self._errors() != check["errors"]:
                return None, None
            return 0, len(check["chain"]) - len(check["lost"])

    def _settle(self) -> None:
        """Wait until the subscriber has drained the socket (events published before
        the response have been read), at most settle_seconds."""
        if self._thread is None:
            return
        asked, end = time.time(), time.time() + self.settle_seconds
        while self._drained_at < asked and time.time() < end:
            time.sleep(0.002)

    def snapshot(self) -> dict:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {"endpoint": self.endpoint, "connected": alive, **self.stats,
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
