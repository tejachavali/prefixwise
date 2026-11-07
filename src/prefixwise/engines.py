"""Engine join for vLLM: its KV-cache events tell an eviction from everything else.

Behind one self-hosted engine, a request whose prefix is unchanged but was not
reused usually means the engine evicted it (GPU memory full). Two look-alikes
exist: the token prefix differs although the request does not (a chat template
that renders history differently), and data-parallel routing. vLLM's KV events
separate them (docs/ENGINES.md):

- With `--engine vllm` and `--kv-events`, Prefixwise sends its session id in the
  request body and `vllm_xargs: {"kv_cache_report_mode": "full"}`. vLLM then
  reports every block a request uses, reused or new, as BlockStored with that
  session id (vllm/v1/core/kv_cache_manager.py, block_pool.py, 2026-11-06).
- So the blocks of a session's previous request are known. When its next
  request is sent, Prefixwise checks them against the blocks still on the GPU:
  some gone is a confirmed eviction; all there means it was not an eviction.
  The check runs at send time because the evictions that matter happen between
  turns, and by then their events have arrived.

Only full-attention block groups are tracked: on hybrid models, sliding-window
blocks are freed routinely without breaking the prefix. Offload tiers (LMCache)
are not tracked; a prefix they restore is a hit and never examined.

Wire format (vllm/distributed/kv_events.py): ZMQ PUB, multipart [topic, seq
(8 bytes, big-endian), msgpack [ts, events, dp_rank?]], each event a map tagged
by "type". Needs `pip install "prefixwise[engines]"` (pyzmq, msgspec).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

GPU = (None, "GPU")
FULL_KINDS = (None, "full_attention", "mla_attention", "sink_full_attention")


class EngineEvents:
    def __init__(self, endpoint: str | None = None, max_sessions: int = 1000, max_chain: int = 4096,
                 max_alive: int = 1_000_000):
        self.endpoint = endpoint
        self.max_sessions, self.max_chain, self.max_alive = max_sessions, max_chain, max_alive
        self._lock = threading.Lock()
        self._alive: set = set()  # (group, block hash) on the GPU, full-attention groups only
        self._window: OrderedDict = OrderedDict()  # session -> [keys its requests used since its last send]
        self._mark: dict = {}  # session -> error count when its window opened
        self.stats = {"batches": 0, "stored": 0, "removed": 0, "cleared": 0, "engine_restarts": 0,
                      "dropped_messages": 0, "decode_errors": 0, "overflows": 0}
        self._last_seq: int | None = None
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
                    self._alive.clear()
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
                        self.stats["removed"] += 1
                        self._alive.discard((group, h))
                elif kind == "AllBlocksCleared":
                    self.stats["cleared"] += 1
                    self._alive.clear()

    def _store(self, ev: dict) -> None:
        group, hashes, session = ev.get("group_idx") or 0, ev.get("block_hashes") or [], ev.get("session_id")
        self.stats["stored"] += len(hashes)
        keys = [(group, h) for h in hashes]
        self._alive.update(keys)
        if len(self._alive) > self.max_alive:  # removals were missed: stop trusting residency
            self._alive.clear()
            self.stats["overflows"] += 1
        if session:
            window = self._window.get(session)
            if window is None:
                window = self._window[session] = []
                self._mark.setdefault(session, self._errors())
            room = self.max_chain - len(window)
            if room > 0:  # the head of the prefix is what decides a miss
                window.extend(keys[:room])

    # -- per request -----------------------------------------------------------------

    def request_sent(self, session_id: str) -> tuple[int, int]:
        """Call when a session's next request is about to be forwarded. Returns
        (blocks of its previous request no longer on the GPU, blocks still there)
        and starts a new window. If any message was lost or unreadable meanwhile,
        a removal may have been missed: residency is reported as 0 (unknown), so
        the classifier will not claim "not an eviction"."""
        with self._lock:
            chain = self._window.pop(session_id, None) or []
            mark = self._mark.pop(session_id, self._errors())
            lost = sum(1 for k in chain if k not in self._alive)
            resident = len(chain) - lost if self._errors() == mark else 0
            self._window[session_id] = []
            self._mark[session_id] = self._errors()
            while len(self._window) > self.max_sessions:
                old, _ = self._window.popitem(last=False)
                self._mark.pop(old, None)
            return lost, resident

    def snapshot(self) -> dict:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {"endpoint": self.endpoint, "connected": alive, **self.stats,
                    "gpu_blocks_tracked": len(self._alive), "sessions_tracked": len(self._window)}

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
                if not sock.poll(200):
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
