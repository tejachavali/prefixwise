"""Engine join for vLLM: its KV-cache events tell an eviction from everything else.

Behind one self-hosted engine, a request whose prefix is unchanged but was not
reused has three usual causes: the engine evicted the blocks (out of GPU
memory), the token prefix differs although the request does not (a chat template
that renders history differently), or the request went to another data-parallel
rank. The response alone cannot separate them. vLLM's KV events can:

- vLLM puts the request body's `session_id` on every BlockStored event. With
  `--engine vllm` Prefixwise sends its session id there, so this module knows
  which blocks each session holds on the GPU.
- BlockRemoved of one of those blocks between two turns: a confirmed eviction.
- No removal and still a miss: the engine still holds the session's blocks, so it
  was not an eviction (see docs/ENGINES.md).

Wire format (vllm/distributed/kv_events.py, 2026-11-06): ZMQ PUB, multipart
[topic, seq (8 bytes, big-endian), msgpack [ts, events, dp_rank?]], each event a
map tagged by "type". Only GPU removals count; offload tiers (LMCache) are
ignored, and a prefix they restore is a hit anyway.

SGLang publishes the same format, but it reads a body `session_id` as its own
session handle and rejects unknown ones, so its events cannot be attributed to
sessions and are not used.

Needs `pip install "prefixwise[engines]"` (pyzmq, msgspec).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

GPU = (None, "GPU")


class EngineEvents:
    def __init__(self, endpoint: str | None = None, max_blocks: int = 500_000):
        self.endpoint = endpoint
        self.max_blocks = max_blocks  # far above any engine's GPU block count
        self._lock = threading.Lock()
        self._owners: OrderedDict = OrderedDict()  # resident block hash -> session ids that stored it
        self._resident: dict[str, int] = {}  # session -> its blocks still on the GPU
        self._removed: dict[str, int] = {}  # session -> its blocks removed since its last turn
        self.stats = {"batches": 0, "stored": 0, "removed": 0, "cleared": 0, "engine_restarts": 0,
                      "dropped_messages": 0, "decode_errors": 0}
        self._last_seq: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- events ----------------------------------------------------------------------

    def handle(self, batch, seq: int | None = None) -> None:
        """One decoded payload: [ts, events, dp_rank?]."""
        with self._lock:
            if seq is not None and self._last_seq is not None:
                if seq > self._last_seq + 1:  # lost messages: residency is no longer certain
                    self.stats["dropped_messages"] += seq - self._last_seq - 1
                elif seq <= self._last_seq:  # sequence restarted: the engine restarted, its cache is empty
                    self.stats["engine_restarts"] += 1
                    self._clear()
            self._last_seq = seq if seq is not None else self._last_seq
            self.stats["batches"] += 1
            for ev in batch[1]:
                kind = ev.get("type")
                if kind == "BlockStored" and ev.get("medium") in GPU:
                    self._store(ev.get("block_hashes") or [], ev.get("session_id"))
                elif kind == "BlockRemoved" and ev.get("medium") in GPU:
                    for h in ev.get("block_hashes") or []:
                        self._remove(h)
                elif kind == "AllBlocksCleared":
                    self.stats["cleared"] += 1
                    self._clear()

    def _store(self, hashes: list, session: str | None) -> None:
        self.stats["stored"] += len(hashes)
        if not session:
            return
        for h in hashes:
            owners = self._owners.setdefault(h, set())
            if session not in owners:
                owners.add(session)
                self._resident[session] = self._resident.get(session, 0) + 1
            self._owners.move_to_end(h)
        while len(self._owners) > self.max_blocks:  # bound memory if removals were missed
            for s in self._owners.popitem(last=False)[1]:
                self._resident[s] = max(0, self._resident.get(s, 0) - 1)

    def _clear(self) -> None:
        for h in list(self._owners):
            self._remove(h)

    def _remove(self, h) -> None:
        self.stats["removed"] += 1
        for s in self._owners.pop(h, ()):
            self._resident[s] = max(0, self._resident.get(s, 0) - 1)
            self._removed[s] = self._removed.get(s, 0) + 1

    # -- per turn --------------------------------------------------------------------

    def for_turn(self, session_id: str) -> tuple[int, int]:
        """(this session's blocks removed since its last turn, its blocks still resident).
        Consumes the removal count. Once any message was lost or unreadable, a
        removal may have been missed, so residency is reported as 0 (unknown) and
        the classifier falls back to inference instead of ruling eviction out."""
        with self._lock:
            removed = self._removed.pop(session_id, 0)
            trusted = not (self.stats["dropped_messages"] or self.stats["decode_errors"])
            return removed, self._resident.get(session_id, 0) if trusted else 0

    def snapshot(self) -> dict:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {"endpoint": self.endpoint, "connected": alive, **self.stats,
                    "tracked_blocks": len(self._owners), "tracked_sessions": len(self._resident)}

    # -- subscriber ------------------------------------------------------------------

    def start(self) -> "EngineEvents":
        try:
            import msgspec  # noqa: F401
            import zmq  # noqa: F401
        except ImportError as exc:
            raise SystemExit("prefixwise: --kv-events needs pyzmq and msgspec: "
                             "pip install 'prefixwise[engines]'") from exc
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
