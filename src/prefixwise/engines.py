"""Engine join: read a self-hosted engine's KV-cache events, so an engine eviction
is named `evicted` instead of looking like a routing miss.

vLLM and SGLang publish the same stream when started with --kv-events-config:
ZMQ PUB, multipart [topic, seq (8 bytes, big-endian), payload], where payload is
msgpack [ts, events, data_parallel_rank?] and each event is a map tagged by
"type": BlockStored {block_hashes, medium, session_id, ...}, BlockRemoved
{block_hashes, medium}, AllBlocksCleared. (vllm/distributed/kv_events.py,
sglang/srt/disaggregation/kv_events.py, read 2026-11-07.)

Attribution:
- vLLM stamps the request's body `session_id` onto BlockStored. With
  `--engine vllm`, Prefixwise sends its session id there, so a later
  BlockRemoved of those blocks is attributed to exactly that session.
- SGLang reads a body `session_id` as its own stateful-session handle and
  rejects unknown ones, so Prefixwise does not send it; there, evidence is
  engine-wide ("the engine evicted N blocks since this session's last turn").
  That is weaker evidence and is labelled as such.

Only GPU-tier removals (medium "GPU" or unset) count as evictions. Blocks
removed from an offload tier (CPU/STORAGE, e.g. LMCache) are counted
separately. With an offload tier a GPU eviction that is restored shows up as a
hit, and a hit wins before any eviction rule.

Needs `pip install "prefixwise[engines]"` (pyzmq, msgspec). The core proxy
stays standard-library only.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

GPU_MEDIA = (None, "GPU", "gpu")


class EngineEvents:
    """Consumes KV events and answers, per turn: what was evicted since this
    session's previous turn? Thread-safe; bounded memory."""

    def __init__(self, endpoint: str | None = None, topic: str = "", max_blocks: int = 2_000_000,
                 max_sessions: int = 10_000):
        self.endpoint, self.topic = endpoint, topic
        self.max_blocks, self.max_sessions = max_blocks, max_sessions
        self._lock = threading.Lock()
        self._block_sessions: OrderedDict = OrderedDict()  # block hash -> set of session ids
        self._evicted: OrderedDict = OrderedDict()  # session -> its GPU blocks removed since last turn
        self._attributed: OrderedDict = OrderedDict()  # sessions seen in BlockStored.session_id
        self._mark: OrderedDict = OrderedDict()  # session -> removed_gpu at its last turn
        self.stats = {"batches": 0, "events": 0, "stored": 0, "removed_gpu": 0, "removed_offload": 0,
                      "cleared": 0, "decode_errors": 0, "dropped_messages": 0, "last_event_at": None}
        self._last_seq: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- events --------------------------------------------------------------------

    @staticmethod
    def _bounded_set(d: OrderedDict, key, limit: int, value=True):
        d[key] = value
        d.move_to_end(key)
        while len(d) > limit:
            d.popitem(last=False)

    def handle_batch(self, batch, seq: int | None = None) -> None:
        """One decoded payload: [ts, events, dp_rank?] (or a dict with 'events')."""
        events = batch.get("events") if isinstance(batch, dict) else (batch[1] if len(batch) > 1 else [])
        with self._lock:
            if seq is not None:
                if self._last_seq is not None and seq > self._last_seq + 1:
                    self.stats["dropped_messages"] += seq - self._last_seq - 1
                self._last_seq = seq
            self.stats["batches"] += 1
            self.stats["last_event_at"] = time.time()
            for ev in events or []:
                self.stats["events"] += 1
                kind = ev.get("type") if isinstance(ev, dict) else None
                if kind == "BlockStored":
                    self._stored(ev)
                elif kind == "BlockRemoved":
                    self._removed(ev)
                elif kind == "AllBlocksCleared":
                    self._cleared()

    def _stored(self, ev: dict) -> None:
        hashes = ev.get("block_hashes") or []
        self.stats["stored"] += len(hashes)
        sid = ev.get("session_id")
        if not sid or ev.get("medium") not in GPU_MEDIA:
            return
        self._bounded_set(self._attributed, sid, self.max_sessions)
        for h in hashes:
            owners = self._block_sessions.get(h)
            if owners is None:
                owners = set()
            owners.add(sid)
            self._bounded_set(self._block_sessions, h, self.max_blocks, owners)

    def _removed(self, ev: dict) -> None:
        hashes = ev.get("block_hashes") or []
        if ev.get("medium") not in GPU_MEDIA:
            self.stats["removed_offload"] += len(hashes)
            return
        self.stats["removed_gpu"] += len(hashes)
        for h in hashes:
            for sid in self._block_sessions.pop(h, ()):
                self._bounded_set(self._evicted, sid, self.max_sessions, self._evicted.get(sid, 0) + 1)

    def _cleared(self) -> None:
        self.stats["cleared"] += 1
        n = max(1, len(self._block_sessions))
        self.stats["removed_gpu"] += n
        for owners in self._block_sessions.values():
            for sid in owners:
                self._bounded_set(self._evicted, sid, self.max_sessions, self._evicted.get(sid, 0) + 1)
        self._block_sessions.clear()

    # -- per turn --------------------------------------------------------------------

    def for_turn(self, session_id: str) -> tuple[int, int, bool]:
        """Consume what happened since this session's previous turn.

        Returns (attributed, engine_wide, attributable): this session's GPU blocks
        removed; GPU blocks the engine removed in total; and whether this session's
        blocks are attributed at all (vLLM with session ids) so callers can prefer
        the precise signal and ignore the engine-wide one."""
        with self._lock:
            attributed = self._evicted.pop(session_id, 0)
            total = self.stats["removed_gpu"]
            prev = self._mark.get(session_id)
            self._bounded_set(self._mark, session_id, self.max_sessions, total)
            engine_wide = total - prev if prev is not None else 0
            return attributed, engine_wide, session_id in self._attributed

    def snapshot(self) -> dict:
        with self._lock:
            return {"endpoint": self.endpoint, "connected": self._thread is not None and self._thread.is_alive(),
                    **self.stats, "tracked_blocks": len(self._block_sessions),
                    "attributed_sessions": len(self._attributed)}

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

        decoder = msgspec.msgpack.Decoder()
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SUBSCRIBE, self.topic.encode())
        sock.connect(self.endpoint)
        ready.set()
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if not dict(poller.poll(200)):
                    continue
                frames = sock.recv_multipart()
                payload = frames[-1]
                seq = int.from_bytes(frames[-2], "big", signed=True) if len(frames) >= 3 else None
                try:
                    self.handle_batch(decoder.decode(payload), seq)
                except Exception:  # a malformed batch must never stop the subscriber
                    with self._lock:
                        self.stats["decode_errors"] += 1
        finally:
            sock.close(0)
