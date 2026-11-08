"""Engine join for vLLM and SGLang: KV-cache events tell an eviction from everything else.

SGLang (>= 0.5.20) is `SGLangEvents`, below; the rest of this note is about vLLM.

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
Offload tiers (LMCache, vLLM CPU offload, SGLang HiCache) are not tracked as caches: a
prefix they restore is a hit (`cached_tokens` counts it). On such a hit, blocks of the
previous request that the read covered yet had been removed from the GPU are reported
as the evicted count, and the collector names the hit's detail `host_restored`
(prefixwise.details): served from a host or offload tier after a GPU eviction, no
prefill recomputed. Claimed only when the stream has shown the tier's own events (a
BlockStored on a CPU or storage medium) and, on vLLM, no other session stored a lost
block while the request was in flight: a request sharing the prefix that recomputed
it looks the same otherwise. One cache only; with several, which cache served is
unknown.

Wire format (vllm/distributed/kv_events.py): ZMQ PUB, multipart [topic, seq
(8 bytes, big-endian), msgpack [ts, events, dp_rank?]], each event a map tagged
by "type". Needs `pip install "prefixwise[engines]"` (pyzmq, msgspec).
"""

from __future__ import annotations

import itertools
import threading
import time
from array import array
from collections import OrderedDict

GPU = (None, "GPU")
FULL_KINDS = (None, "full_attention", "mla_attention", "sink_full_attention")
NOTHING = (None, None)


class _Window:
    """Blocks one session's requests reported since its last send, by position."""

    __slots__ = ("at", "block_size", "mixed", "seen")

    def __init__(self, mixed: bool = False):
        self.at: dict = {}  # position -> (group, hash); a later report of a position wins
        self.block_size: int | None = None
        self.mixed = mixed  # concurrent requests reported into it
        self.seen = 0  # blocks reported for this session since the send: this engine served it


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
        self.offload_seen = False  # a BlockStored on a non-GPU medium: an offload tier (CPU, storage) exists
        self.stats = {"batches": 0, "stored": 0, "removed": 0, "cleared": 0, "engine_restarts": 0,
                      "dropped_messages": 0, "decode_errors": 0, "overflows": 0, "disconnects": 0,
                      "wrong_rank": 0}
        self.rank: int | None = None  # the data-parallel rank this stream must carry (batch[2]), if known
        self._connected = False  # the subscriber's TCP connection to the publisher is up
        self._last_seq: int | None = None
        self._latest_ts = 0.0  # engine timestamp of the last batch read
        self._clock_offset: float | None = None  # min(receive time - engine ts): skew + least latency
        self._drained_at = 0.0  # last time the subscriber found the socket empty
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _errors(self) -> int:
        s = self.stats
        return s["dropped_messages"] + s["decode_errors"] + s["overflows"] + s["disconnects"] + s["wrong_rank"]

    def _forget(self) -> None:
        """The stream was cut (the engine died, was replaced, or the network dropped): removals
        may have been missed, so nothing known about residency stands. In-flight checks keep
        only removals actually seen; their negative claims stop (errors changed)."""
        self._alive.clear()
        self._windows.clear()

    # -- events ----------------------------------------------------------------------

    def handle(self, batch, seq: int | None = None) -> None:
        """One decoded payload: [ts, events, dp_rank?]."""
        with self._lock:
            if (self.rank is not None and len(batch) > 2 and batch[2] is not None
                    and batch[2] != self.rank):  # another rank's stream: a misconfiguration
                self.stats["wrong_rank"] += 1
                return
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
                self._event(ev)
            if isinstance(batch[0], (int, float)):
                ts = float(batch[0])
                self._latest_ts = max(self._latest_ts, ts)
                offset = time.time() - ts  # the engine may run on another host with another clock
                self._clock_offset = offset if self._clock_offset is None else min(self._clock_offset, offset)

    def _event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "BlockStored" and ev.get("medium") not in GPU:
            self.offload_seen = True  # the tier's own events (CPU offload, LMCache): not tracked, but it exists
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

    def _store(self, ev: dict) -> None:
        group, hashes, session = ev.get("group_idx") or 0, ev.get("block_hashes") or [], ev.get("session_id")
        self.stats["stored"] += len(hashes)
        parent = ev.get("parent_block_hash")
        parent_pos = self._alive.get((group, parent)) if parent is not None else -1
        pos = None if parent_pos is None else parent_pos + 1  # unknown parent: unplaced
        w = self._window(session) if session else None
        if w is not None:
            w.block_size = w.block_size or ev.get("block_size")
            w.seen += len(hashes)
        for h in hashes:
            key = (group, h)
            self._alive[key] = pos
            if w is not None and pos is not None and pos < self.max_chain:
                w.at[pos] = key
            pos = None if pos is None else pos + 1
            for check in self._checks.values():  # in-flight requests: few
                if key in check["lost"] and session != check["session"]:
                    check["restored_elsewhere"] = True  # another request recomputed it: not a tier restore

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
            window = self._windows[session_id] = _Window(mixed=concurrent)
            token = next(self._tokens)
            at = dict(prev.at) if prev else {}
            keys = set(at.values())
            self._checks[token] = {"session": session_id, "at": at, "keys": keys, "window": window,
                                   "lost": {k for k in keys if k not in self._alive},
                                   "block_size": prev.block_size if prev else None,
                                   "concurrent": concurrent or bool(prev and prev.mixed),
                                   "restored_elsewhere": False, "errors": self._errors()}
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
            if prompt_tokens and cached_tokens >= prompt_tokens * 0.5:
                # A hit: no eviction to judge. Blocks of the previous request that this read
                # covered yet had been removed from the GPU came back from a host or offload
                # tier (details: host_restored). Claimed only from a clean, uncontended stream.
                stop = max(0, cached_tokens) // check["block_size"]
                restored = sum(1 for pos, k in check["at"].items() if pos < stop and k in check["lost"])
                # Only when an offload tier exists (its events were seen) and no other request
                # stored a lost block meanwhile: otherwise a re-computation looks the same.
                clean = (self.offload_seen and not check["concurrent"] and not check["restored_elsewhere"]
                         and self._errors() == check["errors"])
                self._checks.pop(token, None)
                return (restored, 0) if restored and clean else NOTHING
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

    # -- several caches (EngineFleet) -----------------------------------------------

    def _served_previous(self, token: int) -> bool:
        """This engine reported blocks for the session's previous request."""
        with self._lock:
            check = self._checks.get(token)
            return bool(check and check["at"])

    def _served_now(self, token: int) -> bool:
        """This engine reported blocks for the request this token was sent for."""
        with self._lock:
            check = self._checks.get(token)
            return bool(check and check["window"].seen)

    def _stored_at(self, token: int, cached_tokens: int) -> bool:
        """This engine served the request this token was sent for (vLLM reports every block
        a request uses, so any report of the session since the send says so)."""
        return self._served_now(token)

    def _reported(self, token: int, cached_tokens: int) -> tuple:
        """(key, block size) this engine reported for the request at the block where its cache
        read stopped: the first block it did not have, which it then computed."""
        with self._lock:
            check = self._checks.get(token)
            w = check["window"] if check else None
            if w is None or not w.block_size:
                return None, None
            return w.at.get(max(0, cached_tokens) // w.block_size), w.block_size

    def held_here(self, token: int, cached_tokens: int | None, prompt_tokens: int | None = None,
                  needed: tuple = (None, None)) -> int | None:
        """The session's previous request was served here; this one by another engine, which
        reported `needed` = (key, block size) for the block where its cache read stopped. If this
        engine still holds that very block, the prefix this request needed was cached, on the
        wrong engine: returns how many of the previous request's blocks are still resident here.
        None when that cannot be said for sure (another block there, a changed prefix, a side
        call, residency unsure). Call after _settle()."""
        with self._lock:
            check = self._checks.pop(token, None)
            if check is None or cached_tokens is None or not check["at"] or not check["block_size"]:
                return None
            if prompt_tokens and cached_tokens >= prompt_tokens * 0.5:  # a hit: never examined
                return None
            key = check["at"].get(max(0, cached_tokens) // check["block_size"])
            if key is None or key in check["lost"]:
                return None
            if needed[0] is None or needed != (key, check["block_size"]):
                return None  # the request needed another block there: the prefix differs
            if self.hybrid or check["concurrent"] or self._errors() != check["errors"]:
                return None  # residency could be wrong, or another request of the session used it
            return len(check["keys"] - check["lost"])

    def _settle(self, asked: float | None = None, end: float | None = None) -> bool:
        """Wait until the stream has caught up with `asked` (default now): a batch stamped
        later was read (busy engine), or the socket was found empty (idle engine). False if
        that did not happen by `end` (default settle_seconds from now)."""
        if self._thread is None:
            return True
        asked = time.time() if asked is None else asked
        end = asked + self.settle_seconds if end is None else end
        while True:
            offset = self._clock_offset or 0.0
            if self._drained_at >= asked or self._latest_ts + offset >= asked:
                return True
            if time.time() >= end:
                return False
            time.sleep(0.002)

    def snapshot(self) -> dict:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive() and self._connected
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

        from zmq.utils.monitor import recv_monitor_message

        decode = msgspec.msgpack.Decoder().decode
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        if "[" in (self.endpoint or ""):
            sock.setsockopt(zmq.IPV6, 1)
        # A SUB socket never notices on its own that its publisher is gone; the monitor does.
        monitor = sock.get_monitor_socket(zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        poller.register(monitor, zmq.POLLIN)
        sock.connect(self.endpoint)
        ready.set()
        try:
            while not self._stop.is_set():
                ready_socks = dict(poller.poll(10))
                if monitor in ready_socks:
                    event = recv_monitor_message(monitor)["event"]
                    with self._lock:
                        if event == zmq.EVENT_DISCONNECTED:
                            self._connected = False
                            self.stats["disconnects"] += 1
                            self._forget()
                        elif event == zmq.EVENT_CONNECTED:
                            self._connected = True
                if sock not in ready_socks:
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
            sock.disable_monitor()
            monitor.close(0)
            sock.close(0)


class EngineFleet:
    """Several KV caches behind one upstream: vLLM data-parallel ranks or replicas behind a
    router, or SGLang data-parallel ranks. One events reader per cache (`make`), the same
    interface as one.

    No router has to say which cache served a request: vLLM reports every block a request uses
    with its session (kv_cache_report_mode "full"); SGLang reports the blocks a request newly
    stores, with its session, starting exactly where its cache read stopped (`_stored_at`). So
    the cache whose stream reported this request's blocks served it, and the one that reported
    them for the previous request served that. At the response:

    - the same cache served both: that cache's own verdict (evicted, or still resident);
    - another cache served this one, and the previous one still holds the block where this
      one's cache read stopped: a confirmed routing miss, reported as
      (None, None, (previous cache, serving cache, blocks still resident there));
    - anything else (a cache that cannot be told, a stream behind, lost messages, concurrent
      requests of the session): nothing, and the classifier makes no engine claim.
    """

    def __init__(self, sources: list[tuple[str, str]], max_sessions: int = 1000, make=None, **kw):
        if len(sources) < 2:
            raise ValueError("EngineFleet needs at least two KV-event sources")
        self.labels = [label for label, _ in sources]
        make = make or (lambda endpoint: EngineEvents(endpoint, max_sessions=max_sessions, **kw))
        self.sources = [make(endpoint) for _, endpoint in sources]
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        self._checks: OrderedDict = OrderedDict()  # token -> (session, per-source tokens)
        self._tokens = itertools.count(1)

    @property
    def hybrid(self) -> bool:
        return any(s.hybrid for s in self.sources)

    @hybrid.setter
    def hybrid(self, value: bool) -> None:
        for s in self.sources:
            s.hybrid = value

    def start(self) -> "EngineFleet":
        for s in self.sources:
            s.start()
        return self

    def stop(self) -> None:
        for s in self.sources:
            s.stop()

    def request_sent(self, session_id: str) -> int:
        tokens = [s.request_sent(session_id) for s in self.sources]
        with self._lock:
            token = next(self._tokens)
            self._checks[token] = (session_id, tokens)
            while len(self._checks) > self.max_sessions:
                _, (_, stale) = self._checks.popitem(last=False)
                for s, t in zip(self.sources, stale):
                    s.discard(t)
            return token

    def discard(self, token: int | None) -> None:
        if token is None:
            return
        with self._lock:
            entry = self._checks.pop(token, None)
        if entry is not None:
            for s, t in zip(self.sources, entry[1]):
                s.discard(t)

    def request_done(self, token: int, cached_tokens: int | None, prompt_tokens: int | None = None) -> tuple:
        with self._lock:
            entry = self._checks.pop(token, None)
        if entry is None:
            return NOTHING
        session, tokens = entry
        verdict = NOTHING
        hit = cached_tokens is None or bool(prompt_tokens and cached_tokens >= prompt_tokens * 0.5)
        # Every stream caught up with now (one deadline for all), so this request's own reports
        # have arrived. A hit, or no cached_tokens, is never examined: no wait.
        stop = max(0, cached_tokens or 0)
        # Which cache served the previous turn is fixed at the send (chains are frozen then).
        before = [i for i, (s, t) in enumerate(zip(self.sources, tokens)) if s._served_previous(t)]
        if not hit and len(before) == 1 and all(s._thread is not None for s in self.sources):
            # SGLang publishes a request's store after its first output: wait for one, but only
            # when a claim is possible at all (one cache served the previous turn).
            wait = time.time() + max(getattr(s, "barrier_seconds", 0.0) for s in self.sources)
            while time.time() < wait and not any(s._stored_at(t, stop) for s, t in zip(self.sources, tokens)):
                time.sleep(0.002)
        asked = time.time()
        end = asked + max(s.settle_seconds for s in self.sources)
        if not hit and len(before) <= 1 and all(s._settle(asked, end) for s in self.sources):
            now = [i for i, (s, t) in enumerate(zip(self.sources, tokens)) if s._stored_at(t, stop)]
            if len(now) == 1:
                c = now[0]
                if not before or before[0] == c:
                    verdict = self.sources[c].request_done(tokens[c], cached_tokens, prompt_tokens)
                else:
                    p = before[0]
                    needed = self.sources[c]._reported(tokens[c], cached_tokens)
                    held = self.sources[p].held_here(tokens[p], cached_tokens, prompt_tokens, needed)
                    if held:
                        verdict = (None, None, (self.labels[p], self.labels[c], held))
        for s, t in zip(self.sources, tokens):
            s.discard(t)  # no-op for the token already judged
        return verdict

    def snapshot(self) -> dict:
        snaps = [s.snapshot() for s in self.sources]
        total = {k: sum(int(x.get(k) or 0) for x in snaps)
                 for k in ("batches", "stored", "removed", "cleared", "engine_restarts", "dropped_messages",
                           "decode_errors", "overflows", "disconnects", "wrong_rank", "gpu_blocks_tracked",
                           "requests_in_flight")}
        return {"endpoint": ", ".join(str(x["endpoint"]) for x in snaps),
                "connected": all(x["connected"] for x in snaps), **total, "hybrid_model": self.hybrid,
                "caches": [{"label": label, **x} for label, x in zip(self.labels, snaps)]}


class _Seg:
    """Blocks SGLang stored in one event, in chain order: one radix-tree node or part of one.

    SGLang reports only the blocks it newly stores, each run linked to its parent
    block, so a request's full chain is this run plus its ancestors (`parent`,
    up to block `parent_last`). Positions are in tokens."""

    __slots__ = ("ambiguous", "block", "born", "cut", "hashes", "parent", "parent_last", "start", "tokens")

    def __init__(self, parent, parent_last, start, block, tokens, hashes, born):
        self.parent, self.parent_last = parent, parent_last
        self.start = start  # token position of block 0; None if an ancestor was never seen
        self.block, self.tokens = block, tokens  # tokens per block, tokens in the run
        self.hashes = hashes  # packed; the index maps a hash to its run only, to save memory
        self.born = born  # event counter when stored
        self.cut: int | None = None  # first block removed from the GPU (SGLang evicts leaves first)
        self.ambiguous = False  # a hash of it is also on another subtree: removals cannot be told apart

    def end(self, last: int) -> int:
        """Token position after block `last`."""
        return self.start + min(self.tokens, (last + 1) * self.block)


def _chain(tail) -> list:
    """[(run, last block used)] from the root to tail."""
    out = []
    seg, last = tail
    while seg is not None:
        out.append((seg, last))
        seg, last = seg.parent, seg.parent_last
    out.reverse()
    return out


class SGLangEvents(EngineEvents):
    """Engine join for SGLang >= 0.5.20 (docs/ENGINES.md, section "SGLang").

    A body `session_id` is stamped on the blocks a request newly stores; blocks it
    reuses are not reported again, so Prefixwise keeps a copy of the engine's tree
    of GPU blocks and walks a session's chain through parent links. Blocks are one
    token by default (`--page-size`). GPU eviction removes leaves first, so a
    removal on a chain removes everything after it. Events carry no attention
    kind: on sliding-window or Mamba models SGLang frees those layers without an
    event, so "not an eviction" is claimed only for a model declared
    full-attention. SGLang queues events in order and publishes them after a
    step's output, so the negative claim waits for the request's own store, which
    follows every eviction made to admit it."""

    def __init__(self, endpoint: str | None = None, full_attention: bool = False,
                 max_alive: int = 2_000_000, max_held_tokens: int = 16_000_000,
                 barrier_seconds: float = 0.5, **kw):
        super().__init__(endpoint, max_alive=max_alive, **kw)
        self.full_attention = full_attention
        self.hybrid = not full_attention
        self.max_held_tokens = max_held_tokens  # tokens of chains kept for sessions (8 bytes each)
        self.barrier_seconds = barrier_seconds
        self._alive: dict = {}  # block hash on the GPU -> its run
        # session -> [(run, last index) or None, mixed, positions where its stores started]
        self._tails: OrderedDict = OrderedDict()
        self._held = 0
        self._born = itertools.count(1)
        self.stats["duplicate_hashes"] = 0

    # -- events ----------------------------------------------------------------------

    def _event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "BlockStored" and ev.get("medium") not in GPU:
            self.offload_seen = True  # HiCache's host or storage tier exists
        if kind == "BlockStored" and ev.get("medium") in GPU:
            self._store(ev)
        elif kind == "BlockRemoved" and ev.get("medium") in GPU:
            self._remove(ev.get("block_hashes") or [])
        elif kind == "AllBlocksCleared":
            self.stats["cleared"] += 1
            self._clear()

    def _hold(self, tail, sign: int) -> None:
        if tail is not None:
            seg, last = tail
            self._held += sign * min(seg.tokens, (last + 1) * seg.block) + sign * (seg.start or 0)

    def _store(self, ev: dict) -> None:
        hashes = ev.get("block_hashes") or []
        if not hashes:
            return
        self.stats["stored"] += len(hashes)
        tokens = len(ev.get("token_ids") or ()) or len(hashes) * (ev.get("block_size") or 1)
        block = ev.get("block_size") or max(1, tokens // len(hashes))
        try:
            packed = array("q", hashes)  # SGLang's hashes are signed 64-bit
        except (TypeError, OverflowError):
            packed = list(hashes)
        born = next(self._born)
        parent_hash = ev.get("parent_block_hash")
        p = None if parent_hash is None else self._alive.get(parent_hash)
        if parent_hash is None:
            seg = _Seg(None, -1, 0, block, tokens, packed, born)
        elif p is None:  # stored before Prefixwise subscribed: this chain cannot be placed
            seg = _Seg(None, -1, None, block, tokens, packed, born)
        else:
            i = len(p.hashes) - 1 if p.hashes[-1] == parent_hash else p.hashes.index(parent_hash)
            seg = _Seg(p, i, None if p.start is None else p.end(i), block, tokens, packed, born)
        alive = self._alive
        for h in hashes:
            old = alive.get(h)
            if old is not None and old is not seg:
                # The same hash on two subtrees (SGLang leaves extra_key out of event hashes):
                # a removal cannot be told apart, so neither chain is judged.
                old.ambiguous = seg.ambiguous = True
                self.stats["duplicate_hashes"] += 1
            alive[h] = seg
        session = ev.get("session_id")
        if session:
            tail = self._tails.get(session)
            if tail is None:  # no window opened by request_sent (dropped, or another client): mixed
                tail = self._tails[session] = [None, True, {}]
            self._tails.move_to_end(session)
            if seg.start is not None:
                tail[2].setdefault(seg.start, (packed[0], block))  # where a store began: its first block
            self._hold(tail[0], -1)
            tail[0] = (seg, len(packed) - 1)  # stores come in chain order: the latest is the deepest
            self._hold(tail[0], 1)
            self._trim()
        if len(alive) > self.max_alive:  # more blocks than the GPU holds: removals were missed
            self._forget()
            self.stats["overflows"] += 1

    def _forget(self) -> None:
        """Residency is unknown (missed removals, a cut stream): judge nothing from it."""
        self._alive.clear()
        for t in list(self._tails.values()):
            self._hold(t[0], -1)
        self._tails.clear()
        for check in self._checks.values():
            check["overflowed"] = True

    def _trim(self) -> None:
        while self._tails and (len(self._tails) > self.max_sessions or self._held > self.max_held_tokens):
            _, t = self._tails.popitem(last=False)
            self._hold(t[0], -1)

    def _remove(self, hashes) -> None:
        self.stats["removed"] += len(hashes)
        prev, i = None, -1
        for h in hashes:
            seg = self._alive.pop(h, None)
            if seg is None:
                prev = None
                continue
            # A node's blocks come in chain order; look a position up only where a run restarts.
            if not (seg is prev and i + 1 < len(seg.hashes) and seg.hashes[i + 1] == h):
                i = seg.hashes.index(h)
                seg.cut = i if seg.cut is None else min(seg.cut, i)
            else:
                i += 1
            prev = seg

    def _clear(self) -> None:
        for seg in set(self._alive.values()):
            seg.cut = 0
        self._alive.clear()
        for check in self._checks.values():  # chains not indexed any more (stored long ago) too
            check["cleared"] = True

    # -- per request -----------------------------------------------------------------

    def _drop_check(self, token: int) -> None:
        check = self._checks.pop(token, None)
        if check is not None:
            self._hold(check["tail"], -1)

    def request_sent(self, session_id: str) -> int:
        with self._lock:
            concurrent = False
            for check in self._checks.values():
                if check["session"] == session_id:
                    check["concurrent"] = concurrent = True
            prev = self._tails.pop(session_id, None)  # its held tokens move to the check
            window = self._tails[session_id] = [None, concurrent, {}]
            token = next(self._tokens)
            self._checks[token] = {"session": session_id, "tail": prev[0] if prev else None,
                                   "window": window, "sent": next(self._born) if prev else 0,
                                   "concurrent": concurrent or bool(prev and prev[1]),
                                   "cleared": False, "overflowed": False, "errors": self._errors()}
            while len(self._checks) > self.max_sessions:  # requests that never completed
                self._hold(self._checks.popitem(last=False)[1]["tail"], -1)
            self._trim()
            return token

    def discard(self, token: int | None) -> None:
        if token is not None:
            with self._lock:
                self._drop_check(token)

    def _removed(self, check, seg, j) -> bool:
        """Was the stop block removed between the previous request and this response?"""
        if check["cleared"]:
            return True
        if seg.cut is None or seg.cut > j:
            return False
        other = self._alive.get(seg.hashes[j])  # stored again before this request was sent: present
        return not (other is not None and other is not seg and other.born < check["sent"])

    def _restored(self, check, cached: int) -> int:
        """On a hit: blocks of the previous request's chain before the read's end (token position
        `cached`) that had been removed from the GPU, so this request read them from another
        tier (details: host_restored). 0 when nothing can be said (an unplaced or ambiguous
        chain, a concurrent request, an overflow or a cut stream)."""
        if (not self.offload_seen or check["tail"] is None or check["overflowed"] or check["concurrent"]
                or self._errors() != check["errors"]):
            return 0
        chain = _chain(check["tail"])
        if any(seg.start is None or seg.ambiguous for seg, _ in chain):
            return 0
        n = 0
        for seg, last in chain:
            for j in range(last + 1):
                if seg.start + j * seg.block >= cached:
                    return n
                if self._removed(check, seg, j):
                    n += 1
        return n

    def _judge(self, check, cached: int):
        """(stop run, block, blocks on chain, blocks removed), or None."""
        if check["tail"] is None or check["overflowed"]:
            return None
        chain = _chain(check["tail"])
        if any(seg.start is None or seg.ambiguous for seg, _ in chain):
            return None
        total, first_cut, stop, offset = 0, None, None, 0
        for seg, last in chain:
            n = last + 1
            if seg.cut is not None and seg.cut <= last and first_cut is None:
                first_cut = offset + seg.cut
            if stop is None and seg.start <= cached < seg.end(last):
                stop = (seg, (cached - seg.start) // seg.block)
            offset += n
        if stop is None:  # the read stopped beyond the previous chain
            return None
        total = offset
        gone = total - first_cut if first_cut is not None else 0
        return stop[0], stop[1], total, gone

    def request_done(self, token: int, cached_tokens: int | None,
                     prompt_tokens: int | None = None) -> tuple[int | None, int | None]:
        with self._lock:
            check = self._checks.get(token)
            if check is None or cached_tokens is None:
                self._drop_check(token)
                return NOTHING
            if prompt_tokens and cached_tokens >= prompt_tokens * 0.5:  # a hit: restored blocks only
                restored = self._restored(check, max(0, cached_tokens))
                self._drop_check(token)
                return (restored, 0) if restored else NOTHING
            judged = self._judge(check, max(0, cached_tokens))
            if judged is None:
                self._drop_check(token)
                return NOTHING
            seg, j, total, gone = judged
            if self._removed(check, seg, j):
                self._drop_check(token)
                return max(gone, 1), 0
            if self.hybrid or check["concurrent"]:
                self._drop_check(token)
                return NOTHING
        # Not removed so far. Wait for this request's own store: SGLang queues events in
        # order, so it follows every removal made to admit this request. Its first store
        # starts exactly where its read stopped; the previous request's late finish store
        # (published after its response) starts elsewhere, so it cannot pass for it.
        stop = max(0, cached_tokens)
        end = time.time() + self.barrier_seconds
        while stop not in check["window"][2] and time.time() < end and self._thread is not None:
            time.sleep(0.002)
        with self._lock:
            self._drop_check(token)
            judged = self._judge(check, max(0, cached_tokens))
            if judged is None:
                return NOTHING
            seg, j, total, gone = judged
            if self._removed(check, seg, j):
                return max(gone, 1), 0
            barrier = stop in check["window"][2] or self._thread is None
            if (not barrier or check["concurrent"] or check["overflowed"]
                    or self._errors() != check["errors"]):
                return NOTHING
            return 0, total

    # -- several ranks (EngineFleet) -------------------------------------------------

    def _served_previous(self, token: int) -> bool:
        with self._lock:
            check = self._checks.get(token)
            return bool(check and check["tail"] is not None)

    def _stored_at(self, token: int, cached_tokens: int) -> bool:
        """This rank stored the session's blocks from exactly where the request's cache read
        stopped: it computed the rest, so it served the request. (SGLang reports only new
        blocks. The previous request's late finish store starts where that request's prompt
        ended; if that happens to be this stop too, two ranks match and nothing is claimed.)"""
        with self._lock:
            check = self._checks.get(token)
            return bool(check and cached_tokens in check["window"][2])

    def _reported(self, token: int, cached_tokens: int) -> tuple:
        with self._lock:
            check = self._checks.get(token)
            return check["window"][2].get(cached_tokens, (None, None)) if check else (None, None)

    def held_here(self, token: int, cached_tokens: int | None, prompt_tokens: int | None = None,
                  needed: tuple = (None, None)) -> int | None:
        """The session's previous request was served by this rank, this one by another, which
        computed `needed` = (hash, block size) where its read stopped. SGLang's block hashes
        are the same on every rank (SHA-256 over the tokens, chained, no per-process seed), so
        if this rank still holds that very block the prefix was cached here: returns the
        previous request's blocks on this rank, else None."""
        with self._lock:
            check = self._checks.get(token)
            if (check is None or cached_tokens is None or needed[0] is None
                    or (prompt_tokens and cached_tokens >= prompt_tokens * 0.5)):
                self._drop_check(token)
                return None
            judged = self._judge(check, max(0, cached_tokens))
            if judged is None:
                self._drop_check(token)
                return None
            seg, j, total, gone = judged
            same = (seg.hashes[j], seg.block) == tuple(needed) and seg.start + j * seg.block == cached_tokens
            ok = (same and not self._removed(check, seg, j) and not self.hybrid and not check["concurrent"]
                  and not check["overflowed"] and self._errors() == check["errors"])
            self._drop_check(token)
            return total - gone if ok else None

    def snapshot(self) -> dict:
        snap = super().snapshot()
        with self._lock:
            snap.update(engine="sglang", full_attention=self.full_attention, gpu_blocks_tracked=len(self._alive),
                        sessions_tracked=len(self._tails), chain_tokens_held=self._held)
        return snap
