"""The record log: a session-partitioned Redis Stream between edges and several collectors.

    prefixwise collect --log redis://redis:6379/0 --partitions 0-7     # one collector per partition set
    prefixwise collect --log redis://redis:6379/0 --partitions 8-15
    PREFIXWISE_COLLECTOR=redis://redis:6379/0 prefixwise serve ...     # edges write to the log

Why a log and not HTTP: with several collectors, a conversation's turns must reach the one that
holds its state, in order, and a collector that restarts must find its state again. The stream
`prefixwise:records:<p>` holds the records of every session whose id hashes to partition p
(`partition_of`), in the order edges appended them; a collector owns a set of partitions and reads
each in order through a consumer group, so one collector sees all of a session's turns, in the
order they reached the log, and no two collectors ever see the same session. (An edge appends in
its own send order; a backed-up edge can still append turn N after another edge appended N+1, the
case the collector names `unknown` rather than mis-classifies.) A record is acknowledged only after
its ledger row is written (at-least-once); the collector's idempotency by request id, seeded from
the ledger at start, makes a redelivery a no-op, and a record delivered but never written is
written when it is delivered again. At start a collector replays its partitions' retained records
of the last day (`Collector.ingest_many(..., replay=True)`: state only, no rows) so the lanes are
as they were, and the first turn after a deploy is not a `first_write`.

Edges and collectors must agree on the partition count (`PREFIXWISE_LOG_PARTITIONS`, default 16):
changing it moves sessions between partitions, so change it with every collector stopped. The
stream is capped (`maxlen`, approximate, default 1,000,000 records per partition): what falls off
unread is lost, and that is the operator's capacity decision, as with any log.

Needs `pip install 'prefixwise[log]'` (redis-py) and Redis 7.0 or later. The core stays standard library.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import zlib
from urllib.parse import urlsplit, urlunsplit

from .records import RecordError
from .transport import Reject, Retry

STREAM_PREFIX = "prefixwise:records"
GROUP = "prefixwise"
PARTITIONS = int(os.environ.get("PREFIXWISE_LOG_PARTITIONS", "16"))
MAXLEN = int(os.environ.get("PREFIXWISE_LOG_MAXLEN", "1000000"))  # records retained per partition
CLAIM_IDLE_MS = 60_000  # a record another consumer took but never acknowledged for this long is taken over
# At start, a collector rebuilds its conversation state from the records of the last this many
# seconds (a conversation idle longer than a day compares as new; the records are still in the ledger).
REPLAY_SECONDS = float(os.environ.get("PREFIXWISE_LOG_REPLAY_SECONDS", str(24 * 3600)))
MIN_REDIS = (7, 0)  # XAUTOCLAIM drops trimmed entries from the pending list only from 7.0


def partition_of(session_id: str, partitions: int = PARTITIONS) -> int:
    return zlib.crc32(session_id.encode("utf-8")) % max(1, partitions)


def stream_name(partition: int, prefix: str = STREAM_PREFIX) -> str:
    return f"{prefix}:{partition}"


def redacted(url: str) -> str:
    """The URL without its password, for logs and /health."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    userinfo, _, hostport = parts.netloc.rpartition("@")  # hostport keeps IPv6 brackets as written
    user = userinfo.split(":", 1)[0]
    return urlunsplit((parts.scheme, f"{user}:***@{hostport}", parts.path, parts.query, parts.fragment))


def parse_partitions(spec: str | None, partitions: int = PARTITIONS) -> list[int]:
    """'0-7', '0,3,5', '0-3,12' or None (all)."""
    if not spec:
        return list(range(partitions))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"partition range {part!r} is backwards")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    bad = [p for p in out if not 0 <= p < partitions]
    if bad:
        raise ValueError(f"partitions {bad} are outside 0-{partitions - 1} (PREFIXWISE_LOG_PARTITIONS={partitions})")
    return sorted(set(out))


def _client(url: str):
    try:
        import redis
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError("the record log needs redis-py: pip install 'prefixwise[log]'") from None
    return redis.Redis.from_url(url, socket_timeout=10, socket_connect_timeout=5, decode_responses=False)


class RedisBackend:
    """The edge's side: a Sender backend that appends each record to its session's partition."""

    def __init__(self, url: str, partitions: int = PARTITIONS, maxlen: int = MAXLEN, prefix: str = STREAM_PREFIX):
        self.url, self.partitions, self.maxlen, self.prefix = url, max(1, partitions), maxlen, prefix
        self._redis = _client(url)

    @property
    def name(self) -> str:
        return redacted(self.url)

    def deliver(self, batch: list[dict]) -> dict:
        try:
            import redis
        except ImportError:  # pragma: no cover
            raise Reject("redis-py is not installed") from None
        pipe = self._redis.pipeline(transaction=False)
        for rec in batch:
            session = str(rec.get("session_id") or "")
            if not session:
                raise Reject("a record without a session id cannot be partitioned")
            pipe.xadd(stream_name(partition_of(session, self.partitions), self.prefix),
                      {"r": json.dumps(rec, separators=(",", ":"))}, maxlen=self.maxlen, approximate=True)
        try:
            pipe.execute()
        except redis.exceptions.AuthenticationError as exc:
            raise Reject(f"redis refused the credentials: {exc}") from None
        except (redis.exceptions.RedisError, OSError) as exc:
            raise Retry(f"{type(exc).__name__}: {exc}") from None
        return {}


class LogConsumer:
    """A collector's side: reads its partitions in order, ingests, acknowledges after the row is
    written. One thread per partition; `stats()` for /health and /metrics."""

    def __init__(self, collector, url: str, partitions: list[int] | None = None, *, group: str = GROUP,
                 consumer: str | None = None, prefix: str = STREAM_PREFIX, total_partitions: int = PARTITIONS,
                 batch: int = 200, block_ms: int = 1000, replay: bool = True):
        self.collector = collector
        self.url = url
        self.partitions = partitions if partitions is not None else list(range(total_partitions))
        self.group, self.prefix = group, prefix
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        from .collector import MAX_BATCH
        self.batch, self.block_ms, self.replay = max(1, min(batch, MAX_BATCH)), block_ms, replay
        self._redis = _client(url)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.stats = {"consumed": 0, "acked": 0, "claimed": 0, "replayed": 0, "replaying": 0, "errors": 0,
                      "last_error": None, "conflicts": [], "partitions": list(self.partitions),
                      "consumer": self.consumer}

    @property
    def name(self) -> str:
        return redacted(self.url)

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        """Create the groups, check the server, then one thread per partition: each replays its
        partition (state only) and then consumes it. Returns at once; /health shows `replaying`."""
        import redis

        version = str(self._redis.info("server").get("redis_version", "0"))
        parts = tuple(int(x) for x in version.split(".")[:2] if x.isdigit()) or (0,)
        if parts < MIN_REDIS:
            raise ValueError(f"the record log needs Redis {MIN_REDIS[0]}.{MIN_REDIS[1]} or later (XAUTOCLAIM over trimmed "
                             f"entries); this server is {version}")
        if self.replay:
            seed = getattr(self.collector.ledger, "recent_request_ids", None)
            if seed is not None:
                self.collector.seed_seen(seed())
        for p in self.partitions:
            stream = stream_name(p, self.prefix)
            try:  # id "0": a new group starts from the oldest retained record, so a backlog is not skipped
                self._redis.xgroup_create(stream, self.group, id="0", mkstream=True)
            except redis.exceptions.ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise
            self._check_alone(stream)
        with self._lock:
            self.stats["replaying"] = len(self.partitions) if self.replay else 0
        for p in self.partitions:
            t = threading.Thread(target=self._run, args=(p,), name=f"prefixwise-log-{p}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, deadline - time.monotonic()))

    def _check_alone(self, stream: str) -> None:
        """Every partition must have exactly one collector: another consumer active on this stream
        in the last minute would split a session's turns between two collectors. Said once, in
        stderr and /health; the operator decides (it may be the instance being replaced)."""
        try:
            others = [c for c in self._redis.xinfo_consumers(stream, self.group)
                      if (c.get("name") or c.get(b"name")) not in (self.consumer, self.consumer.encode())
                      and int(c.get("idle") or c.get(b"idle") or 0) < CLAIM_IDLE_MS]
        except Exception:  # noqa: BLE001 - informational
            return
        for c in others:
            name = c.get("name") or c.get(b"name")
            name = name.decode() if isinstance(name, bytes) else str(name)
            with self._lock:
                self.stats["conflicts"].append(f"{stream}: {name}")
            print(f"prefixwise: another collector ({name}) read {stream} in the last minute: a partition must "
                  "have one collector, or a session's turns are split between two", file=sys.stderr, flush=True)

    # -- state after a restart ----------------------------------------------------------------

    def _replay(self, partition: int) -> None:
        """Rebuild this partition's conversation state from the records this group already turned
        into rows (state only, nothing written again). Pending entries (delivered, never
        acknowledged) are left out: they were not written, and they are processed for real next.
        Bounded to the last REPLAY_SECONDS by the entries' ids (milliseconds)."""
        stream = stream_name(partition, self.prefix)
        last = None
        for g in self._redis.xinfo_groups(stream):
            if (g.get("name") or g.get(b"name")) in (self.group, self.group.encode()):
                last = g.get("last-delivered-id") or g.get(b"last-delivered-id")
        last = last.decode() if isinstance(last, bytes) else last
        if not last or last == "0-0":
            return
        pending = set()
        summary = self._redis.xpending(stream, self.group)
        if summary and summary.get("pending"):
            for e in self._redis.xpending_range(stream, self.group, min="-", max="+", count=summary["pending"]):
                mid = e.get("message_id")
                pending.add(mid.decode() if isinstance(mid, bytes) else mid)
        start = f"{int((time.time() - REPLAY_SECONDS) * 1000)}-0"
        while not self._stop.is_set():
            entries = self._redis.xrange(stream, min=start, max=last, count=self.batch)
            if not entries:
                break
            records = []
            for entry_id, fields in entries:
                eid = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                if eid in pending:
                    continue
                rec = self._decode(fields)
                if rec is not None:
                    records.append(rec)
            if records:
                self.collector.ingest_many(records, replay=True)
            with self._lock:
                self.stats["replayed"] += len(records)
            last_id = entries[-1][0].decode() if isinstance(entries[-1][0], bytes) else entries[-1][0]
            if last_id == last or len(entries) < self.batch:
                break
            ms, seq = last_id.split("-")
            start = f"{ms}-{int(seq) + 1}"

    # -- the partition loop --------------------------------------------------------------------

    def _run(self, partition: int) -> None:
        import redis

        stream = stream_name(partition, self.prefix)
        if self.replay:
            try:
                self._replay(partition)
            except Exception as exc:  # noqa: BLE001 - a failed replay costs state, never the consumer
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"replay: {type(exc).__name__}: {exc}"
                print(f"prefixwise: replay of {stream} failed ({type(exc).__name__}: {exc}); consuming without it",
                      file=sys.stderr, flush=True)
            finally:
                with self._lock:
                    self.stats["replaying"] -= 1
        pending_first = True
        starting = True  # the predecessor's unacknowledged records are ours now, idle or not
        while not self._stop.is_set():
            try:
                if pending_first:  # what this consumer took before a crash, then what others left behind
                    entries = self._read(stream, "0")
                    if not entries:
                        pending_first = False
                        entries = self._claim(stream, min_idle_ms=0 if starting else CLAIM_IDLE_MS)
                        starting = False
                else:
                    entries = self._read(stream, ">")
                    if not entries:
                        entries = self._claim(stream)
                if entries:
                    self._process(stream, entries)
            except (redis.exceptions.RedisError, OSError) as exc:
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"{type(exc).__name__}: {exc}"
                pending_first = True  # what was read but not acknowledged is read again first
                time.sleep(1.0)
            except Exception as exc:  # noqa: BLE001 - a broken ledger, a bad record, a bug: the loop goes on
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"{type(exc).__name__}: {exc}"
                print(f"prefixwise: log consumer error on {stream}: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                pending_first = True  # the unacknowledged batch is retried before anything new
                time.sleep(1.0)

    def _read(self, stream: str, start: str) -> list:
        reply = self._redis.xreadgroup(self.group, self.consumer, {stream: start}, count=self.batch,
                                       block=self.block_ms if start == ">" else None)
        return reply[0][1] if reply else []

    def _claim(self, stream: str, min_idle_ms: int = CLAIM_IDLE_MS) -> list:
        """Records another consumer took and never acknowledged (it died): ours now."""
        reply = self._redis.xautoclaim(stream, self.group, self.consumer, min_idle_time=min_idle_ms,
                                       start_id="0-0", count=self.batch)
        entries = reply[1] if isinstance(reply, (list, tuple)) and len(reply) > 1 else []
        entries = [e for e in entries if e and e[0] is not None]  # a trimmed entry (Redis < 7) has no id
        if entries:
            with self._lock:
                self.stats["claimed"] += len(entries)
        return entries

    def _process(self, stream: str, entries: list) -> None:
        """Ingest a batch, then acknowledge it: a crash in between redelivers, and the collector's
        idempotency makes the second time a no-op. A broken ledger (ingest_many raises) leaves the
        batch unacknowledged to be retried."""
        ids, records = [], []
        for entry_id, fields in entries:
            ids.append(entry_id)
            rec = self._decode(fields)
            if rec is not None:
                records.append(rec)
        self.collector.ingest_many(records)
        self._redis.xack(stream, self.group, *ids)
        with self._lock:
            self.stats["consumed"] += len(entries)
            self.stats["acked"] += len(ids)

    @staticmethod
    def _decode(fields) -> dict | None:
        raw = fields.get(b"r") if isinstance(fields, dict) else None
        if raw is None:
            return None
        try:
            rec = json.loads(raw)
        except ValueError:
            return None
        return rec if isinstance(rec, dict) else None

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.stats)


__all__ = ["LogConsumer", "RedisBackend", "RecordError", "parse_partitions", "partition_of", "redacted",
           "stream_name"]
