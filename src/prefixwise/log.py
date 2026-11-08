"""The record log: a session-partitioned Redis Stream between edges and several collectors.

    prefixwise collect --log redis://redis:6379/0 --partitions 0-7     # one collector per partition set
    prefixwise collect --log redis://redis:6379/0 --partitions 8-15
    PREFIXWISE_COLLECTOR=redis://redis:6379/0 prefixwise serve ...     # edges write to the log

Why a log and not HTTP: with several collectors, a conversation's turns must reach the one that
holds its state, in order, and a collector that restarts must find its state again. The stream
`prefixwise:records:<p>` holds the records of every session whose id hashes to partition p
(`partition_of`), in the order edges wrote them; a collector owns a set of partitions and reads
each in order through a consumer group, so a conversation's turns are compared in sequence
whatever edge they came through, and no two collectors ever see the same session. A record is
acknowledged only after its ledger row is written (at-least-once); the collector's idempotency by
request id, seeded from the ledger at start, makes a redelivery a no-op. At start a collector
replays its partitions' retained records (`Collector.ingest_many(..., replay=True)`: state only,
no rows) so the lanes are as they were, and the first turn after a deploy is not a `first_write`.

Edges and collectors must agree on the partition count (`PREFIXWISE_LOG_PARTITIONS`, default 16):
changing it moves sessions between partitions, so change it with every collector stopped. The
stream is capped (`maxlen`, approximate, default 1,000,000 records per partition): what falls off
is already in the ledger if a collector was reading; if none was, it is lost and that is the
operator's capacity decision, as with any log.

Needs `pip install 'prefixwise[log]'` (redis-py). The core stays standard library.
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


def partition_of(session_id: str, partitions: int = PARTITIONS) -> int:
    return zlib.crc32(session_id.encode("utf-8")) % max(1, partitions)


def stream_name(partition: int, prefix: str = STREAM_PREFIX) -> str:
    return f"{prefix}:{partition}"


def redacted(url: str) -> str:
    """The URL without its password, for logs and /health."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.hostname or ""
    netloc = (f"{parts.username}:***@" if parts.username else ":***@") + host + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


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
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
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
        self.batch, self.block_ms, self.replay = max(1, batch), block_ms, replay
        self._redis = _client(url)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.stats = {"consumed": 0, "acked": 0, "claimed": 0, "replayed": 0, "errors": 0, "last_error": None,
                      "partitions": list(self.partitions), "consumer": self.consumer}

    @property
    def name(self) -> str:
        return redacted(self.url)

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        import redis

        for p in self.partitions:
            stream = stream_name(p, self.prefix)
            try:  # id "0": a new group starts from the oldest retained record, so a backlog is not skipped
                self._redis.xgroup_create(stream, self.group, id="0", mkstream=True)
            except redis.exceptions.ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise
        if self.replay:
            self._replay()
        for p in self.partitions:
            t = threading.Thread(target=self._run, args=(p,), name=f"prefixwise-log-{p}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)

    # -- state after a restart ----------------------------------------------------------------

    def _replay(self) -> None:
        """Rebuild the collector's conversation state from the records this group already
        consumed (they are in the ledger; nothing is written again), and seed its idempotency set."""
        seed = getattr(self.collector.ledger, "recent_request_ids", None)
        if seed is not None:
            self.collector.seed_seen(seed())
        for p in self.partitions:
            stream = stream_name(p, self.prefix)
            last = None
            for g in self._redis.xinfo_groups(stream):
                if g.get("name") == self.group.encode() or g.get("name") == self.group:
                    last = g.get("last-delivered-id")
            if not last or last in (b"0-0", "0-0"):
                continue
            last = last.decode() if isinstance(last, bytes) else last
            start = "-"
            while True:
                entries = self._redis.xrange(stream, min=start, max=last, count=self.batch)
                if not entries:
                    break
                records = [self._decode(fields) for _, fields in entries]
                self.collector.ingest_many([r for r in records if r is not None], replay=True)
                with self._lock:
                    self.stats["replayed"] += len(entries)
                last_id = entries[-1][0].decode() if isinstance(entries[-1][0], bytes) else entries[-1][0]
                if last_id == last or len(entries) < self.batch:
                    break
                ms, seq = last_id.split("-")
                start = f"{ms}-{int(seq) + 1}"

    # -- the partition loop --------------------------------------------------------------------

    def _run(self, partition: int) -> None:
        import redis

        stream = stream_name(partition, self.prefix)
        pending_first = True
        while not self._stop.is_set():
            try:
                if pending_first:  # what this consumer took before a crash, then what others left idle
                    entries = self._read(stream, "0")
                    if not entries:
                        pending_first = False
                        entries = self._claim(stream)
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

    def _claim(self, stream: str) -> list:
        """Records another consumer took and never acknowledged (it died): ours now."""
        reply = self._redis.xautoclaim(stream, self.group, self.consumer, min_idle_time=CLAIM_IDLE_MS,
                                       start_id="0-0", count=self.batch)
        entries = reply[1] if isinstance(reply, (list, tuple)) and len(reply) > 1 else []
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
