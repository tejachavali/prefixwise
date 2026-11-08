"""Ledger sinks: a copy of every row, where a fleet keeps its data.

    prefixwise collect --sink file:///var/log/prefixwise/rows.jsonl
    prefixwise collect --sink clickhouse://user:pass@clickhouse:8123/prefixwise.entries

The SQLite ledger stays the store `prefixwise report` and `/api/v1` read; a sink receives each
row after it is committed there, so the ledger never waits on a warehouse and a sink that is down
loses nothing from the ledger. Rows hold counts, causes and the caller's ids, never prompt text
(docs/DATA.md), so a sink is as private as the ledger.

- `JSONLSink`: one JSON object per line, appended; a file collectors like Vector, Fluent Bit or
  a log shipper already read.
- `ClickHouseSink`: batches to ClickHouse's HTTP interface (`INSERT ... FORMAT JSONEachRow`) on
  the same bounded sender as the record transport (`prefixwise.transport.Sender`): never blocks,
  retries, drops and counts. The table is created by the operator (docs/DATA.md has the DDL);
  columns match the ledger's.

Postgres is not built in: ship the JSONL file, or write a `LedgerSink` (two methods) with your
driver and pass it to `Ledger.add_sink`.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict

from .transport import Reject, Retry, Sender

_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
_RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
# ClickHouse answers most query errors with HTTP 500 and names the error in X-ClickHouse-Exception-Code.
# These codes mean the rows or the table are wrong (a schema mismatch, a parse error, no such table):
# the batch is dropped and counted rather than retried six times. Anything else (a busy or
# read-only replica, Keeper, memory, timeouts) is retried as a 5xx normally is.
_CLICKHOUSE_PERMANENT = {"16", "27", "47", "53", "60", "62", "81", "117", "497", "516",  # no such column/type
                         "6", "26", "38", "41", "72", "130", "131"}  # cannot parse (text, quoted, date, number, ...)


def row_dict(entry, at: float | None = None) -> dict:
    """A ledger row as a sink receives it: the entry's fields (booleans as 0/1, as in the ledger)
    plus recorded_at."""
    d = {k: (int(v) if isinstance(v, bool) else v) for k, v in asdict(entry).items()}
    d["recorded_at"] = at if at is not None else time.time()
    return d


class JSONLSink:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")  # noqa: SIM115 - held open for the process

    @property
    def name(self) -> str:
        return f"file://{self.path}"

    def record(self, entry, at: float | None = None) -> None:
        line = json.dumps(row_dict(entry, at), separators=(",", ":"), default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def stats(self) -> dict:
        return {}

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._fh.close()


class ClickHouseBackend:
    """Rows to ClickHouse over HTTP: INSERT INTO <table> FORMAT JSONEachRow."""

    def __init__(self, url: str, table: str, user: str | None = None, password: str | None = None,
                 timeout: float = 10.0):
        if not _TABLE.match(table):
            raise ValueError(f"not a table name: {table!r}")
        self.url, self.table, self.user, self.password, self.timeout = url.rstrip("/"), table, user, password, timeout

    @property
    def name(self) -> str:
        return f"{self.url}/{self.table}"

    def deliver(self, batch: list[dict]) -> dict:
        body = "\n".join(json.dumps(r, separators=(",", ":"), default=str) for r in batch).encode() + b"\n"
        query = urllib.parse.urlencode({"query": f"INSERT INTO {self.table} FORMAT JSONEachRow"})
        headers = {"Content-Type": "application/x-ndjson", "User-Agent": "prefixwise-sink"}
        if self.user:
            headers["X-ClickHouse-User"] = self.user
        if self.password:
            headers["X-ClickHouse-Key"] = self.password
        try:
            req = urllib.request.Request(f"{self.url}/?{query}", data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                r.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:200].decode("utf-8", "replace")
            code = exc.headers.get("X-ClickHouse-Exception-Code")
            if code is not None and code in _CLICKHOUSE_PERMANENT:
                raise Reject(f"ClickHouse error {code} (HTTP {exc.code}): {detail}") from None
            if exc.code in _RETRY_STATUSES:
                raise Retry(f"HTTP {exc.code}: {detail}") from None
            raise Reject(f"HTTP {exc.code} from ClickHouse: {detail}") from None
        except Exception as exc:  # noqa: BLE001
            raise Retry(f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}") from None
        return {}


class ClickHouseSink:
    def __init__(self, url: str, table: str, user: str | None = None, password: str | None = None, **sender_kw):
        self.sender = Sender(ClickHouseBackend(url, table, user, password), **sender_kw)

    @property
    def name(self) -> str:
        return self.sender.backend.name

    def record(self, entry, at: float | None = None) -> None:
        self.sender.send(row_dict(entry, at))

    def stats(self) -> dict:
        return self.sender.stats()

    def close(self, timeout: float = 5.0) -> None:
        self.sender.close(timeout)


def sink_from_url(url: str):
    """file:///path.jsonl | /path.jsonl -> JSONLSink;
    clickhouse://[user[:password]@]host[:8123]/db.table (clickhouses:// for TLS) -> ClickHouseSink."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme in ("", "file"):
        path = parts.path if parts.scheme else url
        if not path:
            raise ValueError(f"no path in sink {url!r}")
        return JSONLSink(path)
    if parts.scheme in ("clickhouse", "clickhouses"):
        scheme = "https" if parts.scheme == "clickhouses" else "http"
        host = parts.hostname or "127.0.0.1"
        port = parts.port or (8443 if scheme == "https" else 8123)
        table = parts.path.lstrip("/")
        if not table:
            raise ValueError(f"sink {url!r} names no table: clickhouse://host:8123/db.table")
        unq = urllib.parse.unquote
        return ClickHouseSink(f"{scheme}://{host}:{port}", table, unq(parts.username) if parts.username else None,
                              unq(parts.password) if parts.password else None)
    raise ValueError(f"unknown sink {url!r}: file:///path.jsonl or clickhouse://host:8123/db.table")


def warn_once(state: set, key: str, msg: str) -> None:
    if key not in state:
        state.add(key)
        print(f"prefixwise: {msg}", file=sys.stderr, flush=True)
