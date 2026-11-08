"""The record log (prefixwise.log): edges write a session-partitioned Redis Stream, several
collectors read it in order, and a collector finds its state again after a restart.

Runs against PREFIXWISE_TEST_REDIS if set (CI starts a Redis service), else a redis-server started
here on a free port; skipped when neither redis-py nor a server is available.
"""

import os
import shutil
import socket
import subprocess
import time
import uuid

import pytest
from test_transport import SESSION, rows, scripted, turns, wait_for

redis = pytest.importorskip("redis")

from prefixwise.collector import Collector
from prefixwise.ledger import Ledger
from prefixwise.log import LogConsumer, RedisBackend, parse_partitions, partition_of, redacted
from prefixwise.pricing import PriceTable
from prefixwise.proxy import PrefixwiseProxy
from prefixwise.transport import RemoteCollector, Sender


@pytest.fixture(scope="module")
def redis_url():
    url = os.environ.get("PREFIXWISE_TEST_REDIS")
    if url:
        yield url
        return
    if not shutil.which("redis-server"):
        pytest.skip("no redis-server and no PREFIXWISE_TEST_REDIS")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(["redis-server", "--port", str(port), "--bind", "127.0.0.1", "--save", "",
                             "--appendonly", "no", "--loglevel", "warning"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    deadline = time.monotonic() + 10
    while True:
        try:
            redis.Redis.from_url(url).ping()
            break
        except redis.exceptions.RedisError:
            assert time.monotonic() < deadline, "redis-server did not start"
            time.sleep(0.05)
    yield url
    proc.terminate()
    proc.wait(5)


@pytest.fixture
def prefix(redis_url):
    """A fresh stream namespace per test, cleaned up after."""
    p = f"prefixwise:test:{uuid.uuid4().hex[:8]}"
    yield p
    r = redis.Redis.from_url(redis_url)
    for key in r.scan_iter(p + ":*"):
        r.delete(key)


def edge(redis_url, prefix, partitions=4):
    p = scripted(PrefixwiseProxy(upstream_url="http://unused", api_key="k", collector_url=""), SESSION * 3)
    p.collector = RemoteCollector(Sender(RedisBackend(redis_url, partitions=partitions, prefix=prefix),
                                         flush_seconds=0.02))
    return p


def consumer(redis_url, prefix, partitions, ledger=None, **kw):
    c = Collector(ledger or Ledger(), PriceTable.load())
    lc = LogConsumer(c, redis_url, partitions, prefix=prefix, total_partitions=4, block_ms=100, **kw)
    lc.start()
    return c, lc


def test_partitioning_is_stable_and_specs_parse():
    assert partition_of("s", 16) == partition_of("s", 16) and 0 <= partition_of("anything", 7) < 7
    assert parse_partitions("0-3,12", 16) == [0, 1, 2, 3, 12] and parse_partitions(None, 4) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        parse_partitions("0-16", 16)
    assert redacted("redis://u:secret@host:6379/0") == "redis://u:***@host:6379/0"
    assert redacted("redis://:secret@[::1]:6379/0") == "redis://:***@[::1]:6379/0"
    assert redacted("redis://host:6379/0") == "redis://host:6379/0"
    with pytest.raises(ValueError):
        parse_partitions("3-1", 16)


def test_edges_write_the_log_and_collectors_split_it_by_session(redis_url, prefix):
    local = scripted(PrefixwiseProxy(upstream_url="http://unused", api_key="k", collector_url=""), SESSION * 3)
    e = edge(redis_url, prefix)
    sessions = ["a", "b", "c"]
    owners = {s: partition_of(s, 4) for s in sessions}
    low = [p for p in range(4) if p < 2]
    c1, lc1 = consumer(redis_url, prefix, low)
    c2, lc2 = consumer(redis_url, prefix, [p for p in range(4) if p >= 2])
    try:
        for s in sessions:
            for body, headers in turns(s):
                res, _ = e.handle("anthropic", dict(body), headers)
                assert res["queued"]
                local.handle("anthropic", dict(body), headers)
        wait_for(lambda: c1.stats["records"] + c2.stats["records"] == 9)
        got = rows(c1.ledger) + rows(c2.ledger)
        assert sorted(got) == sorted(rows(local.ledger))
        for s in sessions:  # a session's turns all reached the collector owning its partition
            mine = c1 if owners[s] in low else c2
            assert [r for r in rows(mine.ledger) if r[0] == s][0:1]
        # Everything acknowledged: nothing pending in either group.
        r = redis.Redis.from_url(redis_url)
        assert all(r.xpending(f"{prefix}:{p}", "prefixwise")["pending"] == 0 for p in range(4)
                   if r.exists(f"{prefix}:{p}"))
    finally:
        e.close(1)
        lc1.stop(1)
        lc2.stop(1)


def test_a_restarted_collector_finds_its_state_and_adds_no_duplicate_rows(redis_url, prefix):
    e = edge(redis_url, prefix)
    ledger = Ledger()
    c1, lc1 = consumer(redis_url, prefix, [0, 1, 2, 3], ledger=ledger)
    bodies = turns("s")
    try:
        for body, headers in bodies[:2]:
            e.handle("anthropic", dict(body), headers)
        wait_for(lambda: c1.stats["records"] == 2)
    finally:
        lc1.stop(1)
    # The collector process restarts: a new Collector over the same ledger, state rebuilt from the log.
    c2, lc2 = consumer(redis_url, prefix, [0, 1, 2, 3], ledger=ledger)
    try:
        wait_for(lambda: lc2.snapshot()["replaying"] == 0)  # replay runs in the partition threads
        assert lc2.snapshot()["replayed"] == 2 and c2.lane_threads  # lanes are back
        body, headers = bodies[2]
        e.handle("anthropic", dict(body), headers)
        wait_for(lambda: c2.stats["records"] == 1)
        got = rows(ledger)
        assert [r[1] for r in got] == ["first_write", "none", "system_edit"]  # not first_write again
        assert len(got) == 3  # the replay wrote nothing
    finally:
        e.close(1)
        lc2.stop(1)


def _dead(monkeypatch):
    """Liveness is recent activity: make a consumer that stopped 0.1 s ago count as dead."""
    import prefixwise.log as log_mod

    monkeypatch.setattr(log_mod, "CLAIM_IDLE_MS", 50)
    time.sleep(0.1)


def test_a_record_a_dead_consumer_took_is_processed_once(redis_url, prefix, monkeypatch):
    import json

    r = redis.Redis.from_url(redis_url)
    stream = f"{prefix}:{partition_of('s', 4)}"
    r.xgroup_create(stream, "prefixwise", id="0", mkstream=True)
    from prefixwise.records import TurnRecord
    rec = TurnRecord(provider="openai", api="chat", model="gpt-5.6", session_id="s", agent_id="a", request_id="r-1",
                     usage={"prompt_tokens": 2000, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}})
    r.xadd(stream, {"r": json.dumps(rec.to_dict())})
    # A consumer that died after reading, before acknowledging: the record sits in its pending list.
    r.xreadgroup("prefixwise", "dead-consumer", {stream: ">"}, count=10)
    assert r.xpending(stream, "prefixwise")["pending"] == 1
    _dead(monkeypatch)
    # A starting collector takes over everything a dead predecessor left pending at once: one
    # collector per partition means they are its own.
    c, lc = consumer(redis_url, prefix, [partition_of("s", 4)], replay=False)
    try:
        wait_for(lambda: c.stats["records"] == 1)
        assert lc.snapshot()["claimed"] == 1
        wait_for(lambda: r.xpending(stream, "prefixwise")["pending"] == 0)
        assert len(rows(c.ledger)) == 1
    finally:
        lc.stop(1)


def test_an_unwritable_ledger_leaves_records_unacknowledged_for_a_retry(redis_url, prefix):
    e = edge(redis_url, prefix)
    c, lc = consumer(redis_url, prefix, [0, 1, 2, 3])
    real = c.ledger.record

    def boom(entry, at=None):
        raise RuntimeError("disk full")
    c.ledger.record = boom
    try:
        body, headers = turns("s")[0]
        e.handle("anthropic", dict(body), headers)
        wait_for(lambda: lc.snapshot()["errors"] >= 1)
        assert c.stats["records"] == 0 or len(rows(c.ledger)) == 0
        c.ledger.record = real  # the disk is back
        wait_for(lambda: len(rows(c.ledger)) == 1, timeout=15)
        assert lc.snapshot()["acked"] == 1
        # The failed attempt left no state behind: the retry is still the conversation's first turn.
        assert rows(c.ledger)[0][1] == "first_write" and c.stats["records"] == 1
    finally:
        e.close(1)
        lc.stop(1)


def test_the_proxy_takes_a_redis_collector_url(redis_url, prefix, monkeypatch):
    monkeypatch.setenv("PREFIXWISE_COLLECTOR", redis_url)
    p = PrefixwiseProxy(mock=True)
    assert getattr(p.collector, "remote", False) and isinstance(p.collector.sender.backend, RedisBackend)
    assert p.collector.url == redacted(redis_url)
    p.close(1)
    from prefixwise.cli import build_parser, cmd_collect
    args = build_parser().parse_args(["collect", "--log", redis_url, "--partitions", "0-3"])
    assert args.func is cmd_collect and args.log == redis_url and args.partitions == "0-3"


def test_records_a_crashed_collector_took_but_never_wrote_are_written_after_the_restart(redis_url, prefix, monkeypatch):
    # Review: replay must not mark delivered-but-unwritten records as seen, or they are acked as
    # duplicates and lost. Three records sit in a dead consumer's pending list; the restarted
    # collector replays (nothing to replay: nothing was written), claims them, writes all three.
    import json

    from prefixwise.records import TurnRecord

    r = redis.Redis.from_url(redis_url)
    stream = f"{prefix}:{partition_of('s', 4)}"
    r.xgroup_create(stream, "prefixwise", id="0", mkstream=True)
    for i in range(3):
        rec = TurnRecord(provider="openai", api="chat", model="gpt-5.6", session_id="s", agent_id="a",
                         request_id=f"r-{i}", usage={"prompt_tokens": 2000, "completion_tokens": 1,
                                                      "prompt_tokens_details": {"cached_tokens": 0}})
        r.xadd(stream, {"r": json.dumps(rec.to_dict())})
    r.xreadgroup("prefixwise", "dead-consumer", {stream: ">"}, count=10)  # taken, then the process died
    _dead(monkeypatch)
    c, lc = consumer(redis_url, prefix, [partition_of("s", 4)])  # replay on
    try:
        wait_for(lambda: len(rows(c.ledger)) == 3)
        assert c.stats["duplicates"] == 0 and r.xpending(stream, "prefixwise")["pending"] == 0
    finally:
        lc.stop(1)


def test_a_record_that_once_broke_the_classifier_does_not_break_every_restart(redis_url, prefix):
    import json

    from prefixwise.records import TurnRecord

    r = redis.Redis.from_url(redis_url)
    stream = f"{prefix}:{partition_of('s', 4)}"
    bad = TurnRecord(provider="openai", api="chat", model="gpt-5.6", session_id="s", agent_id="a", request_id="bad",
                     usage={"prompt_tokens": "not a number"}).to_dict()  # normalize_usage raises
    r.xadd(stream, {"r": json.dumps(bad)})
    ledger = Ledger()
    c1, lc1 = consumer(redis_url, prefix, [partition_of("s", 4)], ledger=ledger)
    try:
        wait_for(lambda: len(rows(ledger)) == 1)
        assert rows(ledger)[0][14] == "not_recorded"  # a failure row, acknowledged
    finally:
        lc1.stop(1)
    c2, lc2 = consumer(redis_url, prefix, [partition_of("s", 4)], ledger=ledger)  # restart: replays the bad record
    try:
        wait_for(lambda: lc2.snapshot()["replaying"] == 0)
        assert c2.stats["errors"] == 1 and len(rows(ledger)) == 1  # skipped, counted, nothing crashed
    finally:
        lc2.stop(1)


def test_a_second_collector_on_the_same_partition_is_reported(redis_url, prefix):
    c1, lc1 = consumer(redis_url, prefix, [0], consumer="one")
    try:
        c2, lc2 = consumer(redis_url, prefix, [0], consumer="two")
        try:
            assert any("one" in x for x in lc2.snapshot()["conflicts"])
        finally:
            lc2.stop(1)
    finally:
        lc1.stop(1)


def test_seeding_keeps_the_newest_ids():
    import prefixwise.collector as mod

    c = Collector(Ledger(), PriceTable.load())
    old = mod.SEEN_REQUESTS
    mod.SEEN_REQUESTS = 2
    try:
        c.seed_seen(["r3", "r2", "r1"])  # newest first, as the ledger gives them
        assert list(c._seen) == ["r2", "r3"]
    finally:
        mod.SEEN_REQUESTS = old


def test_a_live_predecessors_batch_is_not_taken_during_a_rolling_restart(redis_url, prefix):
    # Review: XACK is per group, so taking a live consumer's in-flight batch writes its rows twice.
    import json

    from prefixwise.records import TurnRecord

    r = redis.Redis.from_url(redis_url)
    stream = f"{prefix}:{partition_of('s', 4)}"
    r.xgroup_create(stream, "prefixwise", id="0", mkstream=True)
    for i in range(2):
        rec = TurnRecord(provider="openai", api="chat", model="gpt-5.6", session_id="s", agent_id="a",
                         request_id=f"r-{i}", usage={"prompt_tokens": 2000, "completion_tokens": 1,
                                                      "prompt_tokens_details": {"cached_tokens": 0}})
        r.xadd(stream, {"r": json.dumps(rec.to_dict())})
    r.xreadgroup("prefixwise", "old-pod", {stream: ">"}, count=10)  # the old collector, mid-batch, still alive
    c, lc = consumer(redis_url, prefix, [partition_of("s", 4)], replay=False, consumer="new-pod")
    try:
        time.sleep(0.5)
        assert lc.snapshot()["conflicts"] and lc.snapshot()["claimed"] == 0 and len(rows(c.ledger)) == 0
        assert r.xpending(stream, "prefixwise")["pending"] == 2  # still the old pod's to acknowledge
    finally:
        lc.stop(1)


def test_startup_takeover_takes_everything_a_dead_predecessor_left(redis_url, prefix, monkeypatch):
    import json

    from prefixwise.records import TurnRecord

    r = redis.Redis.from_url(redis_url)
    stream = f"{prefix}:{partition_of('s', 4)}"
    r.xgroup_create(stream, "prefixwise", id="0", mkstream=True)
    n = 45
    for i in range(n):
        rec = TurnRecord(provider="openai", api="chat", model="gpt-5.6", session_id=f"s{i % 3}", agent_id="a",
                         request_id=f"r-{i}", usage={"prompt_tokens": 2000, "completion_tokens": 1,
                                                      "prompt_tokens_details": {"cached_tokens": 0}})
        r.xadd(stream, {"r": json.dumps(rec.to_dict())})
    r.xreadgroup("prefixwise", "dead", {stream: ">"}, count=n)  # all pending, consumer gone
    _dead(monkeypatch)
    c, lc = consumer(redis_url, prefix, [partition_of("s", 4)], replay=False, batch=20)  # smaller than the backlog
    try:
        wait_for(lambda: len(rows(c.ledger)) == n)  # not just the first batch, and not after a minute
        assert lc.snapshot()["claimed"] == n and r.xpending(stream, "prefixwise")["pending"] == 0
    finally:
        lc.stop(1)
