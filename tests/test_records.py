"""The turn record (prefixwise.records): the edge → collector boundary.

A record carries everything the collector needs and no prompt text; it survives the wire
(to_dict / from_dict) without changing a verdict; a collector refuses what it cannot read.
"""

import json

import pytest
from test_details import MODEL, USER, a_usage, body, proxy

from prefixwise import metrics
from prefixwise.collector import Collector
from prefixwise.ledger import Ledger
from prefixwise.pricing import PriceTable
from prefixwise.records import RECORD_VERSION, RecordError, TurnRecord

SECRET = "zqx-the-quick-brown-fox-7f3a"  # a string that appears nowhere but in the prompt


def _rows(ledger):
    with ledger.reader() as c:
        c.row_factory = None
        return c.execute("SELECT session_id, agent_id, provider, model, api, stream, status, error, task_id, cause, "
                         "legit, native_reason, prompt_tokens, cached_tokens, uncached_tokens, cache_write_tokens, "
                         "completion_tokens, avoidable_tokens, avoidable_cost, cost, price_as_of, evidence, detail, "
                         "secondary_causes, divergence_role, divergence_index, cache_ttl_seconds, fixes, "
                         "unread_write_tokens, unread_write_cost, gap_seconds IS NOT NULL, latency_ms IS NOT NULL "
                         "FROM entries ORDER BY id").fetchall()


def _records(p, bodies):
    """Drive a proxy, capturing the records it hands its collector."""
    seen = []
    real = p.collector.ingest

    def spy(rec):
        seen.append(rec)
        return real(rec)
    p.collector.ingest = spy
    for b in bodies:
        p.handle("anthropic", b, {"x-prefixwise-session-id": "s"})
    return seen


def test_a_record_holds_no_prompt_text():
    b = body([{"role": "user", "content": f"please {SECRET} now"}])
    b["system"] = [{"type": "text", "text": f"You know {SECRET}.", "cache_control": {"type": "ephemeral"}}]
    b["tools"] = [{"name": "t", "description": SECRET, "input_schema": {"type": "object", "properties": {}}}]
    # Review finding: request parameters and the cache key can hold text too; they travel as digests.
    b["tool_choice"] = {"type": "tool", "name": f"tool_{SECRET}"}
    b["output_format"] = {"type": "json_schema", "schema": {"description": SECRET}}
    b["prompt_cache_key"] = f"key_{SECRET}"
    p = proxy([a_usage(write5=8_000)])
    [rec] = _records(p, [b])
    wire = json.dumps(rec.to_dict())
    assert SECRET not in wire and "quick" not in wire and "You know" not in wire
    assert set(rec.params) == {"tool_choice", "output_format"} and rec.facts.cache_key
    assert len(rec.blocks) == 3 and all(len(h) == 64 for blk in rec.blocks for h in (blk.raw, blk.canonical, blk.masked))


def test_round_trip_through_the_wire_gives_the_same_rows():
    bodies = [body([USER]), body([USER, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}])]
    bodies[0]["system"][0]["text"] += " Today is 2026-11-07."
    bodies[1]["system"][0]["text"] += " Today is 2026-11-08."  # a volatile change
    usages = [a_usage(write5=8_000), a_usage(write5=8_100, fresh=100)]
    direct = proxy(list(usages))
    recs = _records(direct, bodies)
    wire = [json.loads(json.dumps(r.to_dict())) for r in recs]
    again = Collector(Ledger(), PriceTable.load())
    for d in wire:
        again.ingest(TurnRecord.from_dict(d))
    assert _rows(again.ledger) == _rows(direct.ledger)
    assert _rows(again.ledger)[1][9] == "volatile_system"  # the cause column


def test_the_collector_refuses_what_it_cannot_read():
    good = TurnRecord(provider="openai", api="chat", model="m", session_id="s", agent_id="a", request_id="r").to_dict()
    with pytest.raises(RecordError):
        TurnRecord.from_dict({**good, "version": RECORD_VERSION + 1})
    with pytest.raises(RecordError):
        TurnRecord.from_dict({**good, "extra_field": 1})
    with pytest.raises(RecordError):
        TurnRecord.from_dict({**good, "session_id": ""})
    with pytest.raises(RecordError):
        TurnRecord.from_dict({**good, "blocks": [{"role": "system", "index": 0, "raw": "short", "canonical": "x",
                                                   "masked": "y", "byte_length": 1}]})
    with pytest.raises(RecordError):
        TurnRecord.from_dict("not an object")
    assert TurnRecord.from_dict(good).request_id == "r"


def test_a_failure_record_leaves_a_row_with_no_verdict():
    c = Collector(Ledger(), PriceTable.load())
    rec = TurnRecord(provider="anthropic", api="chat", model=MODEL, session_id="s", agent_id="a", request_id="r",
                     status=502, error="upstream_unreachable")
    e = c.ingest_failure(rec)
    assert e.cause is None and e.status == 502 and e.error == "upstream_unreachable"
    assert metrics.query(c.ledger, ["requests_failed"])[0]["requests_failed"] == 1
    assert c.stats["failures"] == 1 and c.stats["records"] == 0
