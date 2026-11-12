"""Importing a gateway's request log gives the verdicts the live proxy gives.

The same traffic goes through two paths: the live proxy (a fake upstream returns each response, the
clock set to each request's start and end), and a JSON Lines log of those requests replayed by
`prefixwise import gateway`. Every request must come out with the same cause, detail, avoidable
tokens and cost. Plus: overlapping requests stay overlapping, and the log formats gateways use."""

import json
import random

import pytest

from prefixwise.gateway_import import _Clock, _replay_clock, parse_time, read_log, replay
from prefixwise.ledger import Ledger
from prefixwise.proxy import PrefixwiseProxy
from test_invariants import MUTATIONS, _append_turn, _body

T0 = 1_762_900_000.0  # 2025-11-11T22:26:40Z


def _usage(provider, prompt, cached):
    if provider == "anthropic":
        return {"input_tokens": prompt - cached, "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0, "output_tokens": 5}
    return {"prompt_tokens": prompt, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": cached}}


def _response(provider, i, prompt, cached):
    if provider == "anthropic":
        return {"id": f"msg_{i}", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}], "usage": _usage(provider, prompt, cached)}
    return {"id": f"chatcmpl-{i}", "object": "chat.completion", "model": "gpt-5",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": _usage(provider, prompt, cached)}


def _traffic(seed, n_turns=8):
    """Two sessions interleaved, each turn a random change or plain growth, random reads and gaps."""
    rng = random.Random(seed)
    provider = rng.choice(["anthropic", "openai"])
    traffic, t = [], T0
    bodies = {s: _body(rng, provider, rng.randint(2, 4), 2, "2026-11-12T09:00:00Z") for s in ("conv-a", "conv-b")}
    prompts = {s: 20_000 for s in bodies}
    for i in range(n_turns):
        s = rng.choice(list(bodies))
        nxt = _append_turn(bodies[s])
        if rng.random() < 0.5:
            changed = rng.choice(MUTATIONS)(rng, nxt, provider)
            nxt = changed[0] if changed else nxt
        prompts[s] += 400
        read = rng.choice([0, prompts[s] // 2, prompts[s] - 400])
        t += rng.choice([2, 8, 30, 400])  # 400 s outlives a 5-minute cache
        traffic.append({"start": t, "end": t + 1.5, "provider": provider, "body": nxt,
                        "headers": {"x-prefixwise-session-id": s},
                        "response": _response(provider, i, prompts[s], read)})
        bodies[s] = nxt
        t += 1.5
    return traffic


def _live(traffic, ledger_path):
    clock = _Clock()
    with _replay_clock(clock):
        p = PrefixwiseProxy(ledger_path=ledger_path, mock=False, upstream_url="http://live.invalid",
                            engine="", kv_events="", fixes=(), otel="", collector_url="")
        for e in traffic:
            clock.now = e["start"]

            def fake(provider, body, request_id, session_id, client_headers, api="chat", e=e):
                clock.now = e["end"]
                return 200, e["response"]
            p._forward = fake
            p.handle(e["provider"], e["body"], dict(e["headers"]))
        p.close()


def _rows(path):
    with Ledger(path).reader() as conn:
        return [tuple(r) for r in conn.execute(
            "SELECT session_id, cause, detail, prompt_tokens, cached_tokens, avoidable_tokens, "
            "ROUND(COALESCE(avoidable_cost, -1), 9), recorded_at FROM entries ORDER BY recorded_at, id")]


@pytest.mark.parametrize("seed", range(40))
def test_an_imported_log_gives_the_live_verdicts(seed, tmp_path):
    traffic = _traffic(seed)
    _live(traffic, str(tmp_path / "live.db"))
    log = tmp_path / "gateway.jsonl"
    with open(log, "w") as fh:
        for e in traffic:
            fh.write(json.dumps({"start_time": e["start"], "end_time": e["end"], "request": e["body"],
                                 "response": e["response"], "request_headers": e["headers"]}) + "\n")
    entries, bad = read_log(log)
    assert bad == 0 and len(entries) == len(traffic)
    stats = replay(entries, str(tmp_path / "import.db"))
    assert stats["imported"] == len(traffic)
    live, imported = _rows(str(tmp_path / "live.db")), _rows(str(tmp_path / "import.db"))
    assert live == imported


def test_overlapping_requests_overlap_in_the_replay(tmp_path):
    """Two requests of one session in flight at once, logged in the wrong order: the replay runs them
    as they happened (start and end events in time order), so the in-flight check sees the overlap."""
    rng = random.Random(3)
    body = _body(rng, "openai", 3, 2, None)
    nxt = _append_turn(body)
    lines = [
        {"timestamp": T0 + 5, "duration_ms": 1000, "request": nxt, "headers": {"x-session-id": "c"},
         "response": _response("openai", 2, 20_400, 0)},
        {"timestamp": T0, "duration_ms": 8000, "request": body, "headers": {"x-session-id": "c"},
         "response": _response("openai", 1, 20_000, 0)},
    ]
    log = tmp_path / "g.jsonl"
    log.write_text("\n".join(json.dumps(x) for x in lines))
    entries, _ = read_log(log)
    replay(entries, str(tmp_path / "i.db"))
    rows = _rows(str(tmp_path / "i.db"))
    # The short request finished first and is recorded first (its lane was empty: first_write); the
    # long one, in flight the whole time, is then compared with a turn it overlapped: no cause claimed.
    assert [r[1] for r in rows] == ["first_write", "unknown"]


def test_log_formats():
    assert parse_time(1_762_900_000) == 1_762_900_000.0
    assert parse_time(1_762_900_000_123) == pytest.approx(1_762_900_000.123)
    assert parse_time("2025-11-11T22:26:40Z") == T0
    assert parse_time("2025-11-11T22:26:40") == T0  # no zone: UTC
    assert parse_time("2025-11-11T22:26:40.500+00:00") == T0 + 0.5
    assert parse_time("not a time") is None and parse_time(True) is None


def test_bodies_as_strings_mixed_case_keys_and_bad_lines(tmp_path):
    rng = random.Random(4)
    body = _body(rng, "anthropic", 2, 2, None)
    good = {"StartTime": "2025-11-11T22:26:40Z", "Latency_MS": 900, "Request_Body": json.dumps(body),
            "Response": json.dumps(_response("anthropic", 1, 20_000, 0)), "Status": 200}
    log = tmp_path / "g.jsonl"
    log.write_text("\n".join([json.dumps(good), "not json", json.dumps({"request": body}), ""]))
    entries, bad = read_log(log)
    assert len(entries) == 1 and bad == 2
    e = entries[0]
    assert e["provider"] == "anthropic" and e["end"] == pytest.approx(T0 + 0.9) and e["body"] == body


def test_the_cli_refuses_to_count_a_log_twice(tmp_path, monkeypatch, capsys):
    from prefixwise.cli import main
    rng = random.Random(5)
    body = _body(rng, "openai", 2, 2, None)
    log = tmp_path / "g.jsonl"
    log.write_text(json.dumps({"ts": T0, "request": body, "response": _response("openai", 1, 20_000, 0)}))
    monkeypatch.chdir(tmp_path)
    assert main(["import", "gateway", "--path", str(log)]) == 0
    out = capsys.readouterr().out
    assert "Imported 1 requests from 1 a gateway log" in out or "Imported 1 requests" in out
    assert main(["import", "gateway", "--path", str(log)]) == 1
    assert "would count it twice" in capsys.readouterr().err
