"""POST /api/v1/observe, the observer behind it: live gateway events give the import's verdicts.

The traffic of tests/test_gateway_import.py, posted as live events (in order, shuffled, from many
threads at once), must come out of the ledger exactly as importing it as a log does: same causes,
details, avoidable tokens and cost, same times. Plus: bad events are refused with a reason and
nothing else is affected, a stream's final usage object is a response, and a full queue refuses."""

import json
import random
import threading

from prefixwise.gateway_import import read_log, replay
from prefixwise.observe import Observer
from prefixwise.proxy import PrefixwiseProxy
from test_gateway_import import T0, _append_turn, _body, _response, _rows, _traffic

HOLD = 0.3  # seconds; long enough for a test's posts to arrive together, short enough to be quick


def _event(e):
    return {"start_time": e["start"], "end_time": e["end"], "request": e["body"], "response": e["response"],
            "request_headers": e["headers"]}


def _proxy(path):
    return PrefixwiseProxy(ledger_path=path, mock=False, upstream_url="http://observe.invalid",
                           engine="", kv_events="", fixes=(), otel="", collector_url="")


def _imported(traffic, tmp_path):
    log = tmp_path / "gateway.jsonl"
    log.write_text("\n".join(json.dumps(_event(e)) for e in traffic) + "\n")
    replay(read_log(log)[0], str(tmp_path / "import.db"))
    return _rows(str(tmp_path / "import.db"))


def _observed(posts, path, hold=HOLD, threads=1):
    """Post each payload to an observer (from `threads` threads at once), wait, close."""
    proxy = _proxy(path)
    obs = Observer(proxy, hold=hold)
    replies = []
    if threads == 1:
        replies = [obs.post(p) for p in posts]
    else:
        lock, barrier = threading.Lock(), threading.Barrier(threads)

        def run(chunk):
            barrier.wait()
            for p in chunk:
                r = obs.post(p)
                with lock:
                    replies.append(r)
        ts = [threading.Thread(target=run, args=(posts[i::threads],)) for i in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert obs.flush(10)
    stats = obs.snapshot()
    obs.close()
    proxy.close()
    return replies, stats


def test_posted_traffic_gives_the_imported_verdicts(tmp_path):
    for seed in range(20):
        traffic = _traffic(seed)
        d = tmp_path / str(seed)
        d.mkdir()
        replies, stats = _observed([_event(e) for e in traffic], str(d / "observe.db"), hold=0)
        assert all(code == 202 for code, _ in replies)
        assert stats["recorded"] == len(traffic) and stats["failed"] == 0
        assert _rows(str(d / "observe.db")) == _imported(traffic, d), seed


def test_a_batch_gives_the_imported_verdicts(tmp_path):
    traffic = _traffic(7)
    replies, _ = _observed([{"events": [_event(e) for e in traffic]}], str(tmp_path / "o.db"), hold=0)
    assert replies == [(202, {"accepted": len(traffic)})]
    assert _rows(str(tmp_path / "o.db")) == _imported(traffic, tmp_path)


def test_out_of_order_and_concurrent_posts_are_recorded_in_turn_order(tmp_path):
    """Shuffled, and from eight threads at once: within the hold, the events are put back in time
    order, so every turn is compared with the turn before it, as in the import."""
    for seed in range(10):
        traffic = _traffic(seed, n_turns=12)
        d = tmp_path / str(seed)
        d.mkdir()
        posts = [_event(e) for e in traffic]
        random.Random(seed).shuffle(posts)
        _, stats = _observed(posts, str(d / "observe.db"), threads=8)
        assert stats["recorded"] == len(traffic) and stats["late"] == 0
        assert _rows(str(d / "observe.db")) == _imported(traffic, d), seed


def test_an_event_later_than_the_hold_is_still_recorded_and_counted_late(tmp_path):
    traffic = _traffic(1, n_turns=4)
    proxy = _proxy(str(tmp_path / "o.db"))
    obs = Observer(proxy, hold=0)
    for e in traffic[1:]:
        assert obs.post(_event(e))[0] == 202
    assert obs.flush(5)
    assert obs.post(_event(traffic[0]))[0] == 202  # the first turn, after the others were recorded
    assert obs.flush(5)
    assert obs.snapshot()["late"] == 1 and obs.snapshot()["recorded"] == len(traffic)
    obs.close()
    proxy.close()


def test_overlapping_requests_overlap_as_in_the_import(tmp_path):
    """The import's overlap case, live: the long request was in flight while the short one ran."""
    rng = random.Random(3)
    body = _body(rng, "openai", 3, 2, None)
    nxt = _append_turn(body)
    events = [{"timestamp": T0 + 5, "duration_ms": 1000, "request": nxt, "headers": {"x-session-id": "c"},
               "response": _response("openai", 2, 20_400, 0)},
              {"timestamp": T0, "duration_ms": 8000, "request": body, "headers": {"x-session-id": "c"},
               "response": _response("openai", 1, 20_000, 0)}]
    _observed(events, str(tmp_path / "o.db"))
    assert [r[1] for r in _rows(str(tmp_path / "o.db"))] == ["first_write", "unknown"]


def test_a_malformed_event_is_refused_with_its_reason_and_nothing_else_is_affected(tmp_path):
    traffic = _traffic(5)
    good = [_event(e) for e in traffic]
    bad = [{"request": good[0]["request"], "response": good[0]["response"]},  # no start time
           {"start_time": T0, "request": "not json", "response": {}},
           "not an object"]
    batch = good[:3] + bad[:1] + good[3:] + bad[1:]
    replies, stats = _observed([{"events": batch}], str(tmp_path / "o.db"), hold=0)
    (code, reply), = replies
    assert code == 400 and reply["accepted"] == len(good)
    assert [r["index"] for r in reply["rejected"]] == [3, len(batch) - 2, len(batch) - 1]
    assert "start time" in reply["rejected"][0]["reason"]
    assert "request body" in reply["rejected"][1]["reason"]
    assert "JSON object" in reply["rejected"][2]["reason"]
    assert reply["error"]["message"]
    assert stats["rejected"] == 3 and stats["recorded"] == len(good)
    assert _rows(str(tmp_path / "o.db")) == _imported(traffic, tmp_path)  # as if the bad ones never came


def test_a_bad_envelope_is_refused():
    obs = Observer(_proxy(":memory:"), hold=0)
    assert obs.post({"events": "x"})[0] == 400
    assert obs.post({"events": []})[0] == 400
    assert obs.post({"events": [{}] * 1001})[0] == 400
    assert obs.snapshot()["accepted"] == 0
    obs.close()


def test_a_stream_reported_by_its_final_usage_object_alone(tmp_path):
    """A streamed call has no response body to log: the gateway posts the final usage on its own."""
    traffic = _traffic(11)
    streamed = []
    for e in traffic:
        ev = _event(e)
        ev["request"] = {**ev["request"], "stream": True}
        ev["response"] = e["response"]["usage"]  # only the usage object
        streamed.append(ev)
    replies, stats = _observed(streamed, str(tmp_path / "o.db"), hold=0)
    assert all(code == 202 for code, _ in replies) and stats["recorded"] == len(traffic)
    observed, imported = _rows(str(tmp_path / "o.db")), _imported(traffic, tmp_path)
    assert [r[:7] for r in observed] == [r[:7] for r in imported]  # same verdicts and tokens
    assert all(r[3] > 0 for r in observed)  # the usage was read, not taken as empty


def test_a_full_queue_refuses_and_queues_nothing():
    traffic = _traffic(2, n_turns=3)
    proxy = _proxy(":memory:")
    obs = Observer(proxy, hold=60, max_pending=2)  # held: nothing is recorded during the test
    assert obs.post({"events": [_event(e) for e in traffic[:2]]})[0] == 202
    code, reply = obs.post(_event(traffic[2]))
    assert code == 503 and "retry" in reply["error"]["message"]
    assert obs.snapshot()["pending"] == 2 and obs.snapshot()["refused"] == 1
    obs.close(5)  # closing records what is held at once
    assert obs.snapshot()["recorded"] == 2 and obs.snapshot()["pending"] == 0
    assert obs.post(_event(traffic[2]))[0] == 503  # closed
    proxy.close()


def test_the_proxy_records_what_is_waiting_when_it_closes(tmp_path):
    traffic = _traffic(4, n_turns=3)
    proxy = _proxy(str(tmp_path / "o.db"))
    proxy.observer.hold = 60
    assert proxy.observer.post({"events": [_event(e) for e in traffic]})[0] == 202
    proxy.close()
    assert len(_rows(str(tmp_path / "o.db"))) == len(traffic)


# -- over HTTP: POST /api/v1/observe ----------------------------------------------------------

def _serve(proxy):
    from prefixwise.proxy import PrefixwiseHTTPServer, make_handler
    srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url, payload, headers=None, raw=None):
    import urllib.error
    import urllib.request
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(url + "/api/v1/observe", data=data, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_posting_over_http_gives_the_imported_verdicts(tmp_path):
    traffic = _traffic(3, n_turns=10)
    proxy = _proxy(str(tmp_path / "o.db"))
    proxy.observer.hold = HOLD
    srv, url = _serve(proxy)
    try:
        posts = [_event(e) for e in traffic]
        random.Random(3).shuffle(posts)
        for p in posts:
            assert _post(url, p) == (202, {"accepted": 1})
        assert proxy.observer.flush(10)
        import urllib.request
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            health = json.loads(r.read())
        assert health["observe"]["recorded"] == len(traffic) and health["observe"]["pending"] == 0
    finally:
        srv.shutdown()
        proxy.close()
    assert _rows(str(tmp_path / "o.db")) == _imported(traffic, tmp_path)


def test_the_token_is_checked_when_set():
    import urllib.error
    import urllib.request
    ev = _event(_traffic(0, n_turns=1)[0])
    proxy = _proxy(":memory:")
    proxy.observer.hold = 0
    srv, url = _serve(proxy)
    try:
        assert _post(url, ev)[0] == 202  # no token set: open, like the rest of the API
        proxy.observe_token = "gateway-secret"
        assert _post(url, ev)[0] == 401
        assert _post(url, ev, {"Authorization": "Bearer wrong"})[0] == 401
        assert _post(url, ev, {"Authorization": "Bearer gateway-secret"})[0] == 202
        assert _post(url, ev, {"X-Prefixwise-Token": "gateway-secret"})[0] == 202
        proxy.auth_token = "admin-secret"  # the proxy's own token is accepted here too
        assert _post(url, ev, {"Authorization": "Bearer admin-secret"})[0] == 202
        assert _post(url, ev, {"Authorization": "Bearer gateway-secret"})[0] == 202
        assert _post(url, ev)[0] == 401
        # the observe token opens only /observe, not the rest of the API
        req = urllib.request.Request(url + "/api/v1/summary", headers={"X-Prefixwise-Token": "gateway-secret"})
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("the observe token opened /api/v1/summary")
        except urllib.error.HTTPError as e:
            assert e.code == 401
        assert proxy.observer.flush(5) and proxy.observer.snapshot()["recorded"] == 5
    finally:
        srv.shutdown()
        proxy.close()


def test_malformed_and_oversized_bodies_are_refused_with_a_reason(monkeypatch):
    from prefixwise import proxy as proxy_mod
    proxy = _proxy(":memory:")
    proxy.observer.hold = 0
    srv, url = _serve(proxy)
    try:
        code, reply = _post(url, None, raw=b"{not json")
        assert code == 400 and "invalid JSON" in reply["error"]["message"]
        code, reply = _post(url, [1, 2])
        assert code == 400 and "object" in reply["error"]["message"]
        code, reply = _post(url, {"events": [{"request": {}, "response": {}}]})
        assert code == 400 and "start time" in reply["rejected"][0]["reason"]
        monkeypatch.setattr(proxy_mod, "OBSERVE_MAX_BYTES", 100)
        code, reply = _post(url, _event(_traffic(0, n_turns=1)[0]))
        assert code == 413 and "100 bytes" in reply["error"]["message"]
        assert proxy.observer.snapshot()["accepted"] == 0
    finally:
        srv.shutdown()
        proxy.close()


def test_an_edge_observes_and_its_collector_records(tmp_path):
    """An edge (prefixwise serve --collector) serves /observe too: its records go to the collector."""
    traffic = _traffic(6, n_turns=6)
    coll = PrefixwiseProxy(ledger_path=str(tmp_path / "c.db"), mock=True, collector_url="")
    coll.collect_only = True
    csrv, curl = _serve(coll)
    edge = PrefixwiseProxy(mock=True, collector_url=curl, engine="", kv_events="", fixes=(), otel="")
    edge.observer.hold = 0
    esrv, eurl = _serve(edge)
    try:
        assert _post(eurl, {"events": [_event(e) for e in traffic]})[0] == 202
        assert edge.observer.flush(10)
        edge.close()  # sends what the edge still holds
        imported = _imported(traffic, tmp_path)
        # The same verdicts; the collector stamps a row when the record arrives (recorded_at).
        assert [r[:7] for r in _rows(str(tmp_path / "c.db"))] == [r[:7] for r in imported]
    finally:
        esrv.shutdown()
        csrv.shutdown()
        coll.close()
