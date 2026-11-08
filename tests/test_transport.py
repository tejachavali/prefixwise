"""Records over the wire (prefixwise.transport): an edge sends, a collector classifies.

The fleet product is the single-process product with a wire in the middle: the same traffic
through an edge and a collector yields the same ledger rows as through one proxy.
"""

import json
import threading
import time
import urllib.error
import urllib.request

from prefixwise.cli import build_parser, cmd_collect
from prefixwise.collector import Collector
from prefixwise.ledger import Ledger
from prefixwise.pricing import PriceTable
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler
from prefixwise.records import TurnRecord
from prefixwise.transport import RemoteCollector, Sender

SYSTEM = [{"type": "text", "text": "You are careful. " * 300, "cache_control": {"type": "ephemeral"}}]


def serve(proxy):
    srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def scripted(proxy, usages):
    """Answer every forwarded request with the next usage, in order (the same for two proxies)."""
    queue = list(usages)

    def fake(provider, body, request_id, session_id, client_headers, api="chat"):
        return 200, {"id": "msg_" + request_id[:8], "model": body["model"], "usage": queue.pop(0)}
    proxy._forward = fake
    return proxy


def usage(read=0, write=0, fresh=200):
    return {"input_tokens": fresh, "output_tokens": 5, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write,
            "cache_creation": {"ephemeral_5m_input_tokens": write, "ephemeral_1h_input_tokens": 0}}


SESSION = [usage(write=1200), usage(read=1200, fresh=60), usage(write=1300, fresh=0)]  # write, hit, system edit


def collector_server(token=None):
    c = PrefixwiseProxy(mock=True, collector_url="")
    c.collect_only, c.auth_token = True, token
    srv, url = serve(c)
    return c, srv, url


def http(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), dict(e.headers)


def wait_for(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def turns(session):
    msgs = [{"role": "user", "content": "start"}]
    bodies = []
    for i in range(3):
        body = {"model": "claude-haiku-5-5", "system": SYSTEM, "messages": list(msgs), "max_tokens": 8}
        if i == 2:
            body["system"] = [{**SYSTEM[0], "text": SYSTEM[0]["text"] + " today 2026-11-08"}]  # a system edit
        bodies.append((body, {"x-prefixwise-session-id": session}))
        msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": f"go {i}"}]
    return bodies


def rows(ledger):
    with ledger.reader() as c:
        c.row_factory = None
        return c.execute("SELECT session_id, cause, prompt_tokens, cached_tokens, avoidable_tokens, avoidable_cost, "
                         "cost, detail, secondary_causes, divergence_role, divergence_index, cache_ttl_seconds, "
                         "unread_write_tokens, status, error FROM entries ORDER BY id").fetchall()


def test_edge_and_collector_give_the_rows_one_proxy_gives():
    coll, srv, url = collector_server(token="secret")
    edge = scripted(PrefixwiseProxy(upstream_url="http://unused", api_key="k", collector_url=url,
                                    collector_token="secret"), SESSION * 2)
    local = scripted(PrefixwiseProxy(upstream_url="http://unused", api_key="k", collector_url=""), SESSION * 2)
    try:
        results = []
        for session in ("a", "b"):
            for body, headers in turns(session):
                results.append(edge.handle("anthropic", dict(body), headers)[0])
                local.handle("anthropic", dict(body), headers)
        assert all(r["queued"] and r["cause"] is None and r["status"] == 200 for r in results)
        wait_for(lambda: coll.collector.stats["records"] == 6)
        assert rows(coll.ledger) == rows(local.ledger)
        assert [r[1] for r in rows(coll.ledger)][:3] == ["first_write", "none", "system_edit"]
        edge.close()
        assert edge.collector.stats()["sent"] == 6 and edge.collector.stats()["dropped"] == 0
    finally:
        srv.shutdown()


def test_a_wrong_token_drops_records_and_the_request_is_unaffected():
    coll, srv, url = collector_server(token="secret")
    edge = PrefixwiseProxy(mock=True, collector_url=url, collector_token="wrong")
    try:
        body, headers = turns("s")[0]
        res, _ = edge.handle("anthropic", body, headers)
        assert res["status"] == 200 and res["queued"]
        wait_for(lambda: edge.collector.stats()["dropped"] == 1)  # 401: not retried
        assert coll.collector.stats["records"] == 0 and "HTTP 401" in edge.collector.stats()["last_error"]
    finally:
        srv.shutdown()


def test_an_unreachable_collector_never_blocks_a_request():
    edge = PrefixwiseProxy(mock=True, collector_url="")
    edge.collector = RemoteCollector(Sender("http://127.0.0.1:1", queue_size=5, max_attempts=1, timeout=0.2))
    body, headers = turns("s")[0]
    t0 = time.monotonic()
    for _ in range(20):
        assert edge.handle("anthropic", dict(body), headers)[0]["status"] == 200
    assert time.monotonic() - t0 < 2.0
    wait_for(lambda: edge.collector.stats()["dropped"] >= 15 and edge.collector.stats()["pending"] <= 5)
    edge.close(timeout=1)


def test_ingest_is_idempotent_and_names_malformed_records():
    coll, srv, url = collector_server()
    try:
        good = TurnRecord(provider="openai", api="chat", model="gpt-5.6", session_id="s", agent_id="a",
                          request_id="r-1", usage={"prompt_tokens": 2000, "completion_tokens": 1,
                                                  "prompt_tokens_details": {"cached_tokens": 0}}).to_dict()
        bad = {**good, "request_id": "r-2", "version": 99}
        s, reply, _ = http("POST", url + "/api/v1/ingest", {"records": [good, bad]})
        assert s == 200 and reply["accepted"] == 1 and reply["duplicates"] == 0
        assert reply["rejected"] == [{"index": 1, "error": "record version 99; this collector reads version 1"}]
        s, reply, _ = http("POST", url + "/api/v1/ingest", {"records": [good]})
        assert s == 200 and reply == {"accepted": 0, "duplicates": 1, "rejected": []}
        assert len(rows(coll.ledger)) == 1 and coll.collector.stats["duplicates"] == 1
        s, reply, _ = http("POST", url + "/api/v1/ingest", {"records": "nope"})
        assert s == 400 and "list" in reply["error"]["message"]
    finally:
        srv.shutdown()


def test_an_edge_has_no_ledger_and_a_collector_has_no_upstream():
    coll, srv, url = collector_server()
    edge = PrefixwiseProxy(mock=True, collector_url=url)
    esrv, eurl = serve(edge)
    try:
        s, err, _ = http("GET", eurl + "/api/v1/summary")
        assert s == 409 and url in err["error"]["message"]
        assert http("GET", eurl + "/ledger")[0] == 409
        assert http("POST", eurl + "/tasks", {"task_id": "t", "success": True})[0] == 409
        s, err, _ = http("POST", eurl + "/api/v1/ingest", {"records": []})
        assert s == 404 and "edge" in err["error"]["message"]
        s, err, _ = http("POST", url + "/v1/chat/completions", {"model": "m", "messages": []})
        assert s == 404 and "collector" in err["error"]["message"]
        s, health, _ = http("GET", eurl + "/health")
        assert health["role"] == "edge" and health["collector"]["url"] == url
        s, health, _ = http("GET", url + "/health")
        assert health["role"] == "collector" and "records" in health["ingest"]
    finally:
        esrv.shutdown()
        srv.shutdown()


def test_metrics_expose_the_transport_counters():
    coll, srv, url = collector_server()
    edge = PrefixwiseProxy(mock=True, collector_url=url)
    esrv, eurl = serve(edge)
    try:
        body, headers = turns("s")[0]
        http("POST", eurl + "/v1/messages", {**body, "session_id": "s"})
        wait_for(lambda: coll.collector.stats["records"] == 1)
        text = urllib.request.urlopen(eurl + "/metrics", timeout=5).read().decode()
        assert "prefixwise_records_sent_total 1" in text and "prefixwise_records_dropped_total 0" in text
        text = urllib.request.urlopen(url + "/metrics", timeout=5).read().decode()
        assert "prefixwise_ingest_records_total 1" in text
    finally:
        esrv.shutdown()
        srv.shutdown()


def test_a_turn_sent_before_the_compared_turn_ended_is_unknown():
    c = Collector(Ledger(), PriceTable.load())
    blocks = PrefixwiseProxy(mock=True, collector_url="").fingerprinter.fingerprint_blocks(
        [("system", "Be careful."), ("messages", {"role": "user", "content": "start"})])
    usage = {"input_tokens": 2000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 1}

    def rec(rid, sent, done, cached=0):
        u = {**usage, "cache_read_input_tokens": cached, "input_tokens": 2000 - cached}
        return TurnRecord(provider="anthropic", api="chat", model="claude-haiku-5-5", session_id="s", agent_id="a",
                          request_id=rid, blocks=blocks, has_breakpoint=True, ttl_seconds=300.0, usage=u,
                          sent_at=sent, done_at=done)
    c.ingest(rec("a", 100.0, 105.0))
    res, e = c.ingest(rec("b", 102.0, 106.0))  # in flight alongside a: could not have read a's write
    assert e.cause == "unknown" and "3.0 s before" in e.evidence and c.stats["concurrent"] == 1
    res, e = c.ingest(rec("c", 110.0, 111.0, cached=1000))  # after a: compared with a, a hit
    assert e.cause == "none"


def test_the_collect_command_is_wired():
    args = build_parser().parse_args(["collect", "--port", "9", "--ledger", "x.db"])
    assert args.func is cmd_collect and args.port == 9 and args.ledger == "x.db"
    args = build_parser().parse_args(["serve", "--collector", "http://c:8081"])
    assert args.collector == "http://c:8081"
