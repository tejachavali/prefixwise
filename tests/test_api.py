"""/api/v1: every number over HTTP, from the same definitions as the CLI."""

import json
import urllib.error
import urllib.request

from test_proxy import BODY, post, run_proxy


def get(port, path, token=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 headers={"X-Prefixwise-Token": token} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def seeded():
    proxy, server, port = run_proxy()  # mock upstream: first_write, then cache hits
    for i in range(3):
        post(port, "/v1/chat/completions", {**BODY, "session_id": f"s{i % 2}"}, {"X-Prefixwise-Task-Id": "t1"})
    return proxy, server, port


def test_catalog_summary_breakdown_timeseries():
    proxy, server, port = seeded()
    try:
        s, cat = get(port, "/api/v1/metrics")
        names = {m["name"] for m in cat["metrics"]}
        assert s == 200 and {"cache_hit_rate", "tokens_per_successful_task", "error_rate"} <= names
        assert "session" in cat["dimensions"]
        s, summ = get(port, "/api/v1/summary")
        assert s == 200 and summ["summary"]["requests"] == 3 and summ["summary"]["requests_failed"] == 0
        assert summ["summary"]["prompt_tokens"] == proxy.ledger.summary()["prompt_tokens"]  # one definition
        s, b = get(port, "/api/v1/breakdown?by=session&metrics=requests,cache_hit_rate&order=-requests")
        assert s == 200 and [r["session"] for r in b["rows"]] == ["s0", "s1"]
        assert set(b["rows"][0]) == {"session", "requests", "cache_hit_rate"}
        s, t = get(port, "/api/v1/timeseries?bucket=day&metrics=requests_all")
        assert s == 200 and len(t["rows"]) == 1 and t["rows"][0]["requests_all"] == 3
        s, w = get(port, "/api/v1/summary?since=1h&session=s1&metrics=requests")
        assert s == 200 and w["summary"]["requests"] == 1
    finally:
        server.shutdown()


def test_requests_paging_detail_tasks_fixes_prices_explain():
    proxy, server, port = seeded()
    try:
        s, page = get(port, "/api/v1/requests?limit=2")
        assert s == 200 and len(page["requests"]) == 2 and page["next_before"]
        s, rest = get(port, f"/api/v1/requests?limit=2&before={page['next_before']}")
        assert len(rest["requests"]) == 1 and rest["next_before"] is None
        rid = page["requests"][0]["request_id"]
        s, one = get(port, f"/api/v1/requests/{rid}")
        assert s == 200 and one["request"]["request_id"] == rid and "evidence" in one["request"]
        assert get(port, "/api/v1/requests/nope")[0] == 404
        assert post(port, "/api/v1/tasks", {"task_id": "t1", "success": True})[0] == 200
        s, summ = get(port, "/api/v1/summary?metrics=tasks_successful,tokens_per_successful_task")
        assert summ["summary"]["tasks_successful"] == 1 and summ["summary"]["tokens_per_successful_task"] > 0
        assert get(port, "/api/v1/fixes")[0] == 200
        s, prices = get(port, "/api/v1/prices")
        assert s == 200 and prices["as_of"] and "claude-sonnet-5-5" in prices["models"]
        prev = {"model": "m", "messages": [{"role": "system", "content": "Today is 2026-11-06."},
                                           {"role": "user", "content": "hi"}]}
        curr = {"model": "m", "messages": [{"role": "system", "content": "Today is 2026-11-07."},
                                           {"role": "user", "content": "hi"}]}
        s, ex, _ = post(port, "/api/v1/explain", {"previous": prev, "current": curr})
        assert s == 200 and ex["explanation"]["cause"] == "volatile_system"
    finally:
        server.shutdown()


def test_errors_and_auth():
    proxy, server, port = seeded()
    try:
        assert get(port, "/api/v1/breakdown?by=colour")[0] == 400
        assert get(port, "/api/v1/summary?metrics=vibes")[0] == 400
        assert get(port, "/api/v1/summary?since=yesterday")[0] == 400
        assert get(port, "/api/v1/timeseries?bucket=week")[0] == 400
        assert get(port, "/api/v1/nope")[0] == 404
        assert post(port, "/api/v1/tasks", {"task_id": "x"})[0] == 400
        assert post(port, "/api/v1/explain", [1])[0] == 400
        proxy.auth_token = "secret"
        assert get(port, "/api/v1/summary")[0] == 401
        assert get(port, "/api/v1/summary", token="secret")[0] == 200
        assert post(port, "/api/v1/tasks", {"task_id": "x", "success": True})[0] == 401
    finally:
        server.shutdown()
