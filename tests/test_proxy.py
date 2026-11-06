import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from prefixwise.proxy import PrefixwiseProxy, make_handler


def run_proxy():
    proxy = PrefixwiseProxy()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return proxy, server


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_proxy_first_then_second_turn():
    proxy, server = run_proxy()
    try:
        port = server.server_address[1]
        body = {"model": "gpt-5.6", "session_id": "s1",
                "messages": [{"role": "system", "content": "You are helpful."},
                             {"role": "user", "content": "hi"}]}
        first = post(port, "/v1/chat/completions", body)
        assert first["cause"] == "first_write"
        second = post(port, "/v1/chat/completions", body)
        assert second["cause"] == "none"  # mock caches 1,000 of 2,000 tokens
        assert proxy.ledger.tokens_per_successful_task() is None  # no task_success yet
        assert len(proxy.ledger.rollup_by_cause()) >= 1
    finally:
        server.shutdown()
