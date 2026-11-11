"""Where the bill went: re-reading cached context is billed every turn, and on a long, well-cached session
it is most of the bill though no miss caused it. From a gateway engineer on Reddit (r/LLMDevs): five calls
from their DeepSeek dashboard, every one a near-total hit, split 62% cache re-reads, 9% new input, 28% output."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prefixwise.findings import bill_line, findings, floor_advice, render, render_share
from prefixwise.pricing import PriceTable
from prefixwise.proxy import PrefixwiseProxy

# The rates (USD per 1M tokens) and five calls from a real DeepSeek dashboard, oldest first: (input, cached, output).
# The dashboard billed the first at $0.004736; these rates give $0.004741.
RATES = {"models": {"deepseek-v4.1-flash": {"input": 0.204619, "cache_read": 0.004094, "output": 0.818476}}}
CALLS = [(565800, 563500, 2400), (568200, 565800, 816), (569200, 568200, 1400), (570800, 569100, 782),
         (571900, 570800, 1100)]


def test_the_cost_split_sums_to_the_cost():
    t = PriceTable(models=RATES["models"])
    usage = {"prompt_tokens": 571900, "completion_tokens": 900, "prompt_tokens_details": {"cached_tokens": 570800}}
    parts = t.request_cost_parts("deepseek-v4.1-flash", "openai", usage)
    assert abs(sum(parts.values()) - t.request_cost("deepseek-v4.1-flash", "openai", usage)) < 1e-7
    assert parts["cache_read"] > parts["input"] and parts["cache_write"] == 0
    assert t.request_cost_parts("unknown-model", "openai", usage) is None
    anth = {"input_tokens": 50, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 200, "output_tokens": 10}
    a = PriceTable.load(None).request_cost_parts("claude-sonnet-4-6", "anthropic", anth)
    assert a["cache_write"] > 0 and a["cache_read"] > 0  # writes priced as their own part


class Upstream:
    def __init__(self, calls):
        self.calls = list(calls)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                inp, cached, out = outer.calls.pop(0)
                b = json.dumps({"id": "x", "model": "deepseek-v4.1-flash",
                                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                "usage": {"prompt_tokens": inp, "completion_tokens": out,
                                          "prompt_tokens_details": {"cached_tokens": cached}}}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def test_a_well_cached_deepseek_session_shows_the_floor(tmp_path, capsys):
    prices = tmp_path / "rates.json"
    prices.write_text(json.dumps(RATES))
    up = Upstream(CALLS)
    p = PrefixwiseProxy(upstream_url=up.url, mock=False, prices=str(prices))
    msgs = [{"role": "system", "content": "agent"}, {"role": "user", "content": "go"}]
    try:
        for i in range(len(CALLS)):
            msgs = msgs + [{"role": "assistant", "content": f"step {i}"}, {"role": "user", "content": f"result {i}"}]
            p.handle("openai", {"model": "deepseek-v4.1-flash", "messages": msgs}, {"x-prefixwise-session-id": "long"})
    finally:
        up.server.shutdown()
    f = findings(p.ledger)
    b = f["bill"]
    assert b["requests"] == 5 == b["requests_priced"]
    share = {k: b[k] / b["total"] for k in ("cache_read", "input", "output")}
    assert abs(share["cache_read"] - 0.622) < 0.002 and abs(share["input"] - 0.093) < 0.002
    assert abs(share["output"] - 0.285) < 0.002 and b["cache_write"] == 0
    first = p.ledger.turns("long")[0]["cost"]
    assert abs(first - 0.004736) / 0.004736 < 0.002  # the dashboard's bill for that call
    text = render(f)
    assert "Where the bill went: cache re-reads" in text and "(62%)" in text
    assert "a better hit rate would not lower it" in text  # the re-read is the biggest part: say where the lever is
    assert "bill: 62% cache re-reads, 9% new input, 28% output" in render_share(f)
    details = render(f, headline=False)  # import prints the short block first: the split once, the advice kept
    assert "Where the bill went" not in details and "a better hit rate would not lower it" in details
    from prefixwise.cli import _report_session
    _report_session(p.ledger, "long", as_json=False)
    out = capsys.readouterr().out
    assert "Where the bill went:" in out and "Re-read per turn: 5 turns read from the cache" in out
    assert "at most 570,800" in out  # the first call read from the cache too: it counts


def test_no_split_no_line_and_older_rows_are_counted_out():
    assert bill_line(None) is None and floor_advice(None) is None
    b = {"cache_read": 1.0, "input": 0.0, "cache_write": 0.0, "output": 1.5, "total": 2.5, "requests": 3,
         "requests_priced": 5}
    assert "(3 of 5 priced requests; older rows have no split)" in bill_line(b)
    assert bill_line(b, short=True).startswith("bill (3 of 5 priced requests): ")  # a pasted block says it too
    assert floor_advice(b) is None  # output is the biggest part here, not the re-read
    assert "<1%" in bill_line({**b, "input": 0.001, "total": 2.501})
