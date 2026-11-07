"""Live overhead check against a real provider: direct vs. through Prefixwise.

Sends the same requests both ways, interleaved (direct, proxy, direct, ...),
so network and provider load affect both equally. Measures:
  - added latency (median and p90 of proxy minus direct, paired by position)
  - cache fidelity: cached-token share direct vs. through the proxy must match
  - that every proxied request is in the ledger

Cost: about 40 small requests (a ~4k-token cached prefix, max_tokens 1).
Haiku 5.5: well under $0.10. Keys come from the environment only.

    ANTHROPIC_API_KEY=... python benchmarks/live_overhead.py --provider anthropic
    OPENAI_API_KEY=...    python benchmarks/live_overhead.py --provider openai
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench import Proxy, Session  # noqa: E402

PROVIDERS = {
    "anthropic": {"url": "https://api.anthropic.com", "path": "/v1/messages", "key": "ANTHROPIC_API_KEY",
                  "model": "claude-haiku-5-5"},
    "openai": {"url": "https://api.openai.com", "path": "/v1/chat/completions", "key": "OPENAI_API_KEY",
               "model": "gpt-5.6-luna"},
}


def post(url: str, body: dict, headers: dict) -> tuple[int, float, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
            return r.status, time.perf_counter() - t0, payload
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t0, {"error": e.read()[:300].decode(errors="replace")}


def cached_share(provider: str, usage: dict) -> float:
    if provider == "anthropic":
        read = usage.get("cache_read_input_tokens", 0) or 0
        total = read + (usage.get("cache_creation_input_tokens", 0) or 0) + (usage.get("input_tokens", 0) or 0)
    else:
        total = usage.get("prompt_tokens", 0) or 0
        read = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return read / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=PROVIDERS, required=True)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--model")
    ap.add_argument("--url", help="override the provider base URL (testing against a fake upstream)")
    a = ap.parse_args()
    cfg = dict(PROVIDERS[a.provider])
    if a.url:
        cfg["url"] = a.url.rstrip("/")
    key = os.environ.get(cfg["key"])
    if not key:
        sys.exit(f"set {cfg['key']} in the environment")
    model = a.model or cfg["model"]
    auth = ({"x-api-key": key, "anthropic-version": "2023-06-01"} if a.provider == "anthropic"
            else {"authorization": f"Bearer {key}"})
    tmp = tempfile.mkdtemp()
    # The proxy forwards the client's own auth header; no key in its environment.
    os.environ.pop("PREFIXWISE_API_KEY", None)
    px = Proxy(cfg["url"], f"{tmp}/ledger.db")

    # Two sessions of the same shape and size but different text, so each
    # warms only its own cache: the cache-hit comparison is then a real check
    # that the proxy forwards what the provider can reuse.
    sd = Session("live-direct", a.provider, 4000, 40, 7)
    sp = Session("live-proxy", a.provider, 4000, 40, 8)
    rows = []
    try:
        for i in range(a.pairs):
            # Deep copies: the breakpoint below must not leak into the sessions'
            # stored history, or breakpoints pile up (Anthropic allows 4).
            bd, bp = copy.deepcopy(sd.next_body(model=model)), copy.deepcopy(sp.next_body(model=model))
            for b in (bd, bp):
                b.pop("user", None)
                b.pop("metadata", None)
                b["max_tokens" if a.provider == "anthropic" else "max_completion_tokens"] = 16
            if a.provider == "anthropic":  # explicit breakpoint on the last message
                for b in (bd, bp):
                    last = b["messages"][-1]
                    last["content"] = [{"type": "text", "text": last["content"],
                                        "cache_control": {"type": "ephemeral"}}]
                    assert json.dumps(b).count('"cache_control"') == 1, "exactly one breakpoint per request"
            first, second = (("direct", cfg["url"], bd, {}),
                             ("proxy", f"http://127.0.0.1:{px.port}", bp, {"x-prefixwise-session-id": "live-proxy"}))
            if i % 2:  # alternate order so neither side always goes first
                first, second = second, first
            res = {}
            for side, base, body, extra in (first, second):
                st, secs, payload = post(base + cfg["path"], body, {**auth, **extra})
                if st != 200:
                    sys.exit(f"{side} request {i} failed: {st} {payload}")
                res[side] = (secs, cached_share(a.provider, payload.get("usage") or {}))
            rows.append(res)
            print(f"  pair {i + 1:2}: direct {res['direct'][0] * 1000:7.0f} ms ({res['direct'][1]:.0%} cached)  "
                  f"proxy {res['proxy'][0] * 1000:7.0f} ms ({res['proxy'][1]:.0%} cached)", flush=True)
    finally:
        px.stop()
    added = sorted((r["proxy"][0] - r["direct"][0]) * 1000 for r in rows[1:])  # skip the cold first pair
    cd = statistics.mean(r["direct"][1] for r in rows[1:])
    cp = statistics.mean(r["proxy"][1] for r in rows[1:])
    led = sqlite3.connect(px.ledger).execute("select count(*) from entries").fetchone()[0]
    print(f"\nprovider {a.provider}, model {model}, {len(rows)} pairs")
    print(f"added latency: median {statistics.median(added):+.0f} ms, p90 {added[int(len(added) * .9)]:+.0f} ms "
          f"(direct median {statistics.median(r['direct'][0] for r in rows) * 1000:.0f} ms)")
    print(f"cache hit share: direct {cd:.1%}, proxy {cp:.1%}  ->  "
          f"{'MATCH' if abs(cd - cp) < 0.05 else 'MISMATCH'}")
    print(f"ledger rows: {led}/{len(rows)}")
    return 0 if abs(cd - cp) < 0.05 and led == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
