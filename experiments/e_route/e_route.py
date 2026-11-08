"""E-route: two vLLM replicas behind a router. Does Prefixwise confirm a routing miss from the
replicas' KV events, with no help from the router?

Per repetition, one session, through the real Prefixwise proxy (in-process) and a small router:
  T1  warm the session (a ~PREFIX-token system prompt + question)   -> first_write
  T2  append a turn                                                 -> the measurement
  T3  append another turn                                           -> recorded, not scored

Router modes:
  alternate   each request goes to the other replica. T2 lands on the replica that has never
              seen the session. Correct cause: routing_miss, confirmed from KV events
              ("served by B, while A ... still held the prefix").
  sticky      a session always goes to the same replica (by session id). Control: T2 is a hit.

Replicas: two vLLM servers on one GPU, each publishing KV events (see README.md), or --mock
(two simulated replicas: an HTTP server and a ZMQ publisher each, following vLLM's report
format), which needs no GPU.

    python experiments/e_route/e_route.py --mock
    python experiments/e_route/e_route.py --replicas http://127.0.0.1:8001,http://127.0.0.1:8002 \\
        --kv-events tcp://127.0.0.1:5557,tcp://127.0.0.1:5558 --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import itertools
import json
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prefixwise.proxy import PrefixwiseProxy  # noqa: E402

WORDS = ("cache prefix token block engine memory route replica session request model tool result "
         "agent system prompt history turn layer attention key value page chunk router worker "
         "affinity balance schedule batch decode prefill latency throughput").split()


def text(seed: int, words: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(words))


def body(model: str, system: str, turns: list[str]) -> dict:
    msgs = [{"role": "system", "content": system}]
    for i, t in enumerate(turns):
        if i:
            msgs.append({"role": "assistant", "content": "Noted."})
        msgs.append({"role": "user", "content": t})
    return {"model": model, "messages": msgs, "max_tokens": 4, "temperature": 0}


class Router:
    """The smallest router: forwards each POST to one replica, alternating or by session."""

    def __init__(self, replicas: list[str], mode: str):
        self.replicas, self.mode = replicas, mode
        self.next = itertools.count()
        self.served: list[str] = []
        router = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                data = json.loads(raw or b"{}")
                if router.mode == "sticky":
                    key = str(data.get("session_id") or "")
                    i = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(router.replicas)
                else:
                    i = next(router.next) % len(router.replicas)
                router.served.append(router.replicas[i])
                headers = {"Content-Type": "application/json"}
                if self.headers.get("authorization"):
                    headers["Authorization"] = self.headers["authorization"]
                req = urllib.request.Request(router.replicas[i] + self.path, data=raw, headers=headers,
                                             method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    out, status = r.read(), r.status
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


class MockReplica:
    """One simulated vLLM replica: a prefix cache of 16-word blocks, an OpenAI chat endpoint that
    reports cached tokens, and a ZMQ publisher sending vLLM's events, every block the request uses
    tagged with its session_id (kv_cache_report_mode "full")."""

    BLOCK = 16

    def __init__(self):
        import msgspec
        import zmq

        self.enc = msgspec.msgpack.Encoder()
        self.pub = zmq.Context.instance().socket(zmq.PUB)
        port = self.pub.bind_to_random_port("tcp://127.0.0.1")
        self.events = f"tcp://127.0.0.1:{port}"
        self.seq = itertools.count(1)
        self.cached: set[int] = set()
        self.lock = threading.Lock()
        replica = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
                out = json.dumps(replica.serve(data)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def serve(self, data: dict) -> dict:
        words = " ".join(f"{m['role']}: {m['content']}" for m in data["messages"]).split()
        hashes, parent = [], b""
        for k in range(len(words) // self.BLOCK):
            parent = hashlib.sha256(parent + " ".join(words[k * self.BLOCK:(k + 1) * self.BLOCK]).encode()).digest()
            hashes.append(int.from_bytes(parent[:8], "big", signed=True))
        with self.lock:
            hit = 0
            while hit < len(hashes) and hashes[hit] in self.cached:
                hit += 1
            self.cached.update(hashes)
        if hashes:  # vLLM "full" mode: the reused blocks and the new ones, with the session
            ev = {"type": "BlockStored", "block_hashes": hashes, "parent_block_hash": None,
                  "token_ids": [], "block_size": self.BLOCK, "lora_id": None, "medium": "GPU",
                  "lora_name": None, "session_id": data.get("session_id")}
            self.pub.send_multipart([b"kv-events", next(self.seq).to_bytes(8, "big"),
                                     self.enc.encode([time.time(), [ev]])])
            time.sleep(0.02)
        return {"id": f"cmpl-{random.random()}", "object": "chat.completion", "model": data["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Noted."}}],
                "usage": {"prompt_tokens": len(words), "completion_tokens": 1,
                          "prompt_tokens_details": {"cached_tokens": hit * self.BLOCK}}}


def run(mode: str, replicas: list[str], kv_events: str, model: str, reps: int, prefix: int) -> list[dict]:
    router = Router(replicas, mode)
    proxy = PrefixwiseProxy(upstream_url=router.url, engine="vllm", kv_events=kv_events)
    time.sleep(1.0)  # ZMQ subscriptions propagate
    rows = []
    try:
        for rep in range(reps):
            session = f"e-route-{mode}-{rep}-{random.randrange(10**9)}"
            system = text(hash(session) % 10**9, int(prefix / 1.3))
            turns = [text(rep * 3 + k, 20) for k in range(3)]
            h = {"x-prefixwise-session-id": session}
            for t in range(3):
                res, _ = proxy.handle("openai", body(model, system, turns[:t + 1]), h)
                rows.append({"mode": mode, "rep": rep, "turn": t + 1, "replica": router.served[-1],
                             "cause": res.get("cause"), "cached": res.get("cached_tokens"),
                             "prompt": res.get("prompt_tokens"), "evidence": res.get("evidence")})
                time.sleep(0.3)  # let the replica's events arrive before the next turn
    finally:
        proxy.engine_events and proxy.engine_events.stop()
        router.server.shutdown()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="two simulated replicas; no GPU")
    ap.add_argument("--replicas", help="comma-separated replica base URLs")
    ap.add_argument("--kv-events", help="comma-separated KV-event endpoints, in the same order")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--prefix", type=int, default=2000, help="system prompt size, tokens (approx.)")
    ap.add_argument("--out", help="write the rows as JSON here")
    args = ap.parse_args()
    if args.mock:
        mocks = [MockReplica(), MockReplica()]
        replicas, kv_events = [m.url for m in mocks], ",".join(m.events for m in mocks)
    else:
        if not (args.replicas and args.kv_events):
            ap.error("--replicas and --kv-events (or --mock)")
        replicas, kv_events = [u.rstrip("/") for u in args.replicas.split(",")], args.kv_events
    rows = run("alternate", replicas, kv_events, args.model, args.reps, args.prefix)
    rows += run("sticky", replicas, kv_events, args.model, args.reps, args.prefix)

    def t2(mode):
        return [r for r in rows if r["mode"] == mode and r["turn"] == 2]
    routed = [r for r in t2("alternate") if r["cause"] == "routing_miss" and "confirmed" in (r["evidence"] or "")]
    hits = [r for r in t2("sticky") if r["cause"] == "none"]
    for r in rows:
        print(f"{r['mode']:9} rep {r['rep']} T{r['turn']} -> {r['replica'].rsplit(':', 1)[-1]:>5}  "
              f"cached {r['cached']}/{r['prompt']}  {r['cause']}: {r['evidence']}")
    print(f"\nalternate T2: routing_miss confirmed {len(routed)}/{len(t2('alternate'))}")
    print(f"sticky T2 (control): hit {len(hits)}/{len(t2('sticky'))}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    ok = len(routed) == len(t2("alternate")) and len(hits) == len(t2("sticky"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
