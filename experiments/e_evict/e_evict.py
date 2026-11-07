"""E-evict: what happens to a session's prefix when the engine runs out of KV memory,
with and without a KV offload tier (LMCache, SGLang HiCache), and what Prefixwise says.

Per repetition, through the real Prefixwise proxy (in-process, like E1-open):
  T1  warm a session (a ~PREFIX-token system prompt + question)      -> first_write
  T2  append a turn                                                  -> control: a hit
  --  flood the engine directly with unrelated prompts totalling
      FLOOD_FACTOR x the GPU KV capacity, so T1's blocks are evicted from GPU
  T3  append another turn                                            -> the measurement

Expected on T3:
  vllm / sglang (GPU cache only)      cached ~0: the prefix was evicted.
                                      Correct cause: evicted. With --kv-events (the engine
                                      join) Prefixwise should say so; without it, it cannot
                                      see an eviction and says routing_miss.
  vllm-lmcache / sglang-hicache       cached high: the prefix came back from CPU memory.
                                      Correct cause: none.

The GPU KV capacity must be known and small: start the engine with
  vLLM:    --block-size 16 --num-gpu-blocks-override 2048     (= 32,768 tokens)
  SGLang:  --max-total-tokens 32768
and give the offload tier more room than the flood (LMCache L1_GB=20 holds ~130k
Qwen3-4B tokens; the default flood is 3 x 32,768 = ~98k).

    python experiments/e_evict/e_evict.py --mock
    python experiments/e_evict/e_evict.py --engine vllm --url http://127.0.0.1:8000 --model Qwen/Qwen3-4B
    python experiments/e_evict/e_evict.py --engine vllm-lmcache --url http://127.0.0.1:8000 --model Qwen/Qwen3-4B
    python experiments/e_evict/e_evict.py --engine sglang --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prefixwise.proxy import PrefixwiseProxy  # noqa: E402

ENGINES = ("vllm", "vllm-lmcache", "sglang", "sglang-hicache")
OFFLOAD = {"vllm-lmcache", "sglang-hicache"}
WORDS = ("cache prefix token block engine memory evict offload session request model tool result "
         "agent system prompt history turn layer attention key value page chunk radix tree host "
         "device transfer schedule batch decode prefill latency throughput").split()


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


def flood(url: str, model: str, total_tokens: int, each: int, seed: int, mock_state: dict | None) -> int:
    """Unrelated prompts straight to the engine (not through Prefixwise)."""
    n = max(1, -(-total_tokens // each))
    if mock_state is not None:
        mock_state["flooded"] = True
        return n
    for k in range(n):
        b = body(model, text(seed * 100_000 + k, int(each / 1.3)), ["Reply with one word."])
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
    return n


def run(args) -> dict:
    engine = "vllm" if args.engine.startswith("vllm") else "sglang"
    proxy = (PrefixwiseProxy(upstream_url=args.url, api_key="sk-dummy", engine=engine, kv_events=args.kv_events)
             if not args.mock else PrefixwiseProxy(engine=engine))
    mock_state = None
    if args.mock:
        # Simulated engine: hits until flooded; after a flood only an offload tier keeps the prefix.
        mock_state = {"flooded": False}

        def fake_forward(provider, b, request_id, session_id, client_headers, api="chat"):
            prompt = sum(len(m["content"].split()) for m in b["messages"]) * 13 // 10
            first = len(b["messages"]) <= 2
            if first:
                cached = 0
            elif mock_state["flooded"] and args.engine not in OFFLOAD:
                cached = 0
            else:
                cached = int(prompt * 0.9)
            mock_state["flooded"] = False
            return 200, {"id": request_id, "usage": {"prompt_tokens": prompt, "completion_tokens": 1,
                                                     "prompt_tokens_details": {"cached_tokens": cached}}}
        proxy._forward = fake_forward  # type: ignore[method-assign]

    flood_tokens = int(args.gpu_kv_tokens * args.flood_factor)
    reps = []
    for r in range(args.repeats):
        sid = f"evict-{args.engine}-{r}-{int(time.time())}"
        system = text(r, int(args.prefix_tokens / 1.3))
        turns = [f"Question {r}.1: summarise the cache in one word."]
        row = {"rep": r}
        for label in ("t1", "t2", "flood", "t3"):
            if label == "flood":
                t0 = time.time()
                row["flood_requests"] = flood(args.url, args.model, flood_tokens, args.flood_each,
                                              r + 1, mock_state)
                row["flood_seconds"] = round(time.time() - t0, 1)
                continue
            if label != "t1":
                turns.append(f"Question {r}.{len(turns) + 1}: and now?")
            res, _ = proxy.handle("vllm", {**body(args.model, system, turns), "session_id": sid})
            if res["status"] >= 400:
                raise SystemExit(f"rep {r} {label}: upstream {res['status']}: {json.dumps(res['response'])[:300]}")
            row[label] = {"cause": res["cause"], "evidence": res.get("evidence"),
                          "prompt": res["prompt_tokens"], "cached": res["cached_tokens"],
                          "share": round(res["cached_tokens"] / res["prompt_tokens"], 3) if res["prompt_tokens"] else 0}
        reps.append(row)
        print(f"rep {r}: T2 {row['t2']['share']:.0%} cached ({row['t2']['cause']}); flood {row['flood_requests']} "
              f"requests in {row['flood_seconds']} s; T3 {row['t3']['cached']}/{row['t3']['prompt']} cached "
              f"-> Prefixwise: {row['t3']['cause']} ({(row['t3']['evidence'] or '').split(':')[0]})", flush=True)

    offload = args.engine in OFFLOAD
    control_ok = all(x["t2"]["share"] >= 0.5 for x in reps)
    t3_hits = sum(x["t3"]["share"] >= 0.5 for x in reps)
    correct = "none" if offload else "evicted"
    verdicts: dict[str, int] = {}
    for x in reps:
        verdicts[x["t3"]["cause"]] = verdicts.get(x["t3"]["cause"], 0) + 1
    if proxy.engine_events is not None:
        proxy.engine_events.stop()
    return {"engine": args.engine, "model": args.model, "mock": args.mock, "repeats": args.repeats,
            "kv_events": args.kv_events,
            "prefix_tokens": args.prefix_tokens, "gpu_kv_tokens": args.gpu_kv_tokens,
            "flood_tokens": flood_tokens, "control_hits": control_ok,
            "t3_hits": t3_hits, "engine_behaved_as_expected": control_ok and (
                t3_hits == len(reps) if offload else t3_hits == 0),
            "correct_cause": correct, "prefixwise_t3": verdicts,
            "prefixwise_correct": verdicts.get(correct, 0), "reps": reps}


def write(res: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"e_evict_{res['engine']}_{time.strftime('%Y%m%d-%H%M%S')}"
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1))
    lines = [f"# E-evict: {res['engine']}{' (mock)' if res['mock'] else ''}", "",
             f"Model {res['model']}; prefix ~{res['prefix_tokens']} tokens; GPU KV {res['gpu_kv_tokens']} tokens; "
             f"flood {res['flood_tokens']} tokens; {res['repeats']} repetitions.", "",
             f"- Control (T2, before the flood) hit every time: **{'yes' if res['control_hits'] else 'NO'}**",
             f"- T3 (after the flood) hit: **{res['t3_hits']}/{res['repeats']}** "
             f"(expected {'all' if res['correct_cause'] == 'none' else 'none'})",
             f"- Engine behaved as expected: **{'yes' if res['engine_behaved_as_expected'] else 'NO'}**",
             f"- Correct cause on T3: `{res['correct_cause']}`; Prefixwise said: "
             + ", ".join(f"`{k}` x{v}" for k, v in res["prefixwise_t3"].items())
             + f" ({res['prefixwise_correct']}/{res['repeats']} correct)", "",
             "| rep | T2 cached | flood | T3 cached | Prefixwise on T3 |", "|---:|---:|---|---:|---|"]
    for x in res["reps"]:
        lines.append(f"| {x['rep']} | {x['t2']['share']:.0%} | {x['flood_requests']} req, {x['flood_seconds']} s | "
                     f"{x['t3']['cached']}/{x['t3']['prompt']} | `{x['t3']['cause']}`: {x['t3'].get('evidence') or ''} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return stem.with_suffix(".md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=ENGINES, default="vllm")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--gpu-kv-tokens", type=int, default=32768, help="the engine's GPU KV capacity (see docstring)")
    ap.add_argument("--flood-factor", type=float, default=3.0)
    ap.add_argument("--flood-each", type=int, default=4000, help="tokens per flood request")
    ap.add_argument("--prefix-tokens", type=int, default=4000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--mock", action="store_true", help="simulated engine, no GPU")
    ap.add_argument("--kv-events", help="the engine's KV-event endpoint (e.g. tcp://127.0.0.1:5557): with it, "
                    "Prefixwise should name the GPU-only eviction 'evicted' (the engine join)")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()
    res = run(args)
    md = write(res, Path(args.output_dir))
    print(f"\n{md.read_text()}")
    return 0 if res["control_hits"] and res["engine_behaved_as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
