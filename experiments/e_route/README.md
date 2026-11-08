# E-route: two vLLM replicas behind a router, and what Prefixwise says

**Question.** When a router sends a session's next turn to a different replica, the turn misses
although nothing in the request changed. Can Prefixwise confirm that from the replicas' KV events
alone, with no help from the router, and stay quiet when the session stays on one replica?

**Method.** Two vLLM replicas, each publishing KV events; a minimal router in front (in the
script); Prefixwise in front of the router, subscribed to both streams
(`--kv-events tcp://...:5557,tcp://...:5558`). Per repetition one session, three turns.

| router mode | turn 2 lands on | correct Prefixwise cause for turn 2 |
|---|---|---|
| `alternate` | the replica that never saw the session | `routing_miss`, **confirmed** from KV events |
| `sticky` (control) | the same replica | `none` (a hit) |

Turn 3 is recorded, not scored (in `alternate` it returns to the first replica, which still
holds the session's first turn).

**Live result (2026-11-08, Colab L4, vLLM 0.33.0):** PASS, routing misses confirmed 5/5, control hits 5/5 ([record](runs/2026-11-08-colab-l4.md)).

**Also checked (no GPU):** `python experiments/e_route/e_route.py --mock` runs two simulated
replicas that follow vLLM's report format (every block a request uses, with its session) and
passes; it is also a test (`tests/test_engines.py`).

## Run on Colab (one L4, two replicas of Qwen3-1.7B)

Two vLLM servers share the GPU, each with its own HTTP port, KV-event port and memory share.
Start the second only after the first is ready.

```bash
pip install -e ".[engines]" "vllm>=0.30"
MODEL=Qwen/Qwen3-1.7B PORT=8001 EVENTS_PORT=5557 \
  VLLM_ARGS="--gpu-memory-utilization 0.42 --max-model-len 8192" \
  ./experiments/e1_open/vllm_server.sh > /content/replica-a.log 2>&1 &
# wait until: curl -s http://127.0.0.1:8001/v1/models   answers
MODEL=Qwen/Qwen3-1.7B PORT=8002 EVENTS_PORT=5558 \
  VLLM_ARGS="--gpu-memory-utilization 0.42 --max-model-len 8192" \
  ./experiments/e1_open/vllm_server.sh > /content/replica-b.log 2>&1 &
# wait until: curl -s http://127.0.0.1:8002/v1/models   answers
python experiments/e_route/e_route.py \
  --replicas http://127.0.0.1:8001,http://127.0.0.1:8002 \
  --kv-events tcp://127.0.0.1:5557,tcp://127.0.0.1:5558 \
  --model Qwen/Qwen3-1.7B --reps 5 --out /content/e_route.json
```

Expected: `alternate T2: routing_miss confirmed 5/5`, `sticky T2 (control): hit 5/5`, `PASS`.
Paste the full output (and `vllm --version`) into the run record. Nothing here needs an API key.
