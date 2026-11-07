# E-evict: engine eviction, KV offload (LMCache, SGLang HiCache), and what Prefixwise says

**Question.** When a self-hosted engine runs out of GPU KV memory and evicts a session's
prefix, does the next turn miss? Does an offload tier (LMCache on vLLM, HiCache on SGLang)
bring it back? And does Prefixwise name the right cause?

**Method.** Per repetition: warm a session (T1), confirm a hit (T2), flood the engine with
unrelated prompts totalling 3x its GPU KV capacity, then return to the session (T3).

| engine | T3 should be | correct Prefixwise cause |
|---|---|---|
| `vllm`, `sglang` (GPU cache only) | a miss | `evicted` |
| `vllm-lmcache`, `sglang-hicache` | a hit, restored from CPU memory | `none` |

**What we already know (mock, and from the engines' source).** With an offload tier the
hit is visible to Prefixwise: vLLM's `cached_tokens` counts tokens loaded from LMCache, and
SGLang's counts host-tier hits. Without offload, Prefixwise today says `routing_miss`: it
has no engine events, so it cannot tell an eviction from a request routed to another
replica. That is the gap the engine join (KV events) closes; this experiment measures it on
real engines before we build it.

## Run on Colab (L4, Qwen3-4B)

vLLM and SGLang pin different torch versions: use a separate runtime (or venv) for each.
The GPU KV pool is made small and exact so the flood is short and the eviction certain.

**Runtime 1: vLLM, then vLLM + LMCache**

```bash
pip install -e . "vllm>=0.30" lmcache
# 1. vLLM alone, 32,768-token GPU KV
MODEL=Qwen/Qwen3-4B VLLM_ARGS="--block-size 16 --num-gpu-blocks-override 2048 --max-model-len 16384" \
  ./experiments/e1_open/vllm_server.sh &          # wait for "Application startup complete"
python experiments/e_evict/e_evict.py --engine vllm --url http://127.0.0.1:8000 --model Qwen/Qwen3-4B
# stop the server, then:
# 2. vLLM + LMCache (20 GB CPU tier, same GPU KV)
MODEL=Qwen/Qwen3-4B GPU_BLOCKS=2048 L1_GB=20 ./experiments/e1_open/vllm_lmcache_server.sh &
python experiments/e_evict/e_evict.py --engine vllm-lmcache --url http://127.0.0.1:8000 --model Qwen/Qwen3-4B
python experiments/e1_open/e1_open.py --engine vllm-lmcache --vllm-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B --model-alias Qwen/Qwen3-4B-alias
```

**Runtime 2: SGLang, then SGLang + HiCache**

```bash
pip install -e . "sglang[all]"
# 3. SGLang alone
MODEL=Qwen/Qwen3-4B MAX_TOTAL_TOKENS=32768 ./experiments/e1_open/sglang_server.sh &
python experiments/e1_open/e1_open.py --engine sglang --vllm-url http://127.0.0.1:30000 --model Qwen/Qwen3-4B
python experiments/e_evict/e_evict.py --engine sglang --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B
# 4. SGLang + HiCache (host tier = 4 x GPU)
MODEL=Qwen/Qwen3-4B MAX_TOTAL_TOKENS=32768 HICACHE=1 ./experiments/e1_open/sglang_server.sh &
python experiments/e_evict/e_evict.py --engine sglang-hicache --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B
```

**With the engine join** (the point of the experiment): repeat runs 1 and 3 with the engine's
KV events on and `--kv-events tcp://127.0.0.1:5557` added to `e_evict.py` (`pip install -e ".[engines]"`
first). vLLM: `vllm_server.sh` already publishes events. SGLang: `KV_EVENTS=1 ./experiments/e1_open/sglang_server.sh`.
Expected: Prefixwise names the T3 miss `evicted` (vLLM: attributed to the session; SGLang:
engine-wide), where without events it says `routing_miss`.

Each run writes `results/*.md` and `.json`. If a server flag is rejected, check
`--help` for that release and note the change; don't change the harness to make a run pass.

## Engine notes (verified in source, 2026-11-07)

- **SGLang** needs `--enable-cache-report`, or `cached_tokens` is absent and every turn
  looks like a miss. It reads a body `session_id` as its own stateful-session handle and
  rejects unknown ones, so Prefixwise does not forward it to SGLang. It serves any model
  name with its one model, so E1's `model_switch` turns are a label change on the same weights.
- **LMCache** now runs as its own server (`lmcache server`), connected to vLLM by
  `LMCacheMPConnector`; the old in-process mode and `LMCACHE_*` variables are gone. It stores
  whole 256-token chunks: a prefix shorter than a chunk never hits LMCache, and a change
  inside a chunk loses that whole chunk.
- **vLLM** `cached_tokens` = local GPU hits + external (LMCache) tokens, decided when the
  request is admitted; a failed LMCache load can over-report.
- Both engines publish `BlockStored` / `BlockRemoved` KV events over ZMQ
  (`--kv-events-config`), the input for the engine join.
