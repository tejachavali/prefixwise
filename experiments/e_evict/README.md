# E-evict: engine eviction, KV offload (LMCache, SGLang HiCache), and what Prefixwise says

**Question.** When a self-hosted engine runs out of GPU KV memory and evicts a session's
prefix, does the next turn miss? Does an offload tier (LMCache on vLLM, HiCache on SGLang)
bring it back? And does Prefixwise name the right cause?

**Method.** Per repetition: warm a session (T1), confirm a hit (T2), flood the engine with
unrelated prompts totalling 3x its GPU KV capacity, then return to the session (T3).

| engine | T3 should be | correct Prefixwise cause |
|---|---|---|
| `vllm`, `sglang` (GPU cache only) | a miss | `evicted` |
| `vllm-lmcache`, `sglang-hicache` | a hit, restored from CPU memory | `none`; with `--kv-events`, detail `host_restored` |

**What we already know (mock, and from the engines' source).** With an offload tier the
hit is visible to Prefixwise: vLLM's `cached_tokens` counts tokens loaded from LMCache, and
SGLang's counts host-tier hits. Without offload, Prefixwise with `--engine` says `evicted`
(inferred), and with vLLM KV events, confirmed. The live runs check both on real engines.

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

SGLang 0.5.23 pins `outlines==0.1.11`, whose `outlines_core 0.1.26` has prebuilt wheels only
for Python 3.9-3.12. On a Python 3.13 runtime pip tries to compile it from Rust source and fails,
so run SGLang in a Python 3.12 environment:

```bash
pip install -q uv && uv venv -p 3.12 /content/sg312 && source /content/sg312/bin/activate
uv pip install --python /content/sg312/bin/python --prerelease=allow --only-binary outlines_core "sglang==0.5.23" -e ".[engines]"
# 3. SGLang alone
MODEL=Qwen/Qwen3-4B MAX_TOTAL_TOKENS=32768 ./experiments/e1_open/sglang_server.sh &
python experiments/e1_open/e1_open.py --engine sglang --vllm-url http://127.0.0.1:30000 --model Qwen/Qwen3-4B
python experiments/e_evict/e_evict.py --engine sglang --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B
# 4. SGLang + HiCache (host tier = 4 x GPU)
MODEL=Qwen/Qwen3-4B MAX_TOTAL_TOKENS=32768 HICACHE=1 ./experiments/e1_open/sglang_server.sh &
python experiments/e_evict/e_evict.py --engine sglang-hicache --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B
```

**With KV events:** repeat run 1 with `--kv-events tcp://127.0.0.1:5557` added to
`e_evict.py` (`pip install -e ".[engines]"` first; `vllm_server.sh` already publishes events).
For the offload runs, start the server with `KV_EVENTS=1` (`vllm_lmcache_server.sh`,
`sglang_server.sh`) and add `--kv-events tcp://127.0.0.1:5557` (SGLang: also `--full-attention`):
each restored hit should then carry the detail `host_restored`, printed in brackets after the
cause, and the engine-check line should show `offload_tier_seen: true`.

**Prefixwise inside vLLM** (`vllm serve ... --middleware prefixwise.PrefixwiseMiddleware`, with
`PREFIXWISE_LEDGER=/path/mw.db`): `e_evict.py --engine vllm --url http://127.0.0.1:8000 --in-engine
/path/mw.db` sends each turn straight to the engine and reads its verdict from the middleware's
ledger by the `X-Prefixwise-Request-Id` header; the KV-event notes are then in the server log.
Expected: T3 is `evicted` with evidence starting "confirmed", instead of "inferred".

For SGLang (0.5.20 or later), repeat run 3 with the server publishing events:

```bash
MODEL=Qwen/Qwen3-4B MAX_TOTAL_TOKENS=32768 KV_EVENTS=1 ./experiments/e1_open/sglang_server.sh &
python experiments/e_evict/e_evict.py --engine sglang --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B \
  --kv-events tcp://127.0.0.1:5557 --full-attention
```

Expected: T3 `evicted`, evidence starting "confirmed"; T2 a hit.

Each run writes `results/*.md` and `.json`. If a server flag is rejected, check
`--help` for that release and note the change; don't change the harness to make a run pass.

## Engine notes (verified in source, 2026-11-07)

- **SGLang** needs `--enable-cache-report`, or `cached_tokens` is absent and every turn
  looks like a miss. Since 0.5.20 it writes a body `session_id` on the blocks it stores (an
  attribution label; `session_params` is the stateful-session handle), which Prefixwise sends
  with `--kv-events`. It serves any model
  name with its one model, so E1's `model_switch` turns are a label change on the same weights.
- **LMCache** now runs as its own server (`lmcache server`), connected to vLLM by
  `LMCacheMPConnector`; the old in-process mode and `LMCACHE_*` variables are gone. It stores
  whole 256-token chunks: a prefix shorter than a chunk never hits LMCache, and a change
  inside a chunk loses that whole chunk.
- **vLLM** `cached_tokens` = local GPU hits + external (LMCache) tokens, decided when the
  request is admitted; a failed LMCache load can over-report.
- Both engines publish `BlockStored` / `BlockRemoved` KV events over ZMQ
  (`--kv-events-config`), tagged with the request's session (SGLang from 0.5.20). See docs/ENGINES.md.
