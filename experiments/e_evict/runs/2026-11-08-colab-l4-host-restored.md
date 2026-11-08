# E-evict with KV events on the offload tiers: `host_restored` live (Colab L4, 2026-11-08)

Run by Muse on main at 266b2f6 (PR #67: the harness records each turn's detail). Qwen3-4B;
session prefix ~4,000 tokens; GPU KV capacity 32,768 tokens; flood 98,304 tokens; 5 repetitions per
run; L4, high-RAM runtime. vLLM 0.33.0, LMCache 0.5.7, SGLang 0.5.23, Prefixwise 0.9.0. No code
changed, nothing pushed from Colab. Transcribed by Claude from Muse's report.

**Result: both runs as expected.** After the flood evicted the session's prefix from the GPU, the
offload tier restored it and every T3 was a hit; with the engines' KV events on, Prefixwise named
each hit `none` with the detail **`host_restored`**, 5 of 5 on both tiers. The same runs on
2026-11-07, before #64, recorded plain `none` ([record](2026-11-07-colab-l4.md), B2 and D).

## vLLM 0.33.0 + LMCache 0.5.7 + KV events

`MODEL=Qwen/Qwen3-4B GPU_BLOCKS=2048 L1_GB=20 KV_EVENTS=1 vllm_lmcache_server.sh`;
`e_evict.py --engine vllm-lmcache --kv-events tcp://127.0.0.1:5557`.

| rep | T2 cached | flood | T3 cached | Prefixwise on T3 |
|---:|---:|---|---:|---|
| 0 | 99% | 25 req, 17.3 s | 3328/3404 | `none` [host_restored] |
| 1 | 99% | 25 req, 17.5 s | 3328/3433 | `none` [host_restored] |
| 2 | 99% | 25 req, 17.5 s | 3328/3398 | `none` [host_restored] |
| 3 | 99% | 25 req, 17.4 s | 3328/3436 | `none` [host_restored] |
| 4 | 99% | 25 req, 17.3 s | 3328/3401 | `none` [host_restored] |

Evidence (rep 0): "append-only turn; cache_read at least half of the prior prefix; restored from
the host or offload tier: 208 of the previous request's KV blocks had been removed from GPU memory
(KV events; the tier's own stores were seen), yet this request read 3,328 of 3,404 prompt tokens
from the cache without recomputing them". 3,328 is 13 LMCache chunks of 256 tokens: the sub-chunk
tail is recomputed, as LMCache's chunking predicts (98%). Engine check notes: none. KV-event stream
at the end: 569 batches, 31,478 blocks stored, 26,279 removed, 0 dropped messages, 0 decode errors,
0 overflows, 0 disconnects, `offload_tier_seen: true`, 2,046 GPU blocks tracked, 5 sessions.

## SGLang 0.5.23 + HiCache + KV events

`MODEL=Qwen/Qwen3-4B MAX_TOTAL_TOKENS=32768 HICACHE=1 KV_EVENTS=1 sglang_server.sh` (Python 3.12
venv); `e_evict.py --engine sglang-hicache --kv-events tcp://127.0.0.1:5557 --full-attention`.

| rep | T2 cached | flood | T3 cached | Prefixwise on T3 |
|---:|---:|---|---:|---|
| 0 | 99% | 25 req, 18.2 s | 3382/3404 | `none` [host_restored] |
| 1 | 99% | 25 req, 18.3 s | 3411/3433 | `none` [host_restored] |
| 2 | 99% | 25 req, 18.5 s | 3376/3398 | `none` [host_restored] |
| 3 | 99% | 25 req, 18.6 s | 3414/3436 | `none` [host_restored] |
| 4 | 99% | 25 req, 18.5 s | 3379/3401 | `none` [host_restored] |

Evidence (rep 0): "the cache served all of the reusable prefix (3382 of 3382 tokens); restored from
the host or offload tier: 3379 of the previous request's KV blocks had been removed from GPU memory
(KV events; the tier's own stores were seen), yet this request read 3,382 of 3,404 prompt tokens
from the cache without recomputing them" (reps 1-4: 3,407 / 3,372 / 3,411 / 3,375 blocks removed).
Blocks are one token at SGLang's page size 1, so the removed count is about the whole restored
prefix. Engine check notes: none. KV-event stream at the end: 284 batches, 453,787 blocks stored,
423,502 removed, 0 dropped messages, 0 decode errors, 0 overflows, 0 disconnects, 0 duplicate hashes,
`offload_tier_seen: true`, `full_attention: true`, 30,299 GPU blocks tracked, 5 sessions, 17,087
chain tokens held.

**What this shows**

- LMCache 0.5.7 publishes its tier's stores into vLLM's KV-event stream (`offload_tier_seen`), and
  the restore re-stores the blocks on the GPU: the frozen chain's GPU removals are seen, the read
  covers them, and the hit is attributed to the tier, not the GPU.
- HiCache's host tier does the same on SGLang (`CPU_PINNED` medium), with the chain rebuilt from
  new-block-only reports.
- Nothing else was claimed: no `evicted`, no `unknown`, no spurious detail on T2 (a plain GPU hit).
- Execution notes, no effect on results: servers were launched from the Colab terminal rather than
  notebook cells; the SGLang install was rerun with the venv's Python after a first attempt on the
  runtime's 3.13.
