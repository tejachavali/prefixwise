# E-evict and E1-open on SGLang with KV events: Colab L4, 2026-11-07

Run by Muse on branch `claude/sglang-events` at ebd0a01 (PR #34). Qwen3-4B; session prefix
~4,000 tokens; GPU KV capacity 32,768 tokens; flood 98,304 tokens; 5 repetitions. SGLang 0.5.23
in a Python 3.12.3 environment (the runtime's own Python was 3.13.16, for which SGLang's
`outlines_core` has no wheel), started by `KV_EVENTS=1 sglang_server.sh`. Prefixwise with
`--kv-events tcp://127.0.0.1:5557 --full-attention`. No code changed. Transcribed by Claude
from Muse's report.

| rep | T2 cached | flood | T3 cached | Prefixwise on T3 |
|---:|---:|---|---:|---|
| 0 | 99% | 25 req, 18.0 s | 3/3404 | `evicted`: confirmed (3,382 of the previous request's KV blocks removed) |
| 1 | 99% | 25 req, 18.1 s | 4/3433 | `evicted`: confirmed (3,411 removed) |
| 2 | 99% | 25 req, 17.8 s | 4/3398 | `evicted`: confirmed (3,376 removed) |
| 3 | 99% | 25 req, 17.7 s | 3/3436 | `evicted`: confirmed (3,414 removed) |
| 4 | 99% | 25 req, 17.7 s | 4/3401 | `evicted`: confirmed (3,379 removed) |

Control (T2) hit 5/5; T3 hit 0/5 (expected); Prefixwise **`evicted`, confirmed, 5/5**
(before this change: `evicted`, inferred, 5/5; [record](2026-11-07-colab-l4.md)).

Engine check notes: none (version, events, one data-parallel rank, no session radix cache:
all as required). KV-event stream at the end: 280 batches, 453,787 blocks stored, 423,501
removed, 0 dropped messages, 0 decode errors, 0 overflows, 0 duplicate hashes, 30,299 GPU
blocks tracked, 5 sessions, 17,087 chain tokens held.

**E1-open, same server** (Prefixwise now sends a body `session_id` to SGLang): 120 turns,
agreement 100.0%, UNKNOWN 0.0%, Pass YES; every category 100%, including retry_replay 10/10
and model_switch 5/5. Sending session ids changed nothing.

**What this shows**

- SGLang's events attribute blocks to sessions (from a plain body `session_id`), and the chain
  Prefixwise rebuilds from new-block-only reports is right: the read stopped 3-4 tokens in
  (the shared chat-template opening survived the flood), and the block there was removed.
- Blocks are one token at SGLang's default page size, so "blocks removed" counts tokens: about
  the previous request's whole chain (prompt plus output).
- The own-store barrier held: every verdict was made after the request's own store arrived,
  with no lost messages.
