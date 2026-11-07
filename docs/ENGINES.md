# Self-hosted engines: how Prefixwise names a miss behind vLLM, SGLang and LMCache

Design note, 2026-11-07. Facts about the engines were read from their source:
vllm-project/vllm at 8176238, sgl-project/sglang at a40b25a, LMCache/LMCache at bd9f0f4.

## The question

On a hosted API, a request whose prefix is unchanged but was not reused is usually a
**routing miss**: it went to a machine that did not hold the cache. Behind one
self-hosted engine there is one cache, so that explanation does not apply. What does?

## Candidate causes (one engine, prefix unchanged in the request, low `cached_tokens`)

| cause | mechanism | how common |
|---|---|---|
| **Eviction** | GPU KV memory full; LRU blocks freed | The usual one in long agent sessions; it is what offload tiers (LMCache, HiCache) exist to absorb |
| **Token-level divergence** | The request's messages are unchanged, but the chat template renders history differently (e.g. dropping earlier reasoning). The engine caches tokens, not JSON | Model-dependent. Usually changes only the tail, so most of the prefix still hits and Prefixwise already calls it a hit |
| **Data-parallel routing** | `--data-parallel-size > 1`: each rank has its own cache | Only in DP deployments |
| **Cache reset / restart** | `/reset_prefix_cache`, engine restart | Rare; an eviction in effect |

## What evidence can separate them

| source | what it gives | cost |
|---|---|---|
| `cached_tokens` in the response | Hit or miss, not why | Free; needs `--enable-prompt-tokens-details` (vLLM) or `--enable-cache-report` (SGLang) |
| Engine metrics (`/metrics`) | SGLang: `sglang:evicted_tokens_total`, engine-wide. vLLM: only a sampled histogram (`vllm:kv_block_idle_before_evict_seconds`, behind `--kv-cache-metrics`); no plain eviction counter | Polling; engine-wide only |
| KV events (`--kv-events-config`, ZMQ + msgpack) | `BlockStored` / `BlockRemoved` / `AllBlocksCleared`. **vLLM stamps the request's `session_id` on `BlockStored`**, so blocks can be attributed to sessions. SGLang has the same field, but it is SGLang's stateful-session handle: it rejects ids it did not open, so a proxy cannot set it | A subscriber thread and two dependencies |
| SGLang `return_cached_tokens_details` | Per request: tokens served from device / host / storage (LMCache) | A request field; tiers, not causes |

Only attributed KV events can say "this session's blocks were removed" **and** the
converse, "this session's blocks are all still there, so it was not an eviction". Engine-wide
signals cannot do the second, and on a busy engine evictions happen all the time, so they
add little over inference.

## Decision

1. **`prefixwise serve --engine vllm|sglang`** declares one self-hosted engine. An
   unchanged-prefix miss is named `evicted`, and the evidence says it is **inferred** and
   names the look-alikes. No dependencies. (Without the flag Prefixwise assumes a hosted
   API and says `routing_miss`, which is wrong behind one engine.)
2. **`--kv-events` (vLLM only)** turns the inference into evidence:
   - Prefixwise sends its session id in the request body and
     `vllm_xargs: {"kv_cache_report_mode": "full"}` (merged with the client's). By default
     vLLM reports only *newly stored* blocks; in `full` mode it also reports the blocks a
     request *reused* (`kv_cache_manager.py`), so the whole prefix chain of each request is
     known, tagged with the session.
   - When a session's next request is sent, the blocks its **previous request** used are
     frozen as the chain to judge, in order. Removals of those blocks are recorded from then
     until the response, so evictions while the request waits in vLLM's queue count.
   - At the response, `cached_tokens` says where the cache read stopped: block
     `cached_tokens // block_size` of the chain. **Only that block decides:**
     - removed: `evicted`, **confirmed** (the eviction is what cut the hit short)
     - still on the GPU: the engine held the block yet did not use it, so the request's
       tokens changed at that point. **Not an eviction**: `unknown`, with the two remaining
       explanations (token-level divergence, DP routing)
     - beyond the chain, or nothing known: the inference stands
   - Judging only the stop point is what makes it robust: output blocks a template never
     reuses, and older chains, get evicted routinely and cannot confirm anything.
   - Before deciding, Prefixwise waits until the subscriber has drained the socket (at most
     50 ms), so removals published before the response have been read.
   - Only full-attention block groups are tracked, keyed by (group, hash). On hybrid models,
     sliding-window blocks are freed routinely without breaking the prefix.
3. **LMCache and HiCache need nothing extra.** vLLM's `cached_tokens` includes tokens loaded
   from LMCache (`PrefillStats`: local + external), and SGLang's includes host-tier hits. A
   prefix restored from CPU memory is a hit and is never examined for eviction. Only GPU
   removals count.

Not built, on purpose: SGLang event support (cannot be attributed), engine-wide eviction
counting (weak evidence), and metrics polling (engine-wide).

## Failure modes, and what the code does about them

- **Lost event messages** (ZMQ PUB drops under backpressure): a missed `BlockRemoved` would
  make a session look resident. Gaps in the sequence number are counted; after any gap or
  undecodable message, Prefixwise stops claiming "not an eviction" and falls back to inference.
- **Engine restart:** the sequence number restarts; Prefixwise treats that as a cleared cache.
- **Late subscription / tight loops:** if a session's previous request reported no blocks
  (Prefixwise subscribed later, or the next request followed within milliseconds), there is
  nothing to judge and the inference stands. Never a false claim.
- **Several replicas or DP ranks:** each publishes its own stream (vLLM offsets the port per
  DP rank). Prefixwise subscribes to one; run one Prefixwise per replica, or route each
  session to one replica.
- **Memory:** the set of live blocks is bounded by the engine's GPU capacity (entries leave on
  `BlockRemoved`; a hard cap of 1,000,000 stops trusting residency if removals were missed).
  Per session only the head of the previous request's chain is kept (4,096 blocks: the head
  is what decides a low cache read), for at most 1,000 sessions.

- **Concurrent requests in one session** share a window, so a chain can mix two branches.
  Agents rarely send parallel requests in one session; the effect is a wrong position at worst.

Two rounds of independent review shaped this. The first version judged any removed block a
session ever held (stale chains, output blocks), ignored hybrid models, raced the event
stream, and mismatched session ids on retries; the second judged at send time and missed
evictions while a request waits in vLLM's queue. The stop-point rule above answers all of them.

## How it is validated

- `tests/test_engine_join.py` encodes events with vLLM's own struct definitions and sends
  them over a real ZMQ socket: confirmed eviction, blocks still resident, offload-restored
  hit, queue-time eviction, output blocks and older chains, hybrid groups, duplicates,
  malformed payload, restart, lost messages.
- `experiments/e_evict` (live; on hold): floods an engine past its GPU KV capacity and
  checks T3, with and without LMCache/HiCache and with and without `--kv-events`.
- Open: a live test of token-level divergence (a template that re-renders history), to
  measure how often "not an eviction" occurs in real agent traffic.
