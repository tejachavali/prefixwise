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
2. **`--kv-events` (vLLM only)** turns the inference into evidence. Prefixwise sends its
   session id in the request body, vLLM puts it on `BlockStored`, and per session
   Prefixwise tracks blocks resident on the GPU and blocks removed since the last turn:
   - removed > 0: `evicted`, **confirmed**, with the block count
   - none removed and blocks still resident: **not an eviction**, so `unknown`, with the
     two remaining explanations (token-level divergence, DP routing)
   - nothing attributed (session never seen in events): back to the inference
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
- **Late subscription:** blocks stored before Prefixwise subscribed are not attributed, so
  those sessions fall back to inference (never a false claim).
- **Event lag:** events are published per scheduler step; a removal could land just after
  the turn it explains. Then the verdict is inferred `evicted`, still the right answer.
- **Several replicas or DP ranks:** each publishes its own stream (vLLM offsets the port per
  DP rank). Prefixwise subscribes to one; run one Prefixwise per replica, or route each
  session to one replica.
- **Memory:** tracking is bounded by blocks resident on the GPU (entries leave on
  `BlockRemoved`), with a hard cap of 500,000 blocks.

## How it is validated

- `tests/test_engine_join.py` encodes events with vLLM's own struct definitions and sends
  them over a real ZMQ socket: confirmed eviction, blocks still resident, offload-restored
  hit, malformed payload, restart, lost messages.
- `experiments/e_evict` (live; on hold): floods an engine past its GPU KV capacity and
  checks T3, with and without LMCache/HiCache and with and without `--kv-events`.
- Open: a live test of token-level divergence (a template that re-renders history), to
  measure how often "not an eviction" occurs in real agent traffic.
