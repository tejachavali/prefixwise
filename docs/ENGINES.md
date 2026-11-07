# Self-hosted engines: how Prefixwise names a miss behind vLLM, SGLang and LMCache

Design note, 2026-11-07. Facts about the engines were read from their source:
vllm-project/vllm at 8176238, sgl-project/sglang at a40b25a, LMCache/LMCache at bd9f0f4.
SGLang's KV events re-read the same day at the v0.5.23 release (297e54f) and main (278fd83),
which corrected an earlier conclusion (section "SGLang", below).

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
| KV events (`--kv-events-config`, ZMQ + msgpack) | `BlockStored` / `BlockRemoved` / `AllBlocksCleared`. **vLLM stamps the request's `session_id` on `BlockStored`**, so blocks can be attributed to sessions. SGLang (>= 0.5.20) stamps a body `session_id` too, on the blocks a request newly stores (section "SGLang") | A subscriber thread and two dependencies |
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
2. **`--kv-events` (vLLM)** turns the inference into evidence:
   - Prefixwise sends its session id in the request body and
     `vllm_xargs: {"kv_cache_report_mode": "full"}` (merged with the client's). By default
     vLLM reports only *newly stored* blocks; in `full` mode it also reports the blocks a
     request *reused* (`kv_cache_manager.py`), so the whole prefix chain of each request is
     known, tagged with the session.
   - Each block's position comes from its parent (`parent_block_hash`; vLLM reports reused
     blocks from block 0 with no parent, then new blocks chained onto them), so late or
     re-reported events cannot shift positions.
   - When a session's next request is sent, the blocks its **previous request** used are
     frozen as the chain to judge. Removals of those blocks are recorded from then
     until the response, so evictions while the request waits in vLLM's queue count.
   - At the response, `cached_tokens` says where the cache read stopped: block
     `cached_tokens // block_size` of the chain. **Only that block decides:**
     - removed: `evicted`, **confirmed** (the eviction is what cut the hit short)
     - still on the GPU: the engine held the block yet did not use it, so the request's
       tokens changed at that point. **Not an eviction**: `unknown`, with the two remaining
       explanations (token-level divergence, DP routing)
     - beyond the chain, or nothing known: the inference stands
   - **A positive claim needs one removal actually seen. The negative claim needs all of:**
     the stream caught up with the response (a batch stamped after it was read, or the
     socket was empty, within 50 ms), no message lost meanwhile, a full-attention-only
     model (on hybrid models the hit stops at the shortest group, so a sliding-window
     eviction can leave the full-attention block in place), one request in flight for the
     session, and vLLM reporting cached tokens. Otherwise: no claim, inference stands.
   - Hits are never examined, so they never wait for the stream.
   - Judging only the stop point is what makes it robust: output blocks a template never
     reuses, and older chains, get evicted routinely and cannot confirm anything.
   - Only full-attention block groups are tracked, keyed by (group, hash). On hybrid models,
     sliding-window blocks are freed routinely without breaking the prefix.
3. **LMCache and HiCache need nothing extra.** vLLM's `cached_tokens` includes tokens loaded
   from LMCache (`PrefillStats`: local + external), and SGLang's includes host-tier hits. A
   prefix restored from CPU memory is a hit and is never examined for eviction. Only GPU
   removals count.

4. **`--kv-events` on SGLang** gives the same two verdicts, built differently because SGLang's
   events differ (section "SGLang").

Not built, on purpose: engine-wide eviction counting (weak evidence) and metrics polling
(engine-wide).

## SGLang

**Correction.** The first version of this note said SGLang's `session_id` is its stateful-session
handle and rejects ids it did not open. That is true of `session_params`. Since v0.5.20 a request
also takes a top-level `session_id`, documented as a "stable identity shared by requests in the
same session. Unlike session_params, this does not alter or reconstruct the prompt"
(`managers/io_struct.py`). The scheduler treats such a request as a normal one
(`scheduler.py`, `_handle_generate_request`), and the default cache (`UnifiedRadixCache`,
including HiCache and LMCache) passes it to `record_store`, which writes it on `BlockStored`.
Checked per release: 0.5.18 and 0.5.19 do not put it on events; 0.5.20 to 0.5.23 and main do.
Its only other effect is with `--enable-session-radix-cache` (off by default), where it pins a
session's cache; Prefixwise reads that flag from `/server_info` and then sends no session id.

**How SGLang's events differ from vLLM's, and what each changes:**

- **Only new blocks are reported.** `record_store` runs when a radix node is added, never for
  blocks a request reuses, and there is no `kv_cache_report_mode`. So Prefixwise keeps its own
  copy of the tree of GPU blocks (each stored run linked to its parent block) and finds a
  session's chain by walking up from the last block its previous request stored.
- **One block is one token by default** (`--page-size 1`). The copy is one dictionary entry per
  cached token pointing to its run, with each run's hashes in a packed array: about 80 bytes per
  token (measured), 80 MB per million. Past 2 million tokens Prefixwise forgets the tree and
  judges nothing, as on vLLM's cap.
- **Eviction removes leaves first.** On the GPU, SGLang evicts only device leaves
  (`evict_device_leaf`, `_delete_unbacked_device_leaf`, `_demote` emit `BlockRemoved`). A removal
  on a chain therefore also removed everything after it, so each run keeps one number, the
  first block removed, and the stop-point rule reads it. Node splits emit nothing and keep the
  hashes, so a split only means a later removal cuts a run in the middle.
- **No attention kind on events.** On sliding-window and Mamba models SGLang frees those layers
  from internal nodes without an event, so the cache read can stop at a block whose
  full-attention KV is still there. A removed block at the stop point still proves an eviction;
  "not an eviction" is claimed only with `--full-attention`. `/server_info` shows sliding-window
  capacity (`token_capacity_swa`), which turns the flag off; it has nothing for Mamba, hence the
  flag.
- **The same wire format and stream.** `[topic, seq, msgpack [ts, events, attn_dp_rank]]`,
  sequence numbers, a publisher that sends only when something changed (so an empty socket means
  the stream has caught up), one port per DP rank.

The rest is the vLLM design: freeze the previous request's chain when the next is sent, judge
only the block where the cache read stopped, make the negative claim only when the stream has
caught up with no lost messages and one request in flight. `AllBlocksCleared` (`/flush_cache`)
and an engine restart count as evictions.

**Shared limit, vLLM and SGLang:** a removal is counted from the previous request until the
response. If the cache read stopped because the tokens changed, and the engine then evicted that
same block while this request ran (it is not in use, so it can be evicted), the verdict reads
"confirmed eviction" although the eviction came after the miss. It needs a token change and an
eviction of exactly that block within one request's lifetime.

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
  `BlockRemoved`). Past a hard cap of 1,000,000 (removals must have been missed) Prefixwise
  forgets the saved chains rather than read every block as removed.
- **Clocks:** catching up compares the engine's batch timestamps with the proxy's clock, after
  correcting for the offset between them (learned from the batches), so an engine on another
  host with a skewed clock cannot make a stale stream look current.
- **Failed requests** (upstream error, cut-off stream) release their check, so they do not make
  the session look like it has concurrent requests.
  Per session only the head of the previous request's chain is kept (4,096 blocks: the head
  is what decides a low cache read), for at most 1,000 sessions.

- **Concurrent requests in one session** share a window, so a chain can mix two branches:
  detected, and those requests get no claim.

Four rounds of independent review shaped this. The first version judged any removed block a
session ever held (stale chains, output blocks), ignored hybrid models, raced the event
stream, and mismatched session ids on retries; the second judged at send time and missed
evictions while a request waits in vLLM's queue; the third found hybrid models, a busy
engine and concurrent requests could still produce a false "not an eviction", and that
arrival order could shift positions; the fourth found only rare cases (cap overflow, clock
skew, failed requests), fixed above. The rules above answer all of them.

## How it is validated

- `tests/test_engine_join.py` encodes events with vLLM's own struct definitions and sends
  them over a real ZMQ socket: confirmed eviction, blocks still resident, offload-restored
  hit, queue-time eviction, output blocks and older chains, hybrid groups, duplicates,
  malformed payload, restart, lost messages.
- `experiments/e_evict`, **run live 2026-11-07** (Colab L4, Qwen3-4B; vLLM 0.33.0, LMCache 0.5.7,
  SGLang 0.5.23): all eight runs as expected. vLLM eviction named 5/5, confirmed from KV
  events ("211 of the previous request's KV blocks removed"); LMCache and HiCache restores
  (98-99% of the prefix) counted as hits 5/5; E1-open 120/120 on vLLM + LMCache and on SGLang.
  Record: `experiments/e_evict/runs/2026-11-07-colab-l4.md`.
- `tests/test_sglang_events.py` does the same with SGLang's structs (v0.5.23): confirmed and
  ruled-out evictions, a prefix stored by another session, split nodes, paged blocks, host-tier
  events, cache flush, restart, concurrency, lost messages, overflow, and the `/server_info`
  checks. **Not yet run against a live SGLang**: E-evict with `--kv-events` is the test.
- Open: a live test of token-level divergence (a template that re-renders history), to
  measure how often "not an eviction" occurs in real agent traffic.
