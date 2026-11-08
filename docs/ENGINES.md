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
- **The same wire format.** `[topic, seq, msgpack [ts, events, attn_dp_rank]]` with sequence
  numbers, one port per DP rank.
- **A different publishing schedule.** SGLang publishes a step's events after it has sent that
  step's output (after a prefill result, every 40 decode steps, and when idle), so a response
  can reach Prefixwise before the removal that caused its miss, and an empty socket proves
  nothing. But events are queued in order, and a missed request stores its new blocks (tagged
  with its session) after every eviction made to admit it. So "not an eviction" waits, up to
  0.5 s, for the request's own store; without it there is no claim. The store must start
  exactly where the read stopped: the previous request's finish store can itself be published
  after its response, land in this request's window, and would otherwise pass for it.
- **Event hashes leave out `extra_key`** (`compute_node_event_hash_values`), so identical tokens
  sent with different `extra_key` values sit in two subtrees under the same hashes. A hash
  stored while already present marks both runs, and chains through them are not judged.
- **Blocks stored again** (another session, a HiCache load-back) before this request was sent
  count as present, as on vLLM; stored again after it was sent (by this request itself), they
  do not.

The rest is the vLLM design: freeze the previous request's chain when the next is sent, judge
only the block where the cache read stopped, make the negative claim only with no lost messages
and one request in flight. `AllBlocksCleared` (`/flush_cache`) and an engine restart count as
evictions.

**What `/server_info` gates.** Prefixwise sends a session id, and so judges anything, only when
`/server_info` (read with `PREFIXWISE_API_KEY`, since SGLang's `--api-key` protects it) shows
version >= 0.5.20, events published, no `--enable-session-radix-cache`, and one data-parallel
rank (SGLang routes requests across ranks without session affinity, and each rank has its own
cache and stream). It also sizes the block index from the cache capacity (`token_capacity` /
`page_size`, plus a quarter), so a large cache is not mistaken for missed removals, and a
sliding-window capacity turns "not an eviction" off. Chains kept for sessions are capped by the
sum of their lengths, 16 million tokens (oldest sessions dropped first); a prefix shared by many
sessions counts once per session there but is stored once, so memory stays under 8 bytes times
that. A session's window that is dropped while its request runs comes back marked mixed, so no
negative claim is built on it. Requests with `n > 1` are not
judged (the choices' stores would mix).

A second review found that the previous request's late store could pass for this request's
(fixed above), and that a window dropped mid-request lost its concurrency mark (fixed).
Independent review of this design found two ways to a wrong claim (the `extra_key` hash
collision; data-parallel ranks), the publishing schedule above, a crash on index overflow, and
unbounded memory and quadratic time on deep chains; all fixed as described, with tests.

**Shared limit, vLLM and SGLang:** a removal is counted from the previous request until the
response. If the cache read stopped because the tokens changed, and the engine then evicted that
same block while this request ran (it is not in use, so it can be evicted), the verdict reads
"confirmed eviction" although the eviction came after the miss. It needs a token change and an
eviction of exactly that block within one request's lifetime.

## Several KV caches (vLLM): data-parallel ranks, replicas behind a router

Each data-parallel rank, and each replica behind a router, has its own KV cache. A session whose turns land on
different caches misses although nothing about the request changed: a **routing miss**. One
engine's inference ("prefix unchanged but not reused, so evicted") is wrong there, so with more
than one cache known Prefixwise no longer infers an eviction; the events decide.

```bash
prefixwise serve --upstream http://vllm:8000 --engine vllm --kv-events auto       # every DP rank
prefixwise serve --upstream http://router:8080 --engine vllm \
    --kv-events tcp://replica-a:5557,tcp://replica-b:5557,tcp://replica-c:5557     # behind a router
```

- `auto` reads vLLM's `GET /kv_event_sources` (`{rank: KVEventsConfig}`, each rank's endpoint
  with its port already offset; `vllm/entrypoints/serve/instrumentator/basic.py`, Nov 2026) and
  subscribes to every rank. Those are bind addresses: one bound on all interfaces
  (`tcp://*:5558`) is reached at the upstream's host. Ranks are read once, at startup.
- **Which cache served a request needs no router support.** Prefixwise sends its session id,
  and vLLM, with `kv_cache_report_mode: "full"`, reports every block a request uses (reused or
  new) with that session, on the stream of the cache that ran it
  (`vllm/v1/core/kv_cache_manager.py`). The cache whose stream reported the session's blocks
  after the send served it; the one that reported them for the previous request served that.
  None of the routers reports this in its response by default (research, Nov 2026: only Dynamo
  with `nvext.extra_fields` and AIBrix with `target-pod` do).
- **What a router must do:** pass the request body's `session_id` and `vllm_xargs` through to
  vLLM (vLLM accepts both on `/v1/chat/completions`; `kv_cache_report_mode: "full"` can only
  be set through `vllm_xargs`). Checked against each router's source (main, Nov 2026):

  | router | passes both through? | how |
  |---|---|---|
  | llm-d router (EPP), and GAIE's lightweight EPP | **yes** | forwards the client's bytes; any rewrite (model, priority, P/D sidecar or coordinator) re-encodes the whole JSON object (tested) |
  | vLLM production-stack | **yes** | forwards the raw body; its rewrites (alias, priority, both P/D modes) keep every field. `--session-key session_id` makes its session routing key on the field Prefixwise already sends |
  | AIBrix gateway | **yes** | forwards the raw body; edits single keys in place (priority, P/D `max_tokens`/`stream`) |
  | SGLang model gateway, current (`smg-project/smg`) | **yes** | its typed request keeps unknown fields (tested) |
  | SGLang model gateway, the older copy in the sglang repo | **no** | re-serializes a typed request without unknown fields, in every mode (tested): no claim behind it |
  | NVIDIA Dynamo | **no** | its vLLM worker takes the session from `x-dynamo-session-id` / agent headers, not the body (`components/src/dynamo/vllm/handlers.py`): no claim behind it |

  Prefixwise never sends a session header itself: llm-d, the SGLang router and AIBrix route by
  `x-session-id` when session affinity is on, and Prefixwise does not change routing.
- **Prefill/decode disaggregation:** the prefill and the decode worker both report a request's
  blocks, so with both listed no request can be attributed and nothing is claimed.
- **Verdicts at the response:**
  - the same cache served both turns: that cache's own verdict, as for one engine (evicted, or
    still resident: the token prefix changed);
  - another cache served this turn, it computed the block where its cache read stopped (it did
    not have it), and the cache that served the previous turn still holds **that same block**
    (same hash: vLLM's block hashes are comparable across replicas of one model by default):
    **`routing_miss`, confirmed** ("served by rank 1,
    while rank 0 ... still held the prefix"). The prefix was cached, on the other replica; route
    each session to one cache (the routers' session affinity: `x-session-id` for llm-d, the SGLang
    router and AIBrix, `x-dynamo-session-id` for Dynamo, `--session-key` for production-stack);
  - anything else: no engine claim (`unknown`, saying that an eviction and a request sent to
    another cache look the same). That covers a request no cache reported (shorter than a block),
    two caches reporting it, a block that differs where the read stopped (a prefix rendered
    differently, a side call of the same session served elsewhere), a previous cache that lost
    the block too, a stream behind or with lost messages, a hybrid model, and concurrent requests
    of one session. Each request is judged by its own reports, even when the next one was sent
    before it was judged (streaming).
- **A replica that goes away** (crashed, replaced, scaled down): the subscriber sees its
  connection drop, forgets what that replica held, and claims nothing across the gap, so a
  failover is never reported as a routing miss.
- One subscriber thread per cache; `/metrics` sums their counters, and `GET /engine` lists each
  cache. A hit adds no wait; otherwise all streams share one catch-up deadline.

## Failure modes, and what the code does about them

- **Lost event messages** (ZMQ PUB drops under backpressure): a missed `BlockRemoved` would
  make a session look resident. Gaps in the sequence number are counted; after any gap or
  undecodable message, Prefixwise stops claiming "not an eviction" and falls back to inference.
- **Engine restart:** the sequence number restarts; Prefixwise treats that as a cleared cache.
- **Late subscription / tight loops:** if a session's previous request reported no blocks
  (Prefixwise subscribed later, or the next request followed within milliseconds), there is
  nothing to judge and the inference stands. Never a false claim.
- **Several replicas or DP ranks (vLLM):** each has its own cache and stream. See
  [Several KV caches](#several-kv-caches-vllm-data-parallel-ranks-replicas-behind-a-router) below.
  SGLang: one data-parallel rank for now.
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
  checks. **Live:** E-evict on SGLang 0.5.23 with KV events, Qwen3-4B on an L4: `evicted`, confirmed, 5 of
  5, with no lost messages; E1-open still 120/120 with session ids sent
  (`experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md`).
- `tests/test_engine_fleet.py`: several caches, each verdict and each no-claim case (a prefix
  rendered differently, a side call served elsewhere, the next request sent before this one was
  judged, a stream behind, two caches reporting one request, a replica that went away), over
  real ZMQ sockets too, plus `--kv-events auto` against a stub `/kv_event_sources`.
  **Live:** `experiments/e_route` on two vLLM 0.33.0 replicas (Qwen3-1.7B, one L4) behind a
  router that never says which replica it used: routing misses confirmed 5/5 when it
  alternates, 5/5 hits when it is sticky, no false claim
  (`experiments/e_route/runs/2026-11-08-colab-l4.md`). Not yet run live: data-parallel ranks
  inside one server (`auto`), real routers.
- Open: a live test of token-level divergence (a template that re-renders history), to
  measure how often "not an eviction" occurs in real agent traffic.
