# Prefixwise for SGLang: RadixAttention, HiCache and data-parallel ranks

Prefixwise sits in front of SGLang as a local proxy. For every request it reads the cache hit
(`cached_tokens`) and, when the prefix was not reused, names **why**, says whether it was waste,
and prices it in tokens. With SGLang's KV-cache events it confirms evictions, tells a host-tier
restore from a GPU hit, and, across data-parallel ranks, confirms routing misses. It stores
counts and hashes, never prompt text.

Companion guides: [VLLM.md](VLLM.md) (vLLM, the same join), [LMCACHE.md](LMCACHE.md) (vLLM
with an offload tier). Design and failure modes: [ENGINES.md](ENGINES.md).

## Setup

```bash
pip install "prefixwise[engines]"          # Python 3.10+; the extra adds pyzmq + msgspec

python -m sglang.launch_server --model-path <model> --enable-cache-report \
  --kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:5557"}'       # SGLang >= 0.5.20

prefixwise serve --upstream http://127.0.0.1:30000 --full-attention   # detects SGLang and its KV events
```

Point your client at `http://127.0.0.1:8080/v1` instead of SGLang, run your workload, then
`prefixwise report`.

`prefixwise serve --upstream URL` (or `PREFIXWISE_UPSTREAM_URL`) reads SGLang's `/server_info` at
startup: it sets `--engine sglang`, sets `--kv-events auto` when the server publishes KV events
over tcp (every data-parallel rank; the engines extra must be installed), warns when
`--enable-cache-report` is off, and reminds you to add `--full-attention` if the model has no
sliding-window or Mamba layers (that one is never guessed). Any flag you pass wins; `--engine
sglang --kv-events tcp://127.0.0.1:5557` is the explicit form. The SGLang model gateway relays its
workers' `/server_info`, so a router looks like one engine: behind one, pass `--engine none`
(evictions cannot be told from routing there, and SGLang replicas behind a router are not joined).

**Session ids: the `x-session-id` header and the body `session_id`.** Prefixwise forwards a
client's `x-session-id` header unchanged and sends none of its own (since 0.11.3; earlier versions
replaced it with their own session id, which put every session on one routing key). It also uses the
client's value to group turns into sessions when no `X-Prefixwise-Session-Id` is given. With
`--enable-session-radix-cache`, a client's own body `session_id` reaches SGLang unchanged (before
this release it was removed, since Prefixwise's KV-event join is off with that flag); otherwise it stays a
Prefixwise field, as before.

The SGLang worker reads the session only from the body `session_id`, and per
sgl-project/sglang#47763 sgl-router does not pass the header on (#45620 would make the worker read it;
it is not merged). Measured on SGLang 0.5.23 with a 16,384-token KV pool flooded between two turns
([record](../experiments/e_route/runs/2026-11-10-colab-l4-sglang-session-header.md)): in the default
configuration the session id changes nothing (3 of 4,012 tokens cached with or without it). With
`--enable-session-radix-cache`, it decides what survives: 4,000 of 4,012 cached with a body
`session_id`, 3 of 4,012 without. So with that flag, send each conversation's id as a body
`session_id`. When the flag is on (read from `/server_info`) and a request names its session only in
the header, an eviction of that conversation's prefix gets the detail `session_header_dropped`, and
Prefixwise warns once. sglang-router 0.3.2 had no policy that routes by a session header; its
`cache_aware` policy kept each conversation on one worker by matching the prompt.

- `--enable-cache-report` is required: it puts `usage.prompt_tokens_details.cached_tokens` in
  every response (and the final stream chunk). Without it every request looks like a miss, and
  Prefixwise warns once.
- `--kv-events-config` and `--kv-events` are optional. Without them, evictions are still named,
  as *inferred*. The endpoint must be bound on all interfaces (`tcp://*:PORT`): with a named host
  SGLang connects out instead of listening, and Prefixwise says so.
- `--engine sglang` tells Prefixwise there is one engine behind it (no routing) and sends its
  session id in the request body (`session_id`, which SGLang 0.5.20+ writes on the blocks a
  request stores). It does not alter the prompt or the cache.
- **`--full-attention`**: pass it when the model has no sliding-window or Mamba layers (Qwen3,
  Llama, most dense models). SGLang frees those layers without an event, so without the flag a
  removed block still *confirms* an eviction, but Prefixwise never says "not an eviction" and
  never claims a routing miss. On a hybrid model, leave it out; SGLang's `/server_info` shows a
  sliding-window capacity, which turns the negative claim off anyway. For a Mamba model it shows
  `uses_mamba_radix_cache`, and Prefixwise then ignores `--full-attention`: SGLang frees Mamba
  state without an event.
- **The radix cache off** (`--disable-radix-cache`: HiSparse requires it, and deterministic
  inference sets it on most attention backends): `/server_info` says so, and Prefixwise records
  every miss as `unknown`, "the engine keeps no prefix cache", with nothing avoidable. Before
  0.11.7 an unchanged prefix that missed was an inferred eviction on every turn.
- At startup Prefixwise reads `/server_info` (with `PREFIXWISE_API_KEY` if SGLang has
  `--api-key`) and judges from events only when it is safe: version 0.5.20 or later, events
  published, no `--enable-session-radix-cache`, and the rank count known. `GET /engine` on the
  proxy shows the stream's health.

Launch script used in our runs: `experiments/e1_open/sglang_server.sh` (`KV_EVENTS=1`;
`HICACHE=1` adds `--enable-hierarchical-cache --hicache-ratio 4`; `MAX_TOTAL_TOKENS` caps the
GPU KV pool).

## Data-parallel ranks

Each rank (`--dp-size N`, or `--attn-dp-size N`) has its own radix cache and its own KV-event
publisher at `endpoint_port_base + rank`, every batch stamped with the rank. A session whose
turns land on different ranks misses although nothing about the request changed: a **routing
miss**, which looks exactly like an eviction from the usage alone.

```bash
python -m sglang.launch_server --model-path <model> --dp-size 2 --enable-cache-report \
  --kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:5557"}'       # ranks on 5557, 5558

prefixwise serve --upstream http://127.0.0.1:30000 --engine sglang --kv-events auto --full-attention
```

`--kv-events auto` reads `/server_info` (`kv_events`: `endpoint_host`, `endpoint_port_base`,
`dp_size`) and follows every rank. With several ranks known, an eviction is never just inferred:
the events decide between `evicted` (the rank that served the previous turn removed the block
where this read stopped), `routing_miss` (another rank served this turn and computed a block the
previous rank still holds; SGLang's block hashes are the same on every rank, so the match is
exact), and `unknown` when neither can be shown. Routing misses need `--full-attention`. A batch
carrying another rank's stamp is counted (`wrong_rank` in `/metrics`) and stops claims, so
crossed endpoints cannot mislead it. One endpoint given for a server with several ranks:
Prefixwise says to use `auto`.

Refused, with the reason: prefill/decode disaggregation, ranks across several nodes
(`/server_info` advertises one event host), and `--enable-session-radix-cache`. The
several-rank path is tested over real ZMQ sockets with SGLang's wire format and a stub
`/server_info`; it has **not yet run live** (a single L4 runs one rank).

## What Prefixwise tells apart

Behind SGLang, a turn whose prompt is unchanged can be served four ways, and a prompt the agent
changed never reaches RadixAttention's match at all. The verdicts, and the evidence each needs:

| what happened | verdict | detail | evidence |
|---|---|---|---|
| **GPU hit.** RadixAttention matched the prefix on the device | `none` | | `cached_tokens` covers the previous prompt |
| **HiCache restore.** The GPU copy was evicted; `--enable-hierarchical-cache` brought the prefix back from host memory, no prefill recomputed (with an L3 storage backend, see below) | `none` | `host_restored` | `BlockRemoved` on the GPU medium for the previous request's blocks, the host tier's own stores seen on the stream (`CPU_PINNED` medium), and this request read those blocks back. Live, host tier only: 5/5, below |
| **Eviction.** The radix tree dropped the prefix (leaves first) and this turn recomputed it | `evicted`, confirmed | | the block where the cache read stopped was removed (`"3,382 of the previous request's KV blocks removed"`) |
| Eviction without events | `evicted`, inferred | | one engine, prefix unchanged, not reused; the evidence says *inferred* |
| **Not an eviction.** The block is still on the GPU, so the token prefix changed although the request did not (a chat template re-rendering history) | `unknown` | | KV events, `--full-attention`, the request's own store arrived (up to 0.5 s) and nothing was lost |
| **Previous output re-prefilled.** The read stopped at the previous prompt: none of the previous turn's output (160+ tokens) was reused, although the earlier messages are unchanged. Qwen3's template drops reasoning from history, so each turn prefills the previous answer again in its rendered form | `none` | `output_rerendered` | usage alone, behind the engine; live the output was recomputed on 24 of 25 turns, 36% of prompt tokens, and the detail (added after the run) would have named 15 of them, the 160-token floor leaving short answers out (SGLang 0.5.23, page size 1, [record](../experiments/e_template/runs/2026-11-09-colab-l4-rerun.md)). With page size 1 the counts are exact: a full read stops at the previous prompt, or 4 tokens before it when thinking is off (the `<think>` stub the history never repeats) |
| **Agent-layer prompt change**, before SGLang can match anything | `volatile_system` (a timestamp or id in the system prompt), `tool_reserialize` (same tools, different order or bytes), `history_edit` (an earlier message rewritten), `tool_result_mutation` (an earlier tool result changed) | | the request's own block hashes; the first block that differs is named (`divergence_role`) |

The agent-layer causes are the ones that get blamed on the cache. They are decided from the
request alone, so they hold with or without events, and each comes with its fix
(`prefixwise report --detail`, `prefixwise fixes`). For `tool_reserialize`, `prefixwise serve
--fix canonical_tools` sorts tools by name with deterministic serialization before forwarding,
so SGLang sees the same bytes every turn; it is **proven**: in the E4 replay the realised saving
was 97.5% of the prediction on vLLM (Qwen3-4B), task success flat by the experiment's bar
(`experiments/e4/runs/2026-11-06-live.md`). For `volatile_system`, the fix is to move the date
or id to the end of the prompt; `prefixwise check` (the CI gate) catches both before they
ship.

`host_restored` says the hit cost a restore from host memory: fine when the host tier is sized
for it, and a sign the GPU cache alone is too small for the session's working set. It is claimed
only once the stream has shown the host tier's own stores (so a tier exists), from a clean,
uncontended stream, and behind one rank; without that, a hit after a GPU removal is a plain
`none` (another request sharing the prefix may have recomputed it).

With an L3 storage backend (`--hicache-storage-backend`) and the default host memory mode
(`--hicache-host-memory-mode cache`), a restore from storage looks the same on the event stream.
SGLang does not yet publish KV events for the storage tier
([sglang#29709](https://github.com/sgl-project/sglang/issues/29709)): a completed storage prefetch
is inserted into the host tier and published as a `CPU_PINNED` store, then loaded back to the GPU,
and the `DISK` and `EXTERNAL` mediums are never emitted (checked in SGLang's source, 2026-11-11).
So there a prefix that storage restored is `host_restored`, not `evicted`; a prefix storage did not
fully restore (a prefetch below SGLang's 256-token threshold, or one cut short by its timeout policy)
can still be a confirmed `evicted`. With `--hicache-host-memory-mode buffer_only`, host memory only
stages transfers and no `CPU_PINNED` store is published, so a restore from storage is a plain `none`
with no detail. SGLang's per-request `return_cached_tokens_details` reports the storage share;
Prefixwise does not read it yet, so from KV events alone it cannot tell host from storage.

## Validated live

Colab L4, Qwen3-4B, SGLang 0.5.23 (sglang-kernel 0.4.10), GPU KV pool capped at 32,768 tokens
(`--max-total-tokens`) and flooded with 3x that; 5 repetitions each
([record](../experiments/e_evict/runs/2026-11-07-colab-l4.md),
[with KV events](../experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md)):

| run | after the flood | Prefixwise |
|---|---|---|
| SGLang | prefix lost, 0/5 hits (3 of 3,404 tokens cached: the chat template's opening survived) | `evicted` 5/5, inferred |
| SGLang + KV events, `--full-attention` | prefix lost, 0/5 hits | `evicted` 5/5, **confirmed** ("3,382 of the previous request's KV blocks removed"; 3,411, 3,376, 3,414, 3,379 on the other repetitions). 280 batches, 453,787 blocks stored, 423,501 removed, 0 dropped messages, 0 overflows |
| SGLang + HiCache (`--hicache-ratio 4`) | **99% restored, 5/5 hits** (3,382 of 3,404 tokens cached) | `none` 5/5 |
| SGLang + HiCache + KV events, `--full-attention` (2026-11-08, Prefixwise 0.9.0) | 99% restored, 5/5 hits | `none` 5/5, each with **`host_restored`** ("3379 of the previous request's KV blocks had been removed from GPU memory (KV events; the tier's own stores were seen), yet this request read 3,382 of 3,404 prompt tokens"); 284 batches, 0 lost messages ([record](../experiments/e_evict/runs/2026-11-08-colab-l4-host-restored.md)) |

Blocks are one token at page size 1, SGLang's default for dense models, so "blocks removed" counts tokens: about the
previous request's whole chain, prompt plus output. The 2026-11-07 HiCache row predates
`host_restored` (#64) and was made without KV events, so its hits were recorded as plain `none`;
the 2026-11-08 row is the same flood with the events on.

The classifier scores **120/120** on injected causes on SGLang (`experiments/e1_open`: every
category 100%, with and without session ids sent), and `tests/test_sglang_events.py` replays
SGLang's own event structs (v0.5.23) over real ZMQ sockets: confirmed and ruled-out evictions, a
prefix stored by another session, split nodes, paged blocks, host-tier events, cache flush,
restart, concurrency, lost messages, overflow, and the `/server_info` checks.

## Limits

- Routing misses and "not an eviction" need `--full-attention`; on a hybrid model only
  evictions are confirmed.
- Requests with `n > 1` are not judged (the choices' stores would mix). `extra_key` is left out
  of SGLang's block hashes, so a changed `extra_key` is judged first, as `param_rendered`.
- A removal is counted from the previous request until the response: a token change and an
  eviction of exactly that block within one request's lifetime reads as a confirmed eviction
  (shared with vLLM; see [ENGINES.md](ENGINES.md)).
- Past 2 million tracked tokens (or 1.25x the GPU cache capacity, if larger) Prefixwise forgets
  its copy of the tree and judges nothing until it rebuilds; chains kept for sessions are capped
  at 16 million tokens.
- Several ranks: tested over the wire, not yet run live (a single L4 runs one rank).
- Token-level divergence ("not an eviction") did not occur live: 0 of 25 turns on SGLang (Qwen3-4B).
  A truncated answer (no closing `</think>`) is kept whole as content by Qwen3's template, so the
  prompt grows by the unfinished thinking and carries it on every later turn (3 of 25 turns).

Design, alternatives considered and failure modes: [ENGINES.md](ENGINES.md). Questions and bug
reports: GitHub issues.
