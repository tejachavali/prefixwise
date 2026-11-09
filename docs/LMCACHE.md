# Prefixwise for LMCache: what the offload tier saved, and what the agent broke

LMCache keeps a CPU-memory copy of the KV vLLM computes, so blocks later evicted from the GPU are
loaded back on the next turn and a session survives a flood that would otherwise cost it a full
prefill. vLLM reports the result as an ordinary cache hit. Prefixwise, in front of vLLM as a
local proxy, answers the two questions that hit does not: when a turn *missed*, was it the
cache or the agent's prompt? And when it *hit*, did the GPU serve it or did LMCache rescue it?
It stores counts and hashes, never prompt text.

Companion guides: [VLLM.md](VLLM.md) (vLLM without an offload tier), [SGLANG.md](SGLANG.md)
(SGLang and HiCache). Design and failure modes: [ENGINES.md](ENGINES.md).

## Setup

LMCache runs as its own process (its L1 is CPU memory, LRU); vLLM connects to it with the
connector LMCache ships. The launch script we validated with is
`experiments/e1_open/vllm_lmcache_server.sh`; in full:

```bash
pip install "prefixwise[engines]" vllm lmcache      # Python 3.10+; the extra adds pyzmq + msgspec

lmcache server --host 127.0.0.1 --port 5555 --http-port 8090 --l1-size-gb 20 --eviction-policy LRU &

vllm serve <model> --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
                         "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                         "kv_connector_extra_config": {"lmcache.mp.host": "127.0.0.1", "lmcache.mp.port": 5555}}' \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'

prefixwise serve --upstream http://127.0.0.1:8000 --engine vllm --kv-events tcp://127.0.0.1:5557
```

Point your client at `http://127.0.0.1:8080/v1` instead of vLLM, run your workload, then
`prefixwise report`.

- `--enable-prompt-tokens-details` is required: it puts `cached_tokens` in every response.
  Without it every request looks like a miss, and Prefixwise warns once.
- The connector name is LMCache's current multi-process one (`LMCacheMPConnector`, verified in
  LMCache's quickstart and vLLM's source, Nov 2026); the in-process `LMCacheConnectorV1` and
  its `LMCACHE_*` variables moved out to the separate LMCache_inprocess repository. If your
  deployment still uses that connector, only the `--kv-transfer-config` line differs; Prefixwise
  reads vLLM, not LMCache. The tier's events come from the LMCache server's event log
  (`--kv-event-log-size`, default 32,768) through vLLM's publisher; `--disable-observability` on
  the server turns them off, and with them `host_restored`.
- `--kv-events-config` and `--kv-events` are optional. Without them, Prefixwise still names
  every agent-side cause and counts LMCache's restores as hits; it just cannot tell a GPU hit
  from a restore, and evictions are *inferred*. LMCache's HTTP port is moved to 8090 in the
  script because its default, 8080, is Prefixwise's.
- Several replicas behind a router, or data-parallel ranks: `--kv-events auto` subscribes to
  every rank (vLLM's `GET /kv_event_sources`); behind a router, list each replica's stream,
  `--kv-events tcp://replica-a:5557,tcp://replica-b:5557`. Then a turn sent away from its cached
  prefix is a confirmed `routing_miss`, and `host_restored` is not claimed (which cache served
  is what the events establish; a restore across caches is not attributed). See
  [ENGINES.md](ENGINES.md) for what a router must pass through.

## The two problems

**1. A miss that was the agent's, blamed on the cache.** An agent that rebuilds its tool list
from a set sends the same tools in a different order; one that stamps the time into its system
prompt changes the first bytes of every request. Neither GPU prefix caching nor LMCache can match
a prefix that is different each turn, so the session misses every time, the dashboards show a
cold cache, and the KV tier gets the blame. Prefixwise decides these from the request's own
block hashes, with or without events: `tool_reserialize` ("tool multiset of canonical hashes
equal; sequence or raw bytes differ"), `volatile_system` ("system raw hash changed, masked hash
equal (dates/ids/numbers only)"), `history_edit`, `tool_result_mutation`; each names the first block that differs and
its fix. No prompt text is stored or sent anywhere: the ledger holds hashes, positions and
counts. `--fix canonical_tools` lets the proxy sort the tools itself (proven in the E4 replay:
97.5% of the predicted saving realised on vLLM, task success flat by the experiment's bar), and `prefixwise check`
gates the same causes in CI.

**2. A hit you cannot attribute.** vLLM's `cached_tokens` is one number: local GPU prefix hits
plus tokens loaded through the connector (`vllm/v1/metrics/stats.py`, `PrefillStats`:
`num_local_cached_tokens + num_external_cached_tokens`). The usage, Prometheus and every tool
that reads it say "hit" for a prefix LMCache fetched from CPU memory after the GPU copy was
evicted, so you cannot see what LMCache is doing for you, or that the GPU cache is too small.
With `--kv-events` Prefixwise sees both halves: vLLM publishes `BlockRemoved` on the GPU medium
when it reclaims a block, and the offload connector publishes its own `BlockStored` on the CPU
medium (LMCache main, Nov 2026; converted to vLLM's event types and sent in the same batches). A hit whose blocks had been removed from the GPU, on a stream that has shown
the tier's own stores, is `none` with the detail **`host_restored`**: "restored from the host or
offload tier: N of the previous request's KV blocks had been removed from GPU memory (KV
events; the tier's own stores were seen), yet this request read X of Y prompt tokens from the
cache without recomputing them".

## What the verdicts mean

| what happened | verdict | detail | evidence |
|---|---|---|---|
| GPU prefix hit | `none` | | `cached_tokens` covers the previous prompt; nothing removed |
| LMCache rescued the prefix after a GPU eviction | `none` | `host_restored` | GPU `BlockRemoved` for the previous request's blocks, the tier's own stores seen, the blocks read back (live: 5/5, below) |
| the prefix was lost from every tier and recomputed | `evicted`, confirmed | | the block where the read stopped was removed from the GPU ("211 of the previous request's KV blocks removed") |
| the same, no events | `evicted`, inferred | | one engine, prefix unchanged, not reused |
| the block is still on the GPU: the token prefix changed although the request did not (chat template) | `unknown`, "not an eviction" | | KV events, stream current, nothing lost |
| the agent changed the prompt | `volatile_system`, `tool_reserialize`, `history_edit`, `tool_result_mutation`, ... | | the request's own block hashes; `divergence_role` names the block |

`prefixwise report --detail` shows the details under each cause; `/api/v1/breakdown?by=detail`
slices by them, so "turns rescued by LMCache" is one query, and the ledger rows carry it for
your own dashboards (`docs/DATA.md`).

What `host_restored` is not claimed on: a hit after a GPU removal when the stream has never
shown a non-GPU store (no tier exists; another request sharing the prefix may have recomputed
it), when another session stored a removed block while this request was in flight, from a
contended or cut stream, or behind several caches. LMCache stores whole chunks (256 tokens by
default), so the sub-chunk tail of a prompt is recomputed on every restore: a restored hit reads
back about 98% of the prefix, not all of it, and a prefix shorter than one chunk never hits
LMCache at all.

## Validated live

Colab L4, Qwen3-4B, vLLM 0.33.0, LMCache 0.5.7 (20 GB CPU tier); GPU KV pool fixed at 32,768
tokens (`--block-size 16 --num-gpu-blocks-override 2048`) and flooded with 98,304 tokens of
unrelated prompts; 5 repetitions each ([record](../experiments/e_evict/runs/2026-11-07-colab-l4.md)):

| run | after the flood | Prefixwise |
|---|---|---|
| vLLM alone | prefix lost, **0/5 hits** (0 of 3,404 tokens cached) | `evicted` 5/5, inferred |
| vLLM + KV events | prefix lost, 0/5 hits | `evicted` 5/5, **confirmed** ("211 of the previous request's KV blocks removed"; the prefix spans about 213 blocks of 16) |
| vLLM + LMCache | **98% restored, 5/5 hits** (3,328 of 3,404 tokens cached, from CPU memory) | `none` 5/5 |
| vLLM + LMCache + KV events (2026-11-08, Prefixwise 0.9.0) | 98% restored, 5/5 hits | `none` 5/5, each with **`host_restored`**: "208 of the previous request's KV blocks had been removed from GPU memory (KV events; the tier's own stores were seen), yet this request read 3,328 of 3,404 prompt tokens from the cache without recomputing them" ([record](../experiments/e_evict/runs/2026-11-08-colab-l4-host-restored.md)) |

That is the ROI line: the same flood, the same session, 0 of 5 turns reusable without LMCache
and 5 of 5 with it, the restored share read straight from `cached_tokens`, and with KV events each
rescued turn named as such. The 2026-11-07 LMCache rows predate `host_restored` (#64) and recorded
plain `none`; the 2026-11-08 run is the same configuration on 0.9.0. It also settles the version
question: LMCache 0.5.7 publishes its tier's stores into vLLM's event stream
(`offload_tier_seen: true`, 569 batches, 0 lost messages). 3,328 tokens is 13 chunks of 256: the
sub-chunk tail is recomputed, as LMCache's chunking predicts.

The classifier also scores **120/120** on injected causes on vLLM + LMCache
(`experiments/e1_open`, 120 turns across ten cause types, two sessions growing turn by turn as
agent traffic does).

## Limits

- `host_restored` is attribution, not a measurement of LMCache's own hit rate: LMCache's
  `/metrics` (port 8090 in the script) has that. Prefixwise says which *turns* were rescued and
  what they would have cost. Validated live on LMCache 0.5.7 (above).
- A restore is claimed only after the tier's own stores have appeared on the stream; the first
  restore after Prefixwise starts, before any offload store was seen, is a plain `none`.
- Prefixwise treats any non-GPU medium as the offload tier and does not tell tiers apart; LMCache
  itself publishes only a CPU medium today.
- Token-level divergence ("not an eviction") did not occur in the live runs (0 of 50 harness turns, 0
  of 8 agent requests; [VLLM.md](VLLM.md)).

Design, alternatives considered and failure modes: [ENGINES.md](ENGINES.md). Questions and bug
reports: GitHub issues.
