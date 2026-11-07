# Prefixwise for vLLM (and SGLang, LMCache)

Prefixwise sits in front of your engine as a local proxy. For every request it reads the
cache hit (`cached_tokens`) and, when the prefix was not reused, names **why**, says
whether it was waste, and prices it in tokens. On vLLM it can confirm evictions from the
engine's own KV-cache events. It stores counts and hashes, never prompt text.

## Setup (vLLM)

```bash
pip install "prefixwise[engines]"          # Python 3.10+; the extra adds pyzmq + msgspec

vllm serve <model> --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'

prefixwise serve --upstream http://127.0.0.1:8000 --engine vllm --kv-events tcp://127.0.0.1:5557
```

Point your client at `http://127.0.0.1:8080/v1` instead of the engine, run your workload,
then `prefixwise report`.

- `--enable-prompt-tokens-details` is required: without it vLLM does not report cached
  tokens and every request looks like a miss (Prefixwise warns once).
- `--kv-events-config` and `--kv-events` are optional. Without them, evictions are still
  named, as *inferred*.
- `--engine vllm` tells Prefixwise there is one engine behind it (no routing) and sends its
  session id in the request body (`session_id`, which vLLM puts on its KV events) plus
  `vllm_xargs: {"kv_cache_report_mode": "full"}` so vLLM also reports reused blocks. Neither
  changes caching.

## What the verdicts mean

When a request's prefix was unchanged but not reused:

| verdict | meaning | evidence |
|---|---|---|
| `evicted`, confirmed | the block where the cache read stopped was removed from GPU memory (before or while the request was queued) | vLLM `BlockRemoved` events for that session's blocks |
| `evicted`, inferred | one engine, prefix unchanged, not reused: usually an eviction (the look-alikes below need KV events to rule out) | no events, or nothing attributable |
| `unknown`, "not an eviction" | the engine still holds that block, so the token prefix changed although the request did not (a chat template re-rendering history), or the request went to another data-parallel rank | KV events; claimed only when the stream was current and nothing was lost |

And when the prompt itself changed, the cause names what changed first: `volatile_system` (a
timestamp or id in the system prompt), `tool_reserialize` (same tools, different order),
`tool_catalog_change`, `history_edit`, `compaction_rewrite`, `model_switch`, and more
(`prefixwise report --detail`). Each comes with its fix; `--fix canonical_tools` lets the
proxy stabilise tool order for you.

## LMCache and SGLang

- **vLLM + LMCache:** vLLM's `cached_tokens` includes tokens loaded from LMCache, so a prefix
  LMCache restores after a GPU eviction counts as a hit. Only prefixes lost from every tier
  are evictions. Launch script: `experiments/e1_open/vllm_lmcache_server.sh`.
- **SGLang (and HiCache):** start it with `--enable-cache-report`, then
  `prefixwise serve --upstream http://127.0.0.1:30000 --engine sglang`. Evictions are named
  as inferred: SGLang reads a body `session_id` as its own session handle, so its KV events
  cannot be attributed to sessions. Launch script: `experiments/e1_open/sglang_server.sh`.

## Validated live

Colab L4, Qwen3-4B, vLLM 0.33.0, LMCache 0.5.7, SGLang 0.5.23; GPU KV pool fixed at 32,768
tokens and flooded with 3x that ([record](../experiments/e_evict/runs/2026-11-07-colab-l4.md)):

| engine | after the flood | Prefixwise |
|---|---|---|
| vLLM | prefix lost, 0/5 hits | `evicted` 5/5 (inferred) |
| vLLM + KV events | prefix lost, 0/5 hits | `evicted` 5/5, confirmed ("211 of the previous request's KV blocks removed") |
| vLLM + LMCache | 98% restored, 5/5 hits | `none` 5/5 |
| SGLang | prefix lost, 0/5 hits | `evicted` 5/5 (inferred) |
| SGLang + HiCache | 99% restored, 5/5 hits | `none` 5/5 |

The classifier also scores 120/120 on injected causes on vLLM, vLLM + LMCache and SGLang
(`experiments/e1_open`).

## Limits

- One Prefixwise per engine replica (or route each session to one replica). vLLM offsets the
  KV-events port per data-parallel rank; Prefixwise subscribes to one stream.
- On hybrid models (sliding-window + full attention), "not an eviction" is never claimed.
- How often token-level divergence happens in real agent traffic has not been measured yet.

Design, alternatives considered and failure modes: [ENGINES.md](ENGINES.md). Questions and
bug reports: GitHub issues.
