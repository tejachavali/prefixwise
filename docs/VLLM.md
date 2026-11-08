# Prefixwise for vLLM

Prefixwise sits in front of your engine as a local proxy. For every request it reads the
cache hit (`cached_tokens`) and, when the prefix was not reused, names **why**, says
whether it was waste, and prices it in tokens. On vLLM it can confirm evictions from the
engine's own KV-cache events, tell a host-tier restore from a GPU hit, and, across
data-parallel ranks or replicas, confirm routing misses. It stores counts and hashes, never
prompt text.

Companion guides: [LMCACHE.md](LMCACHE.md) (vLLM with an offload tier),
[SGLANG.md](SGLANG.md) (SGLang, HiCache, data-parallel ranks). Design and failure modes:
[ENGINES.md](ENGINES.md).

## Setup (vLLM)

```bash
pip install "prefixwise[engines]"          # Python 3.10+; the extra adds pyzmq + msgspec

vllm serve <model> --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'

prefixwise serve --upstream http://127.0.0.1:8000          # detects vLLM and its KV events
```

Point your client at `http://127.0.0.1:8080/v1` instead of the engine, run your workload,
then `prefixwise report`.

`prefixwise serve --upstream URL` (or `PREFIXWISE_UPSTREAM_URL`) asks the upstream what it is at
startup (`GET /version`, then `/kv_event_sources`): a vLLM server sets `--engine vllm`, and one
that publishes KV events sets `--kv-events auto` (every data-parallel rank) when the engines extra
is installed. It prints what it detected, and any flag you pass wins: `--engine vllm --kv-events
tcp://127.0.0.1:5557` is the explicit form of the same thing. A hosted API detects as no engine, and
so does a router that does not relay `/version`; behind one that does, pass `--engine none` (one
engine's verdicts do not hold behind a router) or list the replicas' event endpoints.

- `--enable-prompt-tokens-details` is required: without it vLLM does not report cached
  tokens and every request looks like a miss (Prefixwise warns once).
- `--kv-events-config` and `--kv-events` are optional. Without them, evictions are still
  named, as *inferred*.
- `--engine vllm` tells Prefixwise there is one engine behind it (no routing) and sends its
  session id in the request body (`session_id`, which vLLM puts on its KV events) plus
  `vllm_xargs: {"kv_cache_report_mode": "full"}` so vLLM also reports reused blocks. Neither
  changes caching.
- **Several KV caches.** `--kv-events auto` reads vLLM's `GET /kv_event_sources` and
  subscribes to every data-parallel rank (each rank's port is already offset there). Behind
  a router, list each replica's stream: `--kv-events tcp://replica-a:5557,tcp://replica-b:5557`.
  With more than one cache known, Prefixwise no longer infers an eviction; the events decide
  between `evicted` and `routing_miss`, and a request no cache can be attributed to is
  `unknown`. The router must pass the body's `session_id` and `vllm_xargs` through (llm-d,
  production-stack, AIBrix and the current SGLang model gateway do; Dynamo does not):
  [ENGINES.md](ENGINES.md) has the table.

## Inside vLLM, without a proxy

```bash
pip install "prefixwise[engines]"
vllm serve <model> --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}' \
  --middleware prefixwise.PrefixwiseMiddleware
```

`--middleware` loads Prefixwise as ASGI middleware in vLLM's own API server: no second process,
no port change, clients keep talking to vLLM. It does what the proxy does to each inference
request (session id and `vllm_xargs` for the KV join, `--fix` rewrites, `stream_options` so a
stream carries usage), lets vLLM answer, and records the turn from the response; a stream is
relayed chunk by chunk, never buffered. Non-streaming responses carry the same
`X-Prefixwise-*` verdict headers as the proxy's. Classification and the ledger write run off the
event loop, so a slow disk or a KV-event wait never stalls other streams.

KV events need no endpoint: on the first inference request the middleware reads vLLM's own
per-rank publisher configs (`app.state.engine_client`, what `GET /kv_event_sources` serves) and
subscribes, so data-parallel ranks are covered. Configuration is by environment variable, since
vLLM passes the class only `app`: `PREFIXWISE_LEDGER` (default `prefixwise.db` in the working
directory), `PREFIXWISE_FIX=canonical_tools`, `PREFIXWISE_KV_EVENTS` (an endpoint, or `auto`, the
default), `PREFIXWISE_COLLECTOR`. Read the ledger with `prefixwise report --ledger prefixwise.db`,
or run `prefixwise collect` and point `PREFIXWISE_COLLECTOR` at it; vLLM's port serves no
Prefixwise endpoints.

**With `--api-server-count` above 1, set `PREFIXWISE_COLLECTOR`.** Each API-server process gets
its own middleware; without one collector seeing every turn, a session's turns split across
processes and read as misses. The middleware says so once at startup when it sees that flag
without a collector. The same class works in your own FastAPI or Starlette app
(`app.add_middleware(PrefixwiseMiddleware, ledger="prefixwise.db")`); SGLang has no middleware
flag and its KV join needs `/server_info`, so SGLang users run the proxy ([SGLANG.md](SGLANG.md)).
The middleware sits outside vLLM's API-key check, so an unauthenticated request is fingerprinted
before it is rejected and its 401 is a row without a verdict; bodies over 32 MiB
(`PREFIXWISE_MAX_BODY_BYTES`) pass through unexamined. Tested against a stand-in ASGI engine
(`tests/test_asgi.py`, including a client that leaves mid-stream); not yet run inside a live
`vllm serve`.

## What the verdicts mean

When a request's prefix was unchanged but not reused:

| verdict | meaning | evidence |
|---|---|---|
| `evicted`, confirmed | the block where the cache read stopped was removed from GPU memory (before or while the request was queued) | vLLM `BlockRemoved` events for that session's blocks |
| `evicted`, inferred | one engine, prefix unchanged, not reused: usually an eviction (the look-alikes below need KV events to rule out) | no events, or nothing attributable |
| `routing_miss`, confirmed | another rank or replica served this turn and computed the block where its read stopped, while the cache that served the previous turn still holds that same block | KV events from several caches (`--kv-events auto`, or one endpoint per replica) |
| `unknown`, "not an eviction" | the engine still holds that block, so the token prefix changed although the request did not (a chat template re-rendering history) | KV events; claimed only when the stream was current and nothing was lost |

And when the prefix *was* reused (`none`), one detail separates two hits that look the same in
`cached_tokens`:

| detail | meaning | evidence |
|---|---|---|
| none | the GPU served the prefix | nothing removed |
| `host_restored` | the GPU copy had been evicted and an offload tier (LMCache, vLLM's CPU offload) loaded it back, no prefill recomputed: a hit that cost a restore, and a sign the GPU cache alone is too small for the session | GPU `BlockRemoved` for the previous request's blocks the read covered, the tier's own `BlockStored` seen on the stream, one cache only, a clean stream ([#64](https://github.com/tejachavali/prefixwise/pull/64)) |

And when the prompt itself changed, the cause names what changed first: `volatile_system` (a
timestamp or id in the system prompt), `tool_reserialize` (same tools, different order),
`tool_catalog_change`, `history_edit`, `compaction_rewrite`, `model_switch`, and more
(`prefixwise report --detail`). Each comes with its fix; `--fix canonical_tools` lets the
proxy stabilise tool order for you.

## LMCache and SGLang

- **vLLM + LMCache:** vLLM's `cached_tokens` includes tokens loaded from LMCache, so a prefix
  LMCache restores after a GPU eviction counts as a hit; with `--kv-events` that hit carries
  `host_restored`. Only prefixes lost from every tier are evictions. Setup, the two problems
  it solves and the live numbers: [LMCACHE.md](LMCACHE.md).
- **SGLang (and HiCache):** start it with `--enable-cache-report`, then
  `prefixwise serve --upstream http://127.0.0.1:30000 --engine sglang`. Evictions are
  inferred. To confirm them (SGLang 0.5.20 or later), also start SGLang with
  `--kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:5557"}'` and add
  `--kv-events tcp://127.0.0.1:5557` (and `--full-attention` for a model with no
  sliding-window or Mamba layers). Data-parallel ranks: `--kv-events auto`. All of it:
  [SGLANG.md](SGLANG.md).

## Validated live

Colab L4, Qwen3-4B, vLLM 0.33.0, LMCache 0.5.7, SGLang 0.5.23; GPU KV pool fixed at 32,768
tokens and flooded with 3x that ([record](../experiments/e_evict/runs/2026-11-07-colab-l4.md)):

| engine | after the flood | Prefixwise |
|---|---|---|
| vLLM | prefix lost, 0/5 hits | `evicted` 5/5 (inferred) |
| vLLM + KV events | prefix lost, 0/5 hits | `evicted` 5/5, confirmed ("211 of the previous request's KV blocks removed") |
| vLLM + LMCache | 98% restored, 5/5 hits | `none` 5/5 |
| vLLM + LMCache + KV events | 98% restored, 5/5 hits | `none` 5/5; recorded before #64, so without the detail: with today's code the same hits carry `host_restored` once LMCache's tier events reach the stream (verified in LMCache's current source; the next thing to run live) |
| SGLang | prefix lost, 0/5 hits | `evicted` 5/5 (inferred) |
| SGLang + KV events | prefix lost, 0/5 hits | `evicted` 5/5, confirmed ("3,382 of the previous request's KV blocks removed"; [record](../experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md)) |
| SGLang + HiCache | 99% restored, 5/5 hits | `none` 5/5; made without KV events; with them, `host_restored` (not yet run) |

Two vLLM replicas behind a router that never says which one it used (Qwen3-1.7B, one L4,
[record](../experiments/e_route/runs/2026-11-08-colab-l4.md)): with the router alternating,
`routing_miss` confirmed 5/5 from the replicas' events alone; with it sticky, 5/5 hits; no
false claim.

The classifier also scores 120/120 on injected causes on vLLM, vLLM + LMCache and SGLang
(`experiments/e1_open`).

## Limits

- Several caches need their events: `--kv-events auto` for the data-parallel ranks of one
  server (from `/kv_event_sources`), or one endpoint per replica behind a router. Without
  them, keep one Prefixwise per replica or route each session to one replica, since an
  inferred eviction cannot be told from a routing miss. `auto` on data-parallel ranks is
  tested over the wire and not yet run live; replicas behind a router are.
- `host_restored` is claimed only once the offload tier's own stores have been seen on the
  stream, behind one cache; a first restore before that, or a hit behind several caches, is a
  plain `none`.
- On hybrid models (sliding-window + full attention), "not an eviction" is never claimed.
- How often token-level divergence happens in real agent traffic has not been measured yet.

Design, alternatives considered and failure modes: [ENGINES.md](ENGINES.md). Companion
guides: [LMCACHE.md](LMCACHE.md), [SGLANG.md](SGLANG.md). Questions and bug reports: GitHub
issues.
