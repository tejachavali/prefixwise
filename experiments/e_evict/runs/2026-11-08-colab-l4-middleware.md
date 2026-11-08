# E-evict with Prefixwise inside vLLM (`--middleware prefixwise.PrefixwiseMiddleware`): Colab L4, 2026-11-08

Run by Muse on main at 3732617 (PRs #68, #69), Prefixwise 0.9.0 plus the middleware, vLLM 0.33.0,
Qwen3-4B; GPU KV capacity 32,768 tokens (`--block-size 16 --num-gpu-blocks-override 2048`); flood
98,304 tokens; 5 repetitions. No proxy: `vllm serve ... --kv-events-config '{"enable_kv_cache_events":
true, "publisher": "zmq", "endpoint": "tcp://*:5557"}' --middleware prefixwise.PrefixwiseMiddleware`
with `PREFIXWISE_LEDGER=/content/mw.db`; the harness in `--in-engine /content/mw.db` mode sends each
turn straight to vLLM and reads the verdict the middleware wrote by the `X-Prefixwise-Request-Id`
header. No code changed, nothing pushed from Colab. Transcribed by Claude from Muse's report.

**Result: as the proxy (run A2, [record](2026-11-07-colab-l4.md)).** `--middleware` was accepted;
after the first inference request the server log showed the one Prefixwise line,
`prefixwise: subscribed to the engine's KV events: tcp://127.0.0.1:5557` (the endpoint read from
vLLM's own engine client, not from the command line); every T3 was a confirmed eviction.

| rep | T2 cached | flood | T3 cached | Prefixwise on T3 (from /content/mw.db) |
|---:|---:|---|---:|---|
| 0 | 99% | 25 req, 17.2 s | 0/3404 | `evicted`: confirmed (211 of the previous request's KV blocks removed) |
| 1 | 99% | 25 req, 17.3 s | 0/3433 | `evicted`: confirmed (213 removed) |
| 2 | 99% | 25 req, 17.1 s | 0/3398 | `evicted`: confirmed (211 removed) |
| 3 | 99% | 25 req, 17.1 s | 0/3436 | `evicted`: confirmed (213 removed) |
| 4 | 99% | 25 req, 17.1 s | 0/3401 | `evicted`: confirmed (211 removed) |

Control (T2) hit 5/5; T3 hit 0/5 (expected); `evicted`, confirmed, 5/5.

`prefixwise report --ledger /content/mw.db --detail`: 140 requests, 470,671 prompt tokens, 3.6%
cached, 0 avoidable; by cause `system_edit` 124, `first_write` 6, `evicted` 5, `none` 5 (99.1%
cached). The 124 `system_edit` rows are the flood: in-engine, every request passes through the
middleware, including the harness's unrelated flood prompts, which share the default session (no
session header) and differ in their system prompt; through the proxy the flood goes straight to the
engine and is not recorded. Cost: none, Qwen3-4B is not in the price table.

Verdict headers on a non-streaming response (`curl -si ... | grep -i x-prefixwise`, a 9-token prompt):

```
x-prefixwise-request-id: 09f89f9c-...
x-prefixwise-cause: below_min_prefix
x-prefixwise-legit: none
x-prefixwise-avoidable-tokens: 0
```

**What this shows**

- The in-process edge does what the proxy does on a live engine: the same flood, the same five
  confirmed evictions with the same block counts, written from inside vLLM's API server.
- KV-event discovery from `app.state.engine_client.get_kv_event_sources()` works on vLLM 0.33.0
  (the review's finding that it must read `KVEventsConfig` objects, not dicts, was right to fix).
- Not covered: streaming through the middleware under load, `--api-server-count` above 1 with a
  collector, and the SGLang path (no middleware flag there).
