# Prefixwise inside vLLM, with LMCache, no proxy: Colab L4, 2026-11-10

Run by Muse on main at 4f3ee8fa5f7bf2ad0b18d2ac05c93e488465dc3a (prefixwise 0.11.1), vLLM 0.33.0,
LMCache 0.5.7, Qwen3-4B; Colab L4 (23,034 MiB GPU), high-RAM runtime (52.96 GB). No code changed,
nothing pushed. Transcribed by Claude from Muse's report; the outputs below are as printed.

Two things had each passed live but never together: the middleware inside `vllm serve` (evictions
confirmed 5/5, [2026-11-08](2026-11-08-colab-l4-middleware.md)) and LMCache behind the proxy
(`host_restored` 5/5, [2026-11-08](2026-11-08-colab-l4-host-restored.md)). This run is both at once:
one flag on vLLM, LMCache on, no proxy. Then, on the same server, the `canonical_tools` replay (Part B)
and the proxy's engine detection (Part C).

## Part A: a GPU flood, with Prefixwise inside vLLM and LMCache behind it

    PREFIXWISE_LEDGER=/content/mw-lmcache.db \
    MODEL=Qwen/Qwen3-4B GPU_BLOCKS=2048 L1_GB=20 KV_EVENTS=1 \
    VLLM_ARGS="--middleware prefixwise.PrefixwiseMiddleware" \
    ./experiments/e1_open/vllm_lmcache_server.sh
    python experiments/e_evict/e_evict.py --engine vllm-lmcache --in-engine /content/mw-lmcache.db

```
rep 0: T2 99% cached (none); flood 25 requests in 17.9 s; T3 3328/3404 cached -> Prefixwise: none [host_restored]
rep 1: T2 99% cached (none); flood 25 requests in 18.1 s; T3 3328/3433 cached -> Prefixwise: none [host_restored]
rep 2: T2 99% cached (none); flood 25 requests in 18.3 s; T3 3328/3398 cached -> Prefixwise: none [host_restored]
rep 3: T2 99% cached (none); flood 25 requests in 18.4 s; T3 3328/3436 cached -> Prefixwise: none [host_restored]
rep 4: T2 99% cached (none); flood 25 requests in 18.5 s; T3 3328/3401 cached -> Prefixwise: none [host_restored]
# E-evict: vllm-lmcache
Model Qwen/Qwen3-4B; prefix ~4000 tokens; GPU KV 32768 tokens; flood 98304 tokens; 5 repetitions.
- Control (T2, before the flood) hit every time: yes
- T3 (after the flood) hit: 5/5 (expected all)
- Engine behaved as expected: yes
- Correct cause on T3: `none`; Prefixwise said: `none` x5 (5/5 correct)
- Detail on T3: `host_restored` x5
```

**Pass: the same as behind the proxy.** Every turn after the flood was a hit, 3,328 tokens back from
LMCache (13 chunks of 256), and each was labelled `host_restored` (208 GPU blocks seen evicted). The
server log has exactly one Prefixwise line (line 274), and no LMCache connector error or traceback:

```
prefixwise: subscribed to the engine's KV events: tcp://127.0.0.1:5557
```

The middleware found the event stream by itself: no `--kv-events` was passed. `prefixwise report
--ledger /content/mw-lmcache.db --detail`, as Muse sent it (summarised by Muse in its last two lines):

```
140 requests, 470,671 prompt tokens, 7% served from cache.
Not priced: Qwen/Qwen3-4B (not in the price table; `prefixwise prices` lists it, --prices FILE adds rates).
No avoidable waste found: every miss had a legitimate cause.
Expected misses (not waste): 124 x the system prompt was edited; 6 x first request of a session: nothing to reuse yet.
Requests 140; Prompt tokens 470,671 (cache hit rate 7.1%); Completion tokens 560; Avoidable tokens 0 (0.0% of prompt tokens); ...
By-cause table: system_edit 124 requests, 416,443 prompt, 0.0% cached; none 10 requests, 34,034 prompt, 98.3% cached; first_write 6 requests, 20,194 prompt, 0.0% cached
```

The 124 `system_edit` rows are the harness's flood: inside the engine every request passes through the
middleware, the flood included; its prompts share the default session and differ in their system
prompt. Behind a proxy the flood goes straight to the engine and is not recorded. The harness's closing
line "KV-event stream at the end: null" is its own readout of a stream it does not open in this mode.

## Part B: the `canonical_tools` replay on the same server

    python experiments/e4/e4.py --provider vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-4B

```
# E4 (vllm): canonical_tools
Paired requests 60 (errors off 0, on 0), 29 with tool-order drift  |  Pass: YES
| | OFF (as recorded) | ON (canonical_tools) |
| Prompt tokens | 157,706 | 157,706 |
| Cached tokens | 138,928 | 152,800 |
| Task success | 76.7% | 75.0% |
| Causes | {'first_write': 6, 'none': 25, 'tool_reserialize': 29} | {'first_write': 6, 'none': 54} |
Predicted savings (OFF tool_reserialize avoidable): 14,229 tokens
Realised savings (cached ON - cached OFF): 13,872 tokens (97.5% of predicted; pass needs >= 50%)
Task success flat (ON >= OFF - 5 points): yes
```

**Pass, and every figure is identical, to the token, to the vLLM-alone run of
[2026-11-06](../../e4/runs/2026-11-06-live.md).** With greedy decoding the model made the same tool
choices, so the arms sent the same prompts, and the cache served the same tokens. That is also the
answer to the question this part was asked: the two arms are kept apart by a per-arm `cache_salt`, and
had LMCache's stored chunks ignored the salt, the second arm would have read the first arm's chunks
and cached more than vLLM alone did. It did not. That is what the numbers are consistent with, not a
direct test of LMCache's salt handling: in this replay the GPU cache (32,768 tokens) held most of each
arm's working set, so LMCache was asked for little.

## Part C: engine detection, with no engine flag

    prefixwise serve --upstream http://127.0.0.1:8000 --port 8081 --ledger /content/detect.db

```
prefixwise: upstream is vLLM 0.33.0 (detected): --engine vllm, --kv-events auto (it publishes KV events)
Prefixwise proxy on http://127.0.0.1:8081
  upstream: http://127.0.0.1:8000
  ledger:   /content/detect.db
  fixes:    none (pass-through)
  prices:   20 models, table of 2026-11-09 (built-in prices.json)
  engine:   vllm (one self-hosted engine); KV events: tcp://127.0.0.1:5557
  note:     start vllm with --enable-prompt-tokens-details, or every turn looks like a miss
  auth:     none
  note:     set PREFIXWISE_HMAC_KEY so hashes of short prompt blocks cannot be guessed
POST /v1/chat/completions  POST /v1/responses  POST /v1/messages  POST /tasks
GET  /ledger  GET /requests/<id>  GET /metrics  GET /health
```

The first live record of detection: from a URL alone it named vLLM 0.33.0, set `--engine vllm`, found
the event publisher and subscribed. The `--enable-prompt-tokens-details` note is printed for every
vLLM upstream, since the proxy cannot see the engine's flags; this server had it on.

No screen recording was possible from the runtime, so the outputs are text. The proxy was stopped with
SIGTERM (the browser cannot send Ctrl-C).
