# Prefixwise

**Your agent resends its whole conversation every turn, and one changed byte near the top means the provider charges full price for everything after it. Prefixwise tells you which byte, what it cost, and fixes the common causes for you.**

Providers and engines report *how many* tokens missed the prompt cache, not *why*. Prefixwise sits between your agent and the model and does four things:

- **Identify.** One named cause for every request, with its evidence, its cost and its fix.
- **Fix.** `--optimize` repairs the request before it is sent. Only fixes that passed a replay against live models.
- **Prevent.** `prefixwise check` fails CI when a code change would break the cache.
- **Auto-detect.** It recognises vLLM and SGLang behind a URL and, when the engine publishes KV events, subscribes to them on its own.

It stores hashes and counts, never your prompts.

```bash
pip install prefixwise
prefixwise run -- claude             # any agent with an API key: claude, codex, opencode, your own
prefixwise report --session latest   # the session turn by turn
prefixwise run --optimize -- claude  # the same agent, with the proven fixes applied
```

No API key (Claude Code or Codex on a subscription)? `prefixwise import claude-code` reads the logs already on your machine.

## What you see

After a run (this is `prefixwise run --mock` on an eight-turn agent whose tool list comes back in another order twice; the mock provider's token counts are flat, the verdicts and prices are the real logic):

```
-- prefixwise: this run --
(mock: synthetic responses and usage, so the numbers below are illustrative)
8 requests, 16,000 prompt tokens, 62% served from cache. Cost $0.0256.
Avoidable: 4,000 tokens ($0.0138), 25.0% of prompt tokens. Fix these, biggest first:
  1. The same tools were sent in a different order or format (2 requests, 4,000 tokens, $0.0138).
     Fix: Send tools in a fixed order with deterministic serialization (or let the proxy do it: prefixwise serve --fix canonical_tools).
     Prefixwise can do it for you: prefixwise run --fix canonical_tools (proven)
Expected misses (not waste): 1 x first request of a session: nothing to reuse yet.
```

`prefixwise report --session latest` shows where it happened:

```
Session run-20261110-000152: 8 requests, 16,000 prompt tokens, cache hit rate 62.5%, 4,000 avoidable, cost $0.0256   (claude-sonnet-4-6)

turn  prompt  cached     hit             cause  first changed     cost  avoidable  avoidable $
----  ------  ------  ------  ----------------  -------------  -------  ---------  -----------
1      2,000       0    0.0%       first_write              -  $0.0075          0      $0.0000
2      2,000   2,000  100.0%              none    messages[8]  $0.0006          0      $0.0000
3      2,000   2,000  100.0%              none   messages[10]  $0.0006          0      $0.0000
4      2,000       0    0.0%  tool_reserialize       tools[0]  $0.0075      2,000      $0.0069
5      2,000   2,000  100.0%              none   messages[14]  $0.0006          0      $0.0000
6      2,000       0    0.0%  tool_reserialize       tools[0]  $0.0075      2,000      $0.0069
7      2,000   2,000  100.0%              none   messages[18]  $0.0006          0      $0.0000
8      2,000   2,000  100.0%              none   messages[20]  $0.0006          0      $0.0000
```

"first changed" is the block where the prompt stopped matching the previous request, counted over the whole prompt: tools, then system, then messages. With `--optimize` the same agent reads the cache on every turn after the first: 87.5% served from cache instead of 62.5%, and no avoidable tokens.

## Results

Every number is from a live run against real models, and each links to its record. The full table is in the [guide](docs/GUIDE.md#results).

| Claim | Result |
|---|---|
| Agrees with Anthropic's own miss reason | **32 of 32** misses Anthropic named ([1](experiments/e1_hosted/runs/2026-11-06-anthropic-claude-haiku-5-5.md), [2](experiments/e1_hosted/runs/2026-11-06-round4.md)) |
| Names injected causes on open models | **120 of 120** on vLLM, vLLM + LMCache and SGLang ([vLLM](experiments/e1_open/runs/2026-11-06-colab-l4-qwen3-4b.md), [LMCache and SGLang](experiments/e_evict/runs/2026-11-07-colab-l4.md)) |
| Tells an engine eviction from everything else | **5 of 5**, confirmed from KV events ([vLLM](experiments/e_evict/runs/2026-11-07-colab-l4.md), [SGLang](experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md)); a prefix LMCache or HiCache restored counted as a hit **5 of 5** ([record](experiments/e_evict/runs/2026-11-08-colab-l4-host-restored.md)) |
| The tool-order fix saves what was predicted | **99.9%** on Claude, **97.5%** on vLLM, task success flat (one of 60 turns changed on Qwen3-4B) ([record](experiments/e4/runs/2026-11-06-live.md)) |
| The cache-marker fix pays for itself | **106.8%** of predicted savings, prompt cost down **85.6%** with the write surcharge included, task success flat ([record](experiments/e4/runs/2026-11-09-auto-breakpoints-live.md)) |
| Works with real agents | Claude Code, Codex, opencode, the Anthropic and OpenAI SDKs ([record](experiments/compat/2026-11-07.md), [opencode](experiments/compat/2026-11-09-opencode-run.md)) |
| Costs almost nothing | **3 ms** per request at 2k tokens, 13 ms at 120k ([record](benchmarks/runs/2026-11-07-vm.md)) |

## Ways in

Each takes about thirty seconds. All write the same ledger and print the same report.

| You have | Do this |
|---|---|
| An agent you run from a terminal | `prefixwise run -- <your command>` |
| A service, several agents, or a gateway | `prefixwise serve --upstream https://api.anthropic.com`, and point clients at it |
| Your own vLLM server | `vllm serve <model> ... --middleware prefixwise.PrefixwiseMiddleware`: no proxy, no port change ([docs/VLLM.md](docs/VLLM.md)) |
| vLLM with LMCache, or SGLang with HiCache | `prefixwise serve --upstream <url>` in front; it detects the engine ([LMCache](docs/LMCACHE.md), [SGLang](docs/SGLANG.md)) |
| Your own Python code | `client = prefixwise.wrap(Anthropic())` (or OpenAI) |
| LiteLLM | `litellm.callbacks = [PrefixwiseCallback()]` |
| Claude Code or Codex on a subscription | `prefixwise import claude-code` / `prefixwise import codex` |

## Identify

Nineteen causes, each marked as waste or as a legitimate miss. The ones people meet most:

| Cause | What happened | Fix |
|---|---|---|
| `tool_reserialize` | The same tools, in another order or with other bytes | `--optimize` does it, or sort them yourself |
| `volatile_system` | A timestamp or id in the system prompt changed | Move it to the latest user message |
| `history_edit` | Earlier messages were rewritten | Append, do not rewrite; compact at task boundaries |
| `breakpoint_misplaced` | Anthropic: no `cache_control`, or one that stays too early | `--optimize` places them, or add one on the last message |
| `ttl_expired` | The gap since the last turn outlived the cache | A longer TTL, if the reads pay for it |
| `evicted` | A self-hosted engine dropped the session's blocks | More KV memory, or an offload tier |

Beyond the report: `prefixwise explain --diff a.json b.json` shows what changed between two requests, `prefixwise fixes` ranks fixes by dollars, and every non-streaming response carries an `X-Prefixwise-Cause` header (for a stream, look the request up by its id). All causes and details: [guide](docs/GUIDE.md#what-it-tells-you).

## Fix

`--optimize` (on `serve` and `run`, or `PREFIXWISE_OPTIMIZE=1` for the vLLM middleware) applies every fix that has passed a replay against a live model. A fix earns its place by realising at least half of what Prefixwise predicted with task success flat.

- **`canonical_tools`** sorts the tool list and each tool's keys, so a list that drifts between turns stops breaking the cache.
- **`auto_breakpoints`** (Anthropic's API only) adds `cache_control` markers to a request that has none, once the same prefix was seen within the cache window. A one-shot request never pays for a write nobody reads, and a client's own markers are never touched.

Off by default. If the upstream rejects a request carrying markers Prefixwise placed, the client's original is sent instead; `canonical_tools` changes only the order of tools and keys. `prefixwise fixes` says whether a fix paid off on your traffic. Fixes that would change what the model reads, such as moving a timestamp, stay advice.

## Prevent

```bash
prefixwise check -- pytest tests/agent_scenarios      # exit 2 on a cache regression
```

The costly regressions pass every functional test: a tool list built from a set, a timestamp added to the system prompt, a marker moved. `prefixwise check` runs your agent's scenarios against a local mock provider, with no API calls, and fails on four: tools in a changed order, a volatile system prompt, a misplaced marker, a mutated tool result. As a GitHub Action:

```yaml
- uses: tejachavali/prefixwise@v0.11
  with:
    command: pytest tests/agent_scenarios
```

Details, allow-lists and baselines: [guide](docs/GUIDE.md#gate-it-in-ci).

## Self-hosted engines

On vLLM and SGLang, with `pip install "prefixwise[engines]"` and the engine publishing its KV-cache events, Prefixwise reads them, so an eviction is confirmed and not inferred, and a prefix that LMCache or HiCache restored from CPU memory is labelled `host_restored`: a hit the GPU alone would have missed. It also names what a chat template costs, such as a reasoning model recomputing its previous answer every turn. Step by step: [vLLM](docs/VLLM.md), [LMCache](docs/LMCACHE.md), [SGLang](docs/SGLANG.md), [how the engine join works](docs/ENGINES.md).

## In production

Prometheus metrics with an alert rule and a Grafana dashboard ([docs](docs/prometheus.md)), OpenTelemetry spans ([docs](docs/opentelemetry.md)), an HTTP API ([docs](docs/API.md)), a collector for several proxies, and sinks to JSONL or ClickHouse ([architecture](docs/ARCHITECTURE.md)). Measured overhead is 3 ms per request at 2k tokens.

## Limits

- **One traffic shape per proof.** `canonical_tools` was proven on Claude Haiku 5.5 and Qwen3-4B, `auto_breakpoints` on Haiku 5.5 only; both on scripted multi-turn sessions seconds apart, not an agent's own traffic.
- **OpenAI's own verdict is unchecked.** Its diagnostics answered `unavailable` on every call we made. Prefixwise still classifies OpenAI traffic.
- **A change is located per block, not per token.** An edit at the end of a long system prompt is reported for the whole block.
- **Long sessions and many agents at once** are less tested than short ones.
- **Dollars need a price.** Models in the dated table are priced; others show a dash until you add rates.
- **API keys only.** An agent signed in with a subscription cannot be routed through a proxy; use `prefixwise import` for those.

The full list, with what was and was not verified live: [guide](docs/GUIDE.md#limits).

## More

- [The full guide](docs/GUIDE.md): every way in, every cause, the command and environment reference, production notes.
- [Use cases](docs/USE_CASES.md), [data and metrics](docs/DATA.md), [architecture](docs/ARCHITECTURE.md), [changelog](CHANGELOG.md), [security](SECURITY.md).
- Working on Prefixwise: `pip install -e ".[dev]" && python -m pytest tests -q`. Read [WORKLOG.md](WORKLOG.md) first.

MIT. See LICENSE.
