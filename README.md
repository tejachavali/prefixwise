# Prefixwise

**Your agent resends its whole conversation every turn, and one changed byte near the top means the provider charges full price for everything after it. Prefixwise tells you which byte, what it cost, and what to change.**

```bash
pip install prefixwise
prefixwise import claude-code        # Claude Code on a subscription: reads ~/.claude/projects, nothing leaves your machine
prefixwise run -- claude             # or any agent with an API key: claude, codex, opencode, your own
prefixwise report --share            # a block you can paste
```

```
8 requests, 16,000 prompt tokens, 0% served from cache. Cost $0.1002.
Avoidable: 14,000 tokens ($0.0805), 87.5% of prompt tokens. Fix these, biggest first:
  1. The same tools were sent in a different order or format (7 requests, 14,000 tokens, $0.0805).
     Fix: Send tools in a fixed order with deterministic serialization (or let the proxy do it: prefixwise serve --fix canonical_tools).
     Prefixwise can do it for you: prefixwise run --fix canonical_tools (proven)
Expected misses (not waste): 1 x first request of a session: nothing to reuse yet.
Cache writes never read: 14,000 tokens were written to the cache, and the next turn of the same conversation replaced the prefix before reading them (7 requests); the write surcharge on them cost $0.0175 above sending them uncached. Fixing the misses above saves this too; it is not counted twice.
```

That is the real output of `prefixwise run --mock` on an eight-turn agent that rebuilds its tool list
from a set each turn (the mock provider answers with synthetic usage, so the counts are illustrative;
your run prints your own). On real traffic the same report also prices each cause in dollars and,
per Anthropic model, says whether the 1-hour cache would have paid.

Every turn of an agent resends the conversation. The provider charges a tenth for the part it
cached and full price for the rest, and one changed byte early in the prompt (a timestamp in the
system prompt, tools in a new order, an edited earlier message) means everything after it is paid
for again. Providers and engines report *how many* tokens missed, not *why*. Prefixwise names the
cause of every miss, says whether it was waste or a legitimate change, prices it, and tells you the
fix. It stores hashes and counts, never your prompts.

It runs in front of Claude, OpenAI, or your own vLLM or SGLang, inside vLLM as middleware, inside
your Python, under LiteLLM, or over the logs Claude Code and Codex already keep. On vLLM and SGLang
it reads the engine's own KV events to tell an eviction from everything else.

## Results

Every number below is from a live run against real models; each links to its record.

| What was tested | Setup | Result |
|---|---|---|
| Agrees with Anthropic's own miss reason | Claude Haiku 5.5, 7 kinds of miss, Anthropic's diagnostics chained | **32 of 32** misses Anthropic named, across 6 kinds; on 5 parameter changes Anthropic answered `unavailable` (no verdict) and Prefixwise named the parameter ([1](experiments/e1_hosted/runs/2026-11-06-anthropic-claude-haiku-5-5.md), [2](experiments/e1_hosted/runs/2026-11-06-round4.md)) |
| Names injected causes on open models | 120 turns, 9 causes, Qwen3-4B on vLLM 0.33, vLLM + LMCache, SGLang 0.5.23 | **120 of 120** on each ([vLLM](experiments/e1_open/runs/2026-11-06-colab-l4-qwen3-4b.md), [LMCache and SGLang](experiments/e_evict/runs/2026-11-07-colab-l4.md)) |
| Tells engine eviction from everything else | GPU KV cache flooded 3x; vLLM, SGLang, with and without LMCache / HiCache | evictions **5 of 5**, confirmed from the KV events of vLLM and of SGLang; prefixes restored from CPU counted as hits **5 of 5** ([record](experiments/e_evict/runs/2026-11-07-colab-l4.md), [SGLang events](experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md)) |
| A fix pays what it predicts | `--fix canonical_tools` on 60 paired requests with tool-order drift | **99.9%** (Claude) and **97.5%** (vLLM) of predicted savings realised, task success unchanged ([record](experiments/e4/runs/2026-11-06-live.md)) |
| Works with real agents | `prefixwise run` with Claude Code, Codex CLI, opencode, Anthropic and OpenAI SDKs | all five ([record](experiments/compat/2026-11-07.md), [opencode through `run`](experiments/compat/2026-11-09-opencode-run.md)); opencode building a small CLI: 92% cached, 0 avoidable tokens ([record](experiments/dogfood/2026-11-07-opencode-2.md)) |
| Tells a template's cost from an eviction | multi-turn sessions on Qwen3-4B (vLLM 0.33 and SGLang 0.5.23), a DeepSeek-R1 distill and Qwen2.5 (vLLM), KV events on, plus opencode and Claude Code against vLLM | "not an eviction" **0 of 108** requests; the previous answer recomputed on **47 of 50** Qwen3 turns (25-36% of prompt tokens) and **25 of 25** DeepSeek turns (57%), named `output_rerendered`; reused in full on **25 of 25** Qwen2.5 turns, no detail ([Qwen3](experiments/e_template/runs/2026-11-09-colab-l4-rerun.md), [three templates](experiments/e_template/runs/2026-11-09-colab-l4-models.md)) |
| Costs almost nothing | proxy overhead and throughput, measured on a VM | **3 ms** per request at 2k tokens, 13 ms at 120k; 185 requests/s per process ([record](benchmarks/runs/2026-11-07-vm.md)) |

Not yet measured live: agreement with OpenAI's own miss reason (its diagnostics returned
`unavailable` on every call we made), and TTL expiry and routing misses in real agent traffic.
[Limits](#limits) has the full list.

### Turning on a fix: what Prefixwise predicted, and what it saved

An agent sent the same tools in a drifting order, so every reorder re-paid the whole prefix.
Prefixwise named the cause (`tool_reserialize`) and predicted the saving; turning on
`--fix canonical_tools` then realised almost exactly that, with task success unchanged.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/fix-dark.svg">
  <img alt="Share of prompt tokens served from cache with the fix off and on: Claude Haiku 5.5 from 51% to 97%, Qwen3-4B on vLLM from 88% to 97%" src="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/fix-light.svg" width="720">
</picture>

| | cached, fix off | cached, fix on | predicted saving | realised |
|---|---:|---:|---:|---:|
| Claude Haiku 5.5 | 114,835 of 225,395 (51%) | 219,451 (97%) | 104,732 tokens | 104,616 (**99.9%**) |
| Qwen3-4B on vLLM | 138,928 of 157,706 (88%) | 152,800 (97%) | 14,229 tokens | 13,872 (**97.5%**) |

The saving is larger on Claude because Anthropic puts tools first in the prompt, so a reorder
loses everything after them; Qwen3's template puts tools after the system prompt
([record](experiments/e4/runs/2026-11-06-live.md)). Charts are drawn from the records by
[`docs/figures/make_figures.py`](docs/figures/make_figures.py).

## Why Prefixwise

- **The cause, not the count.** 19 causes, one per miss: the first thing, in prompt order,
  that broke reuse. Each comes with whether it was waste and what to change.
- **Priced.** Avoidable tokens and dollars, from the provider's own usage and a dated price
  table. Join requests to tasks and you get **tokens per successful task**.
- **Evidence, not guesses.** Where the provider explains a miss (Anthropic), Prefixwise checks
  itself against it. Behind vLLM or SGLang it reads the engine's KV-cache events to confirm an
  eviction, or rule it out. When the evidence is missing it says `unknown`.
- **Zero setup.** `prefixwise run -- <your agent>` and read the report when it exits. Or change
  one base URL; responses come back unchanged.
- **Data first.** Every request is a row of facts: an HTTP API (`/api/v1`), JSONL or CSV export,
  and the same numbers in the CLI ([docs/DATA.md](docs/DATA.md)).
- **Private and light.** Hashes are stored, never prompts. Standard library only; Python 3.10+.

## Quickstart

Ways in, each about thirty seconds; all write the same ledger and the same report.

| You have | Do this | Prompts leave your machine? |
|---|---|---|
| An agent you run from a terminal (Claude Code, Codex, opencode, your own) | `prefixwise run -- <your command>` | unchanged: the agent still calls its provider |
| A service, several agents, or a gateway | `prefixwise serve --upstream https://api.anthropic.com` and point clients at it | unchanged |
| Your own vLLM server | `pip install "prefixwise[engines]"`, then `vllm serve <model> ... --middleware prefixwise.PrefixwiseMiddleware`: no proxy, no port change ([docs/VLLM.md](docs/VLLM.md)) | nothing leaves the server |
| Your own Python code | `client = prefixwise.wrap(Anthropic())` (or OpenAI) | unchanged: calls go straight to the provider |
| LiteLLM (SDK or proxy server) | `litellm.callbacks = [PrefixwiseCallback()]` | unchanged |
| Claude Code or Codex on a subscription (no API key) | `prefixwise import claude-code` / `prefixwise import codex` | nothing is sent anywhere: reads the local logs |
| Your CI | `prefixwise check -- pytest tests/agent_scenarios` (or the GitHub Action) | nothing is sent anywhere: a local mock answers |

In every case Prefixwise stores hashes and counts, never the prompt.

```bash
pip install prefixwise
prefixwise run -- claude                     # or any agent: your command after the --
```

`prefixwise run` starts the proxy, points the agent at it (`ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`; the agent keeps its own API key), and when the agent exits prints what that run's cache did, in plain English. If a base URL is a vLLM or SGLang server of your own, it recognises it and reads its KV events, as `serve` does. This is the real output of `prefixwise report` on a ledger from the mock provider (20 requests from four scripted sessions: a date in the system prompt on every turn, tools shuffled on every turn, a well-behaved one, and a Haiku side call before a main thread; the mock answers every request with 2,000 prompt tokens and, like a provider's cache with a breakpoint on the last message, reads all of them when the request continues the previous one with the same tools, system prompt and parameters (on Anthropic, with a cache breakpoint), and on Anthropic writes all of them when it does not, so the token counts are flat and the verdicts, prices and fixes are the real ones):

```
20 requests, 40,000 prompt tokens, 30% served from cache. Cost $0.0666.
Avoidable: 18,000 tokens ($0.0432), 45.0% of prompt tokens. Fix these, biggest first:
  1. The system prompt changed only in a date, time or id (5 requests, 10,000 tokens, $0.0240).
     Fix: Move dates, times and ids out of the system prompt, to the end of the prompt or the latest user message.
  2. The same tools were sent in a different order or format (4 requests, 8,000 tokens, $0.0192).
     Fix: Send tools in a fixed order with deterministic serialization (or let the proxy do it: prefixwise serve --fix canonical_tools).
     Prefixwise can do it for you: prefixwise serve --fix canonical_tools (proven)
Expected misses (not waste): 4 x first request of a session: nothing to reuse yet; 1 x switched model: each model has its own cache.
Cache writes never read: 18,000 tokens were written to the cache, and the next turn of the same conversation replaced the prefix before reading them (9 requests); the write surcharge on them cost $0.0090 above sending them uncached. Fixing the misses above saves this too; it is not counted twice.

Details: prefixwise report --detail   Fixes: prefixwise fixes   Data: prefixwise export
```

`prefixwise report --detail` adds the table behind it:

```
cause             requests  prompt  cached  avoidable  waste reqs  tokens/success     cost  avoidable $
----------------  --------  ------  ------  ---------  ----------  --------------  -------  -----------
volatile_system          5  10,000    0.0%     10,000           5               -  $0.0250      $0.0240
tool_reserialize         4   8,000    0.0%      8,000           4               -  $0.0200      $0.0192
none                     6  12,000  100.0%          0           0               -  $0.0013      $0.0000
first_write              4   8,000    0.0%          0           0               -  $0.0153      $0.0000
model_switch             1   2,000    0.0%          0           0               -  $0.0050      $0.0000
```

On real traffic the same report also says, per Anthropic model, whether the 1-hour cache would have paid. (`prefixwise run --mock -- <agent>` tries it with synthetic responses and no API calls.)

**On a Claude Pro or Max plan?** A proxy cannot sit in front of a subscription login, but Claude Code keeps a log of every session. Read those instead, no setup and nothing sent anywhere:

```bash
prefixwise import claude-code                # reads ~/.claude/projects (subagents included), read-only
```

It reports the same way, plus a line per agent (the main conversation, each subagent type), so subagent cost is visible. The logs record each response's usage (cache reads, and writes kept for 5 minutes or 1 hour), not the prompt itself, so a cause is named only from what the log shows changed in the cached prefix: a model switch, a compaction, a Claude Code upgrade, a changed system prompt, an effort change, an expired cache (`ttl_shortened` when 1 hour was asked and 5 minutes kept), or Anthropic's own diagnosis when present. What Claude Code appends mid-conversation (deferred tools, MCP and skill updates, the date) does not break the cache and is never blamed. Anything else is `unknown`, and says so. Dollars are API prices, what the usage is worth. Rows go to `prefixwise-claude-code.db` (apart from any proxy ledger, so nothing counts twice); run it again later and it adds only new requests.

**Codex, signed in with ChatGPT?** The same, from Codex's own logs: `prefixwise import codex` (reads `~/.codex`). It names a model switch, a compaction, a changed tool catalog, and the same tools sent in another order (waste, priced); OpenAI's cache lifetime is not a fixed number, so other misses stay `unknown`. Checked on real Codex 0.167 logs.

Ctrl-C goes to the agent as usual. If you already set `ANTHROPIC_BASE_URL` or `OPENAI_BASE_URL` (a gateway), Prefixwise forwards there, so the agent reaches the same place. It works with agents that use an **API key**; an agent signed in with a subscription instead cannot be routed through a proxy, and the report says so if the provider rejects it. **opencode** reads `ANTHROPIC_BASE_URL`, so `prefixwise run -- opencode run "..."` needs nothing else (opencode 1.19.5, [record](experiments/compat/2026-11-09-opencode-run.md): 3 requests, a Haiku call then Sonnet, 98.8% cached on the Sonnet turn that followed); it also runs against `prefixwise serve` with the proxy's URL as its provider base URL, without `/v1`. Afterwards: `prefixwise report --session latest`. **Codex** ignores `OPENAI_BASE_URL`, so `prefixwise run -- codex ...` adds `-c` settings that point it at the proxy as a custom provider (plain HTTP, the key from `OPENAI_API_KEY`, no sign-in screen, nothing written to `~/.codex`).

### Gate it in CI

```bash
prefixwise check -- pytest tests/agent_scenarios      # exit 2 on a cache regression, 0 otherwise, 1 if the command fails
```

The costly regressions are code changes that pass every functional test: a tool list built from a set (its order drifts), a timestamp added to the system prompt, a `cache_control` marker moved or dropped, an earlier tool result re-rendered. `prefixwise check` runs your agent's scenarios against a local mock provider (no API calls, nothing leaves the machine), compares each turn with its conversation's previous turn exactly as in production, and fails on `tool_reserialize`, `volatile_system`, `breakpoint_misplaced` and `tool_result_mutation` (`--strict` adds `history_edit`, `system_edit`, `param_rendered`; `--fail-on` names any set). The mock answers a turn that offers tools with a tool call (placeholder arguments the agent must accept; `tool_choice` honoured; never two in a row, so no loop), so an agent's loop runs a second turn, the one the gate judges; it holds each conversation's first response for a second so a per-second timestamp changes between turns, and warns about a date or time it sees in a system prompt that did not change within the run. Under pytest, each test's node id is the scenario name (a bundled plugin), so a finding reads `scenario=tests/test_agent.py::test_refactor`, and `--allow FILE` passes known findings by (scenario, cause, first changed block) while you fix them. With a baseline from production (`prefixwise export --baseline`, read from `.prefixwise-baseline.json`), each finding gets a monthly projection, labelled as the estimate it is; without one, tokens. `--bodies FILE...` judges saved request bodies instead (`prefixwise explain`'s format). As a GitHub Action:

```yaml
- uses: tejachavali/prefixwise@v0.10
  with:
    command: pytest tests/agent_scenarios
```

### Running the proxy yourself

For a long-running proxy, start it and point any OpenAI- or Anthropic-compatible client at it:

```bash
prefixwise serve --mock                      # no API key, synthetic responses; try it first
```


```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-...",
                default_headers={"X-Prefixwise-Task-Id": "fix-bug-42", "X-Prefixwise-Session-Id": "fix-bug-42"})
raw = client.chat.completions.with_raw_response.create(model="gpt-5.6", messages=[...])
print(raw.headers["x-prefixwise-cause"], raw.headers["x-prefixwise-avoidable-tokens"])
```

Then:

```bash
prefixwise task fix-bug-42 --success         # when the task ends (or POST /tasks)
prefixwise report                            # what the cache did, and what to fix (--detail for tables)
prefixwise report --by agent_id              # compare agent / prompt versions per successful task
prefixwise explain examples/session.jsonl    # offline: diagnose a saved session (examples/ is in this repo)
```

For real traffic, drop `--mock` and name the upstream; your client's own API key is passed through:

```bash
prefixwise serve --upstream https://api.openai.com       # OpenAI: /v1/chat/completions, /v1/responses
prefixwise serve --upstream https://api.anthropic.com    # Anthropic: /v1/messages
prefixwise serve --upstream http://127.0.0.1:8000        # vLLM / any OpenAI-compatible server
docker build -t prefixwise . && docker run -p 127.0.0.1:8080:8080 -v pw:/data prefixwise   # mock in Docker
```

### In your own code, or behind LiteLLM: no proxy

Wrap the SDK client you already have. Calls go straight to the provider, unchanged; each completed call is classified and recorded exactly as the proxy would:

```python
import prefixwise
from anthropic import Anthropic                       # or OpenAI(), AsyncOpenAI(), AsyncAnthropic()

client = prefixwise.wrap(Anthropic())                 # ledger: ./prefixwise.db (or PREFIXWISE_LEDGER_PATH)
with prefixwise.session("fix-bug-42", task_id="fix-bug-42"):
    client.messages.create(...)                       # then `prefixwise report`, as above
```

Every call through **LiteLLM** (the SDK or its proxy server), with no change to your calls:

```python
import litellm
from prefixwise.litellm_callback import PrefixwiseCallback
litellm.callbacks = [PrefixwiseCallback()]
```

```yaml
# LiteLLM proxy server, config.yaml
litellm_settings:
  callbacks: prefixwise.litellm_callback.prefixwise_callback
```

Sessions come from LiteLLM's `litellm_session_id`, else `metadata={"session_id": ..., "task_id": ..., "agent_id": ...}`, else `user`. What is classified is the request LiteLLM actually sent the provider, after its translation and with the cache breakpoints it injects; Claude through Anthropic, Vertex AI or Bedrock is judged by Anthropic's rules and prices. Answers from LiteLLM's own response cache reached no provider and are not recorded. `wrap` covers `create` and `parse` (and `with_raw_response`), Anthropic's `beta.messages` too. Both record non-streamed calls for now (streams pass through unrecorded), never change a request (fixes are a proxy feature), and can never break a call: a recording failure is reported once and ignored.

**More: [docs/USE_CASES.md](docs/USE_CASES.md)** walks through eight concrete scenarios: a bill that jumped, a CI gate on prompt changes, A/B-testing prompt versions per successful task, Anthropic TTL choice, self-hosted vLLM, and more.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/architecture-dark.svg">
  <img alt="How a request flows through Prefixwise: four ways in (the proxy, prefixwise.wrap, LiteLLM, agent logs); an edge reads, forwards and fingerprints the request where it is and joins the engine's KV-cache events; a turn record of hashes and counts crosses to a collector, in one process or over HTTP or a Redis Stream partitioned by session, which names one cause and writes one SQLite ledger row; six ways out (Prometheus from the collector, OpenTelemetry from the edge, report, API, export, sinks)" src="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/architecture-light.svg">
</picture>

A request comes in one of four ways (the proxy, `prefixwise.wrap`, the LiteLLM callback, or an agent's own logs). The **edge**, wherever the request is, forwards it as sent (the proxy; only an opted-in `--fix` or the engine join's session tag changes a body) or does not touch it at all (the others), fingerprints its blocks and reads the provider's usage into a **turn record**: hashes, positions and counts, never text. For a self-hosted engine, the edge also joins the request with the engine's KV-cache events and puts the verdict in the record. The **collector** compares the record with the previous turn of its own conversation, names one cause with its evidence, and writes one row to a SQLite ledger. In one process the edge calls the collector directly; in a fleet the record crosses HTTP to `prefixwise collect`, or a Redis Stream partitioned by session so one collector owns each conversation. Six ways out: `prefixwise report`, the HTTP API and `prefixwise export` from the ledger; Prometheus counters from the collector's rows; OpenTelemetry spans from the edge (with the verdict when edge and collector are one process); and sinks that copy every row to JSONL or ClickHouse. The whole flow, part by part: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What it tells you

One cause per request: the first thing, in prefix order, that broke reuse. Each request is compared with the previous turn of **its own conversation, on the same model**, in its session: an agent's small-model side calls (titles, summaries) never mask its main thread, and parallel subagents in one session are each compared with their own earlier turns, not with each other. Waste is avoidable; legitimate changes cost tokens but are not waste.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/decision-dark.svg">
  <img alt="How Prefixwise names a cache miss: a flowchart of the classifier's checks in order, from first turn and changed prompt parts to a hit, engine evidence, retry, expiry, routing, and unknown" src="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/decision-light.svg">
</picture>

| Cause | Meaning | Waste? | Typical fix |
|---|---|---|---|
| `none` | Prefix reused; only new content was uncached | no | - |
| `first_write` | First request of a session | no | - |
| `volatile_system` | Only dates, times, ids or numbers changed in the system prompt (a timestamp in any common format, whatever its format last turn) | **yes** | Move volatile values to the end of the prompt, in one fixed format; the evidence names which kinds the prompt carries, and whether it also carries a file path, which other sessions cannot share |
| `system_edit` | The system prompt's wording changed | depends | Batch prompt edits; avoid per-turn rewording |
| `tool_reserialize` | Same tools, different order or bytes | **yes** | Sort tools; serialize deterministically |
| `tool_catalog_change` | A tool was added or removed | no | Mask tools instead of removing them, if frequent |
| `history_edit` | Earlier messages were rewritten without a compaction | **yes** | Append, do not rewrite history |
| `tool_result_mutation` | An earlier tool result was re-rendered or removed | **yes** | Freeze tool output once sent |
| `compaction_rewrite` | Context was compacted (send `X-Prefixwise-Compaction-Event: 1`) | no | Compact at task boundaries, not per turn |
| `param_rendered` | Blocks equal, but tool_choice / thinking / reasoning / format / chat_template_kwargs changed | depends | Keep those params stable within a session |
| `model_switch` | First call to a different model in the session | no | - |
| `retry_replay` | Identical request within 60 s with nothing cached | **yes** | Check retry logic |
| `breakpoint_misplaced` | Anthropic: no `cache_control` breakpoint, or one that stays early while the stable prompt grows past it (the tail after it is paid uncached every turn) | **yes** | Add `cache_control` after the stable prefix |
| `below_min_prefix` | Prompt below the provider's minimum cacheable length | depends | - |
| `ttl_expired` | Gap since the last turn exceeded the cache TTL | no | Longer TTL (e.g. Anthropic `ttl: "1h"`) if gaps are common |
| `routing_miss` | Prompt and timing fine, still nothing cached | no | Set a stable `prompt_cache_key` / session affinity |
| `fork_boundary` | First turn of a forked session (`X-Prefixwise-Fork-Of`) | no | - |
| `evicted` | Engine evicted this session's blocks | depends | More KV cache memory |
| `unknown` | Not enough evidence; never guessed | - | Send TTL / session headers, or check engine events |

Avoidable tokens for a waste turn = the part of the previous turn's prompt that could have been read from cache but was not: a partial hit counts only the part not read, appended content is never counted, and on Anthropic no more than the previous turn left in the cache (what it read plus what it wrote, up to its last breakpoint) is counted, so a tail that was never cached is not blamed on the change. A change the cache absorbed anyway (it still served the whole reusable prefix) is `none`; one in a request too short to cache, or after the entry had expired, is `below_min_prefix` or `ttl_expired`. The change is kept as a secondary cause either way.

### How it knows, without keeping a prompt

Each block of the request (every tool, the system prompt, every message) gets three hashes, computed in the proxy and kept in memory for the session's next turn ([`fingerprint.py`](src/prefixwise/fingerprint.py)):

| hash | over | equal between two turns means |
|---|---|---|
| `raw` | the bytes as sent | the block is byte-identical: the provider saw the same thing |
| `canonical` | the same content with sorted keys, fixed separators, NFC unicode | the same content, possibly serialized differently |
| `masked` | the canonical form with dates, times, UUIDs, hex ids and numbers replaced by placeholders | the same words; only volatile values differ |

The verdict is the first block, in the provider's prefix order (tools, then system, then messages for Anthropic and OpenAI; system first for a self-hosted engine, whose chat template renders it first), whose hashes disagree, and which of them disagree says why:

| what differs | verdict | why it is proof |
|---|---|---|
| tools: `raw` differs, the multiset of `canonical` hashes is equal | `tool_reserialize` | the same tools reached the provider in different bytes or order; nothing else could have produced that pattern |
| tools: the multiset of `canonical` hashes differs | `tool_catalog_change` | a tool was added, removed or edited |
| system: `raw` differs, `masked` equal | `volatile_system` | every character that changed was inside a date, time, id or number |
| system: `masked` differs | `system_edit` | the wording changed |
| messages: the previous turn's blocks are not a prefix of this turn's | `history_edit` (or `compaction_rewrite` with a compaction event) | earlier messages were rewritten, not appended to (a single-message previous request replaced by a new conversation is judged by what the cache read instead) |
| tool results: an earlier one differs or is gone | `tool_result_mutation` (or `compaction_rewrite`) | a tool's output was re-rendered or removed after it was sent |

Short blocks (under about 256 bytes, such as a four-word user message) would be guessable from a bare hash, so with a per-install key (`PREFIXWISE_HMAC_KEY`) they are keyed hashes (HMAC) instead. The ledger stores the verdict, the role and index of the first changed block, counts and prices; no hash leaves the process, and exports never include them.


**The why behind the cause.** Where the evidence allows, a miss also gets a detail, for the ways agents break the cache in practice (from Claude Code, Codex, opencode and LiteLLM issue reports in 2026). Each is claimed only from facts in hand: the request bodies, the provider's usage, what this proxy saw in flight.

| Detail | Under | What happened | What to do |
|---|---|---|---|
| `ttl_shortened` | `ttl_expired` | Every breakpoint asked for a 1-hour cache; Anthropic's usage shows the write was kept for 5 minutes, and more than 5 minutes passed | Check what downgrades the TTL (client version, settings, gateway) |
| `lookback_exceeded` | `breakpoint_misplaced` | Prompt provably unchanged, gap within 5 minutes, but the previous write is 22+ positions back (Anthropic checks 20; a run of tool calls or results is one) | Add a breakpoint at the start of each turn |
| `markers_ignored` | any miss | `cache_control` on the last block, prompt of 8,192+ tokens, yet nothing written or read | Check that no gateway strips the markers, and that the endpoint caches |
| `sent_before_cache_written` | first request | Sent within a second of another request (same model, key, tools, system) still waiting for its response | Stagger parallel requests (subagents, fan-outs) or warm the prefix first |
| `volatile_prefix_at_start` | first request | Another session (a restart, or another user of the same prompt) completed a request moments earlier with the same model, key and tools and a system prompt that differs only in a date, time or id (a timestamp set at session start, a per-user id): it recomputed a prefix it could have shared. Says which kind differed. Any provider | Move the per-session value out of the system prompt to the end of the prompt; a timestamp can also be set once per day |
| `shared_prefix_not_reused` | first request | Anthropic: another session had just written an entry for the same model, key, tools and system prompt, within reach of a breakpoint here, yet little was read | Check cache keys, TTLs and breakpoint placement between the two |
| `thinking_stripped` | `history_edit` | On a model that keeps earlier thinking (Opus 4.5+, Sonnet 4.6+, Haiku 5.5, Fable, Mythos), the only change is removed thinking blocks | Keep thinking blocks in history (client or gateway setting) |
| `cache_key_changed` | routing, expiry | OpenAI `prompt_cache_key` changed between turns | Keep one key per conversation |
| `host_restored` | `none` (a hit) | Behind one self-hosted engine with `--kv-events`: blocks of the previous request were evicted from the GPU, yet this request read them (vLLM CPU offload, LMCache, SGLang HiCache restored them without recomputing) | A hit that cost a restore: the GPU cache alone is too small for the session; size it, or keep the offload tier |
| `output_rerendered` | `none` (a hit) | Behind a self-hosted engine (vLLM, SGLang): the read covered the previous prompt and stopped there, so none of the previous turn's output (160+ tokens) was reused, although the earlier messages are unchanged. The chat template renders history differently from how it was generated (Qwen3 drops reasoning), or the client did not resend it. Live: on a DeepSeek-R1 distill, all 25 turns (the previous answer 57% of prompt tokens); on Qwen3, the output was recomputed on 47 of 50 turns (25-36%) and the detail, added after that run, would have named 29; on Qwen2.5, whose template keeps history as generated, none ([Qwen3](experiments/e_template/runs/2026-11-09-colab-l4-rerun.md), [three templates](experiments/e_template/runs/2026-11-09-colab-l4-models.md)) | The template's cost, not the client's: choose a model whose template keeps history as generated, or budget for it |

Each detail is claimed only when all of its conditions hold; otherwise there is none. `prefixwise report` lists them under each cause; `/api/v1/breakdown?by=detail` and `?detail=` slice by them.

## Reference

**CLI**

| Command | |
|---|---|
| `prefixwise serve [--upstream URL] [--mock] [--optimize] [--fix canonical_tools] [--engine vllm\|sglang\|none] [--kv-events ENDPOINT\|auto] [--full-attention] [--collector URL] [--sink URL] [--otel] [--host H] [--port P] [--ledger FILE]` | Run the proxy. `--engine`/`--kv-events`/`--full-attention`: see Self-hosted engines (detected from `--upstream` when not given; `none` for a router or a hosted API). `--collector`, `--sink`: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Ledger defaults to `./prefixwise.db`. `--optimize` applies every proven fix; `--fix` applies one by name, opt-in and reversible (see below). |
| `prefixwise run [--mock] [--optimize] [--fix NAME] [--engine vllm\|sglang\|none] [--full-attention] [--ledger FILE] [--port P] [--otel] -- <agent>` | Run an agent through Prefixwise and report on that run when it exits. A vLLM or SGLang behind the agent's base URL is detected (its KV events too); `--engine` says it explicitly. |
| `prefixwise check [--fail-on CAUSES] [--strict] [--allow FILE] [--baseline FILE] [--json] [--json-file FILE] [--first-turn-delay S] -- <command>` / `check --bodies FILE...` | CI gate: run the agent's scenarios through a local mock and exit 2 on a cache regression that is a code bug (tool order, a timestamp in the system prompt, a dropped breakpoint, a mutated tool result). `prefixwise export --baseline` writes the production baseline for its monthly projection. |
| `prefixwise prices [--json]` | The dated price table behind dollar costs, and where each rate came from. |
| `prefixwise prune --older-than 30d` | Delete ledger entries older than that (retention). Add `--include-undated` to also drop rows written before 0.3.1. |
| `prefixwise export [--format jsonl\|csv] [--since 7d] [--where session=s1] [--baseline]` | Every ledger row (facts, never prompt text) for a spreadsheet, notebook or warehouse. |
| `prefixwise fixes [--json]` | The fixes that would save the most, ranked by the avoidable tokens your own ledger attributes to each cause. |
| `prefixwise report [--by cause\|session_id\|agent_id\|model\|provider\|task_id] [--session ID\|latest] [--share] [--json]` | Totals and a breakdown, most avoidable first. `--session`: one session turn by turn: prompt and cached tokens, hit rate, the cause and detail, the block where the prompt first changed, as role[position in the whole prompt: tools, then system, then the conversation] (`tools[0]`, `messages[9]`), the request's cost, avoidable tokens and dollars. Hashes and counts only. |
| `prefixwise task ID --success\|--failure` | Record a task outcome (safe while the proxy runs). |
| `prefixwise request REQUEST_ID [--json]` | One request's ledger entry (from `X-Prefixwise-Request-Id`). |
| `prefixwise explain A.json B.json` / `explain SESSION.jsonl [--json] [--fail-on-waste] [--diff]` | Offline diagnosis of saved request bodies, or `{"request": ..., "usage": ..., "session_id": ..., "gap_seconds": ...}` records. |

**HTTP endpoints:** `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages` (streaming supported on all three; the same paths without `/v1` also work, and query strings are forwarded). Any other path, such as `GET /v1/models` or `count_tokens`, passes through to the upstream unchanged and is not recorded. Also: `POST /tasks {"task_id", "success"}`, `GET /requests/<id>`, `GET /engine` (the KV-event stream's health), `GET /ledger`, `GET /metrics` (Prometheus), `GET /health`.

**Request headers** (all optional; stripped before forwarding):

| Header | Effect |
|---|---|
| `X-Prefixwise-Session-Id` | Groups turns into a session (also: `session_id` body field, or Anthropic `metadata.user_id`). Without it, all traffic is one session. |
| `X-Prefixwise-Task-Id` | Joins the request to a task for tokens per successful task (also: `task_id` body field). |
| `X-Prefixwise-Agent-Id` | Labels the agent or prompt version, for `report --by agent_id`. |
| `X-Prefixwise-Compaction-Event: 1` | This turn follows a context compaction. |
| `X-Prefixwise-Fork-Of: <session>` | This session was forked from another. |
| `X-Prefixwise-Cache-Ttl: <seconds>` | The cache lifetime, when known. |

**Response headers:** `X-Prefixwise-Request-Id` always; `X-Prefixwise-Cause`, `X-Prefixwise-Legit`, `X-Prefixwise-Avoidable-Tokens` on non-streaming responses; `X-Prefixwise-Fixes` when a fix rewrote the request. Streamed requests are classified when the stream ends: read them from `GET /requests/<id>`.

**Environment** (the proxy; the middleware's own variables are in [docs/VLLM.md](docs/VLLM.md)): `PREFIXWISE_UPSTREAM_URL`, `PREFIXWISE_API_KEY` (overrides client auth; `x-api-key` for Anthropic), `PREFIXWISE_LEDGER_PATH`, `PREFIXWISE_HMAC_KEY` (HMAC short blocks so their hashes cannot be dictionary-guessed), `PREFIXWISE_MAX_SESSIONS` (default 10,000), `PREFIXWISE_FIXES` (e.g. `canonical_tools`, or `optimize` for every proven fix), `PREFIXWISE_AUTH_TOKEN` (require `X-Prefixwise-Token`), `PREFIXWISE_MAX_BODY_BYTES`, `PREFIXWISE_PRICES` (your own rates file), `PREFIXWISE_CLIENT_TIMEOUT` (default 120 s), `PREFIXWISE_DRAIN_SECONDS` (default 8), `PORT`.

**Dollar costs.** Each request is priced from the provider's own usage breakdown (uncached input, cache reads, cache writes by TTL, output), using a dated price table: `prefixwise prices` shows it. *Avoidable cost* is the avoidable tokens times (the rate the request paid for its uncached prompt tokens − the cache-read price): on a miss those tokens are usually written to the cache again, at the cache-write rate (Anthropic: 1.25x input for 5 minutes, 2x for 1 hour), and a hit would have read them. `prefixwise report` also says, per Anthropic model, whether the 1-hour cache would have paid on your traffic. A model with no rates shows `-`, never an estimate. To add other models or your negotiated rates, write a JSON file in the same format (`prefixwise prices --json` prints one) and set `PREFIXWISE_PRICES=/path/prices.json`. Your entries override the built-in ones. `prefixwise report` and `prefixwise fixes` then show dollars, and say how many requests were priced.

**Fixes the proxy can apply.** `canonical_tools` sorts tools by name and sorts each tool's keys before forwarding, so a tool list that drifts between turns (plugin reload, dict rebuild) stops breaking the cache. It is off by default; with it off, requests pass through unchanged. `--optimize` (also `PREFIXWISE_OPTIMIZE=1` for the vLLM middleware) turns on every proven fix at once, so a new one joins without a flag change. `--fix auto_breakpoints` (opt-in, Anthropic) adds `cache_control` markers to a request that has none, at the end of tools and system, on the last message, and on the previous user turn when the history has grown past Anthropic's 20-block lookback, only once the same prefix was seen within the cache window, so a one-shot request never pays for a write nobody reads; a client's own markers are never touched. It changes the bill (a write costs 1.25x), so it stays out of `--optimize` until a replay shows the writes pay; meanwhile `prefixwise fixes` says whether the markers it placed have paid for themselves on your traffic (tokens written against tokens read back). Prefixwise classifies the request the provider actually received. Fixes that would change what the model reads, like moving a timestamp out of the system prompt, are advice only: `prefixwise fixes` lists them. `canonical_tools` is **proven**: in [E4](experiments/e4/runs/2026-11-06-live.md), turning it on realised 99.9% (Anthropic Haiku 5.5) and 97.5% (vLLM Qwen3-4B) of the savings Prefixwise predicted, with task success flat.

## Data and API

Every number Prefixwise shows is also data. Each request leaves a row of facts (verdict and its evidence, token breakdown, cost, latency, status; failed requests too), and every metric is defined once and served three ways: `prefixwise report`, the HTTP API at `/api/v1` (summary, breakdowns by cause, session, model, day..., request rows, fixes, prices, explain; [docs/API.md](docs/API.md)), and `prefixwise export`. `GET /api/v1/metrics` lists every metric and what it means. Design: [docs/DATA.md](docs/DATA.md).

**Prometheus and Grafana:** `GET /metrics` exports requests, tokens, avoidable tokens and dollars by provider, model and miss cause, plus Prefixwise's own latency and the KV-event stream's health. Import the Grafana dashboard ([`deploy/grafana/prefixwise.json`](deploy/grafana/prefixwise.json)) and the alert rules (hit rate below 80%, avoidable share above 10%, failures, lost KV events: [`deploy/prometheus/prefixwise-rules.yml`](deploy/prometheus/prefixwise-rules.yml)). Setup and queries: [docs/prometheus.md](docs/prometheus.md).

**OpenTelemetry:** `PREFIXWISE_OTEL=1` (or `prefixwise serve --otel`) exports a span per request, following OpenTelemetry's GenAI semantic conventions (`chat {model}`, `gen_ai.usage.cache_read.input_tokens` and `cache_write.input_tokens`), with Prefixwise's verdict on it: `prefixwise.cache.cause`, `.detail`, `.avoidable_tokens`, `.avoidable_cost_usd`. Works the same from the proxy, `prefixwise.wrap` and the LiteLLM callback. Inside your app, spans go to the tracer provider your app sets up (Prefixwise never installs one there); `serve` and `run` export OTLP themselves, configured by the standard `OTEL_*` variables (`pip install 'prefixwise[otel]'`). Through the proxy, a `traceparent` header puts the span in the caller's trace and is passed on upstream. Already tracing your LLM calls? `PREFIXWISE_OTEL=cause` emits a separate span with only Prefixwise's attributes, so tokens are not counted twice. Details: [docs/opentelemetry.md](docs/opentelemetry.md).

## Security

- **What it stores:** the ledger keeps fingerprints of prompt blocks (hashes), token counts, causes, and your session, task and agent labels. It never stores prompt text or API keys. The ledger file is created readable by its owner only. Set `PREFIXWISE_HMAC_KEY` so hashes of short, guessable blocks are keyed.
- **Where it listens:** `127.0.0.1` by default. To listen beyond localhost (a team server, or Docker with a published port), set `PREFIXWISE_AUTH_TOKEN`; clients then send it as `X-Prefixwise-Token`, and every endpoint but `/health` requires it. With `PREFIXWISE_API_KEY` set, the proxy refuses to listen beyond localhost without a token, because anyone who reached the port could spend your key. Publish Docker ports on `127.0.0.1` (`-p 127.0.0.1:8080:8080`).
- **Keys:** the client's own auth headers pass through to the upstream untouched and are never logged. `X-Prefixwise-*` headers are never forwarded.
- **Limits:** request bodies over 32 MiB (`PREFIXWISE_MAX_BODY_BYTES`) get a 413, and malformed requests get a 400 naming the field.

Found a vulnerability? See [SECURITY.md](SECURITY.md).

## Self-hosted engines: vLLM, SGLang, LMCache

Inside vLLM itself, with no proxy: `vllm serve <model> ... --middleware prefixwise.PrefixwiseMiddleware` loads Prefixwise as ASGI middleware in vLLM's own API server (KV events found from the engine, with the `engines` extra installed; `PREFIXWISE_LEDGER`, `PREFIXWISE_ENGINE`, `PREFIXWISE_KV_EVENTS`, `PREFIXWISE_FIX`, `PREFIXWISE_FULL_ATTENTION`, `PREFIXWISE_COLLECTOR` to configure; [docs/VLLM.md](docs/VLLM.md)).

Step-by-step guides, with what each verdict means and the live numbers: **[docs/VLLM.md](docs/VLLM.md)** (vLLM, KV events, several caches), **[docs/LMCACHE.md](docs/LMCACHE.md)** (vLLM with an offload tier: which hits LMCache rescued, which misses were the agent's), **[docs/SGLANG.md](docs/SGLANG.md)** (SGLang, HiCache, data-parallel ranks).

Prefixwise reads the engine's `cached_tokens` like any provider's: start vLLM with `--enable-prompt-tokens-details`, SGLang with `--enable-cache-report`. It recognises the engine behind `--upstream` at startup (vLLM's `/version`, SGLang's `/server_info`) and, when the engine publishes KV events and the `engines` extra is installed (`pip install "prefixwise[engines]"`), subscribes to them (`--kv-events auto`); `--engine` and `--kv-events` are the explicit form, and `--engine none` says a URL is a router, not one engine:

```bash
prefixwise serve --upstream http://127.0.0.1:8000                    # detected: vLLM or SGLang, KV events if published
prefixwise serve --upstream http://127.0.0.1:8000 --engine vllm      # or --engine sglang, said explicitly
```

One engine means one cache and no routing, so a request whose prefix is unchanged but was not reused is named `evicted` (the engine ran out of GPU memory). That verdict is **inferred**, and the evidence says so: a chat template that renders history differently, or data-parallel routing, looks the same.

**What a chat template costs, model by model.** The same 30-turn harness (5 variants x 6 turns, KV events on) on four templates, Colab L4 ([Qwen3](experiments/e_template/runs/2026-11-09-colab-l4-rerun.md), [the other three](experiments/e_template/runs/2026-11-09-colab-l4-models.md)):

| model | template keeps history as generated? | reuse on the 25 later turns | previous answer prefilled again | Prefixwise |
|---|---|---|---|---|
| Qwen3-4B, vLLM 0.33 | no: drops `<think>` from earlier turns | prompt-only 23, full 2 (after truncated thinking) | 25% of prompt tokens | `none`; `output_rerendered` on 14 (the floor leaves short answers out) |
| Qwen3-4B, SGLang 0.5.23 | no | prompt-only 24, full 1 | 36% | `none`; `output_rerendered` on 15 |
| DeepSeek-R1-Distill-Qwen-1.5B, vLLM | no: drops everything up to `</think>` | prompt-only 25 | 57% (17,987 tokens) | `none` with `output_rerendered` on all 25, live |
| Qwen2.5-3B-Instruct, vLLM | yes | full 25 | 0 | `none`, no detail |
| Llama-3.2-3B-Instruct | yes (template alone: vLLM did not fit its 131k context on the L4) | full, by template | 0 | the date in its system header changes the prefix once a day |

"Not an eviction" (a prefix that changed although the request did not): 0 on every model, 108 requests judged. A truncated answer (no closing `</think>`) is kept whole as history by both reasoning templates and paid for on every later turn.

**vLLM: confirm it with KV events.** vLLM can tag the blocks each request uses with a session, so Prefixwise can check the block where the cache read stopped: removed means a confirmed eviction, still on the GPU means the prompt's tokens changed there:

```bash
pip install "prefixwise[engines]"     # pyzmq + msgspec; the core stays stdlib-only
vllm serve MODEL --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'
prefixwise serve --upstream http://127.0.0.1:8000 --engine vllm --kv-events tcp://127.0.0.1:5557
```

The verdict becomes `evicted` **confirmed** ("the engine removed N of this session's blocks"), or `unknown` with "not an eviction: the engine still holds this session's blocks" when the cause lies elsewhere. `GET /engine` shows the event stream's health.

How the join works ([`engines.py`](src/prefixwise/engines.py)): the events name each stored and removed KV block by the engine's own block hash, tagged with the request's session. When a session's request is forwarded, Prefixwise freezes the block chain its previous request used; at the response it finds the block where the cache read stopped (`cached_tokens / block_size` into that chain) and asks the events what happened to it. Removed before the response: a confirmed eviction. Still resident: not an eviction. Events arrive on their own socket, so before any negative claim Prefixwise waits (up to 50 ms) until the event stream has caught up with the response time, follows the publisher's sequence numbers (a gap means a removal may have been missed, so nothing is claimed; a restart means an empty cache), and judges nothing for concurrent requests of one session. With several caches, the events also say which cache served each request, which is how a routing miss is confirmed: served by B while A, which served the previous turn, still held the very block this turn needed.

**Several caches (vLLM data-parallel ranks, or replicas behind a router): routing misses, confirmed.** Each rank and each replica has its own cache, so a session whose turns land on different ones misses although nothing changed. `--kv-events auto` subscribes to every data-parallel rank (from vLLM's `/kv_event_sources`); behind a router, list each replica's stream: `--kv-events tcp://replica-a:5557,tcp://replica-b:5557`. The events themselves show which cache served each request, so the router only has to pass the request body through: llm-d, production-stack, AIBrix and the current SGLang model gateway do; Dynamo and the older SGLang gateway copy do not ([details](docs/ENGINES.md#several-kv-caches-vllm-data-parallel-ranks-replicas-behind-a-router)). A turn served by another cache while the previous one still held the very block it needed is a **confirmed** `routing_miss`; with several caches known, an eviction is never just inferred. How, and when nothing is claimed: [docs/ENGINES.md](docs/ENGINES.md#several-kv-caches-vllm-data-parallel-ranks-replicas-behind-a-router). Live on two vLLM 0.33.0 replicas behind a router that never says which one it used (Colab L4): routing misses confirmed 5/5, and 5/5 hits with sticky routing, no false claim ([record](experiments/e_route/runs/2026-11-08-colab-l4.md)).

**SGLang (0.5.20 or later): the same.** SGLang also writes a request's `session_id` on the blocks it stores, so Prefixwise rebuilds each session's chain from SGLang's events:

```bash
python -m sglang.launch_server --model-path MODEL --enable-cache-report \
  --kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:5557"}'
prefixwise serve --upstream http://127.0.0.1:30000 --engine sglang --kv-events tcp://127.0.0.1:5557 --full-attention
```

Live on SGLang 0.5.23 (Qwen3-4B, L4): evictions confirmed 5 of 5 ([record](experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md)).

`--full-attention` says the model has no sliding-window or Mamba layers (Qwen3, Llama, most dense models). SGLang frees those layers without an event, so without the flag a missing block still confirms an eviction, but Prefixwise never says "not an eviction". At startup Prefixwise reads SGLang's `/server_info` (with `PREFIXWISE_API_KEY` if SGLang has `--api-key`) and judges only when it is safe: SGLang 0.5.20 or later, events on, no `--enable-session-radix-cache`; otherwise it says why and keeps the inference. With several data-parallel ranks, `--kv-events auto` follows every rank (from `/server_info`), and with `--full-attention` a turn sent to another rank while the previous one still held the very block it needed is a confirmed `routing_miss`, as on vLLM (not with prefill/decode disaggregation, or ranks across several nodes). Memory: about 80 bytes per token in SGLang's GPU cache at its default `--page-size 1` (80 MB per million tokens; `--page-size 16` cuts it 16-fold).

**LMCache and HiCache need nothing extra.** A prefix restored from CPU memory counts as a hit (vLLM's `cached_tokens` includes tokens loaded from LMCache), so only prefixes lost from every tier are evictions; with `--kv-events` such a hit carries the detail `host_restored` (the GPU copy was evicted and the tier loaded it back), so rescued turns are one query apart from GPU hits: live 5/5 on LMCache 0.5.7 and on SGLang HiCache ([record](experiments/e_evict/runs/2026-11-08-colab-l4-host-restored.md); [docs/LMCACHE.md](docs/LMCACHE.md), [docs/SGLANG.md](docs/SGLANG.md)). Why it works this way, the alternatives considered and the known limits (several replicas, lost events): [docs/ENGINES.md](docs/ENGINES.md).


**What it looks like on real engines.** A session's prefix, after flooding the GPU's KV cache
with three times its capacity ([record](experiments/e_evict/runs/2026-11-07-colab-l4.md)):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/evict-dark.svg">
  <img alt="Prefix served from cache after a GPU flood: vLLM 0% (evicted, confirmed from KV events), vLLM + LMCache 98% (hit, restored from CPU), SGLang 0% (evicted, confirmed from KV events), SGLang + HiCache 99% (hit, restored from host memory)" src="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/evict-light.svg" width="720">
</picture>

| engine | prefix cached after the flood | correct cause | Prefixwise |
|---|---:|---|---|
| vLLM | 0 of 3,404 | evicted | `evicted` 5 of 5; with KV events, confirmed ("211 of the previous request's KV blocks removed") |
| vLLM + LMCache | 3,328 of 3,404 | none (restored from CPU) | `none` 5 of 5 |
| SGLang | 3 of 3,404 | evicted | `evicted` 5 of 5 (inferred); with KV events, confirmed ("3,382 of the previous request's KV blocks removed", [record](experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md)) |
| SGLang + HiCache | 3,382 of 3,404 | none (restored from host memory) | `none` 5 of 5 |

## Running in production

Measured by [`benchmarks/bench.py`](benchmarks/README.md) (results in [`benchmarks/runs/`](benchmarks/runs/)); rerun it on your own hardware.

- **Added latency:** about 3 ms per request at a 2k-token prompt, 6 ms at 30k, 13 ms at 120k (p50; p95 under 22 ms). Streaming adds the same to time to first byte. A model call takes seconds, so this is well under 1%.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/overhead-dark.svg">
  <img alt="Latency Prefixwise adds per request: median 2.7 ms at 2k prompt tokens, 5.6 ms at 30k, 13.3 ms at 120k; p95 3.8, 7.5 and 18.8 ms" src="https://raw.githubusercontent.com/tejachavali/prefixwise/main/docs/figures/overhead-light.svg" width="720">
</picture>

  | prompt | added p50 | added p95 | added p99 |
  |---|---:|---:|---:|
  | 2k tokens | 2.7 ms | 3.8 ms | 8.0 ms |
  | 30k tokens | 5.6 ms | 7.5 ms | 8.8 ms |
  | 120k tokens | 13.3 ms | 18.8 ms | 27.0 ms |

  OpenAI request shape, not streamed; the cost grows with the prompt because each block is hashed.
- **Live, against real providers** (records: [Anthropic](benchmarks/runs/2026-11-07-live-anthropic.md), [OpenAI](benchmarks/runs/2026-11-07-live-openai.md)): 20 pairs each, direct vs. through the proxy, from Colab. Cache hit share identical both ways (Anthropic Haiku 5.5: 98.1% and 98.1%; OpenAI GPT-5.6 Luna: 97.9% and 97.9%), every request recorded. Median added latency +29 ms and +43 ms on ~1 s calls, which is within the network variance between two separate calls; the proxy's own cost is 3-6 ms.
- **Capacity:** one proxy process handles about 185 requests/s of agent-sized traffic before CPU-bound, with no errors up to 128 concurrent clients. An agent sends roughly one request every few seconds, so one process serves hundreds of agents at once. Session state lives in the process: to run several, route each session to the same instance (e.g. hash `X-Prefixwise-Session-Id`). Fleet-scale deployment (pods, ledgers, KV events): [docs/specs/hyperscale.md](docs/specs/hyperscale.md).
- **Memory:** about 11 KB per tracked session, capped by `PREFIXWISE_MAX_SESSIONS` (default 10,000: roughly 100-150 MB), plus a bounded cache of block hashes (about 8 MB). In a 15-minute soak (148,000 requests, 0 errors) RSS went 60 to 74 MB; Python-level memory is flat once the caches fill (tracemalloc: 16.2 MB at both 2,400 and 4,800 sessions), so the remainder is the allocator keeping freed request buffers.
- **It fails open.** If the ledger cannot be written (disk full, locked) or classification fails, the client still gets the upstream response, with `X-Prefixwise-Error: not recorded`; the proxy logs why to stderr. Upstream errors (429, 500, ...) reach the client unchanged, with `retry-after`, `x-should-retry`, rate-limit headers and the provider's request id, so SDK retries behave as without the proxy. An unreachable or broken upstream is a 502.
- **Streams:** usage is recorded for streamed responses too. For OpenAI and vLLM the proxy adds `stream_options.include_usage` when the client did not, and strips the extra usage chunk before it reaches that client. Anthropic's `message_start` and `message_delta` usage are merged. A stream the client cancels is recorded as a failure (status 499, `client_disconnected`) with no verdict, and the session's history is left as it was, so the next turn is compared with the last complete one.
- **Timeouts:** a client connection idle for `PREFIXWISE_CLIENT_TIMEOUT` seconds (default 120) is closed, so idle or half-sent connections cannot pin threads. Upstream calls time out after 600 s (long generations stream for minutes).
- **Shutdown:** on SIGTERM (`docker stop`, systemd, Kubernetes) the proxy stops accepting connections, lets in-flight requests and streams finish for up to `PREFIXWISE_DRAIN_SECONDS` (default 8), then exits 0.
- **Ledger growth and retention:** about 450 bytes per request (a million requests is about 450 MB); `prefixwise report` takes about 2 s at a million rows. Delete old rows with `prefixwise prune --older-than 30d`, e.g. from a daily cron. Back it up while the proxy runs with `sqlite3 prefixwise.db ".backup backup.db"`.
- **Several proxies, one ledger:** run the control plane once, `prefixwise collect --port 8081` (set `PREFIXWISE_AUTH_TOKEN`), and start each proxy as an edge: `prefixwise serve --collector http://collector:8081` with `PREFIXWISE_COLLECTOR_TOKEN`. An edge hashes requests where they are and sends one small record per turn (hashes, positions and counts, never text) to the collector, which compares each turn with its conversation's previous one and writes the one ledger `prefixwise report` and `/api/v1` read. The collector holds the conversation state, so matching needs no session affinity at the ingress; for a conversation's turns to arrive in order, let them take the same edge (an edge sends in order; across edges a backed-up sender can deliver a turn late, which the collector then names `unknown` rather than mis-classifying), and keep the edges' clocks in sync (`PREFIXWISE_CLOCK_SKEW`, default 1 s, is the tolerance). The same two variables turn `prefixwise.wrap` and the LiteLLM callback into edges. The sender never delays a request: a collector that is down is retried, then records are dropped and counted (`/health`, `prefixwise_records_dropped_total`); the queue holds 10,000 records, about a minute of one busy edge's traffic, so an outage longer than that loses records, visibly. Scripts using `prefixwise.wrap` flush their last records at exit (`Recorder.close`). **Several collectors:** put a Redis Stream between them, `prefixwise collect --log redis://redis:6379/0 --partitions 0-7` (and `8-15` on another), with edges started as `PREFIXWISE_COLLECTOR=redis://redis:6379/0` (`pip install 'prefixwise[log]'`). Records are partitioned by session, so all of a conversation's turns reach the one collector that owns it, in the order they reached the log; a collector that restarts takes over what a dead predecessor left unfinished, replays the last day to find its state again, and writes no row twice and loses none (stop the old collector before the new one starts: `strategy: Recreate`). Needs Redis 7.0 or later. **Warehouse:** `--sink clickhouse://user:pass@host:8123/db.table` or `--sink file:///path/rows.jsonl` gives a copy of every row (`docs/DATA.md` has the table). Design: [docs/specs/hyperscale.md](docs/specs/hyperscale.md).
- **Not built yet:** request logging (the proxy logs only errors; metrics: `GET /metrics`, traces: OpenTelemetry, above).

## Limits

- **Not verified yet:**
  - **OpenAI native agreement.** OpenAI's `prompt_cache_diagnostics` returned `unavailable` for every comparison on gpt-5.6, -luna and -sol. That held with turn delays of 1 s, 30 s and 120 s, and on a minimal standalone call with a completed baseline. OpenAI's dashboard shows 0 diagnosed responses. Prefixwise still classifies OpenAI traffic; only the cross-check against OpenAI's own verdict is missing.
  - **Causes E1 does not inject:** TTL expiry and tool-result mutation (eviction and routing misses are verified live, below).
  - **Real agent traffic at scale.** opencode and Claude Code have run through it ([compat](experiments/compat/2026-11-07.md), [against vLLM](experiments/e_template/runs/2026-11-09-colab-l4-rerun.md)); long sessions and many agents at once have not.
- **Self-hosted engines:** with `--engine`, an engine eviction is inferred from one cache and no routing; on vLLM and SGLang, `--kv-events` confirms it or rules it out. Validated live on vLLM 0.33 and SGLang 0.5.23 (Colab L4): evictions named 5/5, confirmed with KV events on both ([vLLM](experiments/e_evict/runs/2026-11-07-colab-l4.md), [SGLang](experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md)), LMCache and HiCache restores counted as hits 5/5 ([record](experiments/e_evict/runs/2026-11-07-colab-l4.md)). Token-level divergence measured live: 0 "not an eviction" in 108 requests (Qwen3-4B on vLLM and SGLang, a DeepSeek-R1 distill and Qwen2.5 on vLLM, opencode and Claude Code against vLLM); what the runs did show is the previous answer prefilled again every turn on reasoning templates (`output_rerendered`; [Qwen3](experiments/e_template/runs/2026-11-09-colab-l4-rerun.md), [three templates](experiments/e_template/runs/2026-11-09-colab-l4-models.md)). Several vLLM caches (ranks, replicas behind a router): routing misses confirmed from KV events, live 5/5 on two vLLM replicas ([record](experiments/e_route/runs/2026-11-08-colab-l4.md)); data-parallel ranks inside one server (vLLM and SGLang, `--kv-events auto`) not yet run live.
- **Divergence is located per block, not per token.** An edit at the very end of a long system prompt is reported as `system_edit`, even if the provider could have reused everything before it (a design for locating it inside the block: [docs/specs/hyperscale.md](docs/specs/hyperscale.md#21-chunk-sketches-designed-not-built)). In one live OpenAI call, appending three words to 2,100 tokens of instructions gave 0 cached tokens, which looks more like routing than the edit itself.
- One proxy process keeps session state in memory: run one instance per agent fleet, or send all of a session's turns to the same instance.
- OpenAI's default in-memory cache TTL is not a fixed number, so after a long gap a cold OpenAI cache is `unknown` unless you send `X-Prefixwise-Cache-Ttl`.
- **Dollar costs:** only models in the dated price table are priced: Claude Haiku, Sonnet and Opus 5.5, Fable 5.1 and Sonnet 4.6 (Anthropic's pricing page, 2026-11-06/07), and OpenAI GPT-5.6 Luna, Terra and Sol (OpenAI's pricing page, 2026-11-07; Sol's promotional price expires after 2026-11-30). Everything else shows `-` until you add rates with `PREFIXWISE_PRICES`. OpenAI cache writes are priced at the 30-minute rate. Batch, fast-mode and data-residency multipliers are not modelled.

## Working on Prefixwise

Agents and people working on this repo: read [`WORKLOG.md`](WORKLOG.md) first. It has the current goal, why, what to run, and what happens with each result. `build.log` records outcomes.

The flowcharts are drawn by code: when the request flow or the classifier's order changes, update `docs/figures/make_architecture.py`, run it, look at the result, and commit the SVGs. CI fails if a committed figure differs from what its script draws, or if the decision flowchart leaves out a cause the classifier can return (`tests/test_figures.py`).

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests -q                              # about 490 tests, 2 min
python experiments/e1_open/e1_open.py --mock           # harness smoke test
```

## Design notes

How it is built today (parts, request flow, how a cause is decided, data, rules): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
The original research and architecture, kept as written: [docs/DESIGN.md](docs/DESIGN.md).
Current designs: [docs/DATA.md](docs/DATA.md) (data and metrics), [docs/API.md](docs/API.md),
[docs/ENGINES.md](docs/ENGINES.md) (vLLM and SGLang engine join).

## License

MIT. See LICENSE.
