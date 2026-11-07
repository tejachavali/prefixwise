# Prefixwise

**Find out why your LLM prompt cache missed, what it cost, and how many tokens each successful task really takes.**

Prefixwise is a local proxy and CLI that sits between your agent and the model API (OpenAI, Anthropic, or a self-hosted vLLM). For every request it names the cause of any cache miss (a timestamp in the system prompt, a reordered tool list, a re-rendered tool result, an expired TTL...), says whether that miss was waste or a legitimate change, and prices it in tokens. Join requests to tasks and it reports the number that matters: **tokens per successful task**.

- Drop-in: change your SDK's base URL. Responses come back unchanged; the verdict rides in headers.
- Offline too: `prefixwise explain` diagnoses saved request bodies with no proxy and no API key.
- Standard library only. Hashes are stored, never prompts.

```text
$ prefixwise report
Requests            1,284
Prompt tokens       9,412,330   (cache hit rate 61.0%)
Completion tokens   402,118
Avoidable tokens    2,106,904   (22.4% of prompt tokens)
Tasks               212 succeeded / 240 resolved
Tokens per successful task  46,293

cause                 requests     prompt  cached  avoidable  waste reqs  tokens/success
--------------------  --------  ---------  ------  ---------  ----------  --------------
volatile_system            301  2,288,600   12.0%  1,766,040         301          48,110
tool_result_mutation        88    702,000   55.1%    251,320          88          47,904
none                       790  5,925,000   84.0%          0           0          44,980
...
```
<sub>Illustrative output: a timestamp in the system prompt was costing a fifth of all prompt tokens.</sub>

## Quickstart

```bash
pip install prefixwise
prefixwise serve --mock                      # no API key, synthetic responses; try it first
```

Point any OpenAI- or Anthropic-compatible client at it:

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
prefixwise report                            # where the tokens went
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

**More: [docs/USE_CASES.md](docs/USE_CASES.md)** walks through eight concrete scenarios: a bill that jumped, a CI gate on prompt changes, A/B-testing prompt versions per successful task, Anthropic TTL choice, self-hosted vLLM, and more.

## What it tells you

One cause per request: the first thing, in prefix order, that broke reuse. Each request is compared with the previous request **to the same model** in its session, so an agent's small-model side calls (titles, summaries) never mask what is happening on its main thread. Waste is avoidable; legitimate changes cost tokens but are not waste.

| Cause | Meaning | Waste? | Typical fix |
|---|---|---|---|
| `none` | Prefix reused; only new content was uncached | no | - |
| `first_write` | First request of a session | no | - |
| `volatile_system` | Only dates, times, ids or numbers changed in the system prompt | **yes** | Move volatile values to the end of the prompt |
| `system_edit` | The system prompt's wording changed | depends | Batch prompt edits; avoid per-turn rewording |
| `tool_reserialize` | Same tools, different order or bytes | **yes** | Sort tools; serialize deterministically |
| `tool_catalog_change` | A tool was added or removed | no | Mask tools instead of removing them, if frequent |
| `history_edit` | Earlier messages were rewritten without a compaction | **yes** | Append, do not rewrite history |
| `tool_result_mutation` | An earlier tool result was re-rendered or removed | **yes** | Freeze tool output once sent |
| `compaction_rewrite` | Context was compacted (send `X-Prefixwise-Compaction-Event: 1`) | no | Compact at task boundaries, not per turn |
| `param_rendered` | Blocks equal, but tool_choice / thinking / reasoning / format changed | depends | Keep those params stable within a session |
| `model_switch` | First call to a different model in the session | no | - |
| `retry_replay` | Identical request within 60 s with nothing cached | **yes** | Check retry logic |
| `breakpoint_misplaced` | Anthropic: no `cache_control` breakpoint | **yes** | Add `cache_control` after the stable prefix |
| `below_min_prefix` | Prompt below the provider's minimum cacheable length | depends | - |
| `ttl_expired` | Gap since the last turn exceeded the cache TTL | no | Longer TTL (e.g. Anthropic `ttl: "1h"`) if gaps are common |
| `routing_miss` | Prompt and timing fine, still nothing cached | no | Set a stable `prompt_cache_key` / session affinity |
| `fork_boundary` | First turn of a forked session (`X-Prefixwise-Fork-Of`) | no | - |
| `evicted` | Engine evicted this session's blocks | depends | More KV cache memory |
| `unknown` | Not enough evidence; never guessed | - | Send TTL / session headers, or check engine events |

Avoidable tokens for a waste turn = the part of the previous turn's prompt that could have been read from cache but was not. Appended content is never counted.

## Reference

**CLI**

| Command | |
|---|---|
| `prefixwise serve [--upstream URL] [--mock] [--fix canonical_tools] [--engine vllm\|sglang] [--kv-events ENDPOINT] [--host H] [--port P] [--ledger FILE]` | Run the proxy. `--engine`/`--kv-events`: see Self-hosted engines. Ledger defaults to `./prefixwise.db`. `--fix` applies an opt-in, reversible rewrite to every request (see below). |
| `prefixwise prices [--json]` | The dated price table behind dollar costs, and where each rate came from. |
| `prefixwise prune --older-than 30d` | Delete ledger entries older than that (retention). Add `--include-undated` to also drop rows written before 0.3.1. |
| `prefixwise fixes [--json]` | The fixes that would save the most, ranked by the avoidable tokens your own ledger attributes to each cause. |
| `prefixwise report [--by cause\|session_id\|agent_id\|model\|provider\|task_id] [--json]` | Totals and a breakdown, most avoidable first. |
| `prefixwise task ID --success\|--failure` | Record a task outcome (safe while the proxy runs). |
| `prefixwise request REQUEST_ID [--json]` | One request's ledger entry (from `X-Prefixwise-Request-Id`). |
| `prefixwise explain A.json B.json` / `explain SESSION.jsonl [--json] [--fail-on-waste]` | Offline diagnosis of saved request bodies, or `{"request": ..., "usage": ..., "session_id": ..., "gap_seconds": ...}` records. |

**HTTP endpoints:** `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages` (streaming supported on all three; the same paths without `/v1` also work, and query strings are forwarded). Any other path, such as `GET /v1/models` or `count_tokens`, passes through to the upstream unchanged and is not recorded. Also: `POST /tasks {"task_id", "success"}`, `GET /requests/<id>`, `GET /ledger`, `GET /health`.

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

**Environment:** `PREFIXWISE_UPSTREAM_URL`, `PREFIXWISE_API_KEY` (overrides client auth; `x-api-key` for Anthropic), `PREFIXWISE_LEDGER_PATH`, `PREFIXWISE_HMAC_KEY` (HMAC short blocks so their hashes cannot be dictionary-guessed), `PREFIXWISE_MAX_SESSIONS` (default 10,000), `PREFIXWISE_FIXES` (e.g. `canonical_tools`), `PREFIXWISE_AUTH_TOKEN` (require `X-Prefixwise-Token`), `PREFIXWISE_MAX_BODY_BYTES`, `PREFIXWISE_PRICES` (your own rates file), `PREFIXWISE_CLIENT_TIMEOUT` (default 120 s), `PREFIXWISE_DRAIN_SECONDS` (default 8), `PORT`.

**Dollar costs.** Each request is priced from the provider's own usage breakdown (uncached input, cache reads, cache writes by TTL, output), using a dated price table: `prefixwise prices` shows it. *Avoidable cost* is the avoidable tokens times (input price − cache-read price): what they would have cost as cache hits. A model with no rates shows `-`, never an estimate. To add other models or your negotiated rates, write a JSON file in the same format (`prefixwise prices --json` prints one) and set `PREFIXWISE_PRICES=/path/prices.json`. Your entries override the built-in ones. `prefixwise report` and `prefixwise fixes` then show dollars, and say how many requests were priced.

**Fixes the proxy can apply.** `canonical_tools` sorts tools by name and sorts each tool's keys before forwarding, so a tool list that drifts between turns (plugin reload, dict rebuild) stops breaking the cache. It is off by default; with it off, requests pass through unchanged. Prefixwise classifies the request the provider actually received. Fixes that would change what the model reads, like moving a timestamp out of the system prompt, are advice only: `prefixwise fixes` lists them. `canonical_tools` is **proven**: in [E4](experiments/e4/runs/2026-11-06-live.md), turning it on realised 99.9% (Anthropic Haiku 5.5) and 97.5% (vLLM Qwen3-4B) of the savings Prefixwise predicted, with task success flat.

## Security

- **What it stores:** the ledger keeps fingerprints of prompt blocks (hashes), token counts, causes, and your session, task and agent labels. It never stores prompt text or API keys. The ledger file is created readable by its owner only. Set `PREFIXWISE_HMAC_KEY` so hashes of short, guessable blocks are keyed.
- **Where it listens:** `127.0.0.1` by default. To listen beyond localhost (a team server, or Docker with a published port), set `PREFIXWISE_AUTH_TOKEN`; clients then send it as `X-Prefixwise-Token`, and every endpoint but `/health` requires it. With `PREFIXWISE_API_KEY` set, the proxy refuses to listen beyond localhost without a token, because anyone who reached the port could spend your key. Publish Docker ports on `127.0.0.1` (`-p 127.0.0.1:8080:8080`).
- **Keys:** the client's own auth headers pass through to the upstream untouched and are never logged. `X-Prefixwise-*` headers are never forwarded.
- **Limits:** request bodies over 32 MiB (`PREFIXWISE_MAX_BODY_BYTES`) get a 413, and malformed requests get a 400 naming the field.

Found a vulnerability? See [SECURITY.md](SECURITY.md).

## Self-hosted engines: vLLM, SGLang, LMCache

Prefixwise reads the engine's `cached_tokens` like any provider's: start vLLM with `--enable-prompt-tokens-details`, SGLang with `--enable-cache-report`. Hits an offload tier restores from CPU memory (LMCache on vLLM, HiCache on SGLang) count as hits.

**Engine join.** Give Prefixwise the engine's KV-cache event stream and it can tell an eviction (the engine ran out of GPU memory) from a routing miss:

```bash
pip install "prefixwise[engines]"                 # adds pyzmq + msgspec; the core stays stdlib-only
vllm serve MODEL --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'
prefixwise serve --upstream http://127.0.0.1:8000 --engine vllm --kv-events tcp://127.0.0.1:5557
```

- **vLLM** (with or without LMCache): `--engine vllm` sends Prefixwise's session id in the request body, vLLM stamps it on the blocks it stores, and a later `BlockRemoved` of those blocks names that session's next miss `evicted` with a block count.
- **SGLang:** start it with `--kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:5557"}'` and use `--engine sglang`. SGLang reads a body `session_id` as its own session handle, so Prefixwise does not send one; the evidence is engine-wide ("the engine evicted N blocks since this session's last turn") and says so.
- **LMCache / HiCache:** a GPU eviction that the offload tier restored is a hit, so it stays `none`; only removals from GPU count as evictions.
- `GET /engine` shows the stream: batches, blocks stored and removed, dropped messages. With several engine replicas, run one Prefixwise per replica (or route sessions to the same one).

## Running in production

Measured by [`benchmarks/bench.py`](benchmarks/README.md) (results in [`benchmarks/runs/`](benchmarks/runs/)); rerun it on your own hardware.

- **Added latency:** about 3 ms per request at a 2k-token prompt, 6 ms at 30k, 13 ms at 120k (p50; p95 under 22 ms). Streaming adds the same to time to first byte. A model call takes seconds, so this is well under 1%.
- **Live, against real providers** (records: [Anthropic](benchmarks/runs/2026-11-07-live-anthropic.md), [OpenAI](benchmarks/runs/2026-11-07-live-openai.md)): 20 pairs each, direct vs. through the proxy, from Colab. Cache hit share identical both ways (Anthropic Haiku 5.5: 98.1% and 98.1%; OpenAI GPT-5.6 Luna: 97.9% and 97.9%), every request recorded. Median added latency +29 ms and +43 ms on ~1 s calls, which is within the network variance between two separate calls; the proxy's own cost is 3-6 ms.
- **Capacity:** one proxy process handles about 185 requests/s of agent-sized traffic before CPU-bound, with no errors up to 128 concurrent clients. An agent sends roughly one request every few seconds, so one process serves hundreds of agents at once. Session state lives in the process: to run several, route each session to the same instance (e.g. hash `X-Prefixwise-Session-Id`).
- **Memory:** about 11 KB per tracked session, capped by `PREFIXWISE_MAX_SESSIONS` (default 10,000: roughly 100-150 MB), plus a bounded cache of block hashes (about 8 MB). In a 15-minute soak (148,000 requests, 0 errors) RSS went 60 to 74 MB; Python-level memory is flat once the caches fill (tracemalloc: 16.2 MB at both 2,400 and 4,800 sessions), so the remainder is the allocator keeping freed request buffers.
- **It fails open.** If the ledger cannot be written (disk full, locked) or classification fails, the client still gets the upstream response, with `X-Prefixwise-Error: not recorded`; the proxy logs why to stderr. Upstream errors (429, 500, ...) reach the client unchanged, with `retry-after`, `x-should-retry`, rate-limit headers and the provider's request id, so SDK retries behave as without the proxy. An unreachable or broken upstream is a 502.
- **Timeouts:** a client connection idle for `PREFIXWISE_CLIENT_TIMEOUT` seconds (default 120) is closed, so idle or half-sent connections cannot pin threads. Upstream calls time out after 600 s (long generations stream for minutes).
- **Shutdown:** on SIGTERM (`docker stop`, systemd, Kubernetes) the proxy stops accepting connections, lets in-flight requests and streams finish for up to `PREFIXWISE_DRAIN_SECONDS` (default 8), then exits 0.
- **Ledger growth and retention:** about 450 bytes per request (a million requests is about 450 MB); `prefixwise report` takes about 2 s at a million rows. Delete old rows with `prefixwise prune --older-than 30d`, e.g. from a daily cron. Back it up while the proxy runs with `sqlite3 prefixwise.db ".backup backup.db"`.
- **Not built yet:** shared session state across processes, a metrics endpoint, and request logging (the proxy logs only errors).

## Limits

- **What has been verified live (2026-11-06):**
  - **E1-open, vLLM:** Qwen3-4B on vLLM 0.33.0, Colab L4. 120 turns, 9 injected causes including append-only growth. Agreement 100%, UNKNOWN 0%. See [`experiments/e1_open/runs/`](experiments/e1_open/runs/).
  - **E1-hosted, Anthropic:** claude-haiku-5-5. Across two runs, on 37 misses diagnosed by Anthropic's own API (compaction, model switch, parameter change, system edit, tool catalog change, tool reorder, volatile system), Prefixwise named a cause Anthropic's reason covers 37 out of 37 times. All 60 cache hits were classified as hits. UNKNOWN 0%. See [`experiments/e1_hosted/runs/`](experiments/e1_hosted/runs/).
  - **E4, fixes:** `canonical_tools` realised 99.9% (Anthropic) and 97.5% (vLLM) of its predicted savings with task success flat. On Qwen3-4B, 1 task of 60 changed; on Anthropic, none. See [`experiments/e4/runs/`](experiments/e4/runs/).
- **Not verified yet:**
  - **OpenAI native agreement.** OpenAI's `prompt_cache_diagnostics` returned `unavailable` for every comparison on gpt-5.6, -luna and -sol. That held with turn delays of 1 s, 30 s and 120 s, and on a minimal standalone call with a completed baseline. OpenAI's dashboard shows 0 diagnosed responses. Prefixwise still classifies OpenAI traffic; only the cross-check against OpenAI's own verdict is missing.
  - **Causes E1 does not inject:** TTL expiry, routing, eviction, tool-result mutation.
  - **Real agent traffic,** which is messier than scripted turns.
- **Self-hosted engines:** without `--kv-events`, an engine eviction reads as `routing_miss`, because nothing in the response distinguishes the two; with it, as `evicted`. On SGLang the evidence is engine-wide (blocks cannot be attributed to sessions there). The engine join has been tested against vLLM's wire format, not yet on a live engine.
- **Divergence is located per block, not per token.** An edit at the very end of a long system prompt is reported as `system_edit`, even if the provider could have reused everything before it. In one live OpenAI call, appending three words to 2,100 tokens of instructions gave 0 cached tokens, which looks more like routing than the edit itself.
- One proxy process keeps session state in memory: run one instance per agent fleet, or send all of a session's turns to the same instance.
- `evicted` needs vLLM KV events, which the proxy does not consume yet.
- OpenAI's default in-memory cache TTL is not a fixed number, so after a long gap a cold OpenAI cache is `unknown` unless you send `X-Prefixwise-Cache-Ttl`.
- **Dollar costs:** only models in the dated price table are priced: Claude Haiku, Sonnet and Opus 5.5, Fable 5.1 and Sonnet 4.6 (Anthropic's pricing page, 2026-11-06/07), and OpenAI GPT-5.6 Luna, Terra and Sol (OpenAI's pricing page, 2026-11-07; Sol's promotional price expires after 2026-11-30). Everything else shows `-` until you add rates with `PREFIXWISE_PRICES`. OpenAI cache writes are priced at the 30-minute rate. Batch, fast-mode and data-residency multipliers are not modelled.

## Working on Prefixwise

Agents and people working on this repo: read [`WORKLOG.md`](WORKLOG.md) first. It has the current goal, why, what to run, and what happens with each result. `build.log` records outcomes.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests -q                              # 75 tests, about 10 s
python experiments/e1_open/e1_open.py --mock           # harness smoke test
```

## Design notes

The research and architecture behind Prefixwise.

### What is missing elsewhere

Every provider and engine reports how many tokens missed. None, on its own, names the app-side cause across providers, and none joins spend to task success.

- OpenAI and Anthropic now report a native miss reason (OpenAI Responses, GPT-5.6+, 9 reasons; Anthropic, 6 reasons, GA Sep 23, 2026), but only the earliest divergence, in a coarse taxonomy, and neither names TTL expiry, routing, or eviction.
- vLLM and SGLang report hit rates and per-request cached tokens, not per-request causes.
- LMCache sees chunk hashes, not sessions, agents, or tasks.
- Gemini has no miss diagnostics, hit count only.
- Single-provider cause classification already exists (CacheCanary for Bedrock Claude, Copilot Cost Analyzer for Copilot Chat). The unbuilt part is the cross-provider, cross-harness version, joined to the engine, priced per successful task.

### Architecture (v2, verified Nov 6, 2026)

Design confidence 85/100, proven 79/100. The rise from v1 (79/100) is earned by the verified v2 corrections; 90 and 99 need the E1-E5 experiments below, with running code against real traffic. Full survey and verification: `../repo-survey-and-architecture.md` and `../v2-verification-check.md`.

#### 1. Capture: proxy first, hooks second

Only the proxy sees the exact bytes on the wire, the bytes the provider actually hashes. Hooks see what the harness constructed; the provider may re-serialize before hashing. Hooks (LangGraph middleware, OpenAI Agents SDK, Claude Agent SDK including PreCompact, CrewAI) supply session, agent, and task identity the proxy cannot see. OTel is third, roll-up only.

The proxy also auto-chains native diagnostics: Anthropic `diagnostics.previous_message_id` (no beta header, GA Sep 23, 2026) and OpenAI `prompt_cache_options.comparison_response_id`. Neither changes caching. The provider reason is a partial agreement signal, not ground truth.

#### 2. Fingerprinting: raw, canonical, and masked hashes

Three hashes per block, in prefix order. Order is per provider, not universal: Anthropic is tools, system, messages; OpenAI injects tools before developer instructions.

- `raw`: exact bytes as sent. TOOL_RESERIALIZE is defined as canonical-equal, raw-different, so without raw it cannot be seen.
- `canonical`: sorted keys, fixed separators, NFC unicode.
- `masked`: canonical bytes with volatile values masked (dates, times, ids, numbers). A system block whose raw hash changed but whose masked hash is equal is VOLATILE_SYSTEM, not SYSTEM_EDIT.

Hashes are stored, never raw prompts, by default. A bare SHA-256 of a short or predictable block is dictionary-guessable, so when an HMAC key is supplied, every hash of a short block is an HMAC with the per-install key, replacing bare SHA-256 entirely for that block. The short-block threshold (default 64 tokens, approximate) is a policy choice, not a verified fact. Soft tokens are fingerprinted by type and count only, never hashed as text.

#### 3. Classifier: 19 causes, one per miss, legitimacy separate

First divergence in prefix order wins; the rest are recorded as secondary. UNKNOWN when evidence is absent, never a guess. Legitimacy is a separate column: a legitimate change still costs tokens, but it is not waste.

`FIRST_WRITE`, `MODEL_SWITCH`, `VOLATILE_SYSTEM`, `SYSTEM_EDIT`, `TOOL_RESERIALIZE`, `TOOL_CATALOG_CHANGE`, `COMPACTION_REWRITE`, `HISTORY_EDIT`, `TOOL_RESULT_MUTATION`, `FORK_BOUNDARY`, `PARAM_RENDERED`, `BELOW_MIN_PREFIX`, `BREAKPOINT_MISPLACED`, `TTL_EXPIRED`, `EVICTED`, `ROUTING_MISS`, `RETRY_REPLAY`, `NONE`, `UNKNOWN`

`PARAM_RENDERED` is the case Anthropic reports as `unavailable`: blocks equal, but tool_choice, thinking, context_management, output_config, output_format, or beta headers differ. Minimum cacheable prefix is model-dependent: OpenAI 1,024 tokens, Anthropic 512-4,096, Gemini 3.x 4,096 (2,048 on 2.5).

Two scoring metrics ship with the classifier: agreement (share of turns where Prefixwise's cause maps onto the native reason when both exist) and coverage (share of misses not UNKNOWN).

#### 4. Engine join: request id, not hash recomputation

vLLM >= v0.30.0 (Sep 22, 2026, PR #51381) stamps `session_id` onto every `BlockStored` KV event; the proxy sets `X-Request-Id` and `session_id`. LMCache `--lookup-hash-log-dir` logs the same request id with chunk hashes. SGLang accepts `rid` and `session_id` and, with `return_cached_tokens_details`, reports cached tokens split by device, host, and storage tier. Join on request id; time-window fallback is flagged low confidence only. Hash recomputation stays out: vLLM 16-token blocks, LMCache 256-token chunks, and SGLang radix trees cannot be reproduced from one hash.

The production-stack `LookupMsg` preflight is NOT claimed to work against current LMCache: its import path (`lmcache.v1.cache_controller.message`) 404s at LMCache 012965f (real path `lmcache.inprocess.v1.cache_controller.message`), and the pinned version is unverified.

#### 5. Compaction: frozen boundary, rare and timed rewrites

"Append-only compaction" as written in v1 is incoherent: any compaction that shrinks the window rewrites the prefix after the boundary by definition. The replacement: never touch the system/tool prefix; compact only when the window is near its limit or the cache would expire anyway; prefer task boundaries over per-turn compaction; snapshot the pre-compaction fingerprint at the PreCompact hook and price the rewrite exactly once.

#### 6. Fix toggles and the ledger

Toggles are opt-in and reversible: canonical tool ordering, volatile fields out of the prefix, breakpoint placement, TTL selection. Each ships as "recommended" on predicted savings and graduates to "proven" only after realised savings are measured with task success flat (realised >= 50% of predicted over 50 requests). The reuse-adjusted saving (k = write multiplier / (1 - read multiplier)) is labelled as a model, not a verified fact, because the read multiplier is model-dependent.

Headline metric: tokens per successful task, with cost as a secondary column. Rate tables are versioned snapshots with dates; if a rate is unknown, cost is null, never invented.

#### 7. Distribution: local first

The OSS core is the ledger, diffing, and one engine join, as a local CLI and proxy. Fleet roll-up is hosted and optional, redacted entries only, never raw prompts.

### What Prefixwise will not build

Not a cost dashboard, not a prompt optimiser or compressor, not a router or gateway, not a second observability platform, not a hosted-provider feature. It cannot fix OpenAI's per-machine routing or Anthropic's automatic-caching opacity; it records those misses as ROUTING_MISS and moves on.

### E1-open: classifier validation without API credits

E1-open validates the classifier on open Hugging Face models served by vLLM, with controlled injected causes as ground truth. No Anthropic or OpenAI API credits needed. See `experiments/e1_open/README.md`.

```bash
# Mock mode (no GPU, no server): scoring path smoke test only
PYTHONPATH=src python3 experiments/e1_open/e1_open.py --mock

# Live, on a GPU box (vLLM >= 0.30.0; Qwen3-8B fits on one 24 GB GPU)
MODEL=Qwen/Qwen3-8B bash experiments/e1_open/vllm_server.sh
PYTHONPATH=src python3 experiments/e1_open/e1_open.py \
  --vllm-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B \
  --model-alias Qwen/Qwen3-8B-alias

# Or on Colab (L4 GPU, Qwen3-4B default): open experiments/e1_open/colab_e1_open.ipynb
```

120 turns, 9 injected causes, including two append-only sessions where every turn grows the conversation (FIRST_WRITE, NONE, VOLATILE_SYSTEM, TOOL_RESERIALIZE, TOOL_CATALOG_CHANGE, SYSTEM_EDIT, COMPACTION_REWRITE, RETRY_REPLAY, MODEL_SWITCH). Pass: agreement >= 95%, UNKNOWN <= 10% of misses. Mock mode scores 100% by construction (simulated cached_tokens); only the live run is evidence about vLLM.

Live run Nov 5-6, 2026 (Colab L4, vLLM 0.33.0, Qwen3-4B): agreement 90.0%, Pass NO, only retry_replay failed (0/10). Two fixes since: the server now starts with `--enable-auto-tool-choice --tool-call-parser hermes` (vLLM 0.33.0 rejected tool-carrying requests with HTTP 400 without them), and retry turns use a fresh upstream session + fresh `cache_salt` instead of POST `/reset_prefix_cache` (HTTP 404 on vLLM 0.33.0). Qwen3-8B failed to start on a Colab L4 in that run, so 4B is the Colab default; 8B needs a larger GPU.

### Build order (E1 first)

1. **E1-open: passed live 2026-11-06** (Qwen3-4B, vLLM 0.33.0; 100%). **E1-hosted: Anthropic passed 2026-11-06** (37/37 native agreement over two runs); OpenAI diagnostics returned `unavailable` throughout. Original plan: 120 turns on vLLM + Qwen3-8B with injected causes as ground truth; agreement >= 95%, UNKNOWN <= 10%. No API credits. Then **E1-hosted:** 100 consecutive turns on Anthropic and OpenAI with native diagnostics chained; Prefixwise's first-diverged block agrees with the native reason >= 95% of the time; UNKNOWN <= 10% of misses on real coding-agent traffic. Needs API credits and is still the provider-facing proof.
2. **E2/E3:** vLLM >= v0.30.0 + LMCache on one GPU box, ids set by the proxy; >= 95% of requests join on ids within 5 s; lookup-hash lines join on request id for >= 99% of requests.
3. **E4: passed live 2026-11-06** (`canonical_tools`: 99.9% of predicted savings realised on Anthropic, 97.5% on vLLM; task success flat). Original plan: one toggle (canonical tools or volatile-to-tail) on a recorded agent session replayed live; realised cache_read >= 50% of predicted over 50 requests; task success flat.
4. **E5:** PreCompact snapshot on one harness; rewrite priced exactly once; compaction frequency reduced to task boundaries without a drop in a 10-task pass rate.
5. **SGLang adapter** (rid, session_id, return_cached_tokens_details).
6. Hosted fleet roll-up, only if anyone asks.

## License

MIT. See LICENSE.
