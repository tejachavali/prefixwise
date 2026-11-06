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
pip install "git+https://github.com/tejachavali/prefixwise"
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
docker build -t prefixwise . && docker run -p 8080:8080 -v pw:/data prefixwise   # mock in Docker
```

**More: [docs/USE_CASES.md](docs/USE_CASES.md)** walks through eight concrete scenarios: a bill that jumped, a CI gate on prompt changes, A/B-testing prompt versions per successful task, Anthropic TTL choice, self-hosted vLLM, and more.

## What it tells you

One cause per request: the first thing, in prefix order, that broke reuse. Waste is avoidable; legitimate changes cost tokens but are not waste.

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
| `model_switch` | Different model than the previous turn | no | - |
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
| `prefixwise serve [--upstream URL] [--mock] [--host H] [--port P] [--ledger FILE]` | Run the proxy. Ledger defaults to `./prefixwise.db`. |
| `prefixwise report [--by cause\|session_id\|agent_id\|model\|provider\|task_id] [--json]` | Totals and a breakdown, most avoidable first. |
| `prefixwise task ID --success\|--failure` | Record a task outcome (safe while the proxy runs). |
| `prefixwise request REQUEST_ID [--json]` | One request's ledger entry (from `X-Prefixwise-Request-Id`). |
| `prefixwise explain A.json B.json` / `explain SESSION.jsonl [--json] [--fail-on-waste]` | Offline diagnosis of saved request bodies, or `{"request": ..., "usage": ..., "session_id": ..., "gap_seconds": ...}` records. |

**HTTP endpoints:** `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages` (streaming supported on all three), `POST /tasks {"task_id", "success"}`, `GET /requests/<id>`, `GET /ledger`, `GET /health`.

**Request headers** (all optional; stripped before forwarding):

| Header | Effect |
|---|---|
| `X-Prefixwise-Session-Id` | Groups turns into a session (also: `session_id` body field, or Anthropic `metadata.user_id`). Without it, all traffic is one session. |
| `X-Prefixwise-Task-Id` | Joins the request to a task for tokens per successful task (also: `task_id` body field). |
| `X-Prefixwise-Agent-Id` | Labels the agent or prompt version, for `report --by agent_id`. |
| `X-Prefixwise-Compaction-Event: 1` | This turn follows a context compaction. |
| `X-Prefixwise-Fork-Of: <session>` | This session was forked from another. |
| `X-Prefixwise-Cache-Ttl: <seconds>` | The cache lifetime, when known. |

**Response headers:** `X-Prefixwise-Request-Id` always; `X-Prefixwise-Cause`, `X-Prefixwise-Legit`, `X-Prefixwise-Avoidable-Tokens` on non-streaming responses. Streamed requests are classified when the stream ends: read them from `GET /requests/<id>`.

**Environment:** `PREFIXWISE_UPSTREAM_URL`, `PREFIXWISE_API_KEY` (overrides client auth; `x-api-key` for Anthropic), `PREFIXWISE_LEDGER_PATH`, `PREFIXWISE_HMAC_KEY` (HMAC short blocks so their hashes cannot be dictionary-guessed), `PREFIXWISE_MAX_SESSIONS` (default 10,000), `PORT`.

## Limits

- **Validated in mock and unit tests, not yet against live traffic.** The live E1 runs (vLLM, then OpenAI and Anthropic with native diagnostics) are the next step; see below.
- One proxy process keeps session state in memory: run one instance per agent fleet, or send all of a session's turns to the same instance.
- `evicted` needs vLLM KV events, which the proxy does not consume yet.
- OpenAI's default in-memory cache TTL is not a fixed number, so after a long gap a cold OpenAI cache is `unknown` unless you send `X-Prefixwise-Cache-Ttl`.
- Costs are not priced yet (`avoidable_cost` is null): rates change, and a guessed rate is worse than none.

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

# Or on Colab (L4 GPU): open experiments/e1_open/colab_e1_open.ipynb
```

120 turns, 9 injected causes, including two append-only sessions where every turn grows the conversation (FIRST_WRITE, NONE, VOLATILE_SYSTEM, TOOL_RESERIALIZE, TOOL_CATALOG_CHANGE, SYSTEM_EDIT, COMPACTION_REWRITE, RETRY_REPLAY, MODEL_SWITCH). Pass: agreement >= 95%, UNKNOWN <= 10% of misses. Mock mode scores 100% by construction (simulated cached_tokens); only the live run is evidence about vLLM.

### Build order (E1 first)

1. **E1-open (next):** 120 turns on vLLM + Qwen3-8B with injected causes as ground truth; agreement >= 95%, UNKNOWN <= 10%. No API credits. Then **E1-hosted:** 100 consecutive turns on Anthropic and OpenAI with native diagnostics chained; Prefixwise's first-diverged block agrees with the native reason >= 95% of the time; UNKNOWN <= 10% of misses on real coding-agent traffic. Needs API credits and is still the provider-facing proof.
2. **E2/E3:** vLLM >= v0.30.0 + LMCache on one GPU box, ids set by the proxy; >= 95% of requests join on ids within 5 s; lookup-hash lines join on request id for >= 99% of requests.
3. **E4:** one toggle (canonical tools or volatile-to-tail) on a recorded agent session replayed live; realised cache_read >= 50% of predicted over 50 requests; task success flat.
4. **E5:** PreCompact snapshot on one harness; rewrite priced exactly once; compaction frequency reduced to task boundaries without a drop in a 10-task pass rate.
5. **SGLang adapter** (rid, session_id, return_cached_tokens_details).
6. Hosted fleet roll-up, only if anyone asks.

## Legal note

General information only, not legal advice. For a California employee whose team builds agentic AI cost analytics, Cal. Labor Code §2870 likely does not protect this side project even if built on own time and equipment. Get California employment counsel and written disclosure before production use or incorporating.

## License

MIT. See LICENSE.
