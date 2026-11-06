# Prefixwise

A model-agnostic ledger of avoidable agent token spend.

Prefixwise names *why* a prompt cache missed, prices the miss, and proves whether fixing it keeps task success flat. Efficiency is tokens per successful task. Cost is one output of that, not the product.

## What is missing elsewhere

Every provider and engine reports how many tokens missed. None, on its own, names the app-side cause across providers, and none joins spend to task success.

- OpenAI and Anthropic now report a native miss reason (OpenAI Responses, GPT-5.6+, 9 reasons; Anthropic, 6 reasons, GA Sep 23, 2026), but only the earliest divergence, in a coarse taxonomy, and neither names TTL expiry, routing, or eviction.
- vLLM and SGLang report hit rates and per-request cached tokens, not per-request causes.
- LMCache sees chunk hashes, not sessions, agents, or tasks.
- Gemini has no miss diagnostics, hit count only.
- Single-provider cause classification already exists (CacheCanary for Bedrock Claude, Copilot Cost Analyzer for Copilot Chat). The unbuilt part is the cross-provider, cross-harness version, joined to the engine, priced per successful task.

## Architecture (v2, verified Nov 6, 2026)

Design confidence 85/100, proven 79/100. The rise from v1 (79/100) is earned by the verified v2 corrections; 90 and 99 need the E1-E5 experiments below, with running code against real traffic. Full survey and verification: `../repo-survey-and-architecture.md` and `../v2-verification-check.md`.

### 1. Capture: proxy first, hooks second

Only the proxy sees the exact bytes on the wire, the bytes the provider actually hashes. Hooks see what the harness constructed; the provider may re-serialize before hashing. Hooks (LangGraph middleware, OpenAI Agents SDK, Claude Agent SDK including PreCompact, CrewAI) supply session, agent, and task identity the proxy cannot see. OTel is third, roll-up only.

The proxy also auto-chains native diagnostics: Anthropic `diagnostics.previous_message_id` (no beta header, GA Sep 23, 2026) and OpenAI `prompt_cache_options.comparison_response_id`. Neither changes caching. The provider reason is a partial agreement signal, not ground truth.

### 2. Fingerprinting: raw, canonical, and masked hashes

Three hashes per block, in prefix order. Order is per provider, not universal: Anthropic is tools, system, messages; OpenAI injects tools before developer instructions.

- `raw`: exact bytes as sent. TOOL_RESERIALIZE is defined as canonical-equal, raw-different, so without raw it cannot be seen.
- `canonical`: sorted keys, fixed separators, NFC unicode.
- `masked`: canonical bytes with volatile values masked (dates, times, ids, numbers). A system block whose raw hash changed but whose masked hash is equal is VOLATILE_SYSTEM, not SYSTEM_EDIT.

Hashes are stored, never raw prompts, by default. A bare SHA-256 of a short or predictable block is dictionary-guessable, so when an HMAC key is supplied, every hash of a short block is an HMAC with the per-install key, replacing bare SHA-256 entirely for that block. The short-block threshold (default 64 tokens, approximate) is a policy choice, not a verified fact. Soft tokens are fingerprinted by type and count only, never hashed as text.

### 3. Classifier: 19 causes, one per miss, legitimacy separate

First divergence in prefix order wins; the rest are recorded as secondary. UNKNOWN when evidence is absent, never a guess. Legitimacy is a separate column: a legitimate change still costs tokens, but it is not waste.

`FIRST_WRITE`, `MODEL_SWITCH`, `VOLATILE_SYSTEM`, `SYSTEM_EDIT`, `TOOL_RESERIALIZE`, `TOOL_CATALOG_CHANGE`, `COMPACTION_REWRITE`, `HISTORY_EDIT`, `TOOL_RESULT_MUTATION`, `FORK_BOUNDARY`, `PARAM_RENDERED`, `BELOW_MIN_PREFIX`, `BREAKPOINT_MISPLACED`, `TTL_EXPIRED`, `EVICTED`, `ROUTING_MISS`, `RETRY_REPLAY`, `NONE`, `UNKNOWN`

`PARAM_RENDERED` is the case Anthropic reports as `unavailable`: blocks equal, but tool_choice, thinking, context_management, output_config, output_format, or beta headers differ. Minimum cacheable prefix is model-dependent: OpenAI 1,024 tokens, Anthropic 512-4,096, Gemini 3.x 4,096 (2,048 on 2.5).

Two scoring metrics ship with the classifier: agreement (share of turns where Prefixwise's cause maps onto the native reason when both exist) and coverage (share of misses not UNKNOWN).

### 4. Engine join: request id, not hash recomputation

vLLM >= v0.30.0 (Sep 22, 2026, PR #51381) stamps `session_id` onto every `BlockStored` KV event; the proxy sets `X-Request-Id` and `session_id`. LMCache `--lookup-hash-log-dir` logs the same request id with chunk hashes. SGLang accepts `rid` and `session_id` and, with `return_cached_tokens_details`, reports cached tokens split by device, host, and storage tier. Join on request id; time-window fallback is flagged low confidence only. Hash recomputation stays out: vLLM 16-token blocks, LMCache 256-token chunks, and SGLang radix trees cannot be reproduced from one hash.

The production-stack `LookupMsg` preflight is NOT claimed to work against current LMCache: its import path (`lmcache.v1.cache_controller.message`) 404s at LMCache 012965f (real path `lmcache.inprocess.v1.cache_controller.message`), and the pinned version is unverified.

### 5. Compaction: frozen boundary, rare and timed rewrites

"Append-only compaction" as written in v1 is incoherent: any compaction that shrinks the window rewrites the prefix after the boundary by definition. The replacement: never touch the system/tool prefix; compact only when the window is near its limit or the cache would expire anyway; prefer task boundaries over per-turn compaction; snapshot the pre-compaction fingerprint at the PreCompact hook and price the rewrite exactly once.

### 6. Fix toggles and the ledger

Toggles are opt-in and reversible: canonical tool ordering, volatile fields out of the prefix, breakpoint placement, TTL selection. Each ships as "recommended" on predicted savings and graduates to "proven" only after realised savings are measured with task success flat (realised >= 50% of predicted over 50 requests). The reuse-adjusted saving (k = write multiplier / (1 - read multiplier)) is labelled as a model, not a verified fact, because the read multiplier is model-dependent.

Headline metric: tokens per successful task, with cost as a secondary column. Rate tables are versioned snapshots with dates; if a rate is unknown, cost is null, never invented.

### 7. Distribution: local first

The OSS core is the ledger, diffing, and one engine join, as a local CLI and proxy. Fleet roll-up is hosted and optional, redacted entries only, never raw prompts.

## What Prefixwise will not build

Not a cost dashboard, not a prompt optimiser or compressor, not a router or gateway, not a second observability platform, not a hosted-provider feature. It cannot fix OpenAI's per-machine routing or Anthropic's automatic-caching opacity; it records those misses as ROUTING_MISS and moves on.

## Run it locally

```bash
python3 -m pip install --break-system-packages pytest  # or: pip install -e ".[dev]"
PYTHONPATH=src python3 -m pytest tests/ -q
PORT=8080 python3 -m prefixwise.proxy
```

In mock mode (no `PREFIXWISE_UPSTREAM_URL` set) the proxy returns synthetic usage so the fingerprint, classifier, and ledger path can be exercised with no API key and no cost:

```bash
curl -s http://127.0.0.1:8080/health
curl -si -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.6","session_id":"s1","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"hi"}]}'
curl -s http://127.0.0.1:8080/ledger
```

With a real upstream, set `PREFIXWISE_UPSTREAM_URL` (e.g. `https://api.openai.com`) and `PREFIXWISE_API_KEY`. The proxy forwards with `X-Request-Id` and `X-Session-ID` for the engine join.

The client gets the upstream response and status unchanged, so the proxy is a drop-in base URL. Prefixwise's verdict rides in response headers (`X-Prefixwise-Cause`, `X-Prefixwise-Legit`, `X-Prefixwise-Avoidable-Tokens`, `X-Prefixwise-Request-Id`). Pass session and agent identity as `X-Prefixwise-Session-Id` / `X-Prefixwise-Agent-Id` headers, or as `session_id` / `agent_id` body fields, which are stripped before forwarding. Client auth headers pass through; `PREFIXWISE_API_KEY`, if set, overrides them (`x-api-key` for Anthropic). Streaming (`stream: true`) is relayed as SSE, event by event, unchanged. For OpenAI-compatible upstreams the proxy asks for `stream_options.include_usage` and hides that usage chunk from clients that did not ask for it. Response headers go out before usage is known, so a streamed request's verdict is read afterwards from `GET /requests/<X-Prefixwise-Request-Id>`. A stream cut off before its usage arrives is not recorded. Not yet supported: the OpenAI Responses API, so OpenAI `comparison_response_id` chaining is not sent.

### Tokens per successful task

The headline metric is all tokens (prompt + completion) spent on finished tasks, failed ones included, divided by the number of successful tasks. Tag each request with a task, via the `X-Prefixwise-Task-Id` header or a `task_id` body field (stripped before forwarding), then record the outcome when the task ends:

```bash
curl -s -X POST http://127.0.0.1:8080/tasks -H 'Content-Type: application/json' \
  -d '{"task_id":"fix-bug-123","success":true}'
```

Requests with no task, and tasks with no recorded outcome, are left out of the metric rather than guessed. Recording an outcome again overwrites it (a retried task that passes). Existing ledger files are migrated in place.

## E1-open: classifier validation without API credits

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

## Build order (E1 first)

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
