# Changelog

## Unreleased

- **SGLang KV events:** `prefixwise serve --engine sglang --kv-events tcp://host:5557` confirms evictions on SGLang 0.5.20 or later, as on vLLM. SGLang writes a request's body `session_id` on the blocks it stores (an attribution label; it does not change the prompt), but reports only new blocks, so Prefixwise keeps a compact copy of SGLang's tree of GPU blocks (about 80 bytes per cached token) and walks each session's chain through parent links. A removed block where the cache read stopped is a confirmed eviction on any model; "not an eviction" needs `--full-attention`, because SGLang frees sliding-window and Mamba layers without events. At startup Prefixwise reads SGLang's `/server_info` and judges only when it is safe (0.5.20 or later, events on, one data-parallel rank, no `--enable-session-radix-cache`); otherwise it says why and keeps the inference. "Not an eviction" waits for the request's own store event, which SGLang publishes after every eviction made to admit it. Validated live on SGLang 0.5.23 (Qwen3-4B, L4): evictions confirmed 5 of 5, E1-open unchanged at 120/120. **Correction:** 0.4.0 said SGLang rejects session ids it did not open; that is true only of `session_params`, SGLang's stateful sessions.

## 0.4.0: 2026-11-07

`prefixwise run`, a report that says what to do, every metric as data (API and export), and
self-hosted engines (vLLM, SGLang, LMCache) validated live.

**Python 3.10 is now supported** (the test suite passes on 3.10 to 3.13); vLLM users on 3.10 can install it.

**Upgrading:**
- Existing ledgers migrate in place (schema 3); old rows count as succeeded.
- `prefixwise report` now prints plain-English findings by default. The totals and tables
  are behind `--detail` (or `--by`). Scripts should use `--json`, which keeps its keys and
  adds `findings`.
- Failed requests (upstream errors, cut-off streams) are now recorded as rows without a
  verdict. Token and cost metrics exclude them; `summary` adds `failed_requests`.

- **Self-hosted engines (vLLM, SGLang, LMCache):** `prefixwise serve --engine vllm|sglang` declares one self-hosted engine. With no routing possible, an unchanged-prefix miss is named `evicted` (inferred, and the evidence says so) instead of `routing_miss`. On vLLM, `--kv-events tcp://host:5557` reads the engine's KV-cache events: Prefixwise tags requests with its session id and asks vLLM to report every block a request uses, then checks the block where the cache read stopped (removed: confirmed eviction; still there: not an eviction). The verdict becomes `evicted` confirmed with a block count, or `unknown` ("not an eviction: the engine still holds this session's blocks"). Library callers: `handle("vllm", ...)` sends `session_id` only with `engine="vllm"`. Prefixes restored by LMCache or HiCache count as hits. `GET /engine` shows the stream. Optional extra `prefixwise[engines]`. Design and limits: `docs/ENGINES.md`.
- **SGLang and LMCache:** E1-open takes `--engine vllm | vllm-lmcache | sglang | sglang-hicache`, with launch scripts `sglang_server.sh` and `vllm_lmcache_server.sh` (flags checked against each project's source). Prefixwise sends a body `session_id` only to vLLM; SGLang reads it as its own session handle and rejects unknown ones.
- **E-evict** (`experiments/e_evict`): floods an engine past its GPU KV capacity and checks whether the session's prefix survives, with and without an offload tier (LMCache, HiCache), and with and without KV events.
- **E1-open report:** the misses list compares against the expected cause, so `switch_back` hits no longer appear as misses.
- **Fixed:** `vllm_server.sh` enabled KV events with a field name current vLLM rejects (`enable_kv_cache_events` is the field).
- **`prefixwise run -- <agent>`**: zero setup. Starts the proxy on a free port, points the agent at it (`ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`; the agent keeps its own keys; existing base URLs become the upstreams), runs it, and prints a plain-English report on that run. The agent's exit code is passed through; Ctrl-C goes to the agent (it cancels a turn, it does not kill the agent) and SIGTERM is passed on. Gateway base URLs are kept exactly as set; a base URL that is already a Prefixwise proxy is refused rather than double-counted; `PREFIXWISE_API_KEY` never replaces the agent's keys.
- **A report that says what to do:** `prefixwise report` now leads with findings in plain English (cache share, cost, avoidable waste by cause with the fix for each, expected misses, failures); `--detail` (or `--by`) adds the tables; `--since 24h` narrows the window. The same findings are data at `GET /api/v1/findings`.
- **One proxy, two providers:** per-provider upstreams, so Anthropic and OpenAI requests each reach their own API.
- **Data first** ([docs/DATA.md](docs/DATA.md)): every request leaves a row of facts, failures included. New per-request fields: status and error (upstream errors and cut-off streams are rows now, without a verdict, so rates keep honest denominators), api, stream, uncached and cache-write tokens, upstream latency, Prefixwise's own overhead, the gap since the previous turn, the verdict's evidence and secondary causes, and where the prompt first diverged. Existing ledgers migrate in place.
- **Every metric defined once** (`prefixwise.metrics`): a catalog of names, units and definitions that the CLI, the API and exports all use. Ratios are ratios of sums. Token and cost metrics count requests with a verdict; failures count in `requests_failed` and `error_rate`.
- **HTTP API `/api/v1`** ([docs/API.md](docs/API.md)): `metrics` (the catalog), `summary`, `breakdown?by=`, `timeseries?bucket=`, `requests` (paginated), `requests/{id}`, `fixes`, `prices`, `POST tasks`, `POST explain`; filters and time windows; same token auth as the proxy.
- **`prefixwise export`**: raw rows as JSONL or CSV.
- **Summary** reports `failed_requests`.
- **Fixed (agent compatibility check):** a session mixing Chat Completions and the Responses API on one model had its Responses calls rejected (400): Prefixwise chained a `chatcmpl-` id into `prompt_cache_options`. Diagnostics now chain only to an id from the same API (`resp_`, `msg_`).
- **Fixed:** a short Anthropic prompt (below the model's minimum cacheable prefix: 512 tokens for Claude 5.x, up to 4,096 for older models) was reported as a misplaced cache breakpoint, with avoidable tokens and advice that could not help. It is now `below_min_prefix`, not waste.
- **Fixed (agent compatibility check):** under `prefixwise run`, cancelling a streamed turn with Ctrl-C printed proxy tracebacks (BrokenPipeError) into the agent's terminal. Client disconnects are now silent, other server errors are one line, and while the agent runs the proxy's own messages go to `prefixwise-run.log` next to the ledger.
- **Fixed:** when the previous request was below the provider's minimum cacheable prefix, a change since (e.g. a rewritten message) was reported as waste. Nothing could have been cached, so it is now `below_min_prefix`.
- **Codex works with `prefixwise run`:** Codex 0.167+ ignores `OPENAI_BASE_URL`, so its requests bypassed the proxy. `prefixwise run -- codex ...` now adds `-c` overrides for a custom provider (plain HTTP + SSE instead of WebSocket, key from `OPENAI_API_KEY`, no sign-in screen, nothing written to `~/.codex`). The proxy forwards Codex's headers, including the `x-codex-turn-state` sticky routing that keeps requests on the same backend, and uses its thread id as the session.
- Report wording: "1 request", not "1 requests".

## 0.3.1: 2026-11-07

Production hardening, found by a new benchmark; per-model lanes; OpenAI prices.
**Upgrading:** existing ledgers are migrated in place on first start. Rows written before
0.3.1 have no timestamp, so `prefixwise prune` keeps them unless you add `--include-undated`.

- **Production benchmark** (`benchmarks/bench.py`): added latency, throughput, memory, a soak and 13 failure cases, against the real `prefixwise serve` and a fake upstream. Results in `benchmarks/runs/`; README "Running in production" summarises them. CI runs the failure cases on every push.
- **3-4x lower overhead on long prompts:** masked block hashes are cached, since an agent resends the same prefix every turn. A 120k-token request went from +50 ms to +13 ms (p50).
- **Fails open:** a ledger write or classifier failure no longer turns a paid upstream response into a 500 (or, mid-stream, appends an error to the stream). The response passes through with `X-Prefixwise-Error: not recorded`.
- **Upstream headers relayed:** `retry-after`, `retry-after-ms`, `x-should-retry`, rate-limit headers and the provider's request id now reach the client, on success, errors and streams. They were dropped, so SDKs could not honour the provider's backoff.
- **Idle connections are closed** after `PREFIXWISE_CLIENT_TIMEOUT` (120 s). Before, 300 idle sockets held 300 threads forever.
- **Graceful shutdown:** SIGTERM drains in-flight requests and streams (up to `PREFIXWISE_DRAIN_SECONDS`, 8 s) and exits 0. In Docker the proxy is PID 1, where an unhandled SIGTERM was ignored and `docker stop` killed it mid-stream after 10 s.
- **Ledger retention:** entries record when they were written; `prefixwise prune --older-than 30d` deletes old ones.
- **Per-model lanes within a session.** Each request is compared with the last request to the same model, so small-model side calls no longer mask causes on the main thread. Found by dogfooding opencode, which interleaves Haiku calls with Sonnet. A first call to a new model is still `model_switch`; switching back to a model whose prefix is still cached is a hit. Provider diagnostics chain to the same model's last response.
- **Replacing a one-shot request is not a history edit:** a single-message request followed by a different one (side calls) is judged on its shared-prefix cache reuse.
- **E1-open harness fix:** Session B turns 8–10 now keep the compacted history. They had reverted to the pre-compaction history, hidden by the model switch until lanes exposed it.
- **CLI:** `prefixwise report | head` exits quietly.
- **OpenAI prices:** GPT-5.6 Luna, Terra and Sol are priced from OpenAI's own pricing page (read 2026-11-07; standard speed, short context). The cache-write column is OpenAI's 30-minute write; the 12-hour rate is recorded but not applied. Sol's price is promotional and expires after 2026-11-30, after which it shows `-` until updated. Plain `gpt-5.6` has no published row and stays unpriced.
- **Price matching is stricter:** a table id matches only itself plus a dated snapshot or cloud version suffix, so `gpt-5.6-sol-pro` no longer borrows Sol's price.

## 0.3.0: 2026-11-07

- **Fixes the proxy can apply:** `prefixwise serve --fix canonical_tools` sorts tools so a drifting tool list stops breaking the cache. It is opt-in and reversible, and the ledger records which requests it rewrote.
- **`prefixwise fixes`:** ranks fixes by the tokens your own ledger says they would save.
- **Found by dogfooding (opencode):** `/messages`-style paths without `/v1` are accepted. Query strings are forwarded. Endpoints Prefixwise does not analyse (`/v1/models`, `count_tokens`, ...) pass through unrecorded. `HEAD` is answered. Sonnet 4.6 is priced.
- **Dollar costs:** every request is priced from a dated price table. Built in: Claude Haiku, Sonnet and Opus 5.5 and Fable 5.1, verified from Anthropic's pricing page on 2026-11-06, including Haiku's over-100k tier and 1-hour cache writes. Add your own with `PREFIXWISE_PRICES`. `report` shows cost, avoidable cost and cost per successful task; `fixes` shows savings in dollars; `prefixwise prices` shows the table. Unknown rates stay empty, never estimated.
- **E4 harness** (`experiments/e4`): replays recorded sessions with a fix off and on, and measures predicted vs. realised savings and task success. **`canonical_tools` is proven:** 99.9% (Anthropic) and 97.5% (vLLM) of predicted savings realised, with task success flat. `prefixwise fixes` shows the evidence.
- **E1-hosted:** on Anthropic, Prefixwise agreed with Anthropic's own miss diagnosis on 37 of 37 misses. OpenAI's diagnostics returned `unavailable` (recorded as an open question).
- **Fixed:** the moving Anthropic `cache_control` breakpoint, and string vs. text-block content, were both misread as `history_edit`.
- **Fixed:** Anthropic's real diagnostics shape (`diagnostics.cache_miss_reason`) is now read.
- **Fixed:** a request with no usage is now `unknown`, not `below_min_prefix`.
- **Security:**
  - `PREFIXWISE_AUTH_TOKEN`, and the proxy refuses to expose an API key without one
  - input validation and body limits
  - the ledger file is owner-only
  - `SECURITY.md`

## 0.2.0: 2026-11-06

- **CLI:** `prefixwise serve | report | task | request | explain`.
- **Proxy:** a drop-in pass-through for OpenAI chat completions, the Responses API and Anthropic messages, streaming included. Verdicts come back in `X-Prefixwise-*` headers.
- **Tokens per successful task:** measured per task, not per request, with a `/tasks` endpoint.
- **Causes:** tool-result, TTL and fork causes now fire, and the timing rules no longer guess.
- **E1-open** passed live on vLLM: 120 turns, 100%.
- Docker image and CI on Python 3.11 to 3.13.

## 0.1.0: 2026-11-05

- First version: fingerprinting, the 19-cause classifier, the ledger and the proxy.
