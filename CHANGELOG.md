# Changelog

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
