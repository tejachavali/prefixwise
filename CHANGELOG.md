# Changelog

## 0.3.0: unreleased

Tag `v0.3.0` to publish (see CONTRIBUTING.md, "Releasing").

- **Fixes the proxy can apply:** `prefixwise serve --fix canonical_tools` sorts tools so a drifting tool list stops breaking the cache. It is opt-in and reversible, and the ledger records which requests it rewrote.
- **`prefixwise fixes`:** ranks fixes by the tokens your own ledger says they would save.
- **E4 harness** (`experiments/e4`): replays recorded sessions with a fix off and on, and measures predicted vs. realised savings and task success.
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
