# `prefixwise import codex` on real Codex logs, 2026-11-07

Run by Muse on Colab (branch `claude/import-codex`, PR #41), codex-cli 0.167.0, two
`codex exec` tasks in a fresh runtime. No code changed. Transcribed by Claude.

**Setup note.** `codex exec` reads its API key from `CODEX_API_KEY` for the built-in OpenAI
provider (codex-rs/login/src/auth/manager.rs); with only `OPENAI_API_KEY` set it sends no
credential and gets a 401 "Missing bearer". `prefixwise run -- codex` is unaffected: it defines
its own provider with `env_key="OPENAI_API_KEY"`.

**Logs:** 4 rollout files (two from the failed 401 runs, with no usage records); entry types
`response_item` 42, `event_msg` 31, `token_usage_record` 5, `session_meta` 4, `world_state` 4,
`turn_context` 4: the format the importer was built from.

**Result:** 5 requests from 2 logs, 66,515 prompt tokens, 59.5% from cache.

| cause | requests | prompt tokens | cached |
|---|---:|---:|---:|
| none | 3 | 40,211 | 98.5% |
| first_write | 2 | 26,304 | 0.0% |

Each thread's first request wrote; every later request read 98.5%. Re-running the import
added 0 and skipped 5 (idempotent). The Codex model was not in the price table, so cost
showed `-`; the import now names unpriced models.
