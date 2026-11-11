# DeepSeek and Tencent Hy4 APIs: the cache fields, live (2026-11-11)

Run by Muse from a runbook (no script in this repo); numbers as reported. Keys in Colab Secrets.

## What the providers document (read from their own pages)

| | DeepSeek | Tencent Hy4 (TokenHub) |
|---|---|---|
| OpenAI-compatible base URL | https://api.deepseek.com (the OpenAI SDK uses .../v1) | https://tokenhub-intl.tencentcloudmaas.com/v1 (intl), https://tokenhub.tencentmaas.com/v1 (Guangzhou) |
| Model ids | `deepseek-flash`, `deepseek-v4-pro` | `hy4` |
| Cache field in `usage` | `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`, and `prompt_tokens_details.cached_tokens` (the same number) | `prompt_tokens_details.cached_tokens` |
| Prices | per 1M tokens, peak and off-peak (off-peak half): flash cache hit $0.006 / miss $0.30 / output $1.20 at peak | not on the guide page |

## DeepSeek, raw API (`deepseek-flash`, a ~3,000-token system prompt, three turns)

| Turn | prompt_tokens | cached (both fields) | miss |
|---|---|---|---|
| 1 | 3,032 | 0 | 3,032 |
| 2 | 3,039 | 2,816 | 223 |
| 3 (streamed) | 3,039 | 2,816 | 223 |

Both replies were empty: all 16 completion tokens were reasoning tokens.

## Through Prefixwise

First run: every request returned 404: `prefixwise serve --upstream https://api.deepseek.com/v1`
forwarded to `/v1/v1/chat/completions`. Reproduced locally and fixed in PR #117
(`PrefixwiseProxy.target`). Re-run on the fixed branch, all six turns went through:

| Session | Turn | prompt | cached | cause |
|---|---|---|---|---|
| deepseek-1 (~3,000-token system prompt) | 1 | 3,032 | 2,816 | first_write (the prefix was warm from the raw run) |
| | 2 | 3,039 | 2,816 | none |
| | 3 (streamed) | 3,039 | 2,816 | none |
| deepseek-short (~200-token system prompt) | 1 | 232 | 0 | first_write |
| | 2 | 239 | 0 | unknown |
| | 3 (streamed) | 239 | 0 | retry_replay, 239 avoidable |

DeepSeek cached nothing of the short prompt on any turn, so it does not cache a prompt that short (its
minimum is not documented on the pages read). Turn 3 was an identical repeat and fell to the retry rule,
which claimed waste for a request that could not have been cached: fixed on the branch, a short prompt
on an OpenAI-compatible API whose minimum is unknown is `unknown` before the retry rule is tried.

## Tencent Hy4

Not reached: the raw API answered error 401006 with the key given, before and after activating the
service; the console's Online Inference Service list holds the service id that the model id has to match.
Not re-run.
