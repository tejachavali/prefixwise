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

Every request returned 404: `prefixwise serve --upstream https://api.deepseek.com/v1` forwarded to
`/v1/v1/chat/completions`. Reproduced locally and fixed in PR #117 (`PrefixwiseProxy.target`); the
proxy legs are to be re-run on the fixed branch.

## Tencent Hy4

Not reached: the raw API answered error 401006 with the key given, before and after activating the
service; the console's Online Inference Service list holds the service id that the model id has to match.
Not re-run.
