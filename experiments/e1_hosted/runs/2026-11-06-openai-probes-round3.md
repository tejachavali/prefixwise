# E1-hosted OpenAI probes, round 3, 2026-11-06

Run by Muse on Colab, commit 7cb90ed, no code changes. Every probe returned STOP: OpenAI answered `prompt_cache_diagnostics: {"type": "unavailable"}` on every chained turn.

| Probe | Model / option | Turn 3 (tool_reserialize) | Turn 4 (system_edit) | Input tokens |
|---|---|---|---|---:|
| a | gpt-5.6-luna, no prompt_cache_key | real miss (0/5,905), `unavailable` | real miss (0/5,932), `unavailable` | 29,531 |
| b | gpt-5.6 | **all-zero usage** | real miss (0/5,932), `unavailable` | 17,769 |
| c | gpt-5.6-sol | real miss (0/5,905), `unavailable` | **all-zero usage** | 11,786 |

Turn 2 raw usage, identical in all three: `{"input_tokens": 5881, "input_tokens_details": {"cache_write_tokens": 24, "cached_tokens": 5854}, "output_tokens": 16, ...}`. Caching works, and it was a hit every time.

**OpenAI's documentation** (https://developers.openai.com/api/docs/guides/prompt-caching/diagnostics, as quoted by Muse): `unavailable` means the comparison could not produce a conclusive result, the model does not support diagnostics, or the comparison is not ready. To get verdicts you need:
- the Responses API, on GPT-5.6 or later;
- a recent completed baseline response from the same organization, passed as `comparison_response_id` (records expire; an expired one gives `comparison_response_not_found`);
- a reusable prefix of at least 1,024 tokens.

`mode: "explicit"` disables the implicit breakpoint, so it was not tried. A cache_miss verdict carries `reason` and `cache_missed_tokens`, and may also carry `comparison_reusable_tokens`.

**OpenAI dashboard:** Diagnosed responses 0, "No cache-miss reasons for this selection". Spend for the day was $0.08 across all rounds, and the dashboard was still lagging.

**Hypothesis for the next probe:** the harness chains each turn to a response from about one second earlier, so the baseline's diagnostic record is "not ready". The harness now has `--turn-delay`. Separately, three turns came back with all-zero usage. The harness now records the response `status`, `error` and `incomplete_details` so those can be explained.
