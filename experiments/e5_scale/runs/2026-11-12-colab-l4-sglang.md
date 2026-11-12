# E5 on SGLang, real agent trajectories and generated conversations: Colab L4, 2026-11-12

Run by Muse on branch `claude/e5-agent-traces` at 0670efe. SGLang 0.5.23 in a Python 3.12 venv (the
runtime's Python 3.13 has no `outlines_core` wheel; as on 2026-11-07), started by
`KV_EVENTS=1 experiments/e1_open/sglang_server.sh` (`--enable-cache-report --tool-call-parser qwen25`,
KV events over ZMQ), Qwen3-4B. Prefixwise with `--engine sglang --kv-events tcp://127.0.0.1:5557
--full-attention`. `--limit 150 --turns 10 --seed 2`. No code changed. Transcribed by Claude from
Muse's report.

## Real agent trajectories (`nvidia/Open-SWE-Traces`, split `claudecode`)

```
# E5 accuracy at scale
150 conversations, 1497 requests (0 errors).
- Change turns named exactly: **424 of 434** (0 unknown, 10 seen within one reporting unit, 0 named as another cause)
- False alarms on growth turns: **0 of 913**
- Engine causes on growth turns (evicted / routing): 0
- Unknown on growth turns: 0
| change | injected | named | unknown | precision |
|---|---:|---:|---:|---:|
| tool_reserialize | 63 | 63 | 0 | 63/63 |
| tool_catalog_change | 83 | 83 | 0 | 83/83 |
| volatile_system | 76 | 76 | 0 | 76/76 |
| system_edit | 78 | 78 | 0 | 78/78 |
| history_edit | 73 | 73 | 0 | 73/73 |
| tool_result_mutation | 61 | 51 | 0 | 51/51 |
Confusion (label -> verdict):
- first_write -> first_write: 150
- history_edit -> history_edit: 73
- none -> none: 913
- system_edit -> system_edit: 78
- tool_catalog_change -> tool_catalog_change: 83
- tool_reserialize -> tool_reserialize: 63
- tool_result_mutation -> none: 10
- tool_result_mutation -> tool_result_mutation: 51
- volatile_system -> volatile_system: 76
```

## Generated conversations

```
# E5 accuracy at scale
150 conversations, 1500 requests (0 errors).
- Change turns named exactly: **456 of 456** (0 unknown, 0 seen within one reporting unit, 0 named as another cause)
- False alarms on growth turns: **0 of 894**
- Engine causes on growth turns (evicted / routing): 0
- Unknown on growth turns: 0
| change | injected | named | unknown | precision |
|---|---:|---:|---:|---:|
| tool_reserialize | 75 | 75 | 0 | 75/75 |
| tool_catalog_change | 85 | 85 | 0 | 85/85 |
| volatile_system | 81 | 81 | 0 | 81/81 |
| system_edit | 84 | 84 | 0 | 84/84 |
| history_edit | 59 | 59 | 0 | 59/59 |
| tool_result_mutation | 72 | 72 | 0 | 72/72 |
Confusion (label -> verdict):
- first_write -> first_write: 150
- history_edit -> history_edit: 59
- none -> none: 894
- system_edit -> system_edit: 84
- tool_catalog_change -> tool_catalog_change: 85
- tool_reserialize -> tool_reserialize: 75
- tool_result_mutation -> tool_result_mutation: 72
- volatile_system -> volatile_system: 81
```

**Reading.** The same as on vLLM ([real traces](2026-11-12-colab-l4-vllm-open-swe-traces.md),
[generated](2026-11-12-colab-l4-vllm-generated.md)): every injected change seen, none named as
another cause, no false alarm, no engine cause claimed on a growth turn. The 10
`tool_result_mutation -> none` are the case the vLLM record explains: an edit appended to the latest
tool result falls in the prompt's last, uncached block, so nothing was lost; the edit is kept as a
secondary cause.

## E5 in one table

| engine | traffic | changes named | seen, nothing lost | named as another cause | false alarms | errors |
|---|---|---:|---:|---:|---:|---:|
| vLLM 0.33.0 | generated | 441 of 441 | 0 | 0 | 0 of 909 | 0 |
| vLLM 0.33.0 | Open-SWE-Traces | 425 of 441 | 16 | 0 | 0 of 906 | 0 |
| SGLang 0.5.23 | generated | 456 of 456 | 0 | 0 | 0 of 894 | 0 |
| SGLang 0.5.23 | Open-SWE-Traces | 424 of 434 | 10 | 0 | 0 of 913 | 0 |

1,772 injected changes, 3,622 growth turns, 5,994 requests: every change seen, 0 named as another
cause, 0 false alarms. Not exercised: an edit at the start of the latest tool result (the harness
appends), evictions under memory pressure, other models.
