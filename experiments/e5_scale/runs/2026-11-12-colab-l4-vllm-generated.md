# E5 on vLLM, generated conversations: Colab L4, 2026-11-12

Run by Muse on `main` at 8bb1b13. vLLM 0.33.0 (`--enable-prefix-caching --enable-prompt-tokens-details
--enable-auto-tool-choice --tool-call-parser hermes`, KV events over ZMQ), Qwen3-4B. Prefixwise with
`--engine vllm --kv-events tcp://127.0.0.1:5557 --full-attention`. Generated conversations
(`--source generated --limit 150 --turns 10 --seed 1`). No code changed. Transcribed by Claude from
Muse's report.

Two earlier attempts measured nothing and are not results: the datasets did not load (a harness
assumption of a `train` split), then every request was refused with HTTP 400 (the runbook left out
vLLM's tool-call flags, and the harness did not show the error). Both are fixed in the harness.

```
# E5 accuracy at scale
150 conversations, 1500 requests (0 errors).
- Change turns named exactly: **441 of 441** (0 unknown, 0 seen within one reporting unit, 0 named as another cause)
- False alarms on growth turns: **0 of 909**
- Engine causes on growth turns (evicted / routing): 0
- Unknown on growth turns: 0
| change | injected | named | unknown | precision |
|---|---:|---:|---:|---:|
| tool_reserialize | 85 | 85 | 0 | 85/85 |
| tool_catalog_change | 75 | 75 | 0 | 75/75 |
| volatile_system | 69 | 69 | 0 | 69/69 |
| system_edit | 91 | 91 | 0 | 91/91 |
| history_edit | 67 | 67 | 0 | 67/67 |
| tool_result_mutation | 54 | 54 | 0 | 54/54 |
Confusion (label -> verdict):
- first_write -> first_write: 150
- history_edit -> history_edit: 67
- none -> none: 909
- system_edit -> system_edit: 91
- tool_catalog_change -> tool_catalog_change: 75
- tool_reserialize -> tool_reserialize: 85
- tool_result_mutation -> tool_result_mutation: 54
- volatile_system -> volatile_system: 69
```

**Reading.** Every injected change (441) was named as itself, and no turn that only appended (909)
was given a change cause, against a real engine with its KV events joined. No eviction or routing
miss was claimed on a growth turn: at this load nothing was evicted.

**What this does not show.** The conversations are generated, not real agent traces (their tools,
prompts and tool results are synthetic); the real-trace run is next. One engine (vLLM), one model.
