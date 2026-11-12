# E5 on vLLM, real agent trajectories: Colab L4, 2026-11-12

Run by Muse on branch `claude/e5-agent-traces` at 0670efe. vLLM 0.33.0 (`--enable-prefix-caching
--enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser hermes`, KV events over
ZMQ), Qwen3-4B. Prefixwise with `--engine vllm --kv-events tcp://127.0.0.1:5557 --full-attention`.
150 trajectories from `nvidia/Open-SWE-Traces`, split `claudecode` (Claude Code solving SWE tasks:
one task, then tool calls and results, with the agent's own tools and system prompt), the first 10
model calls of each, `--seed 1`. No code changed. Transcribed by Claude from Muse's report.
The generated run on the same commit gave the numbers of the [earlier record](2026-11-12-colab-l4-vllm-generated.md)
again (441 of 441, 0 of 909).

```
# E5 accuracy at scale
150 conversations, 1497 requests (0 errors).
- Change turns named exactly: **425 of 441** (0 unknown, 16 seen within one reporting unit, 0 named as another cause)
- False alarms on growth turns: **0 of 906**
- Engine causes on growth turns (evicted / routing): 0
- Unknown on growth turns: 0
| change | injected | named | unknown | precision |
|---|---:|---:|---:|---:|
| tool_reserialize | 72 | 72 | 0 | 72/72 |
| tool_catalog_change | 80 | 80 | 0 | 80/80 |
| volatile_system | 87 | 87 | 0 | 87/87 |
| system_edit | 71 | 71 | 0 | 71/71 |
| history_edit | 60 | 60 | 0 | 60/60 |
| tool_result_mutation | 71 | 55 | 0 | 55/55 |
Confusion (label -> verdict):
- first_write -> first_write: 150
- history_edit -> history_edit: 60
- none -> none: 906
- system_edit -> system_edit: 71
- tool_catalog_change -> tool_catalog_change: 80
- tool_reserialize -> tool_reserialize: 72
- tool_result_mutation -> tool_result_mutation: 55
- tool_result_mutation -> none: 16
- volatile_system -> volatile_system: 87
```

**Reading.** On real agent trajectories, every injected change was seen (441 of 441) and none was
named as another cause; 425 were named as the primary cause. No turn that only appended (906) was
given a change cause, and no eviction or routing miss was claimed.

**The 16 `tool_result_mutation -> none`.** These are counted as "seen within one reporting unit":
the verdict is `none` with `tool_result_mutation` kept as a secondary cause. An agent's request ends
with its latest tool result, and the harness edits a tool result by appending text to its end. When
the edited result is the latest one, the edit falls in the prompt's last, partial block, which the
engine does not cache in any case: the next request still read everything the previous one could
offer, so nothing was lost and nothing is avoidable, which is what `none` says (invariant 2 in
docs/ALGORITHM.md). Checked on the mock replay of the same trace shape: the one `none` among 12
tool-result edits was the edit of the latest result; the 11 edits further back were all named.
Generated conversations do not show this, as their requests end with a user message.

**What this does not show.** An edit at the start of the latest tool result (a re-render) would cost
that whole result; the harness only appends, so that case is not exercised here. One engine (vLLM),
one model, the first 10 model calls of each trajectory.
