# E1-hosted, Anthropic: live re-check after per-model lanes (PR #21)

Run by Muse on 2026-11-07, `--providers anthropic --repeats 1`, code at a2c5c13
(pytest 139 passed). No code changed. Transcribed by Claude from Muse's report.

Turns 20, errors 0. Native agreement 100.0% (6 turns with a native reason).
Injected agreement 100.0%. UNKNOWN 0.0%. **Pass: YES.**
Input tokens 174,472, cached 121,684.

| injected | correct |
|---|---:|
| compaction_rewrite | 1/1 |
| first_write | 2/2 |
| model_switch | 1/1 |
| none | 10/10 |
| param_rendered | 1/1 |
| switch_back | 1/1 |
| system_edit | 1/1 |
| tool_catalog_change | 1/1 |
| tool_reserialize | 1/1 |
| volatile_system | 1/1 |

No disagreements. `switch_back` had no native reason (a hit); its expected cause is `none`.
