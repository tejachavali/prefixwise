# E5: accuracy at scale

E1 scored the classifier on 120 scripted turns. E5 replays many real agent conversations (a public
trace dataset) through the real proxy into a real engine and injects one known change at random
turns, which then stays, as an agent's would. Every other turn is plain growth. It reports:

- change turns named exactly, and per change: injected, named, `unknown`, precision;
- false alarms: growth turns given a change cause or a waste cause (the number that must be 0);
- changes seen but within one reporting unit of the prefix (usage cannot tell those from rounding);
- engine causes on growth turns, counted apart (under memory pressure they can be true).

`python3 experiments/e5_scale/e5_scale.py --mock` runs it against an independent prefix cache (the
literal rendered prompt, in 16-token blocks), not Prefixwise's own logic. Mock, 80 generated
conversations, 800 requests: 232 of 232 changes seen, 218 named, 14 within one reporting unit (no
block size known without KV events), 0 named as another cause, 0 false alarms on 488 growth turns.
That checks the harness and the classifier against a cache that behaves like one; it is not evidence
about an engine. Live runs go in `runs/`.

## Baseline: a section-level diagnostic

Each run also scores what a section-level diagnostic could say about the same requests: the first of
tools, system or messages that differs between consecutive requests, which is the level of detail of
provider cache diagnostics and request diffs. It is computed from the two request bodies alone,
independently of Prefixwise. Each section holds two of the injected causes (tools: reordered or catalog
changed; system: a volatile value or an edit; messages: a history edit or a tool-result edit), so it
names the exact cause of none by construction; the reported measure is how often it can still say
whether the miss was waste (only when both causes in its section are waste or both expected). On the
mock, generated conversations: right section 441 of 441, waste known 121 of 441, silent on all 909
growth turns.
