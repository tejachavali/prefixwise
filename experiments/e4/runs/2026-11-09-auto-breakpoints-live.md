# E4 live, 2026-11-09: `auto_breakpoints` on Anthropic

Run by Muse on a Colab runtime, code at 9a94405272618d725cee8659b9fa3f556b422d71, model
`claude-haiku-5-5`, 3:32 to 3:35 PM PT, first run. The mock harness passed first (59 of 60 rewritten).
Transcribed by Claude from Muse's report; the printed report below is verbatim.

    python experiments/e4/e4.py --fix auto_breakpoints --provider anthropic --model claude-haiku-5-5 --max-input-tokens 900000

**Pass.** A client that sets no `cache_control` (six sessions of ten turns, `sessions/no-markers.jsonl`),
replayed as sent and then with `--fix auto_breakpoints`:

```
# E4 (anthropic_auto_breakpoints): auto_breakpoints
Paired requests 60 (errors off 0, on 0), 0 with tool-order drift | Pass: YES
| | OFF (as recorded) | ON (auto_breakpoints) |
| Prompt tokens | 226,115 | 226,115 |
| Cached tokens | 0 | 216,533 |
| Task success | 98.3% | 100.0% |
| Causes | {'breakpoint_misplaced': 54, 'first_write': 6} | {'first_write': 7, 'none': 53} |
Predicted savings (OFF breakpoint_misplaced avoidable): 202,680 tokens
Realised savings (cached ON - cached OFF): 216,533 tokens (106.8% of predicted; pass needs >= 50%)
Task success flat (ON >= OFF - 5 points): yes
Requests rewritten (markers placed): 59 of 60; not rewritten [session, turn]: [[0, 1]] (pass allows one: the arm's first request)
Cache writes: OFF 0 tokens, ON 5,716 (pass needs OFF 0: no marker, nothing written; ON 0 means the provider ignored the markers, e.g. a prompt below the model's minimum)
Prompt-side cost: OFF $0.0226, ON $0.0033, saved $0.0193 (85.6%), write surcharge included; pass needs ON below OFF
Whole-request cost: OFF $0.0281, ON $0.0087 (output tokens OFF 10,928, ON 10,959: they vary by chance, so they are not in the rule)
Markers rejected by the provider: no
```

## Reading it

- **Savings: 106.8% of predicted.** Above 100% because the prediction counts only what the OFF ledger
  calls `breakpoint_misplaced` (turns 2 to 10 of each session), while with the fix on the first turn of
  sessions 2 to 6 also reads the tools and system prompt that session 1 wrote.
- **The bill: prompt-side cost down 85.6%, surcharge included.** By hand from Haiku 5.5's rates ($0.10
  input, $0.01 read, $0.125 five-minute write per million): OFF is 226,115 tokens at the input rate,
  $0.0226; ON is 216,533 read, 5,716 written and 3,866 uncached, $0.0033. One write of the whole prompt on
  session 1's second turn, then about 35 tokens written per turn.
- **Tool choices: unchanged or better.** 59 of 60 right without the fix, 60 of 60 with it. One turn's
  difference is chance, not an effect of the markers: they do not change what the model reads.
- **No request rejected for its markers**; the one request not rewritten is the arm's first, when the
  proxy had seen nothing yet.
- **The OFF arm cached nothing** (0 read, 0 written), so it is the client the experiment is about.

## Limits

- One model (Haiku 5.5), one shape of traffic: multi-turn sessions that share tools and a system prompt,
  turns seconds apart. A client whose turns come more than five minutes apart pays the write and reads
  nothing; the proxy's guard (the same prefix seen within the cache TTL) is what keeps markers off it,
  and that guard is pinned by unit tests, not by this run.
- Scripted sessions, not an agent's own traffic.
- The per-request rows (`.json`) and the log stayed in the Colab runtime.

## What follows

`auto_breakpoints` meets the bar the spec set for a proven fix (realised savings at least half the
prediction, task success flat) and the two this fix added (cheaper in dollars with the surcharge, no
marker rejected). It joins `PROVEN_FIXES`, so `--optimize` turns it on.
