# opencode through `prefixwise run`, 2026-11-09

Run by Muse on prefixwise 0.10.5, opencode 1.19.5. The runbook asked for an Anthropic API key in the
environment and no `opencode.json` setting a provider base URL; Muse's report confirms that opencode "read
the environment Prefixwise sets" and does not say otherwise about the setup. Transcribed by Claude from
Muse's report. The question:
does opencode read the `ANTHROPIC_BASE_URL` that `prefixwise run` sets, or bypass the proxy? Earlier runs
([dogfood 1](../dogfood/2026-11-06-opencode.md), [dogfood 2](../dogfood/2026-11-07-opencode-2.md)) went
through `prefixwise serve` with the proxy's URL set in opencode's provider config, which left this open.

    prefixwise run --ledger /tmp/pw-oc/run.db -- opencode run "Write a file hello.txt containing the word hello, then stop."

**Answer: it routes.** Every request landed in the ledger; the known-good `serve` path was not needed.

`prefixwise run`'s report: 3 requests, 16,733 prompt tokens, 47% served from cache, cost $0.0340, no
avoidable waste (both misses legitimate).

`prefixwise report --session latest`:

| turn | model | prompt | cached | hit | cause |
|---:|---|---:|---:|---:|---|
| 1 | claude-haiku-5-5 | 807 | 0 | 0.0% | first_write |
| 2 | claude-sonnet-4-6 | 7,915 | 0 | 0.0% | model_switch |
| 3 | claude-sonnet-4-6 | 8,011 | 7,912 | 98.8% | none |

Notes:
- opencode's first request went to Haiku (a side call: a title, a summary) before the main thread on
  Sonnet, so the Sonnet thread's first turn is `model_switch`: the first call to another model in the
  session, legitimate, no waste. The same pattern was seen in dogfood 2.
- The per-turn `$` column read $0.0000 on every row while the run total was $0.0340: that column was
  avoidable dollars, not the request's cost. Fixed after this run: the view now has a `cost` column
  and names the other `avoidable $`.
- The scratch directory was deleted after the run; the ledger held hashes and counts only.
