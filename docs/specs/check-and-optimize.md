# Specs: `prefixwise check` and `prefixwise serve --optimize`

Status: **proposed, not built** (2026-11-08). Both keep the core guarantee: the prompt itself is
never stored or sent anywhere; only block hashes, counts and Prefixwise's verdicts are.

## 1. `prefixwise check`: a CI gate for cache regressions

**Problem.** The costly cache regressions are code changes: a tool list built from a set (order
drifts, `tool_reserialize`), a timestamp added to the system prompt (`volatile_system`), a
`cache_control` marker moved or dropped (`breakpoint_misplaced`). They pass every functional test
and show up weeks later on the bill.

**What exists.** `prefixwise explain FILE... --fail-on-waste` already exits 2 on waste in saved
request bodies (`cli.py:501`). `prefixwise run --mock -- <cmd>` already runs any agent against a
mock provider and classifies every turn. Every ledger row already has `divergence_role` and
`divergence_index` (the first changed block). `check` packages these as a gate.

**Command.**

```bash
prefixwise check -- pytest tests/agent_scenarios   # run the agent's scenarios through a mock proxy
prefixwise check --bodies runs/*.jsonl             # or judge saved trajectories (explain's format)
```

| option | meaning |
|---|---|
| `--fail-on CAUSES` | default `tool_reserialize,volatile_system,breakpoint_misplaced`; any cause name |
| `--allow FILE` | known, accepted findings (session, turn, cause), so a gate can be adopted on old code |
| `--baseline LEDGER` | a production ledger, for the dollar projection below |
| `--json` | machine-readable report |

**How it works.** Within one run, each turn is compared with the previous turn of its own
conversation, exactly as in production, so a regression shows up as a cause **inside a single
run**: no comparison with the base branch, and no prompt stored between runs. The mock provider
returns synthetic usage; the three gated causes are decided from the request alone (block hashes,
markers), so they do not depend on it.

**Output** (one line per finding, then a total):

```
FAIL tool_reserialize  scenario=refactor_task turn=3  first changed block: tools[2]
     the same tools were sent in a different order (canonical hashes equal, raw bytes differ)
     projected: ~$1,940/month (prod: 412k turns/month on claude-sonnet-5-5, 18% of turns affected)
     fix: send tools in a fixed order; or prefixwise serve --fix canonical_tools (proven)
```

**Dollar projection** (an estimate, labelled as one): with `--baseline`,
`affected share of turns in the run x production turns per month on that model x mean reusable
prefix tokens per turn x (input price - cache-read price)`, from `pricing.py` and the ledger. Without
a baseline, report tokens only. Never invent a price (the pricing rule).

**Exit codes:** 0 pass, 2 a gated cause found (not in `--allow`), 1 the run itself failed.

**GitHub Action** (composite, in this repo as `action.yml`):

```yaml
- uses: tejachavali/prefixwise@v0.7
  with:
    command: pytest tests/agent_scenarios
    fail-on: tool_reserialize,volatile_system,breakpoint_misplaced
```

It installs the pinned version, runs `prefixwise check`, writes the report to the job summary,
and fails the job on exit 2. Nothing leaves the runner.

**Tests.** Scenario pairs that must fail (tools from a set, a date in the system prompt, a marker
removed) and must pass (append-only turns, a legitimate tool added); the allow-list; exit codes;
a run that crashes is exit 1, never a pass.

## 2. `prefixwise serve --optimize`: opt-in auto-heal

**Contract** (as for today's `--fix canonical_tools`, `fixes.py`): off by default; each transform
is named, applied only when asked, recorded in the row's `fixes` column and in an
`X-Prefixwise-Fixes` response header; never applied by `prefixwise.wrap` or the LiteLLM callback
(they record what was already sent); undone on the spot if the provider rejects the request; and
shipped as "proven" only after an E4 replay shows realised savings of at least half the
prediction with task success flat. `--optimize` turns on the proven set; `--fix NAME` stays for
each one.

### 2.1 Canonical tools: exists, proven

`canonical_tools` sorts tools by name and every key inside them. It eliminates
`tool_reserialize`; E4 (2026-11-06) realised 99.9% (Anthropic) and 97.5% (vLLM) of the predicted
savings with task success flat. It is the first member of `--optimize`. Extending it to sort keys
inside other JSON the client sends (response formats, tool choice) is safe on the same grounds.

### 2.2 Automatic `cache_control` placement: opt-in, not lossless for the bill

Adding or moving a breakpoint changes what the user pays: a 5-minute cache write costs 1.25x
input, a 1-hour write 2x, and a write pays off only if it is read about 1.4 times
(`reuse_adjusted_reads`, `ledger.py`). So this is an opt-in policy, not a silent rewrite:

- Only for Anthropic requests that carry **no** marker (the client's own markers are never moved),
  at most 4 in total (the API's limit).
- Place them where agents reuse: the end of tools + system (shared by every turn), and the latest
  user turn (the rolling conversation point); a third before long tool-result runs when a turn is
  more than 20 blocks past the previous marker (the case `lookback_exceeded` detects).
- Only when the prefix is above the model's minimum cacheable size and the session has had at
  least one earlier turn within the TTL (a one-shot request would pay for a write it never reads).
- Records the predicted break-even and, after the session, the realised reads, so `prefixwise
  fixes` can show whether the policy paid.

### 2.3 Volatile values in the system prompt: warn by default

Moving a timestamp out of the system prompt changes what the model reads, so it is not lossless
(`fixes.py` says so today). Default: **detect and warn**. When the masked hash of the system prompt
is stable but the raw hash changes turn to turn (the same test that names `volatile_system`), the
proxy adds `X-Prefixwise-Advice: volatile-system` and the report names the block. An experimental
`relocate_volatile` transform, behind its own flag, would move only spans that match the masking
patterns (ISO dates, times, UUIDs) from the system prompt to a short note at the start of the
latest user turn; it ships only after an E4 run shows task success unchanged on real tasks.

### Tests and evidence

Each transform: byte-level unit tests (input to output), a proxy test that the rewrite is
recorded and undone on a provider 400, and an E4 run before it joins `--optimize`.
