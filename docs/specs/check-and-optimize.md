# Specs: `prefixwise check` and `prefixwise serve --optimize`

Status: `prefixwise check` **built** (0.9, 2026-11-08, as amended below); `--optimize` **proposed, not built**. Both keep the core guarantee: the prompt itself is
never stored or sent anywhere; only block hashes, counts and Prefixwise's verdicts are.

## 1. `prefixwise check`: a CI gate for cache regressions

**Built as amended (2026-11-08 review):**
- Default `--fail-on` is `tool_reserialize,volatile_system,breakpoint_misplaced,tool_result_mutation`
  (mutating an earlier turn's tool output is always a code bug); `--strict` adds `history_edit`,
  `system_edit`, `param_rendered`, which a scenario that edits history or switches a parameter on
  purpose will trip, so they are not default.
- `--allow FILE` keys on `(scenario, cause, divergence_role)`, not session and turn (test session
  ids change run to run). The scenario is `X-Prefixwise-Session-Id` when the agent sends one; under
  pytest the bundled plugin (`prefixwise.pytest_plugin`, a `pytest11` entry point, inert unless
  `PREFIXWISE_CHECK` and `PREFIXWISE_URL` are set) posts each test's node id to
  `POST /api/v1/check/scenario`, so requests without a session id are recorded under the test.
- The baseline is a small JSON summary, not a ledger on the CI runner: `prefixwise export
  --baseline` writes `.prefixwise-baseline.json` (per model `turns_per_month`,
  `eligible_turns_per_month`, `mean_reusable_tokens`; counts only), and `check` auto-detects it in
  the working directory when `--baseline` is not given.
- The projection's denominator is eligible turns: `affected eligible turns / eligible turns in the
  run` (first turns excluded) x the baseline's `eligible_turns_per_month` x `mean_reusable_tokens` x
  `PriceTable.miss_rate` (the rate a miss pays above a hit per token). Without a baseline, tokens.
- **Multi-turn mock driver.** An agent test that runs one turn tests nothing (`first_write`). The
  mock answers a user turn that offers tools with a call to the first function tool (arguments are
  placeholders by type for the schema's required properties; Anthropic `tool_use`, chat
  `tool_calls`, Responses `function_call`, streamed too), so the loop runs a second turn, the one
  the gate judges; a turn that answers a tool call gets text, so one round per user message, never
  a loop. A run with no turn that had a previous turn warns: `0 cache-eligible multi-turn sequences
  observed`.
- **Sub-second clocks.** Against a fast mock, turn 1 and turn 2 are milliseconds apart, so a
  per-second timestamp never changes. `check` holds each conversation's first mock response for
  `--first-turn-delay` seconds (1.1) so `int(time.time())` and `%S` stamps do change; a date or
  time the fingerprinter's masks find in a system prompt that did not change within the run is a
  `WARN` naming the scenario (a per-minute or per-day stamp would cost a miss per change in
  production; a fixed cutoff date is to be ignored).
- Exit codes: 2 when a gated cause is found and not allowed, else 1 when the command exited
  non-zero, else 0. With `--json` the command's own stdout goes to stderr so stdout is the report.
- The Action pins the version from its own tag (`@v0.9.0` installs `prefixwise==0.9.0`; a moving
  `@v0.9` installs the latest).

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
prefix tokens per turn x (rate paid for those tokens on a miss, the cache-write rate where they are written again - cache-read price)`, from `pricing.py` and the ledger. Without
a baseline, report tokens only. Never invent a price (the pricing rule).

**Exit codes:** 0 pass, 2 a gated cause found (not in `--allow`), 1 the run itself failed.

**GitHub Action** (composite, in this repo as `action.yml`):

```yaml
- uses: tejachavali/prefixwise@v0.9
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
