# WORKLOG: read this first

Shared handoff between the agents working on Prefixwise. It says **what is being attempted, why, what to run, and what happens with each result**. `build.log` records what happened after the fact; this file records intent.

- **Read** the "Now" section before doing anything.
- **Append** an entry to the log at the bottom whenever you start, finish or change direction. Newest last. Never rewrite old entries.
- **Keep "Now" current:** whoever changes the plan rewrites it.

## Who does what

| Agent | Where it runs | Does | Doesn't |
|---|---|---|---|
| **Claude** (Claude Code cloud session) | Repo checkout with GitHub push access | Writes code and tests, opens and merges PRs once CI passes, reads run results, decides the next step, keeps this file current | Hold provider API keys; run GPUs |
| **Muse** | Colab (GPU) and a browser, with the user's API keys as Colab Secrets | Runs experiments exactly as written below, reads provider dashboards and docs, reports outputs as plain text | Change repo code during a run; push (no GitHub login in Colab; Claude commits results); put keys in any file |
| **Teja** | | Decides direction, relays messages between the agents | |

## Ground rules for experiment runs

1. Run from the commit named in "Now". Check it with `git log --oneline -1` and `python -m pytest tests -q`.
2. API keys go in environment variables only, never in a file, report or log.
3. **Don't change code to make a run pass.** A failure is a finding: report it exactly as printed.
4. Report the full printed output (summary line, tables, disagreements, errors) as plain text, plus spend from the provider usage pages.
5. Cheapest step first: `--mock` (free), then probes, then full runs. Stop at the first STOP or failure and report.
6. Claude commits the results to `experiments/<name>/runs/`, after scanning them for keys.

## Where things stand

| Milestone | Status | Evidence |
|---|---|---|
| E1-open: classifier names injected causes on live vLLM | **Pass** 2026-11-06: 120 turns, 100%, Qwen3-4B on vLLM 0.33.0, Colab L4 | `experiments/e1_open/runs/` |
| E1-hosted, Anthropic: agrees with Anthropic's own miss diagnostics | **Pass** 2026-11-06: 37/37 diagnosed misses, all 7 injected miss causes, 60/60 hits, UNKNOWN 0 | `experiments/e1_hosted/runs/` |
| E1-hosted, OpenAI | **Blocked on OpenAI:** `prompt_cache_diagnostics` returns `unavailable` on every comparison (GPT-5.6 Luna/Sol/base, delays up to 120 s, completed baseline). Not a Prefixwise bug. | `experiments/e1_hosted/runs/2026-11-06-round4.md` |
| E4: a fix saves what Prefixwise predicts, tasks still succeed | **Harness built, mock Pass** (99.8% of predicted realised). Live runs pending. | `experiments/e4/` |
| E2/E3 engine join, E5 compaction | Not started | README "Build order" |

## Now

**Goal: prove the `canonical_tools` fix (E4).**

**Why:** E1 showed Prefixwise names causes correctly. Users care whether *acting on* its advice saves money without breaking their agent. `canonical_tools` (`prefixwise serve --fix canonical_tools`) sorts tools so a tool list that drifts between turns stops breaking the cache. It is the first fix the proxy can apply on its own, so it is the first one to prove.

**The bet:**
- **Savings:** with the fix on, the cache reuse Prefixwise predicted actually shows up. The pass bar is at least 50% of predicted; the mock got 99.8%.
- **Tasks:** reordering tools doesn't change which tool the model picks. The pass bar is task success with the fix on no more than 5 points below success with it off.

The second part is the real risk. Models can be sensitive to tool order, and only a live model can show it.

**Code:** PR #11, branch `claude/e4-fix-toggles` (on `main` once merged).

**Muse runs, in order:**
1. `pip install -e ".[dev]"`, `python -m pytest tests -q` (104 passed), `python experiments/e4/e4.py --mock` (Pass YES).
2. **vLLM (free):** start vLLM as for E1-open (Qwen3-4B, `--enable-prefix-caching --enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser hermes`), then
   `python experiments/e4/e4.py --provider vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-4B`
3. **Anthropic (a few cents; waits ~5.5 min between arms so the cache expires):**
   `python experiments/e4/e4.py --provider anthropic --model claude-haiku-5-5 --max-input-tokens 900000`

**Report back for each run:**
- the full printed `.md` report
- any HTTP errors
- every turn where `tool_called` differs between OFF and ON for the same session and turn (from the `.json`)

**What Claude does with each result:**
- **Both pass:** `canonical_tools` is marked *proven* in the README and in `prefixwise fixes`, and the runs are committed. Next fix to test: volatile values out of the system prompt.
- **Savings below 50% of predicted:** check whether the predictor over-counts (compare `avoidable_tokens` per turn with the actual cached gain) and fix it before claiming anything.
- **Task success drops with the fix on:** that's a real finding. `canonical_tools` stays "recommended, with a warning", and the README says so. Look at which tasks flipped.
- **HTTP errors:** fix the harness first, as in E1-hosted rounds 1–3.

## Open questions

- **OpenAI diagnostics:** `unavailable` everywhere. A support note is drafted (Claude's message of 2026-11-06, round 4). Unresolved; low priority.
- Anthropic reports `cache_missed_input_tokens` larger than the request's prompt tokens (9,986 vs 8,687). Unexplained.
- **Prefixwise locates changes per block, not per token.** An edit at the end of a long system prompt reads as `system_edit` even when the provider could reuse most of it. A known limit, in the README.

## Log (append only, newest last)

- **2026-11-05:** Teja: v2 core and E1-open harness (commits d3b25ef, f1da9c4).
- **2026-11-06 early:** Claude, PRs #1–#8:
  - classifier fix for append-only growth; proxy passthrough
  - task metric; streaming; thread safety
  - CLI; Responses API; causes wired up
  - packaging, Docker, CI and docs (v0.2.0)
- **2026-11-06:** Muse ran E1-open live on Colab and found vLLM tool-choice flags, a `/reset_prefix_cache` 404 and Qwen3-8B not fitting. Teja and Muse fixed them (fd4c003). The re-run passed 100% (74310ec).
- **2026-11-06:** Claude built E1-hosted (PR #9).
  - The mock run found two Anthropic false positives, fixed: the moving `cache_control` breakpoint, and string vs text-block content.
  - Muse's probes v1 and v2 stopped at the gate, because the probe was badly designed and had no real misses. Fixed.
  - Live probes showed Anthropic's real diagnostics shape (`diagnostics.cache_miss_reason.type`); the reader was fixed.
- **2026-11-06:** Round 3: Anthropic Pass, 30/30. Round 4: `model_switch` 2/2 (37/37 total). OpenAI `unavailable` throughout (PR #10).
- **2026-11-06:** Claude built E4: the `canonical_tools` toggle, `prefixwise fixes`, and the replay harness. Mock Pass at 99.8% (PR #11). **Next: Muse runs E4 live (see Now).**
- **2026-11-06:** Teja asked that intent always be logged for Muse. This file was created; Claude updates it with every PR and before every request to Muse.
- **2026-11-06:** PR #11 merged (E4 harness, canonical_tools, WORKLOG). Muse started the E4 live runs.
- **2026-11-06:** Teja asked for the personal note at the end of the README to be removed (PR #12). It remains in git history; whether to purge it is Teja's call.
