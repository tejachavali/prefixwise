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
| E4: a fix saves what Prefixwise predicts, tasks still succeed | **Pass** 2026-11-06: `canonical_tools` 99.9% realised (Anthropic), 97.5% (vLLM); success flat (0/60, 1/60 changed) | `experiments/e4/runs/2026-11-06-live.md` |
| Dogfood: real agent traffic | **Done** 2026-11-06: opencode, 25 requests, 92.9% cached, 0 avoidable; 4 rough edges found and fixed (PR #18) | `experiments/dogfood/2026-11-06-opencode.md` |
| E2/E3 engine join, E5 compaction | Not started | README "Build order" |

## Now

**Prefixwise 0.3.0 is public:** `pip install prefixwise`. GitHub release `v0.3.0` (tag on `343ce61`), release workflow run #1 green, PyPI shows 0.3.0 as latest. Muse published it from the GitHub web UI (Claude's session cannot push tags), and Claude verified it by a clean install from PyPI and a smoke test.

**Muse: nothing needed right now.** Optional task C still stands: OpenAI's official prices for `gpt-5.6`, `-luna`, `-sol` and `-terra`, from https://developers.openai.com/api/docs/pricing.

**Next for Claude:**
1. **Per-model lanes within a session**, from the dogfood. A small-model side call should not hide causes on the main thread. Updates E1's model-switch scoring in the same PR.
2. **The product plan proposed to Teja on 2026-11-06:** `prefixwise run -- <cmd>`, a local web UI (money view, session timeline, prefix map), and live fix experiments. Waiting on Teja's go-ahead and choice of first audience.

**Releasing from now on:** Claude prepares the version bump and CHANGELOG in a PR. **Muse (or Teja) creates the GitHub release** in the web UI, which creates the tag and triggers `.github/workflows/release.yml`.

## In parallel (Claude): launch gates

Teja asked when Prefixwise can go public for people to try. The gates, in order:
1. **E4 live pass:** done 2026-11-06, `canonical_tools` proven.
2. **Dogfooding:** done 2026-11-06 (opencode); fixes in PR #18.
3. **Security review:** done, PR #13. See below.
4. **Launch polish:** done, PR #14 (v0.3.0 prepared, release workflow, CONTRIBUTING, CHANGELOG, issue templates). Publishing needs Teja's one-time PyPI setup; see CONTRIBUTING.md "Releasing".

Then a soft launch to a few people who run agents (Omni's founder is a candidate), then wider.

**Security review results (PR #13).** Each finding was confirmed with a proof-of-concept against a live proxy before fixing, and is pinned by a test in `tests/test_security.py`:
- Malformed JSON shapes crashed a handler, dropping the connection. Now a 400 naming the field.
- A negative `Content-Length` hung a thread, and request size was unbounded. Now 400, or 413 above 32 MiB.
- The ledger file was world-readable (644). Now 600. It holds no prompt text; that is verified by a test.
- **No authentication:** the documented Docker command published on all interfaces, so with `PREFIXWISE_API_KEY` set anyone could spend the key. Fixed:
  - `PREFIXWISE_AUTH_TOKEN` (clients send `X-Prefixwise-Token`)
  - the proxy refuses to start exposed with a key and no token
  - the docs publish Docker on `127.0.0.1`
- Checked and fine: no client-controlled upstream URL (no SSRF), SQL inputs whitelisted, no request logging, Docker runs non-root.

## Roadmap (agreed with Teja, 2026-11-06)

**Done:**
- E1-open: vLLM, 100%.
- E1-hosted: Anthropic, 37/37 against its own diagnosis.
- E4: `canonical_tools` proven.
- Proxy, CLI, streaming, Responses API, security hardening, CI, Docker; v0.3.0 ready.

**Now: launch.**
1. Dogfooding (Muse, "Now" B).
2. PyPI release: **done** 2026-11-07, `pip install prefixwise` (0.3.0).
3. A real README example from the dogfood report.
4. Soft launch to a few people who run agents, then wider.

**Next: product, in this order.**
1. **Dollar costs** *(done, PR #17; OpenAI rates waiting on Muse task C)*: a dated price table per model, verified from the providers' own pricing pages, so `avoidable_cost` and request cost are filled in. Unknown rates stay null, never guessed. Users can supply their own rates file.
2. **Framework integrations:** LangGraph, the OpenAI Agents SDK and the Claude Agent SDK set the session, task and compaction headers automatically.
3. **Finer-grained diagnosis:** locate where *inside* a block a change happened. This fixes the "edit at the end of a long prompt" limit.
4. **A second proven fix:** `volatile_system`, timestamps out of the system prompt. It's an app change, so its E4 compares two versions of a recorded session.
5. **Gemini support.** It has no native miss diagnostics, so Prefixwise adds the most there.

**Later, when users ask:**
- E5 compaction (snapshot before compaction, price the rewrite once, compact at task boundaries).
- E2/E3 engine join (vLLM and LMCache cache events by request ID; enables `evicted`).
- Shared state across multiple proxy instances.
- An SGLang adapter.
- A hosted team roll-up.

**Waiting on others:** OpenAI native diagnostics (`unavailable`).

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
- **2026-11-06:** Claude ran a pre-launch security review: 5 real issues found by proof-of-concept and fixed (PR #13). Next for Claude: launch polish (PyPI trusted publishing, CONTRIBUTING, CHANGELOG). E4 live results still pending from Muse.
- **2026-11-06:** Claude: launch polish (PR #14). Version 0.3.0 prepared, not published. The name `prefixwise` is free on PyPI as of today. Trusted-publishing workflow added; Teja does the one-time PyPI pending-publisher setup, then tags `v0.3.0`.
- **2026-11-06:** Teja dropped the legal-clearance gate from the launch plan (Teja's call).
- **2026-11-06:** Muse ran E4 live. Pass on vLLM (97.5% of predicted realised, 1/60 task flip) and Anthropic (99.9%, 0/60). `canonical_tools` marked proven (PR #15). Next: Teja dogfoods on a real agent.
- **2026-11-06:** Teja asked Muse to do the PyPI setup and dogfooding on Teja's behalf. Claude pushes the v0.3.0 tag once Muse confirms the setup.
- **2026-11-06:** Roadmap written down (section above). Claude started dollar costs while Muse dogfoods.
- **2026-11-06:** Claude: dollar costs (PR #17). Anthropic rates are verified from platform.claude.com, the official page (fetched by a Claude helper). OpenAI's pricing pages are unreachable from Claude's environment, so that's Muse task C.
- **2026-11-07:** Muse finished PyPI setup (part A) and dogfooding (part B: opencode, 92.9% cached, 0 waste). Claude fixed the 4 rough edges (PR #18); the release tag follows.
- **2026-11-07:** v0.3.0 released. PR #18 (dogfood fixes) and #19 (CHANGELOG) merged. Claude's tag push got HTTP 403 (this session can't create tags), so Muse created the GitHub release and the release workflow published to PyPI. Claude verified with a clean `pip install prefixwise==0.3.0`. A smoke test found a BrokenPipeError on `report | head`, fixed in PR #20.
