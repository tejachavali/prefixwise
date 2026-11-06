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
| E2/E3 engine join, E5 compaction | Not started | README "Build order" |

## Now

**Goals (Muse, on Teja's behalf):**
- **A.** Set up PyPI publishing, so `pip install prefixwise` works.
- **B.** Dogfood Prefixwise on real agent traffic (launch gate 2; gate 1, E4, passed).

**Why:**
- **A:** trying Prefixwise should be one command.
- **B:** every result so far comes from scripted turns: E1's injected causes and E4's recorded sessions. Real agents are messier: long tool outputs, retries, compaction, streaming, parallel sessions. Real traffic will surface rough edges no experiment can, and gives a real `prefixwise report` to replace the README's illustrative one.

### A. PyPI (about 5 minutes; needs Teja's PyPI and GitHub logins in the browser)
1. On pypi.org, logged in as Teja: **Your projects → Publishing → Add a new pending publisher**, with:
   - PyPI project name `prefixwise`
   - owner `tejachavali`, repository `prefixwise`
   - workflow `release.yml`
   - environment `pypi`
2. On GitHub, logged in as Teja: `tejachavali/prefixwise` → **Settings → Environments → New environment**, named `pypi`, with no secrets.
3. **Don't tag.** Report back that both are done; Claude pushes the `v0.3.0` tag, watches the release workflow and checks `pip install prefixwise`.

### B. Dogfood (an hour or more of real work)
1. `pip install "git+https://github.com/tejachavali/prefixwise"` (from `main`).
2. `prefixwise serve --upstream https://api.anthropic.com --ledger dogfood.db`. It listens on 127.0.0.1:8080. Keys stay in environment variables.
3. Point a **real agent doing real work** at it. Preferably your own model calls, if your client lets you set the API base URL to `http://127.0.0.1:8080` (Anthropic) or `http://127.0.0.1:8080/v1` (OpenAI). Otherwise run an open-source coding agent on a real task. Set `X-Prefixwise-Session-Id` per conversation if you can.
4. Don't script the traffic. The point is behaviour Prefixwise hasn't seen.

**Report back:**
- the output of `prefixwise report`, `prefixwise report --by model` and `prefixwise fixes`
- which agent you used and roughly how long it ran
- anything that broke or looked wrong (errors, odd causes, a large `unknown` share), with the request IDs

The ledger holds no prompt text, so these are safe to share. Don't share `dogfood.db` itself if the work was private.

**What Claude does with it:**
- **Broken things:** fix them first.
- **A large `unknown` or `routing_miss` share:** find the missing evidence (a header, a provider field).
- **A wasteful cause that dominates:** check its advice, and consider a proxy toggle for it (the next E4 candidate).
- **All clean:** the report becomes the README's real example, and then launch.

**Not blocking:** the next fix to prove after dogfooding is probably moving volatile values out of the system prompt. That needs an app change rather than a proxy toggle, so its E4 compares two versions of a recorded session.

## In parallel (Claude): launch gates

Teja asked when Prefixwise can go public for people to try. The gates, in order:
1. **E4 live pass:** done 2026-11-06, `canonical_tools` proven.
2. **Dogfooding:** Muse, on Teja's behalf; see "Now" B.
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
