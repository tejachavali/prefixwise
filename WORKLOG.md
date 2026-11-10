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
| E1-open: classifier names injected causes on live vLLM | **Pass** 2026-11-06: 120 turns, 100%, Qwen3-4B on vLLM 0.33.0, Colab L4. Re-passed 2026-11-07 with per-model lanes (120/120) | `experiments/e1_open/runs/` |
| E1-hosted, Anthropic: agrees with Anthropic's own miss diagnostics | **Pass** 2026-11-06: 32/32 misses Anthropic named (6 causes), plus 5 parameter changes it called `unavailable` that Prefixwise named, 60/60 hits, UNKNOWN 0. Re-passed 2026-11-07 with lanes (20/20) | `experiments/e1_hosted/runs/` |
| E1-hosted, OpenAI | **Blocked on OpenAI:** `prompt_cache_diagnostics` returns `unavailable` on every comparison (GPT-5.6 Luna/Sol/base, delays up to 120 s, completed baseline). Not a Prefixwise bug. | `experiments/e1_hosted/runs/2026-11-06-round4.md` |
| E4: a fix saves what Prefixwise predicts, tasks still succeed | **Pass** 2026-11-06: `canonical_tools` 99.9% realised (Anthropic), 97.5% (vLLM); success flat (0/60, 1/60 changed) | `experiments/e4/runs/2026-11-06-live.md` |
| Dogfood: real agent traffic | **Done** 2026-11-06: opencode, 25 requests, 92.9% cached, 0 avoidable; 4 rough edges found and fixed (PR #18). Dogfood 2 (2026-11-07): 21 requests, 92.2% cached, 0 avoidable, fixes confirmed | `experiments/dogfood/2026-11-06-opencode.md` |
| E2/E3 engine join | **Pass** 2026-11-07/08: evictions confirmed 5/5 from vLLM and SGLang KV events; LMCache and HiCache restores 5/5 as hits, each `host_restored` (2026-11-08); two vLLM replicas behind a router: routing misses 5/5 confirmed; Prefixwise inside `vllm serve` as middleware 5/5 | `experiments/e_evict/runs/`, `experiments/e_route/runs/` |
| `prefixwise check` (CI gate) | **Shipped** 0.9.0 (2026-11-08): mock provider, allow-list, baseline, pytest plugin, GitHub Action `tejachavali/prefixwise@v0.11` | `docs/specs/check-and-optimize.md` |
| E-template: token-level divergence in real traffic | **Measured** 2026-11-09: "not an eviction" 0 of 108 requests (Qwen3-4B on vLLM and SGLang, DeepSeek-R1 distill, Qwen2.5, opencode and Claude Code against vLLM). What the runs showed instead: reasoning templates recompute the previous answer every turn (Qwen3 25-36%, R1 distill 57% of prompt tokens), now the detail `output_rerendered`; Qwen2.5 and Llama 3.2 reuse it in full | `experiments/e_template/runs/` |
| E5 compaction | Not started | |

## Now

**Prefixwise 0.11.4 is released** (2026-11-10; 0.11.0 to 0.11.3 the same day; 0.10.0 to 0.10.7 on 2026-11-09; PyPI and the GitHub Release match, the
Action's moving tag `v0.11` points at it (`v0.10` stays at 0.10.7), the README's links on PyPI resolve to GitHub). Since 0.4.0, in order: 0.5 to 0.8 (sub-reasons, importers for Claude Code and Codex
logs, LiteLLM/OpenTelemetry/SDK middleware, Prometheus and Grafana, the collector and record log, the
multi-replica engine join, SGLang data-parallel ranks over the wire), 0.9.0 (`prefixwise check`,
`host_restored`, docs/SGLANG.md and docs/LMCACHE.md), 0.10.0 (Prefixwise inside `vllm serve` as ASGI
middleware; engine auto-detection for `serve` and `run`; the block-rounded hit fix and the engine's block
size on the record; `output_rerendered`; E-template live on vLLM, SGLang, real agents and four templates).

**Nothing is running.** Open, in rough order of value:
- A Llama 3.2 run on vLLM with `VLLM_ARGS="--max-model-len 32768"` (it did not fit at the script's 131k),
  to see its full reuse live, and a session across the server's midnight on a Llama template, to catch the
  date-line divergence that Prefixwise would call "not an eviction".
- Data-parallel ranks inside one server (`--kv-events auto` on vLLM and SGLang): tested over the wire,
  needs two GPUs to run live. Middleware streaming under load and `--api-server-count` above 1 with a
  collector: not yet run live.
- Outreach beyond the vLLM Slack post (0.4.0): SGLang and LMCache communities have not been told; the
  guides for them (docs/SGLANG.md, docs/LMCACHE.md) are the material. Waits on Teja.

**Standing rules unchanged:** Muse runs only (tags releases); API keys never in files, reports or chat;
no force-pushes; independent review before every merge; nothing changed to make a run pass.

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
- E1-hosted: Anthropic, 32/32 against the reasons it named (plus 5 `unavailable` parameter changes).
- E4: `canonical_tools` proven.
- Proxy, CLI, streaming, Responses API, security hardening, CI, Docker; v0.3.0 ready.

**Now: launch.**
1. Dogfooding (Muse, "Now" B).
2. PyPI release: **done** 2026-11-07, `pip install prefixwise` (0.3.0).
3. A real README example from the dogfood report.
4. Soft launch to a few people who run agents, then wider.

**Next: product, in this order.**
1. **Dollar costs** *(done, PR #17; OpenAI GPT-5.6 Luna/Terra/Sol added in PR #22 from Muse task C)*: a dated price table per model, verified from the providers' own pricing pages, so `avoidable_cost` and request cost are filled in. Unknown rates stay null, never guessed. Users can supply their own rates file.
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
- **Done (0.11.4): `session_header_dropped`**, for sgl-project/sglang#47763 item 1, after a live run settled it (experiments/e_route/runs/2026-11-10-colab-l4-sglang-session-header.md): a session id changes SGLang's caching only with `--enable-session-radix-cache` and memory full, and the miss is an eviction, so it is a detail under `evicted`, gated on /server_info, not a cause. The run also showed what this note had wrong: with that flag Prefixwise's join is off, and Prefixwise removed the client's body session_id instead of carrying it past the router; fixed in #114. Still worth building when a real user needs it: confirmed `routing_miss` for SGLang workers behind a router, as vLLM replicas have.
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
- **2026-11-07:** Claude: per-model lanes (PR #21). They exposed two things: one-shot side calls were being read as `history_edit` (fixed), and E1-open Session B reverted to pre-compaction history after the model switch (harness fixed). All mocks pass; live re-check requested from Muse.
- **2026-11-07:** Muse task C: OpenAI prices read from developers.openai.com. Claude added them (PR #22) with a `valid_until` expiry for Sol's promotional price, and tightened model-id matching so `gpt-5.6-sol-pro` cannot borrow Sol's rates. Muse's Colab retry of the live E1 re-check and dogfood 2 is running.
- **2026-11-07:** Teja: benchmark and make it production-ready before outreach. Claude built `benchmarks/` (overhead, throughput, memory, soak, 13 fault cases; live overhead vs. a real provider for Muse) and fixed what it found (PR #23): mask-hash cache (3-4x lower overhead on long prompts), fail-open on ledger errors, upstream retry and rate-limit headers relayed, client idle timeout, graceful SIGTERM drain, `prefixwise prune`. CI now runs the fault suite and checks `docker stop` exits 0.
- **2026-11-07:** Muse: lanes live re-check passed on all three (E1-hosted Anthropic 100%, E1-open 120/120, dogfood 2 92.2% cached, 0 waste). Records committed by Claude. Open: no live run yet with Haiku calls interleaved between Sonnet turns.
- **2026-11-07:** PR #23 merged (production benchmark: 15-min soak 148k requests, 0 errors; 5 fixes). Teja: test on SGLang and LMCache. Claude researched both from source, added `--engine` to E1-open, launch scripts and E-evict (PR #24); Muse to run them on Colab. Next build after the results: the engine join (KV events -> `evicted`).
- **2026-11-07:** Teja: go ahead with 0.3.1, the live overhead check, then outreach; PR #24 (SGLang/LMCache) stays open and unmerged, its runs on hold. Muse confirmed its standing rule: runs only, no code changes or pushes; its lanes report stayed on Colab (Claude committed a transcription). Claude: version bump PR #25.
- **2026-11-07:** Muse published v0.3.1 (workflow green, PyPI latest). Live overhead check stopped at pair 5: `live_overhead.py` leaked cache breakpoints into the session history (5 > Anthropic's 4). Claude fixed it (deep copies, a one-breakpoint assertion, the fake upstream now enforces the limit, and a test) in PR #26. Re-run after merge.
- **2026-11-07:** Muse re-ran the live overhead check on 0ff0c70. Anthropic PASS: 98.1% cached both direct and through the proxy (MATCH), 20/20 recorded, median +29 ms on ~1 s calls (network variance; local proxy cost is 3-6 ms). OpenAI: OpenAI rejected the direct request (GPT-5.6 Luna refuses tools on chat completions unless `reasoning_effort` is `none`); a script bug, fixed in PR #27. OpenAI re-run pending.
- **2026-11-07:** Muse: OpenAI live overhead PASS on 0ac3cf8 (97.9% cached both ways, MATCH, 20/20, median +43 ms on ~1.1 s calls). Both providers now verified live. Claude recorded it (PR #28). Next: outreach, once Teja adds a line about Omni.
- **2026-11-07:** Teja: outreach waits; build adapters for SGLang and LMCache. Claude built the engine join (PR #29, stacked on #24, not merged): `prefixwise serve --engine vllm|sglang --kv-events ENDPOINT` reads vLLM/SGLang KV-cache events (wire format checked against both projects' source) so engine evictions are `evicted`, attributed per session on vLLM, engine-wide on SGLang; LMCache/HiCache restores stay hits. Found: `vllm_server.sh` used a KV-events field name current vLLM rejects (fixed). Not yet validated on a live engine; the E-evict runs (on hold) would do it.
- **2026-11-07:** Teja: research first, simplest correct design, best quality. Claude re-researched and reviewed PR #29: the first version over-built (SGLang engine-wide events, 2M-block tracking, a placeholder hack) and missed failure modes (engine restart, lost events). Rewritten: `--engine` gives an inferred `evicted` with no dependencies; vLLM `--kv-events` confirms it or rules it out ('not an eviction': blocks still resident, so token-level divergence or DP routing). SGLang events dropped (cannot be attributed to sessions). Also fixed: the merge of main had put #24's CHANGELOG entries under 0.3.1. Design note: docs/ENGINES.md. Still unmerged; not validated live.
- **2026-11-07:** Independent review of the engine join found 4 real defects (judging stale chains -> false 'confirmed'; hybrid-model sliding-window frees; a response-time race -> false 'not an eviction'; a session-id mismatch on retries). Redesigned: judge only the previous request's blocks, checked when the next request is sent; ask vLLM to report reused blocks (`vllm_xargs.kv_cache_report_mode = full`, found in vLLM's source: by default reused blocks are not reported); full-attention groups only; per-window trust. Second review requested.
- **2026-11-07:** Second review: send-time checking missed evictions while a request waits in vLLM's queue, and output blocks could still falsely confirm. Replaced with the stop-point rule: judge only the block where `cached_tokens` says the read stopped, tracking removals from send to response. 165 tests. Third review running.
- **2026-11-07:** Third review: false 'not an eviction' still possible on hybrid models, a busy engine (drain test), concurrent requests; arrival order could shift positions. Fixed: positions from parent hashes; the negative claim only when the stream caught up, no lost messages, full-attention model, one request in flight, cached tokens reported; hits never examined (no added latency). A new test caught a position off-by-one in the fix itself. 168 tests.
- **2026-11-07:** Fourth review: the design holds; three rare cases fixed (cap overflow no longer reads as removals; engine clock offset learned for the catch-up test; failed requests release their check). 170 tests. PR #29 description rewritten. Still unmerged; not validated live.
- **2026-11-07:** Teja approved the live engine runs. Claude added the verdict's evidence to E-evict's output (so 'confirmed' vs 'inferred' is visible) and lifted the hold.
- **2026-11-07:** Muse ran the live engine runs on Colab L4 (eae0684): 7 of 8 done, all as expected. vLLM eviction named 5/5 (inferred), and with KV events 'confirmed: 211 of the previous request's KV blocks removed' 5/5; LMCache turns it into a hit 5/5 (3328/3404 cached, `none`); E1-open 120/120 on vLLM+LMCache and on SGLang; SGLang eviction 5/5 (inferred). D (SGLang + HiCache) pending. Record: experiments/e_evict/runs/2026-11-07-colab-l4.md.
- **2026-11-07:** Muse: run D (SGLang + HiCache) passed: T3 5/5 hits (3382/3404 restored from host memory), Prefixwise `none` 5/5. All eight engine runs matched. PRs #24/#29 await Teja's go-ahead to merge (they need a catch-up merge with main first).
- **2026-11-07:** Teja: API-first and data-first. Claude audited what the ledger records (facts dropped, failures unrecorded, metric logic split between CLI and ledger) and researched the OpenTelemetry GenAI conventions (all experimental; no cost or miss-cause convention). Built on `claude/data-first`: schema 3 facts incl. failure rows; `prefixwise.metrics` (one catalog, one query); `/api/v1`; `prefixwise export`; docs/DATA.md and docs/API.md. Next: Prometheus/OTel export, then the report UX and `prefixwise run` on top.
- **2026-11-07:** Teja agreed the order: merge data-first (PR #30), Muse compatibility check (Claude Code, Codex, SDKs), then `prefixwise run` + the plain-English report, then release 0.4.0 before the Omni message. Built `prefixwise run` (per-provider upstreams, agent keeps its keys, report on that run) and the findings report (also `/api/v1/findings`), on `claude/run-report` stacked on PR #30.
- **2026-11-07:** PRs #30 (data first), #31 (run + report) and #29 (engines; #24 closed as included) merged. Fresh-install check from main passed (run, report, export); fixed mock-mode cost wording and cent precision. Release 0.4.0 prepared; publishing waits on Muse's compatibility check.
- **2026-11-07:** Teja already sent the Omni founder a message; that draft is retired. Next outreach: the vLLM Slack after 0.4.0. Added Python 3.10 support (suite passes on 3.10) and docs/VLLM.md for that audience.
- **2026-11-07:** Muse's compatibility check, 2 of 4: the Anthropic SDK script worked; the OpenAI SDK script found a real bug (Chat Completions then Responses in one session: a chatcmpl- id chained into prompt_cache_options, 400). Also seen: a 33-token Anthropic prompt reported as a misplaced breakpoint (Anthropic's minimum prefix was never applied). Both fixed with tests (PR #33). Claude Code and Codex still to run, on the fixed code.
- **2026-11-07:** Muse, 0.4.0 check so far: OpenAI SDK PASS (Chat Completions then Responses now work). Claude Code: task worked; Ctrl-C mid-stream showed proxy BrokenPipeError tracebacks in the terminal (fixed: silent disconnects, run logs proxy output to prefixwise-run.log). Also fixed: a sub-minimum previous request reported a history_edit as waste. Pending: Claude Code clean exit block (an older session reported 'No requests reached Prefixwise': to confirm on 0.4.0), Codex.
- **2026-11-07:** Codex ignored OPENAI_BASE_URL (research in the Codex source: removed; endpoint via `openai_base_url`, WebSocket first). `prefixwise run -- codex` now injects a custom provider via `-c` (HTTP+SSE, OPENAI_API_KEY, no sign-in, nothing written), and the proxy forwards Codex headers incl. x-codex-turn-state. Muse to re-test Codex on the release branch.
- **2026-11-07:** Compatibility check complete on the 0.4.0 RC: OpenAI SDK (Chat Completions + Responses) pass; Claude Code pass (82% cached, Ctrl-C clean); Codex exec pass on 220eb49 (1 request through the proxy). Release PR #32 to merge.
- **2026-11-07:** 0.4.0 released (GitHub release from 0de0e93, PyPI); verified from PyPI on Python 3.10 and 3.13 (run, report, export, /api/v1, engines extra). Teja posted to the vLLM Slack. README rewritten to lead with measured results and charts (PR #35, merged).
- **2026-11-07:** SGLang KV events (PR #34, ready, NOT merged): research showed SGLang >= 0.5.20 stamps a body `session_id` on BlockStored (0.4.0's "rejects unknown ids" was about `session_params`). Built `SGLangEvents` (tree of GPU blocks, leaf-first eviction, own-store barrier, /server_info gating); two independent reviews, all findings fixed; 217 tests. **Next, Muse:** live E-evict on SGLang with events (`KV_EVENTS=1 sglang_server.sh`, `e_evict.py --engine sglang --kv-events tcp://127.0.0.1:5557 --full-attention`); expected T3 `evicted` "confirmed" 5/5. Runs only; send output on any difference. If it passes: merge #34, add to README results, release 0.4.1.
- **2026-11-07:** Muse ran SGLang 0.5.23 with KV events live (Colab L4, Qwen3-4B; SGLang in a Python 3.12 venv because outlines_core has no 3.13 wheel): T3 `evicted` **confirmed 5/5**, T2 hits 5/5, 0 dropped messages; E1-open 120/120 with session ids sent. Record: experiments/e_evict/runs/2026-11-07-colab-l4-sglang-events.md. PR #34 to merge; then release 0.4.1.
- **2026-11-08:** 0.5.0 to 0.8.0 shipped (see CHANGELOG: sub-reasons, importers, LiteLLM/OTel/SDK middleware, Prometheus, collector and record log, multi-replica join, SGLang DP ranks over the wire). Three review edge cases fixed (#62). `prefixwise check` built to the amended spec (#63). `host_restored` (#64), docs/SGLANG.md and docs/LMCACHE.md (#65). 0.9.0 released (#66); Muse tagged v0.9.0 and the moving v0.9. Upstream auto-detection (#68) and the vLLM ASGI middleware (#69) merged after independent reviews. Muse: `host_restored` live 5/5 on LMCache 0.5.7 and SGLang HiCache (#70); middleware inside `vllm serve` 5/5 confirmed evictions (#71). Teja: hold 0.10.0 for one more validation round (token-level divergence; `--dp-size 2` needs two GPUs, deferred).
- **2026-11-09:** E-template built (#72). Muse ran it: vLLM first run exposed a classifier bug (block-rounded reads with a parameter change read as `param_rendered`), fixed with a strict block comparison and slack capped at an eighth of the prefix (#73), then the engine's block size on the record (#74). The Colab runtime recycled twice; Muse re-ran end to end on a fresh L4: "not an eviction" 0 of 50 harness turns and 0 of 8 agent requests; the previous answer recomputed on 47 of 50 Qwen3 turns. Built `output_rerendered` (engines only, from 160 output tokens, with the hit check), `prefixwise run` engine detection on both base URLs (#75). 0.10.0 released (#76; README audited; Muse tagged v0.10.0 and v0.10, hashes match). Muse swept three more templates (#77): DeepSeek-R1 distill 25/25 `output_rerendered` (57%), Qwen2.5 full reuse 25/25, Llama 3.2 template alone (vLLM did not fit at 131k). Then: PyPI README links made absolute at build time (hatch-fancy-pypi-readme); this file brought up to date after two days without entries.
- **2026-11-09 (later):** Reddit feedback on 0.10.0 turned into two fixes: a timestamp in any format is one volatile value and the `volatile_system` verdict names what the prompt carries (#80, after a review caught word-eating patterns and path masking in the first version); the Codex importer tells a reordered tool list from a changed catalog and prices the waste (#81). Also the PyPI README links (#78), the three-template record and README table (#77, #79). 0.10.1 released (#82); Muse tagged v0.10.1, moved v0.10, verified hashes and the PyPI links (0 relative, 21 to GitHub).
- **2026-11-09 (evening):** Two more Reddit cases turned into fixes: a session restart with a session-start timestamp is named (`volatile_prefix_at_start`, #83, after a review tightened tools-canonical matching, per-kind keys and the Anthropic write condition); unread cache writes are counted on partial hits, 1-hour part first (#84). 0.10.2 released (#85); Muse tagged v0.10.2, moved v0.10, hashes match. Four Reddit replies drafted in plain voice; the two for 0.10.1 can go up now, the two for 0.10.2 now too.
- **2026-11-09 (night):** `prefixwise explain --diff` (#86; review: line-by-line rendering, never silent, cheap on large blocks), released as 0.10.3 (#87; Muse tagged after a corrected precondition, since main had moved on). Then the presentation pass: every current Claude model priced, a family fallback the report names, unpriced models named, `report --share`, `report --prices`, README opening with the bill (#88; review caught a snapshot-as-minor pricing bug and a rank collision). 0.10.4 released (#89; Muse tagged v0.10.4 and moved v0.10 through the GitHub API, since git transport to github.com was being reset from its runtime; PyPI and GitHub hashes match). Then `--optimize` (#90): every proven fix in one flag, `PREFIXWISE_OPTIMIZE` for the middleware, `response_format` compared as a prompt-affecting parameter; the spec's key-sorting of other JSON built and dropped (providers parse it; `properties` order must not change). Reddit post for r/ClaudeCoding drafted; needs Teja's own `import claude-code` screenshot. Then `--fix auto_breakpoints` (#92; review caught the record reading the client's body for its markers, the masked-key guard, and a classifier gap: the first marked turn after unmarked ones is `first_write`). 0.10.5 released (#93); then `report --session` (#94, from r/opencodeCLI; review: time order for imported logs, `--since` honoured), Muse verified opencode through `prefixwise run` (record; the run exposed the view's `$` column as avoidable dollars, fixed with a cost column, #95), and `prefixwise fixes` says whether placed markers paid (#96; review: the reads are what the markers enabled, not only of their own writes). 0.10.6 released (#97). Screenshots for an r/opencodeCLI post showed the view's footer was wrong about the first-changed number (a position in the whole prompt, not a message index) and that first-write and model-switch rows named a meaningless block: fixed (#98), 0.10.7 released (#99). Muse's real opencode session (14 requests, 84.2% cached) led the r/opencode post. Next (done, see 2026-11-10): the E4 plan for `auto_breakpoints` (a Muse run on real Anthropic traffic) so it can join `--optimize`; `relocate_volatile` after that.
- **2026-11-10:** `auto_breakpoints` proven. E4 extended to replay a client without `cache_control`, scored in tokens and dollars (#100; review: the dollar rule on the prompt side, one unrewritten request fails the run). Muse ran it live on Haiku 5.5, first run: 106.8% of predicted savings, prompt-side cost down 85.6% with the surcharge, tool choices 59/60 off and 60/60 on, none rejected. The fix joined `PROVEN_FIXES` (#101; review found markers could reach a self-hosted engine through the middleware with no resend, a resend that needed the error to name the field, and `check` reading fixes from the environment: all fixed). Per-request rows committed beside the record (#102). Released as 0.11.0 (#103): `--optimize` now places markers on Anthropic requests that carry none. Also: an 87-second demo video (mock terminal scenes, the LMCache card from the 2026-11-08 record) and a message for Kuntai Du of LMCache, who replied he will try it; a runbook with Muse for the middleware inside vLLM with LMCache on, and `canonical_tools` on that server, both recorded. **Next:** Muse's LMCache-with-middleware result (then the one-flag path in docs/LMCACHE.md and real footage in the video); `relocate_volatile` only with an E4 plan; the opencode session importer if the threads ask.
- **2026-11-10 (later):** README cut from 510 lines to about 160 around identify, fix, prevent, auto-detect, with the old text kept whole as docs/GUIDE.md (#104; review cut ten overclaims and fixed three wrong record links). Muse ran `--optimize` for real on the Anthropic API (Sonnet 4.6, one scripted agent): 74.6% to 87.1% cached, $0.0356 to $0.0250, 3,375 avoidable tokens to none; the turn that flipped the tool list back read the older cache entry, which the mock had not predicted. Record and README updated (#105). Demo video rebuilt around the four parts with the real run cut in (86 s; the auto-detect line is still against a stand-in server). Released as 0.11.1 (#106), docs only. **With Muse:** the middleware inside vLLM with LMCache, the optimization replay on that server, and a real auto-detection line, all recorded. **Next:** cut that footage in; the one-flag path in docs/LMCACHE.md if it passes.
- **2026-11-10 (evening):** Muse ran the middleware inside vLLM with LMCache (host_restored 5/5, as through the proxy), the canonical_tools replay on that server (identical to vLLM alone), and the first live engine detection; recorded, docs/LMCACHE.md gains the one-flag setup (#107). A first-run audit from a clean PyPI install found Ollama/llama.cpp-style servers (no cached-token count) misread as routing_miss on every unchanged turn; now unknown with an explanation, prompt causes still named (#108; review: vLLM's null details, KV-confirmed evictions kept, no provider minimum). Released as 0.11.2 (#109). Focus decided with Teja: real users on real traffic before new features; engine operators first.
- **2026-11-10 (night):** Muse tagged v0.11.2 and moved v0.11 (step 9b run with python3 and the downloaded wheel). Fixed the proxy replacing the client's `x-session-id` with its own id (`default` for traffic with no Prefixwise session), which behind a session-affinity router would put every session on one worker: the client's value is now forwarded unchanged and used as the session, Codex's own headers rank above it, and Prefixwise sends none of its own (#111; review: header precedence, stale retry comments). The SGLang RFC `session_header_dropped` cause is deferred, with reasons, under Open questions. Released as 0.11.3 (#112).
- **2026-11-10 (late night):** Muse ran the SGLang session repro (0.5.23: a body session_id after a flood, 4,000 of 4,012 cached with `--enable-session-radix-cache`, 3 without; 3 either way with the flag off; sgl-router 0.3.2 has no session-header policy). #114: the client's body session_id passes through with that flag, and `session_header_dropped` names the eviction (review: wire compatibility with 0.11.3 collectors, no advice after a refused field; a Redis test race fixed). #115, from a Reddit question: `required` and `type` lists are sets for the detector and `canonical_tools` (review: a type list holding a schema raised, a 500; the Codex importer agrees). Released as 0.11.4.
