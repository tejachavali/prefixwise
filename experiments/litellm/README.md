# LiteLLM: Prefixwise through LiteLLM, and LiteLLM's own cache behaviour

**Question.** Does Prefixwise name cache misses correctly for traffic that goes through LiteLLM, in
every way LiteLLM is used (SDK sync and async, streamed and not, OpenAI and Anthropic formats, the
proxy server, the Router, the new `agent_eval`)? And, along the way: does LiteLLM itself keep real
agent conversations cache-friendly (translation, routing)?

**Versions.**
- LiteLLM 1.112.0.dev2, the first release with `agent_eval` (`litellm.harness.evaluation`).
- LiteLLM 1.108.3, the stable release, used for test A.
- Prefixwise `main` at the time: 0.12.2 plus the `agent_eval` labels (#144).
- All runs were on 2026-11-22, on one CPU machine. No model was called.

**Providers.** The tests run against simulated providers, never real APIs:

- **`cachemock.py`** caches like the real providers do, by content and not by session:
  - A request reads the longest prefix of itself that an earlier request left in the cache.
  - OpenAI format caches automatically at every block boundary, from 1,024 tokens up.
  - Anthropic format caches only at `cache_control` markers, and bills writes as `cache_creation_input_tokens`.
  - Both serve streamed and non-streamed responses, and usage in streams follows each API.
  - The cache is updated as each request arrives, so results are deterministic.
- **The `agent_eval` runs** use Prefixwise's own mock (`--mock` with tool calls), because the tool loop needs tool calls.

**Real data** (not committed: fetch it as below and set `PREFIXWISE_DATASETS`):

| Dataset | Source | Licence | Used |
|---|---|---|---|
| mini-swe-agent with Gemini 3.5 Flash on SWE-bench Multilingual: 264 trajectories, about 52 calls each, OpenAI chat format with tool calls | `github.com/SWE-bench/20260902_mini-v2.4.6_gemini-3-5-flash`, `trajs/*.traj.json`; the tool schema is `BASH_TOOL` in `github.com/SWE-agent/mini-swe-agent`, `src/minisweagent/models/utils/actions_toolcall.py` | none stated | B, C, D |
| tau-bench historical trajectories: GPT-4o and Claude 3.5 Sonnet, airline and retail, 1,980 in all | `github.com/sierra-research/tau-bench`, `historical_trajectories/` | MIT | B, C, D |
| A Claude Opus 4.6 mini-swe-agent run with Anthropic's real usage per call (`set_cache_control: default_end`) | `github.com/swe-bench/experiments`, `analysis/test_data/claude-4-6-opus.json` | none stated | E |

For tau-bench, the tool schemas live in that repository's Python code, which was not run. Each tool's
schema is rebuilt from the arguments the trajectory used, which is enough for these tests.

## Results

| Test | What | Result |
|---|---|---|
| A. Injected changes through the SDK | 6 cases (no change, tools reordered, timestamp in the system prompt, system edit, history edit, tool result edit) × OpenAI format / Anthropic with markers on the system prompt / Anthropic with markers on the last message × sync/async × streamed/not = 72 cases, `matrix_a.py` | **72/72** on 1.112.0.dev2, and **72/72** on 1.108.3 |
| B. LiteLLM's translation is append-only on real conversations | 664 trajectories (264 mini-swe-agent, 400 tau-bench), 17,981 step pairs per target. Each step translated by LiteLLM (body captured before sending, to a closed port). Check: step N's prompt is an exact prefix of step N+1's, markers ignored. `translate_b.py` | OpenAI, Anthropic, Bedrock invoke and Gemini: **17,981 / 17,981**. Bedrock Converse: **17,976 / 17,981** (finding below). Vertex AI not tested: it needs Google credentials before it builds a request. |
| C. Real conversations replayed through LiteLLM, with Prefixwise | 70 trajectories (30 + 40), replayed unchanged and with one change injected at a random step, in OpenAI and Anthropic formats: 25,450 calls, `replay_c.py` | **0 false alarms** in 25,450 calls; every call recorded. Of 768 injected changes, **738 named**, **10 seen and shown as costing nothing**, **0 missed**. The 10 are OpenAI-format edits whose loss was under one 128-token reporting unit (17 to 103 tokens): Prefixwise treats a read within that tolerance as complete, by design (ALGORITHM.md section 1), and keeps the change as a secondary cause. |
| D. The Router across two deployments | 14 trajectories × {load-balanced, `prompt_caching`, `session_affinity`} × {OpenAI, Anthropic}; two cachemock processes with separate caches; ground truth is the deployment that served each call (`model_id`). `router_d.py` | Turns moved to the other deployment: load-balanced 171/373; `prompt_caching` 176 to 186 / 373; `session_affinity` **0 / 373** (96.7% from cache, against 93.6%). Prefixwise named every move to a cold deployment (nothing read) `routing_miss` (12 per setup), and never named one without a move. A move to a deployment holding an older part of the conversation reads part of it. With at least half read, it counts as a hit (finding below). With less than half read, it is `unknown` (3 to 6 per setup): on the OpenAI format, where the cache lifetime is not known, expiry and routing cannot be told apart. |
| E. Real provider numbers | The Claude Opus 4.6 run's 36 calls, each request rebuilt as LiteLLM sends it, with Anthropic's real usage, through `prefixwise import gateway`. `real_usage_e.py` | `first_write`, then `below_min_prefix` (call 1 was 2,054 tokens, under Opus's 4,096 minimum, and Anthropic indeed cached nothing), then 34 hits; 95% from cache; no false alarm |
| F. `agent_eval` | 3 tasks × 2 trials, the tool loop, direct and through the LiteLLM proxy server. `eval_direct.py`, `eval_gw.py` | Before #144: one session for the whole eval; 5 of 12 calls were named `history_edit` when they were each a new run's first call. After #144, through the proxy server: 6 runs, 6 sessions with task and target, each `first_write` then a hit. Direct mode: LiteLLM passes no labels to the model call yet (finding below). |

## Findings

**In Prefixwise (fixed or documented):**
- `agent_eval` runs were not separated: fixed in #144. The eval's judge is recorded apart as `agent_eval_judge`.
- **A partial read after a routing switch counts as a hit.** With two deployments, a turn moved back to a deployment that holds an older part of the conversation reads most of it, for example 1,821 of 2,200 tokens. Prefixwise counts a read of at least half as a hit (ALGORITHM.md section 1), so the tokens lost to the switch are not named. Misses on a cold deployment are named. Naming the partial loss would change a rule the paper measured, so it is left as is and noted here.
- **Test harnesses must keep the event loop alive.** LiteLLM logs async calls on a background task, so a script that wraps each `acompletion` in its own `asyncio.run()` loses the record. That is LiteLLM's design; long-running services are unaffected.

**In LiteLLM** (observed with the scripts here; for LiteLLM's maintainers to confirm):

1. **Bedrock Converse can rewrite an earlier message.** When an assistant turn is empty (no text, no tool calls), the translation drops it and merges the tool result before it with the next user message. On the turn before, that tool result ended the prompt unmerged. Adding the next user message therefore changes an earlier message, and the cached prefix breaks there.
   - Seen 5 times in 17,981 step pairs, all in tau-bench Claude 3.5 Sonnet trajectories with an empty assistant turn.
   - Anthropic, Bedrock invoke and Gemini translate the same conversations append-only.
2. **The Router's `prompt_caching` check keeps a conversation on its deployment only when the cache markers are in the messages.** It held 12 of 12 turns on one deployment with `cache_control` written by the caller. Turns still moved:
   - with markers added through `cache_control_injection_points`;
   - for OpenAI-format conversations, which have no markers.

   The check's own comment says the routing key should account for markers injected later.
3. **`session_affinity` reads `metadata.session_id`, not `litellm_session_id`.** With the session given only as `litellm_session_id`, turns still moved (as in the first run of D). With `metadata.session_id`, no turn moved.
4. **`agent_eval`'s tool loop labels its calls only through the gateway.** In direct mode, `llms/tool_loop/harness/transformation.py` `_routing_kwargs` passes no `metadata`, so no callback (Prefixwise, or any observability integration) can tell which run, task or target a call belongs to. Passing `metadata=dict(ctx.metadata)` there (one line) was checked to give the same result as the gateway.
5. **`agent_eval`'s scorecard has no cache numbers.** `harness.types.Usage` holds input tokens, output tokens and calls only, so the scorecard cannot show how much of a task's cost was cache reads, writes or misses. Prefixwise's ledger, keyed by the eval's run, task and target (#144), can.

## Rerun

```bash
pip install --pre "litellm[proxy]==1.112.0.dev2" -e .   # in a separate environment
export PREFIXWISE_DATASETS=$PWD/experiments/litellm/datasets
# fetch the datasets above into $PREFIXWISE_DATASETS (sparse git clones), as:
#   mini-gemini35/trajs/*.traj.json
#   SWE-agent_mini-swe-agent/src/minisweagent/models/utils/actions_toolcall.py
#   sierra-research_tau-bench/historical_trajectories/*.json
#   swebench-experiments/analysis/test_data/claude-4-6-opus.json
cd experiments/litellm
python matrix_a.py a.db                        # A
python translate_b.py b.json 264 400           # B, about 20 minutes
python replay_c.py c.json 30 40 11             # C, about 20 minutes
python -I cachemock.py 18601 & python -I cachemock.py 18602 &
python router_d.py d.json 4 10 18601 18602     # D
python real_usage_e.py e.jsonl && prefixwise import gateway --path e.jsonl --ledger e.db   # E
python eval_direct.py f.db                     # F, direct
python mockserve.py 18531 &                    # F, through the proxy server
PREFIXWISE_LEDGER_PATH=$PWD/gw.db LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true litellm --config gw.yaml --port 18532 &
python eval_gw.py gw.db gpt-5.6
```

Records: `runs/2026-11-22-matrix.txt` (A, every case), `runs/2026-11-22-translation.json` (B, with the
five Bedrock Converse examples), `runs/2026-11-22-replay.json` (C), `runs/2026-11-22-router.json` (D).
