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
  - It also serves the OpenAI Responses API (automatic, as chat completions), Claude in chat format (OpenRouter: at `cache_control` markers, writes as `cache_write_tokens`) and Bedrock Converse (at `cachePoint`s; not streamed).
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
| A. Injected changes through the SDK | 6 cases (no change, tools reordered, timestamp in the system prompt, system edit, history edit, tool result edit) × OpenAI format / Anthropic with markers on the system prompt / Anthropic with markers on the last message × sync/async × streamed/not = 72 cases, `matrix_a.py` | **72/72** on 1.112.0.dev2, and **72/72** on 1.108.3. In 64 cases the change is named as the cause. In the other 8 (Anthropic with markers on the system prompt only, history or tool-result edits), nothing after the system prompt was cached, so the edit cost nothing: the expected verdict is `none` with the edit as a secondary cause, and that is what was recorded. |
| B. LiteLLM's translation is append-only on real conversations | 664 trajectories (264 mini-swe-agent, 400 tau-bench), 17,981 step pairs per target. Each step translated by LiteLLM (body captured before sending, to a closed port). Check: step N's prompt is an exact prefix of step N+1's, markers ignored. `translate_b.py` | OpenAI, Anthropic, Bedrock invoke and Gemini: **17,981 / 17,981**. Bedrock Converse: **17,976 / 17,981** (finding below). Vertex AI not tested: it needs Google credentials before it builds a request. |
| C. Real conversations replayed through LiteLLM, with Prefixwise | 70 trajectories (30 + 40), each replayed unchanged and with one change injected at a random step (where the trajectory can carry it), in OpenAI and Anthropic formats: 768 replays, 23,450 calls, `replay_c.py` | **0 false alarms** in 23,450 calls; every call recorded. Of 628 injected changes, **618 named**, **10 seen and shown as costing nothing**, **0 missed**. The 10 are OpenAI-format edits that cost 17 to 107 tokens, under one 128-token reporting unit: Prefixwise treats a read within that tolerance as complete, by design (ALGORITHM.md section 1), and keeps the change as a secondary cause. |
| D. The Router across two deployments | 12 trajectories (4 + 8) × {load-balanced, `prompt_caching`, `session_affinity`} × {OpenAI, Anthropic}, 373 calls each; two cachemock processes with separate caches; ground truth is the deployment that served each call (`model_id`); the session sent as both `litellm_session_id` and `metadata.session_id`. `router_d.py` | Turns moved to the other deployment, of the 361 that could move: load-balanced 171; `prompt_caching` 176 to 186; `session_affinity` **0** (96.7% from cache, against 93.6%). Prefixwise named every move to a cold deployment (nothing read) `routing_miss` (12 per setup and format), and never named one without a move. A move to a deployment holding an older, shorter part of the conversation reads part of it. With at least half read, it counts as a hit (finding below). With less than half read, it is `unknown` (3 to 6 per setup and format), with the evidence saying the reusable prefix was not read and why it cannot be told: on the OpenAI format, expiry or routing (the cache lifetime is not known); on the Anthropic format, no timing or usage evidence names a cause. |
| E. Real provider numbers | The Claude Opus 4.6 run's 36 calls, each request rebuilt as LiteLLM sends it, with Anthropic's real usage, through `prefixwise import gateway`. `real_usage_e.py` | `first_write`, then `below_min_prefix` (call 1 was 2,054 tokens, under Opus's 4,096 minimum, and Anthropic indeed cached nothing), then 34 hits; 94.8% from cache; no false alarm |
| G. Live: the Messages API on real APIs | `live_messages.py`: 4-turn conversations through LiteLLM's Messages API (the SDK, and the proxy server's `/v1/messages` with Claude Code's session header), one change injected at turn 3, streamed and not, on real Anthropic (claude-haiku-4-5) and OpenAI (gpt-4.1-mini); Prefixwise 0.12.4, LiteLLM 1.110.2; `openai_cache_probe.py` for OpenAI's misses | Anthropic: **20 of 20** conversations right, every verdict matching Anthropic's own cache numbers. OpenAI: every injected change named (16 of 16), but OpenAI read nothing on 38 of the 40 turns that only appended, named `routing_miss`; the probe showed those misses are OpenAI's Responses API without `prompt_cache_key` (directly too), not LiteLLM or Prefixwise (finding 6) |
| H. Tests B and C on Hugging Face datasets | `hf_litellm.py` on four public datasets: Open-SWE-Traces (Claude Code), SWE-smith (SWE-agent), SWE-rebench OpenHands, APIGen-MT (tau-bench style); translation of 389 conversations to five targets; replays of 115 conversations through four routes (chat completions to OpenAI and to Claude, the Messages API to Claude and to OpenAI) with one change injected | Translation append-only on **5,054 / 5,054** step pairs per target for OpenAI, Anthropic and both Bedrock formats, and **3,550 / 3,550** for Gemini (LiteLLM built no Gemini request for the OpenHands conversations: a test-loader bug, fixed in I). Replays: **0 false alarms** in 38,016 calls, every call recorded; of 2,300 injected changes, **1,813 named**, 4 seen at no cost, **0 missed**, and 483 after a turn below the model's minimum cacheable prefix (nothing to lose, counted apart) |
| I. More Hugging Face datasets, and Gemini again | `hf_litellm.py` on JetBrains-Research SWE-smith, Kwai-Klear SWE-smith (mini-swe-agent), allenai Sera, and SWE-rebench OpenHands again with the loader fixed; translation of 110 conversations to five targets; replays of 30 conversations through the four routes of H | Translation append-only on **1,144 / 1,144** step pairs for each of the five targets, Gemini included, and on the OpenHands conversations **300 / 300** for Gemini (0 before the fix). Replays: **0 false alarms** in 6,900 calls, every call recorded; of 520 injected changes, **430 named**, **0 missed**, 90 after a turn below the minimum. Five more datasets gave no usable conversations (`runs/2026-11-23-round3.md`) |
| J. Real Claude Code through the proxy server | Claude Code (`claude -p`, claude-haiku-4-5) fixing two bugs through `litellm --config live_proxy.yaml` with Prefixwise's callback, on the real Anthropic API | Claude Code reported both bugs fixed and the tests passing; **7 of 7** calls recorded under Claude Code's session: `first_write`, then 6 turns each reading all of the reusable prefix (97.1% to 98.8% of the prompt); 84.9% from cache, nothing avoidable, $0.0519 |
| F. `agent_eval` | 3 tasks × 2 trials, the tool loop, direct and through the LiteLLM proxy server. `eval_direct.py`, `eval_gw.py` | Before #144 (direct mode): one session for the whole eval; 5 of 12 calls were named `history_edit` when they were each a new run's first call. After #144, through the proxy server: 6 runs, 6 sessions with task and target, each `first_write` then a hit. Direct mode: LiteLLM passes no labels to the model call yet (finding below); with that one line passed, the same as through the proxy server. |

## Findings

**In Prefixwise (fixed or documented):**
- `agent_eval` runs were not separated: fixed in #144. The eval's judge is recorded apart as `agent_eval_judge`.
- **A partial read after a routing switch counts as a hit.** With two deployments, a turn moved back to a deployment that holds an older part of the conversation reads most of it: in test D, for example, a turn of 21,925 tokens read 15,899, where the previous turn on the other deployment had sent 21,541. Prefixwise counts a read of at least half as a hit (ALGORITHM.md section 1), so the tokens lost to the switch are not named. Misses on a cold deployment are named. Naming the partial loss would change a rule the paper measured, so it is left as is and noted here.
- **Replayed agent logs must be paired the way APIs require.** In the first live replays of real conversations (round 3), every replay's first call with a tool call in it was rejected (64 of 128 calls): Claude Code logs each content block of one reply as its own assistant entry, so the test's converter put another assistant message between a tool call and its result. Review found three more such shapes (repeated tool-call ids, from datasets without ids; split entries sharing an id; an assistant entry before the first user message). The offline cache mock does not check pairing, so tests C and H could not see it. `hf_litellm.py` now handles all four, skips and counts any conversation a live API would still reject, and checks seven shapes in `hf_litellm.py selftest`. The live replays are to be run again.
- **Test harnesses must keep the event loop alive.** LiteLLM logs async calls on a background task, so a script that wraps each `acompletion` in its own `asyncio.run()` loses the record. That is LiteLLM's design; long-running services are unaffected.

**In LiteLLM** (observed with the scripts here; for LiteLLM's maintainers to confirm):

1. **Bedrock Converse can rewrite an earlier message.** When an assistant turn is empty (no text, no tool calls), the translation drops it and merges the tool result before it with the next user message. On the turn before, that tool result ended the prompt unmerged. Adding the next user message therefore changes an earlier message, and the cached prefix breaks there.
   - Seen 5 times in 17,981 step pairs, all in tau-bench Claude 3.5 Sonnet trajectories with an empty assistant turn.
   - Anthropic, Bedrock invoke and Gemini translate the same conversations append-only.
2. **The Router's `prompt_caching` check keeps a conversation on its deployment only when the cache markers are in the messages.** With `cache_control` written by the caller, it held 12 of 12 turns on one deployment, for two model names (`runs/2026-11-22-router-prompt-caching.txt`). Turns still moved:
   - with markers added per request through `cache_control_injection_points`;
   - for OpenAI-format conversations, which have no markers.

   The check derives its routing key with `AnthropicCacheControlHook.messages_with_default_injections`, which reproduces LiteLLM's automatic default breakpoints (`enable_prompt_caching`), but not a caller's `cache_control_injection_points`.
3. **`session_affinity` reads `metadata.session_id`, not `litellm_session_id`** (`deployment_affinity_check._get_session_id_from_request_kwargs` reads only the metadata dicts). In a first, smaller run of D with the session given only as `litellm_session_id`, `session_affinity` moved 81 and 83 of the 157 turns that could move, about as many as load-balanced routing in the same run (89 and 69) (`runs/2026-11-22-router-litellm-session-id-only.json`). With `metadata.session_id`, no turn moved.
4. **`agent_eval`'s tool loop labels its calls only through the gateway.** In direct mode, `llms/tool_loop/harness/transformation.py` `_routing_kwargs` passes no `metadata`, so no callback (Prefixwise, or any observability integration) can tell which run, task or target a call belongs to. Passing `metadata=dict(ctx.metadata)` there (one line) was checked to give the same result as the gateway.
5. **`agent_eval`'s scorecard has no cache numbers.** `harness.types.Usage` holds input tokens, output tokens and calls only, so the scorecard cannot show how much of a task's cost was cache reads, writes or misses. Prefixwise's ledger, keyed by the eval's run, task and target (#144), can.
6. **Messages API calls sent on to OpenAI's models carry no `prompt_cache_key`.** LiteLLM sends a `/v1/messages` call for an OpenAI model through the Responses API without one. Live (G), OpenAI then read nothing on 38 of 40 turns that only appended; with a `prompt_cache_key` set on the same requests sent directly, it read the prefix on 3 of 3, and through Chat Completions on 3 of 3 (`runs/2026-11-22-live.md`). LiteLLM has a stable key at hand: the session (`litellm_session_id`, which its proxy server takes from Claude Code's `x-claude-code-session-id`).
7. **The Gemini translation fails on a tool parameter whose schema is `null`**, with `'NoneType' object has no attribute 'get'`, an error that does not name the tool or the parameter. OpenAI, Anthropic and both Bedrock formats pass the same tools through. Datasets stored as Parquet produce such nulls, which is how test H met it; an agent building tool schemas from typed structs could too. Reproduced with 1.110.2 (`runs/2026-11-23-round3.md`).


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
# G, live (provider keys in the environment; see the scripts' docstrings for the proxy route)
python live_messages.py live-anthropic.json --route sdk --model anthropic/claude-haiku-4-5
python openai_cache_probe.py probe-openai.json --model gpt-4.1-mini
# The same tests as B and C on agent conversations from Hugging Face (pip install datasets)
python hf_litellm.py translate t.json --dataset nvidia/Open-SWE-Traces --split claudecode --limit 100
python hf_litellm.py replay r.json --dataset nvidia/Open-SWE-Traces --split claudecode --limit 30
python hf_litellm.py selftest                  # the converter gives conversations a live API accepts
```

H and I ran with the converter as it was before the live-API fixes (at `ff825e0a`); run now, they merge split
assistant entries and rename repeated ids, so their counts differ.

Records:
- A, every case: `runs/2026-11-22-matrix-1.112.0.dev2.txt` and `runs/2026-11-22-matrix-1.108.3.txt`.
- B, with the five Bedrock Converse examples: `runs/2026-11-22-translation.json`.
- C: `runs/2026-11-22-replay.json`.
- D: `runs/2026-11-22-router.json`, with the session in both fields, and `runs/2026-11-22-router-litellm-session-id-only.json`, the earlier run with `litellm_session_id` only.
- Finding 2: `runs/2026-11-22-router-prompt-caching.txt`, from `router_prompt_caching.py`.
- E and F: `runs/2026-11-22-e-f.md`, the ledgers summarised.
- G and finding 6: `runs/2026-11-22-live.md`, Muse's tables and the OpenAI probe.
- H: `runs/2026-11-23-hf.md`, read from the result files (kept by Muse).
- I, J and finding 7: `runs/2026-11-23-round3.md`, with the result files and Claude Code's ledger in `runs/2026-11-23-round3/`.

The F ledger from before #144 through the proxy server was overwritten while testing; its direct-mode counterpart is recorded.
