# E4: does a fix save what Prefixwise predicts, without breaking tasks?

E1 showed Prefixwise names cache-miss causes correctly (vLLM 100%; Anthropic 32/32 against the miss reasons it named). E4 checks the other half of the claim: **apply the fix Prefixwise recommends, and the predicted savings show up while tasks still succeed.**

The fix under test is the proxy's `canonical_tools` toggle (`prefixwise serve --fix canonical_tools`). It sorts tools by name and sorts each tool's keys, so a tool list that drifts between turns produces the same bytes every time.

## The recorded sessions

`sessions/tool-order-drift.jsonl` holds 6 sessions of 10 turns, deterministic from `--seed 20261106`. They reproduce a common agent bug: the tool list is rebuilt between turns (plugin reload, dict rebuild), so tool order and key order drift. That happens on 29 of 60 turns. Each turn asks a task that needs one specific tool, for example "What is 48213 * 9127?" needs `calculator`. **Task success** means the model called that tool.

## Pass

Over at least 50 paired requests:
1. **realised savings ≥ 50% of predicted.** Predicted is the avoidable tokens the fix-off ledger attributes to `tool_reserialize`, which is what `prefixwise fixes` would promise. Realised is cached tokens with the fix on minus cached tokens with it off, on the same turns.
2. **Task success flat:** with the fix on, success is no more than 5 points below success with it off. Savings never excuse broken tasks.

The two arms are isolated so one can't warm the other's cache: on vLLM by a per-arm `cache_salt`, on Anthropic by waiting out the 5-minute TTL between arms (`--arm-gap`, 330 s by default).

## Run it

```bash
python experiments/e4/e4.py --mock          # free: a real prefix-cache simulator, always-correct tool calls

# vLLM (free on a GPU). Start the server as in E1-open (vllm_server.sh or the Colab notebook,
# with --enable-auto-tool-choice --tool-call-parser hermes), then:
python experiments/e4/e4.py --provider vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-4B

# Anthropic (about 120 requests, a few cents on Haiku 5.5; waits ~5.5 min between arms)
python experiments/e4/e4.py --provider anthropic --model claude-haiku-5-5 --max-input-tokens 900000
```

**Live result (2026-11-06): Pass on both.**
- **Anthropic Haiku 5.5:** 99.9% of predicted savings realised (104,616 of 104,732 tokens), 0 of 60 tool choices changed.
- **vLLM Qwen3-4B:** 97.5% realised (13,872 of 14,229), 1 of 60 changed.

See [`runs/2026-11-06-live.md`](runs/2026-11-06-live.md), including why the vLLM saving is smaller (Qwen3 renders tools after the system prompt).

Mock result (2026-11-06): predicted 91,399 tokens, realised 91,184 (99.8%), Pass. The mock's cache is independent of Prefixwise: it computes the longest prefix each request shares with an earlier one. So this shows Prefixwise's avoidable-token accounting matches real prefix-cache behaviour. Task success in the mock is always 100%; only a live model can show whether reordering tools changes which tool it picks.
