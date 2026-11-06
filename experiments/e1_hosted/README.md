# E1-hosted: Prefixwise vs the providers' own cache diagnostics

E1-open showed Prefixwise names injected causes on live vLLM (Pass, 120 turns, 2026-11-06). E1-hosted asks the provider-facing question: on **Anthropic** (Messages API, `diagnostics.previous_message_id` chained) and **OpenAI** (Responses API, `prompt_cache_options.comparison_response_id` chained), does Prefixwise's cause agree with the provider's own miss reason?

**Pass:** native agreement >= 95% over turns that have a native reason, and UNKNOWN <= 10% of misses.

Native reasons are coarse, so agreement means the Prefixwise cause is one the native reason covers (`diagnostics.NATIVE_COVERS`). For example Anthropic's `system_changed` covers both `system_edit` and `volatile_system`, but never `tool_reserialize`.

## Run it, cheapest step first

```bash
python experiments/e1_hosted/e1_hosted.py --mock                 # free, no network: harness smoke test
python experiments/e1_hosted/e1_hosted.py --estimate             # tokens a full run will send
python experiments/e1_hosted/e1_hosted.py --probe --providers anthropic    # 5 real turns, about a cent
python experiments/e1_hosted/e1_hosted.py --providers anthropic --max-input-tokens 900000
```

**Always probe first.** The probe sends 5 turns: a first write, a hit, two deliberate misses (tools reordered, then system prompt edited), and a hit. A miss reason only exists on a miss, so `null` on the first write and the hits is expected. The probe passes when every *real* miss carries a native reason where Prefixwise reads it. It exits 3 and prints `STOP` otherwise, along with every cache- or diagnostics-shaped field it found, so you can see where the provider put the reason.

Keys: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, or none if a network proxy injects them per host. A 401, 403 or 404 stops the run immediately, so a wrong key or model id costs nothing more.

## Cost

A full run is 100 turns per provider. `--estimate` says about 740k input tokens each, but it counts 4 characters per token and the first real probe (2026-11-06) measured about 2.4x fewer tokens, so expect roughly 300k. Roughly half are deliberate misses (prefix re-processed), and half are cache reads. Output is capped at 16 tokens per turn.

| Provider / model | Approx. full-run cost |
|---|---|
| Anthropic Claude Haiku 5.5 (plus Sonnet 5.5 for the 10 model-switch turns) | well under $1 |
| OpenAI GPT-5.6 Luna | about $0.35 |
| OpenAI GPT-5.6 Sol | about $1.75 |

These are estimates from third-party price listings as of Nov 2026. Check the providers' pricing pages. `--max-input-tokens` is a hard stop.

## What is injected

Sessions of 10 turns. A mutation stays in place for the following turns, as a real harness would leave it.

- **A:** first_write, none, volatile_system, none, tool_reserialize, none, tool_catalog_change, none, system_edit, none
- **B:** first_write, none, param_rendered, none, compaction_rewrite, none, model_switch, model_switch (back), none, none

Not injected: `retry_replay` and `ttl_expired`. A hosted cache can't be made cold on demand, and a TTL test means waiting 5+ minutes per turn. Every turn's raw usage and raw diagnostics are saved in the JSON report.
