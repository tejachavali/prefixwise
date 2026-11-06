# E1-open

Validate the Prefixwise classifier without Anthropic or OpenAI API credits, using open Hugging Face models on vLLM, with controlled injected causes as ground truth.

## What it proves

100 turns, each with a KNOWN injected cause. The classifier is scored against what the harness deliberately changed, not against a provider's native reason (open models have none).

Sessions of 10 turns, chained so each turn is scored against the previous turn in the same session:

- Session A x5: FIRST_WRITE, NONE, VOLATILE_SYSTEM, VOLATILE_SYSTEM, TOOL_RESERIALIZE, TOOL_RESERIALIZE, TOOL_CATALOG_CHANGE, TOOL_CATALOG_CHANGE, SYSTEM_EDIT, SYSTEM_EDIT
- Session B x5: FIRST_WRITE, NONE, COMPACTION_REWRITE, COMPACTION_REWRITE, NONE, RETRY_REPLAY, RETRY_REPLAY, MODEL_SWITCH, MODEL_SWITCH, NONE

Totals: FIRST_WRITE 10, NONE 20, VOLATILE_SYSTEM 10, TOOL_RESERIALIZE 10, TOOL_CATALOG_CHANGE 10, SYSTEM_EDIT 10, COMPACTION_REWRITE 10, RETRY_REPLAY 10, MODEL_SWITCH 10.

Pass criteria: agreement >= 95%, UNKNOWN <= 10% of misses (misses = not NONE / FIRST_WRITE).

## How it works

`e1_open.py` drives the real Prefixwise proxy (`src/prefixwise/proxy.py`), which forwards to the vLLM server with `X-Request-Id` and `X-Session-ID` set, so the engine join is exercised. The proxy fingerprints, classifies (real classifier, not a copy), and records a ledger entry for every turn. Results and a score report are written to `results/e1_open_<timestamp>.json` and `.md`.

Injections, all other blocks identical to the previous turn:

- VOLATILE_SYSTEM: system date changes only (masked hash equal).
- TOOL_RESERIALIZE: tool list reversed, same tool set.
- TOOL_CATALOG_CHANGE: one tool removed, then restored.
- COMPACTION_REWRITE: history replaced by a summary, with `prefixwise_compaction_event=True` (PreCompact snapshot).
- MODEL_SWITCH: model field switches to the served-model alias. A single-model vLLM server rejects an unknown model id, so if the upstream rejects the alias the harness falls back to in-process classification of the same request (marked `upstream_error=True` in the report) instead of dropping the turn.
- RETRY_REPLAY: identical request, after a best-effort POST to vLLM `/reset_prefix_cache`. If the reset endpoint does not exist, the turn is scored honestly as whatever the classifier says (usually NONE).
- NONE: identical request, expect a cache hit.

## Run it in mock mode (no GPU, no server)

```bash
cd <repo root>
PYTHONPATH=src python3 experiments/e1_open/e1_open.py --mock
```

This simulates vLLM `cached_tokens` per injected cause and runs the full scoring path (proxy, classifier, ledger). It is a smoke test of the harness, not evidence about vLLM: the mock values are chosen from the mechanism, so a 100% mock score means the harness and classifier agree on controlled inputs, not that vLLM behaves that way.

## Run it live on a GPU box

```bash
# Terminal 1: start vLLM (needs vLLM >= 0.30.0, Sep 22, 2026, for session_id on KV events)
MODEL=Qwen/Qwen3-8B bash experiments/e1_open/vllm_server.sh

# Terminal 2: wait for the server to be ready, then run E1-open
curl -s http://127.0.0.1:8000/health
PYTHONPATH=src python3 experiments/e1_open/e1_open.py \
  --vllm-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B \
  --model-alias Qwen/Qwen3-8B-alias
```

GPU requirements (approximate, FP16/BF16, no quantization):

| Model | GPU |
|---|---|
| Qwen/Qwen3-4B (fallback) | 1x 16 GB (Colab T4) |
| Qwen/Qwen3-8B (default) | 1x 24 GB (Colab L4, RTX 4090) |
| Qwen/Qwen3-32B | ~40 GB+ (1x A100 80GB, or 2x L4) |
| meta-llama/Llama-3.3-70B-Instruct | 2x H100 80GB, or 1x GPU with `--quantization fp8` (`VLLM_ARGS="--quantization fp8 --max-model-len 8192"`) |

No API keys are needed. A Hugging Face token (`HF_TOKEN`) is only needed for gated models; Qwen3-8B is not gated. Llama 3.3 is gated: accept the license on Hugging Face and set `HF_TOKEN` before starting the server.

Each turn generates at most 16 tokens (`max_tokens=16, temperature=0`), so the 100-turn run is mostly prefill, not generation.

## Run it on Colab

Open `colab_e1_open.ipynb` in Google Colab, pick a GPU runtime (Runtime, Change runtime type, T4 or L4 GPU), then Run all. The notebook installs vLLM and Prefixwise, starts the server on the Colab GPU, waits for it, runs `e1_open.py`, and prints the score report. Default model is Qwen3-8B on L4; if 8B does not fit, set `MODEL = "Qwen/Qwen3-4B"` in the first code cell and rerun.

## Interpreting the report

- Agreement < 95%: look at the per-cause table and the misses list. A cause at 0% usually means the injection did not produce the evidence the classifier needs (for example, the cache was not actually reset for RETRY_REPLAY, or the model alias was not served for MODEL_SWITCH); the report marks those turns with `upstream_error` or a low `cached_tokens`.
- UNKNOWN > 10%: the classifier is honest but blind on those turns; that is the signal to add evidence (engine events, TTL, gap), not to guess.
- Mock mode 100% is expected by construction and proves the harness, not vLLM. Only the live run moves the proven score.

## What E1-open does not prove

E1-hosted is still the provider-facing proof later: 100 turns on Anthropic plus OpenAI with native diagnostics chained, agreement >= 95% with the provider's own reason, UNKNOWN <= 10%. E1-open proves the classifier on controlled open-model traffic; E1-hosted proves it against provider ground truth.
