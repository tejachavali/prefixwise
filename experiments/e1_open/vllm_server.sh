#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for E1-open.
#
# Usage:
#   MODEL=Qwen/Qwen3-8B ./experiments/e1_open/vllm_server.sh
#   MODEL=Qwen/Qwen3-32B ./experiments/e1_open/vllm_server.sh
#   MODEL=meta-llama/Llama-3.3-70B-Instruct ./experiments/e1_open/vllm_server.sh
#
# GPU requirements (approximate, FP16/BF16 weights, no quantization):
#   Qwen3-8B            1x 24 GB GPU (RTX 4090) -- default on a GPU box
#   Qwen3-4B            1x 16 GB GPU (Colab T4) -- Colab default (see below)
#   Qwen3-32B           ~40 GB+ (1x A100 80GB, or 2x L4)
#   Llama-3.3-70B       2x H100 80GB, or 1x GPU with --quantization fp8
#                       (add VLLM_ARGS="--quantization fp8 --max-model-len 8192")
#
# Colab note (live run Nov 5-6, 2026, vLLM 0.33.0, L4): Qwen3-8B
# failed to start ("RuntimeError: Engine core initialization failed",
# server never ready in 10 min). Qwen3-4B worked on the same L4.
# So on Colab use MODEL=Qwen/Qwen3-4B; 8B needs a larger GPU
# (RTX 4090 / A100 class) than a Colab L4 in practice, despite the
# 24 GB nominal fit.
#
# The server must be >= v0.30.0 (Sep 22, 2026): that release stamps
# session_id onto every BlockStored KV event (PR #51381). Check with:
#   python3 -c "import vllm; print(vllm.__version__)"
#
# The server is started with --served-model-name MODEL MODEL-alias so
# E1-open's MODEL_SWITCH turns can switch the model field to the alias
# and still be served. A single-model vLLM server rejects an unknown
# model id, so without the alias those turns fall back to in-process
# classification (marked upstream_error=True in the report).
# If your vLLM version rejects multiple --served-model-name values,
# drop the alias and expect the MODEL_SWITCH fallback; the rest of
# E1-open still runs.
#
# No API key is needed for vLLM itself. A Hugging Face token is only
# needed for gated models (HF_TOKEN env var); Qwen3-8B is NOT gated.
#
# Note: --kv-events-config enables the KV event stream used by the
# engine join (prefixwise serve --engine vllm --kv-events tcp://127.0.0.1:5557).
# The field is enable_kv_cache_events (vllm/config/kv_events.py, 2026-11-06;
# extra fields are rejected). Earlier versions of this script used a name
# current vLLM does not know. E1-open itself scores from cached_tokens in the
# response, so if that flag fails on your vLLM version, drop it and
# E1-open still runs; the KV-event join is then out of scope for that run.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_ALIAS="${MODEL_ALIAS:-${MODEL}-alias}"
PORT="${PORT:-8000}"
# KV-event port: give each replica its own when several run on one host (E-route).
EVENTS_PORT="${EVENTS_PORT:-5557}"
HOST="${HOST:-0.0.0.0}"
# Extra vLLM flags, e.g. VLLM_ARGS="--quantization fp8 --max-model-len 8192"
VLLM_ARGS="${VLLM_ARGS:-}"

echo "Model:        $MODEL (alias: $MODEL_ALIAS)"
echo "vLLM server:  http://$HOST:$PORT"
echo "Checking vLLM version..."
python3 -c "import vllm; print('vLLM', vllm.__version__)" || {
  echo "vLLM is not installed. Install it first:"
  echo "  pip install 'vllm>=0.30.0'"
  exit 1
}

# --enable-auto-tool-choice --tool-call-parser hermes (added Nov 2026):
# vLLM 0.33.0 rejects requests that carry tools with HTTP 400
# "`auto` tool choice requires --enable-auto-tool-choice and
# --tool-call-parser to be set". The proxy forwards tool definitions
# without an explicit tool_choice, so vLLM defaults to `auto` and
# refuses the request unless the server enables it. Qwen3 emits
# Hermes-style tool calls, so the hermes parser is the right one
# for Qwen models; for a non-Qwen model, change the parser to the
# one that matches that model's tool-call format.
exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL" "$MODEL_ALIAS" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "topic": "kv-events", "endpoint": "tcp://*:'"$EVENTS_PORT"'"}' \
  $VLLM_ARGS
