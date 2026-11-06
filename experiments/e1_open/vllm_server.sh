#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for E1-open.
#
# Usage:
#   MODEL=Qwen/Qwen3-8B ./experiments/e1_open/vllm_server.sh
#   MODEL=Qwen/Qwen3-32B ./experiments/e1_open/vllm_server.sh
#   MODEL=meta-llama/Llama-3.3-70B-Instruct ./experiments/e1_open/vllm_server.sh
#
# GPU requirements (approximate, FP16/BF16 weights, no quantization):
#   Qwen3-8B            1x 24 GB GPU (Colab L4, RTX 4090) -- default
#   Qwen3-4B            1x 16 GB GPU (Colab T4) -- fallback, set MODEL=Qwen/Qwen3-4B
#   Qwen3-32B           ~40 GB+ (1x A100 80GB, or 2x L4)
#   Llama-3.3-70B       2x H100 80GB, or 1x GPU with --quantization fp8
#                       (add VLLM_ARGS="--quantization fp8 --max-model-len 8192")
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
# E2/E3 engine join. E1-open itself scores from cached_tokens in the
# response, so if that flag fails on your vLLM version, drop it and
# E1-open still runs; the KV-event join is then out of scope for that run.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_ALIAS="${MODEL_ALIAS:-${MODEL}-alias}"
PORT="${PORT:-8000}"
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

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL" "$MODEL_ALIAS" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --kv-events-config '{"enable_prefix_cache_events": true, "publisher": "zmq", "topic": "cache-events", "endpoint": "tcp://127.0.0.1:5557"}' \
  $VLLM_ARGS
