#!/usr/bin/env bash
# Start vLLM with LMCache CPU offload (LMCache multi-process mode).
#
# Usage:
#   MODEL=Qwen/Qwen3-4B ./experiments/e1_open/vllm_lmcache_server.sh
#   MODEL=Qwen/Qwen3-4B GPU_BLOCKS=2048 L1_GB=20 ./experiments/e1_open/vllm_lmcache_server.sh   # E-evict
#
# How it works (verified in LMCache docs/source/getting_started/quickstart.rst
# and vLLM source, 2026-11-07):
#   1. `lmcache server` runs as its own process (L1 = CPU memory, LRU).
#      LMCache dropped the old in-process mode and LMCACHE_* env vars.
#   2. vLLM connects to it with LMCacheMPConnector. On vLLM >= 0.20 the
#      LMCache-shipped connector (kv_connector_module_path) is preferred.
#   3. vLLM's usage.prompt_tokens_details.cached_tokens = local GPU prefix hits
#      + tokens loaded from LMCache (vllm/v1/metrics/stats.py PrefillStats), so
#      Prefixwise sees LMCache recoveries as ordinary cache hits.
#   LMCache stores whole chunks (default 256 tokens): a prefix shorter than one
#   chunk never hits LMCache, and the sub-chunk tail of a prompt is recomputed.
#
# Ports: vLLM $PORT (8000), LMCache requests 5555, LMCache HTTP/metrics 8090
# (moved off its default 8080, which is Prefixwise's default port).
# GPU_BLOCKS sets --num-gpu-blocks-override with --block-size 16, so the GPU
# KV capacity is exactly GPU_BLOCKS * 16 tokens (E-evict needs it known).
# Check `lmcache server --help` and `vllm serve --help` if a flag is rejected.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
L1_GB="${L1_GB:-20}"
python3 -c "import lmcache, vllm; print('LMCache', getattr(lmcache, '__version__', '?'), 'vLLM', vllm.__version__)" || {
  echo "Install first: pip install lmcache vllm"; exit 1; }

lmcache server --host 127.0.0.1 --port 5555 --http-port 8090 \
  --l1-size-gb "$L1_GB" --eviction-policy LRU &
LMCACHE_PID=$!
trap 'kill $LMCACHE_PID 2>/dev/null || true' EXIT
sleep 5

KV='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5555}}'
EXTRA=()
if [[ "${KV_EVENTS:-0}" == 1 ]]; then   # for the engine join: prefixwise ... --kv-events tcp://127.0.0.1:5557
  EXTRA+=(--kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "topic": "kv-events", "endpoint": "tcp://*:5557"}')
fi
if [[ -n "${GPU_BLOCKS:-}" ]]; then
  EXTRA+=(--block-size 16 --num-gpu-blocks-override "$GPU_BLOCKS" --max-model-len "${MAX_MODEL_LEN:-16384}")
fi
echo "vLLM + LMCache: http://$HOST:$PORT  model $MODEL  L1 ${L1_GB} GB ${EXTRA[*]:-}"
# shellcheck disable=SC2086
vllm serve "$MODEL" --host "$HOST" --port "$PORT" \
  --served-model-name "$MODEL" "${MODEL}-alias" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --kv-transfer-config "$KV" "${EXTRA[@]}" ${VLLM_ARGS:-}
