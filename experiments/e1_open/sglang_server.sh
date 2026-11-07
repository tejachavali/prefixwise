#!/usr/bin/env bash
# Start an SGLang OpenAI-compatible server for E1-open and E-evict.
#
# Usage:
#   MODEL=Qwen/Qwen3-4B ./experiments/e1_open/sglang_server.sh
#   MODEL=Qwen/Qwen3-4B HICACHE=1 ./experiments/e1_open/sglang_server.sh    # + host-memory tier
#   MODEL=Qwen/Qwen3-4B MAX_TOTAL_TOKENS=32768 ./experiments/e1_open/sglang_server.sh  # small GPU KV (E-evict)
#
# Flags (verified in sgl-project/sglang source, 2026-11-06):
#   --enable-cache-report   puts usage.prompt_tokens_details.cached_tokens in
#                           every OpenAI response (and the final stream chunk).
#                           Without it Prefixwise sees no cache hits at all.
#   --enable-metrics        Prometheus /metrics (sglang:cache_hit_rate,
#                           sglang:evicted_tokens_total, ...).
#   --tool-call-parser qwen25  parse Qwen tool calls (tools are accepted without it).
#   --enable-hierarchical-cache  HiCache: GPU -> host memory tier. cached_tokens
#                           includes host-tier hits.
#   --max-total-tokens N    caps the GPU KV pool, so eviction is reachable.
# Run `python -m sglang.launch_server --help` if a flag is rejected: SGLang
# renames flags between releases.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
ARGS=(--model-path "$MODEL" --host "$HOST" --port "$PORT"
      --enable-cache-report --enable-metrics --tool-call-parser qwen25)
if [[ "${HICACHE:-0}" == 1 ]]; then
  ARGS+=(--enable-hierarchical-cache --hicache-ratio "${HICACHE_RATIO:-4}")
fi
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi
python3 -c "import sglang; print('SGLang', sglang.__version__)" || {
  echo "SGLang is not installed: pip install 'sglang[all]'"; exit 1; }
echo "SGLang server: http://$HOST:$PORT  model $MODEL  args ${ARGS[*]} ${SGLANG_ARGS:-}"
# shellcheck disable=SC2086
exec python3 -m sglang.launch_server "${ARGS[@]}" ${SGLANG_ARGS:-}
