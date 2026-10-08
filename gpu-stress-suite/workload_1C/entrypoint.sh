#!/usr/bin/env bash
# Quantize-once (deterministic, cached on the mounted model volume) then run
# the 1C benchmark. Every argument is forwarded to benchmark.py.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/model_int8}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
BASE_REVISION="${BASE_REVISION:-}"   # set to a commit SHA to pin the source repo

# --smoke is a pre-delivery shakeout: tiny ungated model, no quantization,
# so it needs neither HF_TOKEN nor the quantize step. Detect it and shortcut.
for _a in "$@"; do
  if [[ "${_a}" == "--smoke" ]]; then
    echo "[entrypoint] SMOKE mode — skipping HF_TOKEN check and quantization."
    exec python /workspace/benchmark.py "$@"
  fi
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN is not set. meta-llama/Meta-Llama-3-8B-Instruct is a"
  echo "       gated repo — accept the license on HuggingFace and pass the"
  echo "       token via -e HF_TOKEN=... (it is never baked into the image)."
  exit 1
fi
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

if [[ ! -f "${MODEL_DIR}/QUANT_OK.json" ]]; then
  echo "[entrypoint] No cached INT8 model — running deterministic quantization."
  python /workspace/quantize.py \
    --base-model "${BASE_MODEL}" \
    ${BASE_REVISION:+--revision "${BASE_REVISION}"} \
    --out-dir "${MODEL_DIR}" \
    --corpus /workspace/frozen_prompts.json
else
  echo "[entrypoint] Reusing cached INT8 model at ${MODEL_DIR}."
fi

exec python /workspace/benchmark.py --model-dir "${MODEL_DIR}" "$@"
