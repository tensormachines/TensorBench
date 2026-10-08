#!/usr/bin/env bash
# Verify HF_TOKEN (gated repo), then run the 2B LoRA fine-tuning. Every
# argument is forwarded to train.py. --smoke shortcuts the HF_TOKEN check.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-}"          # if unset, train.py auto-picks from VRAM_TABLE
BASE_REVISION="${BASE_REVISION:-}"

# --smoke is a pre-delivery shakeout: tiny ungated model, no HF_TOKEN.
for _a in "$@"; do
  if [[ "${_a}" == "--smoke" ]]; then
    echo "[entrypoint] SMOKE mode — skipping HF_TOKEN check."
    exec python /workspace/train.py "$@"
  fi
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN is not set. The 2B base model is gated (Llama-3-8B-"
  echo "       Instruct for 32GB, Llama-3.2-3B-Instruct for 16GB) — accept"
  echo "       the license on HuggingFace and pass -e HF_TOKEN=..."
  exit 1
fi
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

exec python /workspace/train.py \
  ${BASE_MODEL:+--base-model "${BASE_MODEL}"} \
  ${BASE_REVISION:+--revision "${BASE_REVISION}"} \
  "$@"
