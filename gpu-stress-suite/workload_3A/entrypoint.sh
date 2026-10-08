#!/usr/bin/env bash
# meta-llama/Llama-2-13b-hf is a gated repo: HF_TOKEN must be provided by
# environment (never baked into the image). All args forward to benchmark.py.
set -euo pipefail

# --smoke is a pre-delivery shakeout: tiny ungated model, so no HF_TOKEN
# needed. Detect it and skip the gated-repo guard.
for _a in "$@"; do
  if [[ "${_a}" == "--smoke" ]]; then
    echo "[entrypoint] SMOKE mode — skipping HF_TOKEN check."
    exec python /workspace/benchmark.py "$@"
  fi
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN is not set. meta-llama/Llama-2-13b-hf is gated —"
  echo "       accept the license on HuggingFace and pass -e HF_TOKEN=..."
  exit 1
fi
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

exec python /workspace/benchmark.py "$@"
