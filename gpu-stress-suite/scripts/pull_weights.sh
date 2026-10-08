#!/usr/bin/env bash
# pull_weights.sh — download pre-quantized weights from S3 into the local cache.
# Run on a fresh machine BEFORE run_all.sh to skip 1+ hour of quantization.
# Idempotent: re-running only downloads what is missing or changed.
#
# Usage:
#   ./pull_weights.sh                   # pull entire cache
#   ./pull_weights.sh <model_name>      # pull one model only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${SUITE_DIR}/cache"
BUCKET="tensormachines-benchmark-weights"
PREFIX="v100-dgx2/cache"

mkdir -p "${CACHE_DIR}"

if [[ $# -gt 0 ]]; then
  MODEL="$1"
  SRC="s3://${BUCKET}/${PREFIX}/${MODEL}/"
  DEST="${CACHE_DIR}/${MODEL}/"
  mkdir -p "${DEST}"
  echo "=== Pulling ${MODEL}: ${SRC} -> ${DEST} ==="
  aws s3 sync "${SRC}" "${DEST}" --no-progress
  echo "=== Done: ${MODEL} ==="
else
  SRC="s3://${BUCKET}/${PREFIX}/"
  echo "=== Pulling all weights: ${SRC} -> ${CACHE_DIR}/ ==="
  aws s3 sync "${SRC}" "${CACHE_DIR}/" --no-progress
  echo "=== Done ==="
fi

echo ""
echo "=== Local cache contents ==="
du -sh "${CACHE_DIR}"/*/ 2>/dev/null || echo "(empty)"

MISSING=0
for model_dir in "${CACHE_DIR}"/*/; do
  if [[ ! -f "${model_dir}/QUANT_OK.json" ]]; then
    echo "WARNING: ${model_dir} missing QUANT_OK.json — incomplete download?"
    MISSING=1
  fi
done
[[ "${MISSING}" -eq 0 ]] && echo "All models have QUANT_OK.json — ready to run."
