#!/usr/bin/env bash
# =============================================================================
# ONE-COMMAND RUNNER — Workload 2A: LLaMA-2 13B GPTQ-INT8 integrated inference
#
# Usage:
#   HF_TOKEN=hf_xxx ./run.sh                              # batch 4, seq 1024, 15 min
#   HF_TOKEN=hf_xxx ./run.sh --batch 16 --seq 2048
#   HF_TOKEN=hf_xxx ./run.sh --gpus 0,1
#   HF_TOKEN=hf_xxx BASE_REVISION=<sha> ./run.sh          # pin source repo
#   ./run.sh --smoke --gpus 0                             # no token needed
#
# Prereqs:
#   - Docker + NVIDIA Container Toolkit.
#   - HF_TOKEN with meta-llama/Llama-2-13b-hf license accepted.
#
# First run performs a one-time deterministic GPTQ-INT8 quantization (~20-40
# min on one V100, ~26 GB source weights). Cached in ./model_cache;
# subsequent runs skip it and start the 15-minute benchmark.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="gpu-stress-workload9"
RESULTS_DIR="${SCRIPT_DIR}/results"
MODEL_CACHE_DIR="${SCRIPT_DIR}/model_cache"
HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
mkdir -p "${RESULTS_DIR}" "${MODEL_CACHE_DIR}" "${HF_CACHE_DIR}"

# --smoke is a pre-delivery shakeout (tiny ungated model, no quantization);
# it needs no HF_TOKEN. Usage: ./run.sh --smoke --gpus 0
SMOKE=0
for _a in "$@"; do [[ "${_a}" == "--smoke" ]] && SMOKE=1; done

if [[ "${SMOKE}" -eq 0 && -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: set HF_TOKEN (gated meta-llama repo). e.g. HF_TOKEN=hf_xxx ./run.sh"
  echo "       (not required for the shakeout:  ./run.sh --smoke --gpus 0)"
  exit 1
fi

host_path() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) command -v cygpath >/dev/null 2>&1 && { cygpath -m "$1"; return; } ;;
  esac
  printf "%s" "$1"
}

echo "========================================================"
echo "  Checking GPU access in Docker..."
echo "========================================================"
if ! docker run --rm --gpus all --entrypoint nvidia-smi \
        nvcr.io/nvidia/pytorch:24.09-py3 -L 2>/dev/null; then
  echo "  ERROR: Docker cannot see the GPU (need NVIDIA Container Toolkit)."
  exit 1
fi

echo "========================================================"
echo "  Building image: ${IMAGE}"
echo "========================================================"
docker build --tag "${IMAGE}" "${SCRIPT_DIR}"

echo "========================================================"
echo "  Running Workload 2A  (results -> ${RESULTS_DIR})"
echo "========================================================"
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env BASE_REVISION="${BASE_REVISION:-}" \
  --env HF_HOME=/workspace/hf_cache \
  --volume "$(host_path "${RESULTS_DIR}"):/workspace/results" \
  --volume "$(host_path "${MODEL_CACHE_DIR}"):/workspace/model_int8" \
  --volume "$(host_path "${HF_CACHE_DIR}"):/workspace/hf_cache" \
  "${IMAGE}" \
  "$@"

echo ""
echo "  Done. JSON + CSVs written to: ${RESULTS_DIR}/"
