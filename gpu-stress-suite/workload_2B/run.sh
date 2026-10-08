#!/usr/bin/env bash
# =============================================================================
# ONE-COMMAND RUNNER — Workload 2B: Llama-3 fine-tuning (LoRA or full)
#
# Usage:
#   HF_TOKEN=hf_xxx ./run.sh                              # Llama-3.2-3B-Instruct, 200 steps
#   HF_TOKEN=hf_xxx ./run.sh --base-model meta-llama/Meta-Llama-3-8B-Instruct
#   HF_TOKEN=hf_xxx ./run.sh --gpus 0,1
#   ./run.sh --smoke --gpus 0                              # shakeout: tiny, no HF_TOKEN
#
# Prereqs:
#   - Docker + NVIDIA Container Toolkit.
#   - HF_TOKEN with the base model's license accepted.
#
# First run downloads the FP16 base from HuggingFace into ./hf_cache;
# subsequent runs reuse the cache.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="gpu-stress-workload10"
RESULTS_DIR="${SCRIPT_DIR}/results"
HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
mkdir -p "${RESULTS_DIR}" "${HF_CACHE_DIR}"

# --smoke is a pre-delivery shakeout (tiny ungated model);
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
echo "  Running Workload 2B  (results -> ${RESULTS_DIR})"
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
  --volume "$(host_path "${HF_CACHE_DIR}"):/workspace/hf_cache" \
  "${IMAGE}" \
  "$@"

echo ""
echo "  Done. JSON + per-step CSV written to: ${RESULTS_DIR}/"
