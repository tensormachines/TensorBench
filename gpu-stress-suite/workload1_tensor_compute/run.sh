#!/usr/bin/env bash
# =============================================================================
# ONE-COMMAND RUNNER — Workload 1: Tensor Compute Stress
#
# Can be called from any directory:
#   ./workload1_tensor_compute/run.sh
#   cd workload1_tensor_compute && ./run.sh
#
# Usage:
#   ./run.sh                                   # defaults: 60s, fp16, 8192
#   ./run.sh --duration 120 --precision bf16
#   ./run.sh --matrix-size 4096 --duration 30
#   ./run.sh --max-vram-gb 0                   # no VRAM cap (DGX)
#   ./run.sh --sequential                      # run GPUs one at a time
#   ./run.sh --sequential --duration 120
#
# Prerequisites:
#   Docker with NVIDIA Container Toolkit installed.
#   Windows: Docker Desktop (WSL2 backend) + NVIDIA Container Toolkit for WSL2.
#            Run from Git Bash or WSL2 terminal — not CMD or PowerShell.
#   Linux:   Docker Engine + nvidia-container-toolkit package.
#
# Expected performance (FP16, 8192x8192, batch=4):
#   RTX 4090       — Peak ~330 TFLOP/s  |  Sustained ~200–240 TFLOP/s
#   RTX 4080 SUPER — Peak ~244 TFLOP/s  |  Sustained ~150–180 TFLOP/s
#   DGX H100 SXM5  — Peak ~780 TFLOP/s  |  Sustained ~700–730 TFLOP/s
# =============================================================================

set -euo pipefail

# Always resolve paths relative to this script's location — not the caller's cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="gpu-stress-workload1"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

docker_host_path() {
    local path="$1"
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            if command -v cygpath >/dev/null 2>&1; then
                cygpath -m "${path}"
                return
            fi
            ;;
    esac
    printf "%s" "${path}"
}

RESULTS_DIR_DOCKER="$(docker_host_path "${RESULTS_DIR}")"

# Verify Docker can see the GPU before building
echo "========================================================"
echo "  Checking GPU access in Docker..."
echo "========================================================"
if ! docker run --rm --gpus all --entrypoint nvidia-smi \
        nvcr.io/nvidia/pytorch:24.09-py3 -L 2>/dev/null; then
    echo ""
    echo "  ERROR: Docker cannot access your GPU."
    echo "  Make sure Docker Desktop is running with WSL2 backend"
    echo "  and NVIDIA Container Toolkit is installed."
    exit 1
fi

echo ""
echo "========================================================"
echo "  Building Docker image: ${IMAGE}"
echo "========================================================"
docker build --tag "${IMAGE}" "${SCRIPT_DIR}"

echo ""
echo "========================================================"
echo "  Running Tensor Compute Stress Benchmark"
echo "  Results will be written to: ${RESULTS_DIR}"
echo "========================================================"
docker run \
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    --env TZ="$(cat /etc/timezone 2>/dev/null || echo 'America/Los_Angeles')" \
    --volume "${RESULTS_DIR_DOCKER}:/workspace/results" \
    "${IMAGE}" \
    "$@"

echo ""
echo "  Done. JSON log written to: ${RESULTS_DIR}/"
