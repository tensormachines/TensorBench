#!/usr/bin/env bash
# =============================================================================
# ONE-COMMAND RUNNER - Workload 2: Convolution Compute Stress
#
# Can be called from any directory:
#   ./workload2_resnet50_inference/run.sh
#   cd workload2_resnet50_inference && ./run.sh
#
# Usage:
#   ./run.sh [--duration N] [--batch-sizes B1 B2 ...] [--image-size N] [--max-vram-gb N]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="gpu-stress-workload2"
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

echo "========================================================"
echo "  Building Docker image: ${IMAGE}"
echo "========================================================"
docker build --tag "${IMAGE}" "${SCRIPT_DIR}"

echo ""
echo "========================================================"
echo "  Running Convolution Compute Stress Benchmark"
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
