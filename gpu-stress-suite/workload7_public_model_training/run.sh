#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="gpu-stress-workload7"
RESULTS_DIR="${SCRIPT_DIR}/results"
INSTALL_HF_DEPS="${INSTALL_HF_DEPS:-0}"
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

docker build --tag "${IMAGE}" --build-arg "INSTALL_HF_DEPS=${INSTALL_HF_DEPS}" "${SCRIPT_DIR}"

docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --volume "${RESULTS_DIR_DOCKER}:/workspace/results" \
    "${IMAGE}" "$@"
