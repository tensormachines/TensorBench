#!/usr/bin/env bash
# =============================================================================
# GPU STRESS SUITE - LOCAL ASSESSMENT RUNNER
# Runs a recipe-aligned local prototype assessment using the existing synthetic
# workloads plus host-side telemetry capture.
#
# Usage:
#   ./run_all.sh
#   ./run_all.sh --max-vram-gb 8
#   ./run_all.sh --idle-duration 5 --cooldown-duration 5
#   ./run_all.sh --skip 1,2,3,4,5
#   ./run_all.sh --upload
#   ./run_all.sh --upload --contact you@example.com
#   ./run_all.sh --score --oh 4.00 --rh 0.42 --ce 0.10 --uz 0.5
#   ./run_all.sh --dry-run 2
#   ./run_all.sh --upload --skip 1,3,4,5,6,7,8,9,10,11,12,13,14,15 --idle-duration 5 --cooldown-duration 5  2>&1 | tee -i  26-07-15_17-24_run_all.log
#
# Results:
#   suite_results/assessment_<timestamp>/
#     manifest.json, phases.tsv, summary.txt
#     results/              workload results
#     *_telemetry.csv       NVML and BMC telemetry
#     score.json            with --score
# =============================================================================

set -euo pipefail

IDLE_DURATION=120
WARMUP_DURATION=180
COOLDOWN_DURATION=120
MAX_VRAM_GB=""
MAX_VRAM_SOURCE="auto"
SKIP=""
UPLOAD_RESULTS="${UPLOAD_RESULTS:-0}"   # disabled by default; enable with --upload or UPLOAD_RESULTS=1
CONTACT=""   # email saved as contact.txt with the uploaded results; needs --upload
EMAIL_RE='^[A-Za-z0-9._%+-]+@([A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$'
REQUIRE_MULTI_GPU=0
INSTALL_HF_DEPS="${INSTALL_HF_DEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"   # 0 = off, 1 = skip docker image build, 2 = build docker images
DRY_RUN_IDLE_SEC="${DRY_RUN_IDLE_SEC:-1}"
SCORE_ENABLED=0   # enable with --score
# Score inputs; --oh, --rh, --ce and --uz take precedence.
SCORE_OH="${SCORE_OH:-}"   # ownership cost per GPU per hour
SCORE_RH="${SCORE_RH:-}"   # reserved capacity cost per GPU per hour
SCORE_CE="${SCORE_CE:-}"   # electricity cost per kWh
SCORE_UZ="${SCORE_UZ:-}"   # utilization for the second score; asked for when unset
SCORE=""
SCORE_NOTE=""
SCORE_DEFAULTS=()
DEFAULT_CE="0.1000"
HW_FINGERPRINT="${HW_FINGERPRINT:-/opt/tensormachines/hw_fingerprint.json}"
LOGGER_DIR="${LOGGER_DIR:-/opt/tensormachines/loggers}"
# Node identifier (defaults to short hostname); included in each workload's
# output JSON and the suite manifest.
NODE_ID="${NODE_ID:-$(hostname -s 2>/dev/null || echo unknown)}"

detect_gpu_label() {
    local gpu_names
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | sort -u | paste -sd ', ' -)"
    if [[ -z "${gpu_names}" ]]; then
        echo "unknown GPU"
    else
        echo "${gpu_names}"
    fi
}

GPU_LABEL="${GPU_LABEL:-$(detect_gpu_label)}"

# Simple per-phase duration profile (seconds)
declare -A PHASE_SEC=(
    [1A]=240 
    [1B]=300
    [1C]=420 
    [2A]=900 
    [2B]=720 
    [2C]=180 
    [2D]=180 
    [2E]=180 
    [3A]=300 
    [3B]=1200 
    [nccl]=300
)

phase_sec() {
    local key="$1"
    echo "${PHASE_SEC[$key]}"
}

phase_profile_kv() {
    echo "1A=$(phase_sec 1A),1B=$(phase_sec 1B),2A=$(phase_sec 2A),2B=$(phase_sec 2B),2C=$(phase_sec 2C),2D=$(phase_sec 2D),2E=$(phase_sec 2E),3A=$(phase_sec 3A),3B=$(phase_sec 3B),nccl=$(phase_sec nccl)"
}

run_timestamp() {
    date -u +'%y-%m-%d_%H-%M-%S'
}

usage() {
    cat <<USAGE
Usage: ./run_all.sh [options]

Runs the GPU stress assessment on this node. Results are written to
suite_results/assessment_<timestamp>/.

Options:
  --skip LIST              workload numbers to skip, e.g. 1,2,4
  --dry-run [LEVEL]        1 skips docker image builds (default), 2 builds them
  --max-vram-gb GB         VRAM cap per GPU (default: from the GPU profile)
  --idle-duration SEC      idle baseline length (default ${IDLE_DURATION})
  --warmup-duration SEC    warmup length (default ${WARMUP_DURATION})
  --cooldown-duration SEC  cooldown length (default ${COOLDOWN_DURATION})
  --require-multi-gpu      fail the NCCL workload (6) on a single-GPU node instead of skipping it
  --upload                 upload results when the run completes
  --contact EMAIL          save EMAIL as contact.txt in the uploaded results (needs --upload)
  --score                  compute the cost per million tokens score
  --gpu-profile NAME       force a profile from hardware/gpu/
  --platform-profile NAME  force a profile from hardware/platform/
  -h, --help               this message

Environment:
  HW_FINGERPRINT     hardware fingerprint (default ${HW_FINGERPRINT})
  LOGGER_DIR         telemetry loggers (default ${LOGGER_DIR})
  NODE_ID            node name recorded in results (default: short hostname)
  UPLOAD_RESULTS=1   same as --upload
  DRY_RUN=LEVEL      same as --dry-run LEVEL
  DRY_RUN_IDLE_SEC   idle and cooldown length in a dry run (default ${DRY_RUN_IDLE_SEC})
  INSTALL_HF_DEPS=1  install Hugging Face dependencies into the workload 7 image
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --max-vram-gb) MAX_VRAM_GB="$2"; MAX_VRAM_SOURCE="user"; shift 2 ;;
        --idle-duration) IDLE_DURATION="$2"; shift 2 ;;
        --warmup-duration) WARMUP_DURATION="$2"; shift 2 ;;
        --cooldown-duration) COOLDOWN_DURATION="$2"; shift 2 ;;
        --require-multi-gpu) REQUIRE_MULTI_GPU=1; shift ;;
        --skip) SKIP="$2"; shift 2 ;;
        --upload) UPLOAD_RESULTS=1; shift ;;
        --contact)
            [[ -n "${2:-}" ]] || { echo "ERROR: --contact needs an email address"; exit 1; }
            CONTACT="$2"; shift 2 ;;
        --score) SCORE_ENABLED=1; shift ;;
        --gpu-profile) GPU_PROFILE_OVERRIDE="$2"; shift 2 ;;
        --platform-profile) PLATFORM_PROFILE_OVERRIDE="$2"; shift 2 ;;
        --oh) SCORE_OH="$2"; shift 2 ;;
        --rh) SCORE_RH="$2"; shift 2 ;;
        --ce) SCORE_CE="$2"; shift 2 ;;
        --uz|--Uz) SCORE_UZ="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1
            if [[ "${2:-}" =~ ^[0-9]+$ ]]; then DRY_RUN="$2"; shift 2; else shift; fi
            ;;
        *) echo "Unknown argument: $1  (try --help)"; exit 1 ;;
    esac
done

# Validate dry run level
if [[ ! "${DRY_RUN}" =~ ^[012]$ ]]; then
    echo "ERROR: DRY_RUN must be 0 (off), 1 (skip docker image builds) or 2 (build docker images); got: ${DRY_RUN}"
    exit 1
fi

# Validate contact
if [[ -n "${CONTACT}" ]]; then
    if [[ "${UPLOAD_RESULTS}" != "1" ]]; then
        echo "ERROR: --contact works only together with --upload."
        exit 1
    fi
    if [[ ! "${CONTACT}" =~ ${EMAIL_RE} ]]; then
        echo "ERROR: --contact needs a valid email address, not '${CONTACT}'."
        exit 1
    fi
fi

# Validate score prices
for price in "oh=${SCORE_OH}" "rh=${SCORE_RH}" "ce=${SCORE_CE}"; do
    value="${price#*=}"
    if [[ -n "${value}" && ! "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        name="${price%%=*}"
        echo "ERROR: --${name} / SCORE_${name^^} must be a non-negative number; got: ${value}"
        exit 1
    fi
done
# Succeeds when $1 is a number greater than 0 and at most 1.
valid_uz() {
    awk -v v="$1" 'BEGIN { exit !(v ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ && v > 0 && v <= 1) }'
}
if [[ -n "${SCORE_UZ}" ]] && ! valid_uz "${SCORE_UZ}"; then
    echo "ERROR: --uz / SCORE_UZ must be greater than 0 and at most 1; got: ${SCORE_UZ}"
    exit 1
fi

# Unset Oh and Rh default to the GPU class prices, unset Ce to DEFAULT_CE.
if [[ -z "${SCORE_OH}" ]]; then
    SCORE_DEFAULTS+=(Oh)
fi
if [[ -z "${SCORE_CE}" ]]; then
    SCORE_CE="${DEFAULT_CE}"
fi

# Verify hardware fingerprint and resolve hardware profiles
HWDETECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/harness/hwdetect.py"

set +e
python3 "${HWDETECT}" verify "${HW_FINGERPRINT}"
hw_rc=$?
set -e
if [[ "${hw_rc}" -ne 0 ]]; then
    if [[ "${hw_rc}" -eq 3 ]]; then
        echo "  Hardware change detected since last run. Please re-run provisioning steps."
    else
        echo "  Hardware verification failed."
    fi
    echo "  Assessment not started."
    exit "${hw_rc}"
fi

PROFILE_ARGS=("${HW_FINGERPRINT}")
[[ -n "${GPU_PROFILE_OVERRIDE:-}" ]] && PROFILE_ARGS+=(--gpu-profile "${GPU_PROFILE_OVERRIDE}")
[[ -n "${PLATFORM_PROFILE_OVERRIDE:-}" ]] && PROFILE_ARGS+=(--platform-profile "${PLATFORM_PROFILE_OVERRIDE}")

set +e
HW_ENV="$(python3 "${HWDETECT}" profile "${PROFILE_ARGS[@]}")"
hw_rc=$?
set -e
if [[ "${hw_rc}" -ne 0 ]]; then
    echo "  No usable hardware profile. Please add one under hardware/,"
    echo "  then re-run provisioning steps."
    echo "  Assessment not started."
    exit "${hw_rc}"
fi
eval "${HW_ENV}"

BMC_LOGGER_PATH="${LOGGER_DIR}/${HW_BMC_LOGGER}"
NVML_LOGGER_PATH="${LOGGER_DIR}/${HW_NVML_LOGGER}"

if [[ -z "${MAX_VRAM_GB}" ]]; then
    MAX_VRAM_GB="${HW_MAX_VRAM_GB}"
    MAX_VRAM_SOURCE="profile"
fi

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_RESULTS_DIR="${SUITE_DIR}/suite_results"
TIMESTAMP="$(run_timestamp)"
if [[ "${DRY_RUN}" != "0" ]]; then
    ASSESSMENT_DIR="${ROOT_RESULTS_DIR}/assessment_dryrun_${TIMESTAMP}"
else
    ASSESSMENT_DIR="${ROOT_RESULTS_DIR}/assessment_${TIMESTAMP}"
fi
RESULTS_DIR="${ASSESSMENT_DIR}/results"
SUMMARY_FILE="${ASSESSMENT_DIR}/summary.txt"
if [[ "${DRY_RUN}" != "0" ]]; then
    ROOT_INDEX_FILE="${ROOT_RESULTS_DIR}/summary_dryrun_${TIMESTAMP}.txt"
else
    ROOT_INDEX_FILE="${ROOT_RESULTS_DIR}/summary_${TIMESTAMP}.txt"
fi
MANIFEST_FILE="${ASSESSMENT_DIR}/manifest.json"
PHASE_LOG="${ASSESSMENT_DIR}/phases.tsv"

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

should_skip() {
    local workload_num="$1"
    [[ ",${SKIP}," == *",${workload_num},"* ]]
}

phase_name() {
    local value="$1"
    value="${value,,}"
    value="${value// /_}"
    value="${value//-/_}"
    value="${value//\//_}"
    value="${value//__/_}"
    printf "%s" "${value}"
}

require_host_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: Required host command not found: ${cmd}"
        exit 1
    fi
}

require_host_command docker
require_host_command nvidia-smi
require_host_command python3

printf "phase_id\tphase_kind\tphase_label\tstart_time\tend_time\tduration_s\tresult_file\tstatus\n" > "${PHASE_LOG}"

start_nvml() {
    if pgrep -f "${NVML_LOGGER_PATH}" >/dev/null; then
        echo "[TELEMETRY] WARN: gpu_logger already running:"
        pgrep -af "${NVML_LOGGER_PATH}"
        echo "[TELEMETRY] Killing stale instance(s) before starting fresh."
        pkill -f "${NVML_LOGGER_PATH}" || true
        sleep 1
    fi
    nohup /opt/tensormachines/envs/telemetry_env/bin/python -u "${NVML_LOGGER_PATH}" \
        > ${LOGGER_DIR}/gpu_logger.out 2>&1 &
    echo $! > ${LOGGER_DIR}/gpu_logger.pid
    echo "[TELEMETRY] NVML logger started (PID $(cat ${LOGGER_DIR}/gpu_logger.pid))"
}
stop_nvml() {
    local pid
    pid="$(cat ${LOGGER_DIR}/gpu_logger.pid 2>/dev/null)" || return 0
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pid}" 2>/dev/null || true
    # rm -f ${LOGGER_DIR}/gpu_logger.pid
    echo "[TELEMETRY] NVML logger stopped (PID ${pid})"
}

start_bmc() {
    if pgrep -f "${BMC_LOGGER_PATH}" >/dev/null; then
        echo "[TELEMETRY] WARN: bmc_logger already running:"
        pgrep -af "${BMC_LOGGER_PATH}"
        echo "[TELEMETRY] Killing stale instance(s) before starting fresh."
        pkill -f "${BMC_LOGGER_PATH}" || true
        sleep 1
    fi
    nohup /opt/tensormachines/envs/telemetry_env/bin/python -u "${BMC_LOGGER_PATH}" \
        > ${LOGGER_DIR}/bmc_logger.out 2>&1 &
    echo $! > ${LOGGER_DIR}/bmc_logger.pid
    echo "[TELEMETRY] BMC logger started (PID $(cat ${LOGGER_DIR}/bmc_logger.pid))"
}
stop_bmc() {
    local pid
    pid="$(cat ${LOGGER_DIR}/bmc_logger.pid 2>/dev/null)" || return 0
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pid}" 2>/dev/null || true
    # rm -f ${LOGGER_DIR}/bmc_logger.pid
    echo "[TELEMETRY] BMC logger stopped (PID ${pid})"
}

append_phase_log() {
    local phase_id="$1"
    local phase_kind="$2"
    local phase_label="$3"
    local start_time="$4"
    local end_time="$5"
    local duration_s="$6"
    local result_file="$7"
    local status="$8"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${phase_id}" "${phase_kind}" "${phase_label}" "${start_time}" "${end_time}" \
        "${duration_s}" "${result_file}" "${status}" >> "${PHASE_LOG}"
}

run_idle_phase() {
    local phase_id="$1"
    local label="$2"
    local duration_s="$3"
    local slug
    local start_epoch
    local end_epoch
    local start_time
    local end_time
    local elapsed

    slug="$(phase_name "${label}")"

    if [[ "${DRY_RUN}" != "0" ]]; then
        duration_s="${DRY_RUN_IDLE_SEC}"
    fi

    echo ""
    echo "============================================================**"
    printf "  Phase %s: %s\n" "${phase_id}" "${label}"
    printf "  Mode     : idle / observation\n"
    printf "  Duration : %ss\n" "${duration_s}"
    echo "============================================================**"

    start_time="$(run_timestamp)"
    start_epoch="$(date +%s)"
    # start_telemetry "${phase_id}_${slug}"
    sleep "${duration_s}"
    # stop_telemetry
    end_epoch="$(date +%s)"
    end_time="$(run_timestamp)"
    elapsed=$((end_epoch - start_epoch))

    append_phase_log "${phase_id}" "idle" "${label}" "${start_time}" "${end_time}" "${elapsed}" "" "completed"
}

write_mock_result() {
    local workload_num="$1"
    local phase_id="$2"
    local phase_label="$3"
    local target="${RESULTS_DIR}/${TIMESTAMP}_workload${workload_num}_phase${phase_id}_metadata.json"

    python3 - "${target}" "${TIMESTAMP}" "${workload_num}" "${phase_id}" "${phase_label}" \
              "${NODE_ID}" "${MAX_VRAM_GB}" "${DRY_RUN}" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2],
    "workload": f"workload{sys.argv[3]}",
    "phase_id": sys.argv[4],
    "phase_label": sys.argv[5],
    "node_id": sys.argv[6],
    "max_vram_gb": float(sys.argv[7]),
    "status": "completed",
    "dry_run": True,
    "dry_run_level": int(sys.argv[8]),
    "results": [],
    "note": "Mock result - no workload executed (dry run).",
}
target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
    echo "  [DRY RUN] Mock result written: $(basename "${target}")"
}

run_workload_phase() {
    local phase_id="$1"
    local label="$2"
    local workload_num="$3"
    local phase_duration="$4"
    shift 4
    local -a extra_args=("$@")
    local slug
    local start_epoch
    local end_epoch
    local start_time
    local end_time
    local elapsed
    local result_file
    local docker_exit

    slug="$(phase_name "${label}")"

    local -a profile_args=()
    if ! should_skip "${workload_num}"; then
        # mapfile < <(cmd) cannot see cmd's exit status, so capture it first.
        local args_out args_rc
        set +e
        args_out="$(workload_profile_args "${workload_num}")"
        args_rc=$?
        set -e
        if [[ "${args_rc}" -ne 0 ]]; then
            echo "  Assessment stopped."
            exit 1
        fi
        [[ -n "${args_out}" ]] && mapfile -t profile_args <<< "${args_out}"
    fi

    if should_skip "${workload_num}"; then
        echo "  Phase ${phase_id}: ${label} skipped (--skip includes workload ${workload_num})"
        start_time="$(run_timestamp)"
        append_phase_log "${phase_id}" "workload" "${label}" "${start_time}" "${start_time}" "0" "" "skipped"
        return
    fi

    echo ""
    echo "============================================================**"
    printf "  Phase %s: %s\n" "${phase_id}" "${label}"
    printf "  Workload : gpu-stress-workload%s\n" "${workload_num}"
    if [[ -n "${phase_duration}" ]]; then
        printf "  Duration : %ss\n" "${phase_duration}"
    else
        printf "  Duration : benchmark default\n"
    fi
    printf "  VRAM cap : %s GB\n" "${MAX_VRAM_GB}"
    echo "============================================================**"

    start_time="$(run_timestamp)"
    start_epoch="$(date +%s)"
    # start_telemetry "${phase_id}_${slug}"

    # Cache mounts: shared hf_cache for HF downloads (Llama-3-8B is used by
    # both workload 8 and workload 10 — share to avoid double-download);
    # per-workload model dir for quantized artifacts (workloads 8 and 9
    # populate it via quantize.py; 10/11 leave it empty).
    local hf_cache_host="${SUITE_DIR}/cache/hf_cache"
    local model_cache_host="${SUITE_DIR}/cache/workload${workload_num}_model"
    mkdir -p "${hf_cache_host}" "${model_cache_host}"

    if [[ "${DRY_RUN}" != "0" ]]; then
        # Dry run: no container is started, emit a mock result
        write_mock_result "${workload_num}" "${phase_id}" "${label}"
        docker_exit=0
    else
        set +e
        if [[ -n "${phase_duration}" ]]; then
            docker run --rm --gpus all --ipc=host \
                --ulimit memlock=-1 --ulimit stack=67108864 \
                --env HF_TOKEN="${HF_TOKEN:-}" \
                --env HF_HOME=/workspace/hf_cache \
                --env NODE_ID="${NODE_ID}" \
                --volume "${RESULTS_DIR_DOCKER}:/workspace/results" \
                --volume "$(docker_host_path "${hf_cache_host}"):/workspace/hf_cache" \
                --volume "$(docker_host_path "${model_cache_host}"):/workspace/model_int8" \
                "gpu-stress-workload${workload_num}" \
                --duration "${phase_duration}" \
                --max-vram-gb "${MAX_VRAM_GB}" \
                "${profile_args[@]}" \
                "${extra_args[@]}"
        else
            docker run --rm --gpus all --ipc=host \
                --ulimit memlock=-1 --ulimit stack=67108864 \
                --env HF_TOKEN="${HF_TOKEN:-}" \
                --env HF_HOME=/workspace/hf_cache \
                --env NODE_ID="${NODE_ID}" \
                --volume "${RESULTS_DIR_DOCKER}:/workspace/results" \
                --volume "$(docker_host_path "${hf_cache_host}"):/workspace/hf_cache" \
                --volume "$(docker_host_path "${model_cache_host}"):/workspace/model_int8" \
                "gpu-stress-workload${workload_num}" \
                --max-vram-gb "${MAX_VRAM_GB}" \
                "${profile_args[@]}" \
                "${extra_args[@]}"
        fi
        docker_exit=$?
        set -e
    fi

    # stop_telemetry
    end_epoch="$(date +%s)"
    end_time="$(run_timestamp)"
    elapsed=$((end_epoch - start_epoch))
    result_file="$(ls -1t "${RESULTS_DIR}"/*_workload${workload_num}_*.json 2>/dev/null | head -n 1 || true)"

    if [[ "${docker_exit}" -eq 0 ]]; then
        append_phase_log "${phase_id}" "workload" "${label}" "${start_time}" "${end_time}" "${elapsed}" "${result_file}" "completed"
        return
    fi

    result_file="${RESULTS_DIR}/workload${workload_num}_${TIMESTAMP}_phase${phase_id}_failed.json"
    python3 - "${result_file}" "${TIMESTAMP}" "${workload_num}" "${phase_id}" "${label}" "${docker_exit}" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2],
    "workload": f"workload{sys.argv[3]}",
    "phase_id": sys.argv[4],
    "phase_label": sys.argv[5],
    "status": "failed",
    "docker_exit_code": int(sys.argv[6]),
    "results": [],
}
target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

    echo "  [WARN] Phase ${phase_id} failed with exit code ${docker_exit}; continuing assessment."
    append_phase_log "${phase_id}" "workload" "${label}" "${start_time}" "${end_time}" "${elapsed}" "${result_file}" "failed"
}


# Helper: workload number -> workload directory
workload_dir() {
    case "$1" in
        1)  echo "workload1_tensor_compute" ;;
        2)  echo "workload2_conv_compute" ;;
        3)  echo "workload3_random_access_memory" ;;
        4)  echo "workload4_fft_compute" ;;
        5)  echo "workload5_memory_bandwidth" ;;
        6)  echo "workload6_multigpu_nccl" ;;
        7)  echo "workload7_public_model_training" ;;
        8)  echo "workload_1C" ;;   # 1C: LLaMA-3 8B INT8 integrated baseline
        9)  echo "workload_2A" ;;   # 2A: LLaMA-2 13B INT8 diagnostic
        10) echo "workload_2B" ;;   # 2B: LLaMA-3 8B LoRA write-path
        11) echo "workload_3A" ;;   # 3A: LLaMA-2 13B FP16 near-limit boundary
        *)  echo "" ;;
    esac
}

# Parameters the GPU profile supplies for a workload, as a shell array.
workload_profile_args() {
    local num="$1"
    local dir; dir="$(workload_dir "${num}")"
    local ref="HW_WORKLOAD_ARGS_${dir}[@]"
    if [[ -z "${!ref+set}" ]]; then
        echo "ERROR: GPU profile '${HW_GPU_PROFILE_ID}' has no workloads entry for" >&2
        echo "       '${dir}'. Add one to hardware/gpu/${HW_GPU_PROFILE_ID}.json" >&2
        return 1
    fi
    printf '%s\n' "${!ref}"
}

build_all_images() {
    local dir=""
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo ""
        echo "  [DRY RUN] Level 1 - skipping docker image builds."
        return 0
    fi
    echo ""
    echo "  Building all Docker images..."
    for num in 1 2 3 4 5 6 7 8 9 10 11; do
        if should_skip "${num}"; then
            echo "  Skipping build for gpu-stress-workload${num} (--skip)"
            continue
        fi
        dir="$(workload_dir "${num}")"
        echo "  Building gpu-stress-workload${num} (${dir}) ..."
        if [[ "${num}" == "7" ]]; then
            docker build --tag "gpu-stress-workload${num}" \
                --build-arg "INSTALL_HF_DEPS=${INSTALL_HF_DEPS}" \
                "${SUITE_DIR}/${dir}" --quiet 2>&1 | tail -1
        else
            docker build --tag "gpu-stress-workload${num}" "${SUITE_DIR}/${dir}" --quiet 2>&1 | tail -1
        fi
    done
}

render_manifest() {
    python3 - "${PHASE_LOG}" "${MANIFEST_FILE}" "${TIMESTAMP}" "${ASSESSMENT_DIR}" "${MAX_VRAM_GB}" "${NODE_ID}" "$(phase_profile_kv)" "${DRY_RUN}" "${HW_GPU_PROFILE_ID}" "${HW_PLATFORM_PROFILE_ID}" <<'PY'
import csv
import json
import sys
from pathlib import Path

phase_log = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
run_id = sys.argv[3]
assessment_dir = sys.argv[4]
max_vram_gb = float(sys.argv[5])
node_id = sys.argv[6]
phase_profile = {}
for item in sys.argv[7].split(","):
    if not item:
        continue
    key, value = item.split("=", 1)
    phase_profile[key] = int(value)

phases = []
with phase_log.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        phases.append(row)

dry_run_level = int(sys.argv[8])

manifest = {
    "node_id": node_id,
    "gpu_profile": sys.argv[9],
    "platform_profile": sys.argv[10],
    "run_id": run_id,
    "dry_run": dry_run_level != 0,
    "dry_run_level": dry_run_level or None,
    "assessment_type": "local_recipe_aligned_prototype",
    "assessment_dir": assessment_dir,
    "phase_durations_s": phase_profile,
    "max_vram_gb": max_vram_gb,
    "phases": phases,
}

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
}

write_summary() {
    local end_time="$1"
    local elapsed="$2"
    {
        echo "GPU Stress Suite - Local Assessment Summary"
        if [[ "${DRY_RUN}" != "0" ]]; then
            echo "*** DRY RUN (level ${DRY_RUN}) - workload results are MOCK data ***"
        fi
        echo "Node ID        : ${NODE_ID}"
        echo "Run ID         : ${TIMESTAMP}"
        echo "Assessment dir : ${ASSESSMENT_DIR}"
        echo "Local GPU note : ${GPU_LABEL}"
        echo "Phase profile  : $(phase_profile_kv)"
        echo "Warmup         : ${WARMUP_DURATION}s"
        echo "Idle/Cooldown  : ${IDLE_DURATION}s idle, ${COOLDOWN_DURATION}s cooldown"
        echo "VRAM cap       : ${MAX_VRAM_GB} GB"
        echo "Total time     : $((elapsed/60))m $((elapsed%60))s"
        echo ""
        echo "Artifacts:"
        echo "  Manifest : ${MANIFEST_FILE}"
        echo "  Results  : ${RESULTS_DIR}"
        echo ""
        echo "Phase log:"
        cat "${PHASE_LOG}"
    } > "${SUMMARY_FILE}"

    {
        echo "GPU Stress Suite - Local Assessment Summary"
        if [[ "${DRY_RUN}" != "0" ]]; then
            echo "*** DRY RUN (level ${DRY_RUN}) - workload results are MOCK data ***"
        fi
        echo "Run ID    : ${TIMESTAMP}"
        echo "Results   : ${ASSESSMENT_DIR}"
        echo "Total time: $((elapsed/60))m $((elapsed%60))s"
        echo ""
        echo "Latest files:"
        find "${RESULTS_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sort
    } > "${ROOT_INDEX_FILE}"
}

collect_artifacts() {
    local start_epoch="$1"
    local script_timestamp="$2"   # TIMESTAMP set at script start (before image builds)
    local dest_dir="$3"

    echo ""
    echo "  [COLLECT] Gathering run artifacts into assessment dir..."

    python3 - "${start_epoch}" "${script_timestamp}" "${SUITE_DIR}" "${ROOT_RESULTS_DIR}" "${dest_dir}" <<'PY'
import sys, re, shutil
from datetime import datetime, timezone
from pathlib import Path

start_epoch     = int(sys.argv[1])
script_ts_str   = sys.argv[2]      # yy-mm-dd_hh-mm-ss from TIMESTAMP
suite_dir       = Path(sys.argv[3])
results_dir     = Path(sys.argv[4])
dest_dir        = Path(sys.argv[5])
WINDOW          = 120  # ±2 minutes

TS_RE = re.compile(r'(\d{2}-\d{2}-\d{2}_\d{2}-\d{2}(?:-\d{2})?)')

def ts_to_epoch(ts_str):
    for fmt in ('%y-%m-%d_%H-%M-%S', '%y-%m-%d_%H-%M'):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return None

script_epoch = ts_to_epoch(script_ts_str) or start_epoch

def in_window(filename, ref_epoch):
    m = TS_RE.search(filename)
    if not m:
        return False
    epoch = ts_to_epoch(m.group(1))
    return epoch is not None and abs(epoch - ref_epoch) <= WINDOW

def copy_file(src, label):
    try:
        shutil.move(str(src), str(dest_dir / src.name))
        print(f"  [COLLECT]   {label}: {src.name}")
        return 1
    except Exception as e:
        print(f"  [COLLECT]   WARN: {src.name}: {e}")
        return 0

n = 0

# 1. suite_results/summary_<ts>.txt
for f in results_dir.glob('summary_*.txt'):
    if in_window(f.name, start_epoch):
        n += copy_file(f, 'summary')

# 2. ~/loggers/results/ — bmc/nvml metadata + telemetry
#    Anchored to start_epoch: loggers start after image builds complete
loggers_results = Path('/opt/tensormachines') / 'loggers' / 'results'
if loggers_results.is_dir():
    for f in sorted(loggers_results.iterdir()):
        if f.is_file() and f.suffix in ('.txt', '.csv') and in_window(f.name, start_epoch):
            n += copy_file(f, 'telemetry')

# 3. Suite-dir run_all log files  (e.g. 26-07-11_04-51_run_all_loop1.log)
#    Anchored to script_epoch: log is named when loop_run.sh starts (before image builds)
for f in suite_dir.glob('*_run_all*.log'):
    if in_window(f.name, script_epoch):
        n += copy_file(f, 'run log')

print(f"  [COLLECT] Done — {n} file(s) copied to {dest_dir.name}")
PY
}

upload_results() {
    if [[ "${UPLOAD_RESULTS}" != "1" ]]; then
        echo "  [UPLOAD] Disabled (use --upload or UPLOAD_RESULTS=1). Results local: ${ASSESSMENT_DIR}"
        return 0
    fi
    echo ""
    echo "  [UPLOAD] Uploading results to S3..."
    local -a upload_args=("${ASSESSMENT_DIR}")
    if [[ -n "${CONTACT}" ]]; then
        upload_args+=("${CONTACT}")
    fi
    if [[ "${DRY_RUN}" != "0" ]]; then
        echo "  [UPLOAD] Dry run - invoking upload_results.sh with --dryrun (nothing is transferred)."
        upload_args+=(--dryrun)
    fi
    if bash "${SUITE_DIR}/scripts/upload_results.sh" "${upload_args[@]}"; then
        echo "  [UPLOAD] OK"
    else
        echo "  [UPLOAD] WARN: upload failed; results remain local: ${ASSESSMENT_DIR}"
    fi
}

# Scores the run at full utilization.
compute_score() {
    if [[ "${SCORE_ENABLED}" != "1" ]]; then
        SCORE_NOTE="skipped (no --score)"
        return 0
    fi
    if [[ "${DRY_RUN}" != "0" ]]; then
        SCORE_NOTE="skipped (dry run)"
        return 0
    fi
    # Scoring uses phase 4 (1C).
    local status
    status="$(awk -F'\t' '$1 == "4" { print $8 }' "${PHASE_LOG}")"
    if [[ "${status}" == "skipped" ]]; then
        SCORE_NOTE="skipped (1C not run)"
        return 0
    fi
    if [[ "${status}" != "completed" ]]; then
        SCORE_NOTE="skipped (1C did not complete)"
        return 0
    fi
    local -a score_args=(--ce "${SCORE_CE}")
    if [[ -n "${SCORE_OH}" ]]; then
        score_args+=(--oh "${SCORE_OH}")
    fi
    if [[ -n "${SCORE_RH}" ]]; then
        score_args+=(--rh "${SCORE_RH}")
    fi
    if [[ -n "${HW_NODE_DESIGN_POWER_W:-}" ]]; then
        score_args+=(--np "${HW_NODE_DESIGN_POWER_W}")
    fi
    echo ""
    echo "  [SCORE] Computing score..."
    if SCORE="$(python3 "${SUITE_DIR}/harness/score.py" "${ASSESSMENT_DIR}" "${score_args[@]}")"; then
        echo "  [SCORE] Saved: ${ASSESSMENT_DIR}/score.json"
        # Oh and Rh as used, and the pricing class of any defaults applied.
        read -r SCORE_OH SCORE_RH SCORE_PRICING_CLASS SCORE_PRICING_APPLIED SCORE_GPU < <(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
p = d["pricing_defaults"] or {}
applied = " and ".join(n.capitalize() for n in p.get("applied", [])) or "-"
print(d["inputs"]["oh"], d["inputs"]["rh"], p.get("gpu_class") or "-",
      applied.replace(" ", "_"), d["gpu_class"])' \
            "${ASSESSMENT_DIR}/score.json")
    else
        SCORE=""
        SCORE_NOTE="failed (see error above)"
        echo "  [SCORE] WARN: scoring failed"
    fi
}

# Prints the utilization entered at the terminal, nothing when skipped or when
# there is no terminal.
ask_uz() {
    local reply="" try
    { exec 3< /dev/tty; } 2>/dev/null || return 0
    for try in 1 2 3; do
        read -r -p "  Your expected utilization (greater than 0, at most 1), or Enter to skip: " \
            reply <&3 || reply=""
        if [[ -z "${reply}" ]] || valid_uz "${reply}"; then
            break
        fi
        echo "  Enter a number greater than 0 and at most 1, e.g. 0.6" >&2
        reply=""
    done
    exec 3<&-
    printf '%s' "${reply}"
}

# Prints the score block for utilization $1 and score $2.
print_score() {
    echo ""
    echo "============================================================"
    printf "  SCORE at %s%% utilization (USD per million tokens): %s\n" \
        "$(awk -v v="$1" 'BEGIN { printf "%g", v * 100 }')" "$2"
    echo "============================================================"
}

cleanup() { stop_nvml; stop_bmc; }
trap cleanup EXIT
trap 'trap - EXIT; cleanup; exit 130' INT
trap 'trap - EXIT; cleanup; exit 143' TERM HUP

echo ""
echo "============================================================"
echo "  GPU Stress Suite - Local Recipe-Aligned Assessment"
if [[ "${DRY_RUN}" != "0" ]]; then
    printf "  *** DRY RUN at level %s - no workloads will be executed ***\n" "${DRY_RUN}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf "  *** Docker build     : skipped (dry run level 1)\n"
    else
        printf "  *** Docker build     : enabled (dry run level 2)\n"
    fi
    printf "  *** Idle/cooldown    : clamped to %ss (dry run)\n" "${DRY_RUN_IDLE_SEC}"
fi
printf "  Node ID          : %s\n" "${NODE_ID}"
printf "  GPU profile      : %s\n" "${HW_GPU_PROFILE_ID}"
printf "  Platform profile : %s\n" "${HW_PLATFORM_PROFILE_ID}"
printf "  Local GPU        : %s\n" "${GPU_LABEL}"
printf "  Phase profile    : workloads only (see summary/manifest)\n"
printf "  Idle duration    : %ss\n" "${IDLE_DURATION}"
printf "  Warmup duration  : %ss\n" "${WARMUP_DURATION}"
printf "  Cooldown duration: %ss\n" "${COOLDOWN_DURATION}"
printf "  HF deps install  : %s\n" "${INSTALL_HF_DEPS}"
if [[ "${MAX_VRAM_SOURCE}" == "profile" ]]; then
    printf "  VRAM cap         : %s GB per GPU (from profile %s)\n" \
        "${MAX_VRAM_GB}" "${HW_GPU_PROFILE_ID}"
else
    printf "  VRAM cap         : %s GB per GPU (--max-vram-gb)\n" "${MAX_VRAM_GB}"
fi
printf "  Assessment dir   : %s\n" "${ASSESSMENT_DIR}"
echo "============================================================"

ASSESSMENT_START="$(date +%s)"

start_nvml
start_bmc

build_all_images

MULTI_GPU_ARGS=()
if [[ "${REQUIRE_MULTI_GPU}" == "1" ]]; then
    MULTI_GPU_ARGS+=(--require-multi-gpu)
fi


run_idle_phase        "0"  "Idle Baseline"                    "${IDLE_DURATION}"
run_workload_phase    "1"  "Warmup Tensor Compute"            1  "${WARMUP_DURATION}"
run_workload_phase    "2"  "1A GEMM Compute Baseline"         1  "$(phase_sec 1A)" --sequential
run_workload_phase    "3"  "1B Memory Bandwidth Baseline"     5  "$(phase_sec 1B)"
run_workload_phase    "4"  "1C LLM Inference"                 8  "$(phase_sec 1C)"
run_idle_phase        "5"  "Cooldown 1"                       "${COOLDOWN_DURATION}"
run_workload_phase    "6"  "2A Inference Workload"            9  "$(phase_sec 2A)"
run_workload_phase    "7"  "2B Training Workload"           10  "$(phase_sec 2B)"
run_workload_phase    "8"  "2C Convolution Compute"           2  "$(phase_sec 2C)"
run_workload_phase    "9"  "2D FFT Compute"                   4  "$(phase_sec 2D)"
run_workload_phase    "10" "2E Random Access Memory"          3  "$(phase_sec 2E)"
run_idle_phase        "11" "Cooldown 2"                       "${COOLDOWN_DURATION}"
run_workload_phase    "12" "3A Inference Workload"           11  "$(phase_sec 3A)"
run_workload_phase    "13" "3B Sustained Power Proxy GEMM"       1  "$(phase_sec 3B)"
run_workload_phase    "14" "NVLink NCCL Collective Stress"    6  "$(phase_sec nccl)" "${MULTI_GPU_ARGS[@]}"
run_idle_phase        "15" "Final Idle Snapshot"              "${IDLE_DURATION}"

ASSESSMENT_END="$(date +%s)"
TOTAL_ELAPSED=$((ASSESSMENT_END - ASSESSMENT_START))

# Stop loggers before collecting so telemetry files are fully flushed
cleanup
trap - EXIT INT TERM HUP   # already stopped; prevent double-kill

render_manifest
write_summary "${ASSESSMENT_END}" "${TOTAL_ELAPSED}"

echo ""
echo "============================================================"
echo "  ASSESSMENT COMPLETE"
printf "  Total time : %dm %ds\n" $((TOTAL_ELAPSED/60)) $((TOTAL_ELAPSED%60))
printf "  Summary    : %s\n" "${SUMMARY_FILE}"
printf "  Results    : %s\n" "${RESULTS_DIR}"
echo "============================================================"

collect_artifacts "${ASSESSMENT_START}" "${TIMESTAMP}" "${ASSESSMENT_DIR}"
compute_score

if [[ -z "${SCORE}" ]]; then
    upload_results
    echo ""
    echo "============================================================"
    printf "  SCORE: %s\n" "${SCORE_NOTE}"
    echo "============================================================"
    exit 0
fi

echo ""
printf "  Inputs : Oh=%s Rh=%s Ce=%s USD\n" "${SCORE_OH}" "${SCORE_RH}" "${SCORE_CE}"
if [[ "${SCORE_PRICING_CLASS}" == "unknown" ]]; then
    printf "  WARNING: no pricing data for %s. %s set to the average across\n" \
        "${SCORE_GPU}" "${SCORE_PRICING_APPLIED//_/ }"
    printf "           known GPU classes. To use your own prices, set SCORE_OH and SCORE_RH\n"
    printf "           (USD per GPU per hour) in the repo's .env, e.g. SCORE_OH=2.50\n"
elif [[ ${#SCORE_DEFAULTS[@]} -gt 0 ]]; then
    printf "  WARNING: default Oh used, the 75th percentile rental price for this GPU class.\n"
fi
print_score 1 "${SCORE}"
echo "  This is the cost if the GPUs are busy 100% of the time."

uz="${SCORE_UZ}"
if [[ -z "${uz}" ]]; then
    echo ""
    uz="$(ask_uz)"
fi
if [[ -n "${uz}" ]]; then
    if custom_score="$(python3 "${SUITE_DIR}/harness/score_at_uz.py" "${ASSESSMENT_DIR}" "${uz}")"; then
        print_score "${uz}" "${custom_score}"
    else
        echo "  WARNING: scoring at utilization ${uz} failed (see error above)."
    fi
fi

upload_results
echo ""
echo "  Score details: ${ASSESSMENT_DIR}/score.json"
