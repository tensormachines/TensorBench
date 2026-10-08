#!/usr/bin/env bash
#
# Run 1 time with 5-min break (default) 
# ./loop_run.sh 
#
# Run 5 times with 10-min break, passing args to run_all.sh
#./loop_run.sh --runs 5 --break 600 --max-vram-gb 60
#
# Run forever
#./loop_run.sh --runs 0
#
# Run 30 times and upload results to S3
#./loop_run.sh --upload --runs 30 2>&1 | tee -i 26-07-15_17-38_loop_run.log
#
# note: tee -i is used to allow Ctrl-C to interrupt the run without killing the tee process, so loggers can stop gracefully

set -euo pipefail

BREAK_SEC=${BREAK_SEC:-300}
MAX_RUNS=1
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)  MAX_RUNS="$2"; shift 2 ;;
        --break) BREAK_SEC="$2"; shift 2 ;;
        *)       PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

# Unattended runs score at 100% utilization instead of asking for it.
export SCORE_UZ="${SCORE_UZ:-1}"

RUN=0

while true; do
    RUN=$((RUN + 1))
    LOG_NAME="$(date -u +'%y-%m-%d_%H-%M')_run_all_loop${RUN}.log"

    echo ""
    echo "================================================================"
    echo "  LOOP RUN #${RUN}$([ "${MAX_RUNS}" -gt 0 ] && echo " / ${MAX_RUNS}" || echo " (unlimited)")  — $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "  Log: ${LOG_NAME}"
    echo "================================================================"

    bash "${SUITE_DIR}/run_all.sh" "${PASSTHROUGH_ARGS[@]}" 2>&1 | tee -i "${SUITE_DIR}/${LOG_NAME}" \
        || echo "[WARN] run_all.sh exited non-zero; continuing."

    if [[ "${MAX_RUNS}" -gt 0 && "${RUN}" -ge "${MAX_RUNS}" ]]; then
        echo ""
        echo "  Reached --runs ${MAX_RUNS}. Done."
        break
    fi

    echo ""
    echo "  Run #${RUN} complete. Sleeping ${BREAK_SEC}s before next run..."
    sleep "${BREAK_SEC}"
done