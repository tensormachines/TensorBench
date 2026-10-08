#!/usr/bin/env bash
# =============================================================================
# run_real_workloads.sh
#
# Runs the four real (model-based) workloads in recipe Tier order:
#     1C  ->  2A  ->  2B  ->  3A
#
# Each workload's own run.sh handles its image build, GPU check, bind mounts,
# and results dir. This wrapper just sequences them and forwards any CLI
# args (e.g. --gpus 0,1) uniformly. Per-workload results land in each
# workload's own results/ directory.
#
# Usage:
#   HF_TOKEN=hf_xxx ./run_real_workloads.sh                 # full sequence
#   HF_TOKEN=hf_xxx ./run_real_workloads.sh --gpus 0,1     # restrict GPUs
#   ./run_real_workloads.sh --smoke --gpus 0               # full shakeout sequence
#
# Prereqs (same as the individual run.sh's):
#   - Docker + NVIDIA Container Toolkit.
#   - HF_TOKEN with licenses accepted for:
#       meta-llama/Meta-Llama-3-8B-Instruct  (1C, 2B with the 8B base)
#       meta-llama/Llama-2-13b-hf            (2A, 3A)
#       meta-llama/Llama-3.2-3B-Instruct     (2B, the default base)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Recipe Tier order: 1C (integrated baseline) -> 2A (13B inference diagnostic)
# -> 2B (LoRA write-path diagnostic) -> 3A (boundary probe).
WORKLOADS=(workload_1C workload_2A workload_2B workload_3A)

OVERALL_T0=$(date +%s)
for w in "${WORKLOADS[@]}"; do
  if [[ ! -x "${SCRIPT_DIR}/${w}/run.sh" ]]; then
    echo "ERROR: ${SCRIPT_DIR}/${w}/run.sh not found or not executable." >&2
    exit 1
  fi

  echo ""
  echo "########################################################"
  echo "#  ${w}"
  echo "#  $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "########################################################"

  PHASE_T0=$(date +%s)
  (cd "${SCRIPT_DIR}/${w}" && ./run.sh "$@")
  PHASE_DT=$(( $(date +%s) - PHASE_T0 ))
  echo "[wrapper] ${w} done in ${PHASE_DT}s. Results -> ${SCRIPT_DIR}/${w}/results/"
done

OVERALL_DT=$(( $(date +%s) - OVERALL_T0 ))
echo ""
echo "########################################################"
echo "#  All real workloads complete in ${OVERALL_DT}s"
echo "#  ($(printf '%dh%02dm%02ds' $((OVERALL_DT/3600)) $((OVERALL_DT%3600/60)) $((OVERALL_DT%60))))"
echo "########################################################"
