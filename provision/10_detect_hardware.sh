#!/usr/bin/env bash
# 10_detect_hardware.sh — record this node's hardware fingerprint.
#
# Run once per node, after the driver (3), ipmitool (2), dirs (7) and
# python3-venv (9) are in place, and after the suite is deployed. Re-run after
# any hardware change.
#
# The fingerprint holds detected facts only — no profile information. It is
# what run_all.sh checks the node against before every assessment.
#
# After saving it, this script checks that the suite has profiles that resolve
# for this hardware, so a missing profile surfaces now rather than at the start
# of an assessment. A missing GPU profile is fatal; a missing platform profile
# is a warning, since it only affects which BMC logger runs.
#
# Filled-in profiles from the repo replace older copies in the suite first.
# When none matches, a template is written to the repo for the operator.
set -euo pipefail

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

REPO_DIR="${REPO_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
SUITE_DIR="${SUITE_DIR:-/tensormachines/gpu-stress-suite}"
HW_FINGERPRINT="${HW_FINGERPRINT:-/opt/tensormachines/hw_fingerprint.json}"
HWDETECT="${SUITE_DIR}/harness/hwdetect.py"
REPO_PROFILES="${REPO_DIR}/gpu-stress-suite/hardware"

if [[ ! -f "${HWDETECT}" ]]; then
  echo "ERROR: ${HWDETECT} not found."
  echo "       Deploy the suite first: gpu-stress-suite/* -> ${SUITE_DIR}/"
  exit 1
fi

echo "=== Detecting hardware (probes the BMC; takes a few seconds) ==="
python3 "${HWDETECT}" detect --save "${HW_FINGERPRINT}" >/dev/null

python3 - "${HW_FINGERPRINT}" <<'PY'
import json, sys
f = json.load(open(sys.argv[1]))
g, p = f["gpu"], f["platform"]
print(f"  GPUs          : {g['count']} x {g['name']}  ({g['memory_mib']} MiB each)")
print(f"  PCI device id : {g['pci_device_id']}")
print(f"  Chassis       : {p.get('product_name')}")
print(f"  BMC transport : {p.get('bmc_transport')}  ({len(p.get('sensors', []))} sensors)")
if not g["homogeneous"]:
    print("  WARNING: GPUs are NOT homogeneous — workload sizing assumes they are.")
if p.get("bmc_transport") == "none":
    print("  WARNING: no BMC transport answered — BMC telemetry will be unavailable.")
PY

echo ""
echo "=== Checking that the suite has profiles for this hardware ==="
sync_filled_profiles "${REPO_PROFILES}" "${SUITE_DIR}/hardware"
set +e
PROFILE_OUT="$(python3 "${HWDETECT}" profile "${HW_FINGERPRINT}" 2>&1)"
profile_rc=$?
set -e

case "${profile_rc}" in
  0) echo "${PROFILE_OUT}" | sed 's/^/  /' ;;
  2|4|6) echo "${PROFILE_OUT}" | sed 's/^/  /' >&2
     TEMPLATE_OUT="$(python3 "${HWDETECT}" template "${HW_FINGERPRINT}" --hardware-dir "${REPO_PROFILES}")"
     echo ""
     if [[ -n "${TEMPLATE_OUT}" ]]; then
       echo "${TEMPLATE_OUT}" | sed 's/^/  /'
       echo ""
       echo "  Replace every CHANGEME value in the file(s) above, then re-run run.sh."
       echo "  Sensor names for match.requires_sensors: ${HW_FINGERPRINT}"
       echo "  How to fill them in: ${REPO_DIR}/README.md, section \"Adding hardware\""
     else
       echo "  Fix the profile(s) listed above, then re-run run.sh."
     fi
     exit 13 ;;
  *) echo "${PROFILE_OUT}" >&2
     echo ""
     echo "  ERROR: hardware detection failed."
     exit 1 ;;
esac

echo ""
echo "Hardware fingerprint saved to ${HW_FINGERPRINT}"
