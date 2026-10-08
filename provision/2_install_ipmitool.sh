#!/bin/bash
set -e

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

# Pinned ipmitool version.
IPMITOOL_VERSION="1.8.18-11ubuntu2.2"

echo "=== Installing ipmitool (${IPMITOOL_VERSION}) ==="

if ipmitool -V 2>/dev/null | grep -q "1.8.18"; then
    echo "ipmitool 1.8.18 already installed."
else
    apt_get update
    # Try exact pinned version first; fall back to latest available if unavailable
    if sudo apt-cache policy ipmitool | grep -q "${IPMITOOL_VERSION}"; then
        apt_get install ipmitool="${IPMITOOL_VERSION}"
    else
        echo "WARNING: pinned version ${IPMITOOL_VERSION} not in repo, installing available version."
        apt_get install ipmitool
    fi
fi

echo "=== Verification ==="
which ipmitool
ipmitool -V
if sudo ipmitool -I open sdr list >/dev/null 2>&1; then
  echo "Local IPMI access confirmed."
else
  echo "No local IPMI device. BMC telemetry will need remote IPMI credentials"
  echo "(BMC_HOST/BMC_USER/BMC_PASS) or will not run on this node."
fi
echo "=== Done ==="