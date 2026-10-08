#!/bin/bash
# 3_install_nvidia_driver.sh
# Install the NVIDIA server driver, 580 series (reference release 580.159.03).
# Idempotent: if a working driver is already present, exits without reinstalling.
# NOTE: requires REBOOT after fresh install before nvidia-smi works.
set -e

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

DRIVER_SERIES="580"
DRIVER_PKG="nvidia-driver-${DRIVER_SERIES}-server"
REF_VERSION="580.159.03"   # reference point release

# ---------------------------------------------------------------------------
echo "=== [1/5] Checking for existing driver ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  CUR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  if [ -n "$CUR" ]; then
    echo "Driver already installed and responding: $CUR"
    echo "Reference is $REF_VERSION. Nothing to do."
    nvidia-smi --query-gpu=index,name --format=csv,noheader
    exit 0
  fi
  echo "nvidia-smi present but not responding — driver may be installed, pending reboot."
  echo "If you just installed, REBOOT and re-run baseline. Aborting to avoid double-install."
  exit 0
fi
echo "No working driver found. Proceeding with install."

# ---------------------------------------------------------------------------
echo "=== [2/5] Refreshing apt + checking package availability ==="
apt_get update
if ! apt-cache show "$DRIVER_PKG" >/dev/null 2>&1; then
  echo "WARNING: $DRIVER_PKG not found in current repos."
  echo "You likely need the CUDA/graphics repo (should already be present from CUDA 12.6 install)."
  echo "Available nvidia-driver-*-server metapackages:"
  apt-cache search '^nvidia-driver-[0-9]*-server$' || true
  if [ "${ASSUME_YES:-1}" != "1" ]; then
    read -p "Continue anyway with '$DRIVER_PKG'? Type 'yes' to proceed: " CONFIRM
    [ "$CONFIRM" = "yes" ] || { echo "Aborted."; exit 1; }
  fi
fi

# show exact point release apt will install vs reference
echo "--- Candidate version apt will install ---"
apt-cache madison "$DRIVER_PKG" | head -3 || true
echo "Reference point release: $REF_VERSION (minor drift within 580 series is normally fine)"

# ---------------------------------------------------------------------------
echo "=== [3/5] Installing $DRIVER_PKG ==="
apt_get install "$DRIVER_PKG"

# ---------------------------------------------------------------------------
# --- install matching Fabric Manager (required on NVSwitch systems) ---
if lspci | grep -qi nvswitch; then
  echo "NVSwitch detected — installing Fabric Manager"

  echo "=== Installing NVIDIA Fabric Manager (NVSwitch) ==="
  FM_PKG="nvidia-fabricmanager-${DRIVER_SERIES}"
  # try exact point-release match first, fall back to series
  FM_EXACT="${REF_VERSION}-1"
  if apt-cache madison "$FM_PKG" | grep -q "$REF_VERSION"; then
    echo "Installing $FM_PKG matching driver $REF_VERSION"
    apt_get install "$FM_PKG"
  else
    echo "WARNING: exact $REF_VERSION not found for $FM_PKG"
    apt-cache madison "$FM_PKG" || true
    FMC="yes"
    if [ "${ASSUME_YES:-1}" != "1" ]; then
      read -p "Install latest $FM_PKG anyway? Type 'yes': " FMC
    fi
    [ "$FMC" = "yes" ] && apt_get install "$FM_PKG" || echo "Skipped — GPUs will NOT init without matching fabricmanager"
  fi
  sudo systemctl enable nvidia-fabricmanager
  echo "fabricmanager enabled — starts on boot (needs live driver, so effective after reboot)"

else
  echo "No NVSwitch detected — skipping Fabric Manager (not needed on this node)"
fi
# ---------------------------------------------------------------------------
echo "=== [5/5] Post-install ==="
# The driver usually loads without a reboot; only ask for one if it did not.
if nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1; then
  echo "Driver is live, no reboot needed."
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
  exit 0
fi

echo "Driver installed but nvidia-smi is not responding yet."
echo "After a reboot, verify with:"
echo "  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"
echo ""
exit 10
