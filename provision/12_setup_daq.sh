#!/usr/bin/env bash
# 12_setup_daq.sh — install LabJack LJM driver + build daq_env (DAQ-equipped nodes ONLY)
# Run only on nodes physically wired to a LabJack T7.
# Prereqs (drop files):
#   /opt/tensormachines/setup/daq_setup/labjack_ljm_installer.run
#   /opt/tensormachines/setup/pip_daq_env.txt
set -euo pipefail

TM="/opt/tensormachines"
ENVS="$TM/envs"
SETUP="$TM/setup"
INSTALLER="$SETUP/daq_setup/labjack_ljm_installer.run"
PYBIN="python3"

# --- 0. checks -------------------------------------------------------------
echo "=== [0/4] Checks ==="
$PYBIN --version
[ -f "$INSTALLER" ]            || { echo "ERROR: $INSTALLER missing"; exit 1; }
[ -f "$SETUP/pip_daq_env.txt" ]    || { echo "ERROR: $SETUP/pip_daq_env.txt missing"; exit 1; }
sudo chown -R "$(id -un):$(id -gn)" "$TM"
mkdir -p "$ENVS"
chmod +x "$INSTALLER"

# --- 1. install LJM driver (OS-level) --------------------------------------
echo "=== [1/4] Installing LabJack LJM (Makeself self-extracting) ==="
# installs .so -> /usr/local/lib, headers -> /usr/local/include,
# device udev rules + permissions, Kipling -> /opt/labjack_kipling
sudo "$INSTALLER"
sudo ldconfig          # refresh linker cache so libLabJackM.so is found

# --- 2. build daq_env ------------------------------------------------------
echo "=== [2/4] Building daq_env ==="
$PYBIN -m venv "$ENVS/daq_env"
"$ENVS/daq_env/bin/python" -m pip install --upgrade pip
"$ENVS/daq_env/bin/python" -m pip install -r "$SETUP/pip_daq_env.txt"

# --- 3. verify LJM lib + python binding ------------------------------------
echo "=== [3/4] Verifying ==="
ls -l /usr/local/lib/libLabJackM* 2>/dev/null || echo "WARN: libLabJackM not found in /usr/local/lib"
"$ENVS/daq_env/bin/python" -c "from labjack import ljm; print('ljm import ok')" \
  || echo "WARN: ljm import failed — check ldconfig / driver install"

# --- 4. device check (optional, needs T7 plugged in) -----------------------
echo "=== [4/4] Device probe (skip if T7 not yet connected) ==="
"$ENVS/daq_env/bin/python" - <<'PY' || echo "NOTE: no device opened — connect T7 and re-probe"
from labjack import ljm
h = ljm.openS("T7", "ANY", "ANY")   # any connection, any T7
info = ljm.getHandleInfo(h)
print("Opened T7, serial:", info[2])
ljm.close(h)
PY

echo ""
echo "=== Done ==="
echo "daq_env at: $ENVS/daq_env  (point alias at bin/python, no activation)"