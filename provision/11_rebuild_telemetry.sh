#!/usr/bin/env bash
# 11_rebuild_telemetry.sh — build the telemetry virtualenv.
#
# The venv hardcodes its own path, so it is built at its final location. It must
# use Python 3.10; an existing venv on a different version is rebuilt.
set -euo pipefail

TM="${OPT_DIR:-/opt/tensormachines}"
ENVS="$TM/envs"
REQ_DIR="$TM/setup"
PY_SERIES="3.10"
PYBIN="python${PY_SERIES}"
VENV="$ENVS/telemetry_env"

# --- 0. checks -------------------------------------------------------------
echo "=== [0/3] Checks ==="
command -v "$PYBIN" >/dev/null 2>&1 || {
  echo "ERROR: $PYBIN not found. Run 9_setup_python.sh first."
  exit 1
}
"$PYBIN" --version
sudo chown -R "$(id -un):$(id -gn)" "$TM"
mkdir -p "$ENVS"
[ -f "$REQ_DIR/pip_telemetry_env.txt" ] || { echo "ERROR: $REQ_DIR/pip_telemetry_env.txt missing"; exit 1; }

# --- 1. discard a venv built on the wrong interpreter -----------------------
echo "=== [1/3] Existing environment ==="
if [ -x "$VENV/bin/python" ]; then
  CURRENT="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "unknown")"
  if [ "$CURRENT" = "$PY_SERIES" ]; then
    echo "Existing telemetry_env is on Python $CURRENT — rebuilding in place."
  else
    echo "Existing telemetry_env is on Python $CURRENT, not $PY_SERIES. Removing."
    rm -rf "$VENV"
  fi
else
  echo "No existing telemetry_env."
fi

# --- 2. build --------------------------------------------------------------
echo "=== [2/3] Building telemetry_env ==="
[ -x "$VENV/bin/python" ] || "$PYBIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REQ_DIR/pip_telemetry_env.txt"

# --- 3. verify -------------------------------------------------------------
echo "=== [3/3] Verifying ==="
BUILT="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$BUILT" = "$PY_SERIES" ] || { echo "ERROR: telemetry_env is on Python $BUILT, expected $PY_SERIES"; exit 1; }
echo "telemetry_env on Python $BUILT"
"$VENV/bin/python" -c "import pynvml; print('nvml ok')" || echo "WARN: pynvml import failed"

echo ""
echo "=== Done — $VENV ==="
