#!/usr/bin/env bash
# 9_setup_python.sh — make sure Python 3.10 with venv support is available.
#
# Prefers apt. On releases that no longer carry Python 3.10 (Ubuntu 24.04 and
# later) it builds from source with `make altinstall`, which adds python3.10
# alongside the system interpreter without replacing it.
set -euo pipefail

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

PY_SERIES="3.10"
PY_VERSION="3.10.21"
SRC_URL="https://www.python.org/ftp/python/${PY_VERSION}/Python-${PY_VERSION}.tgz"
BUILD_DIR="/tmp/python-${PY_VERSION}-build"
BUILD_LOG="/tmp/python-${PY_VERSION}-build.log"

# Usable means: the interpreter runs and can create a virtual environment.
python310_usable() {
  command -v python${PY_SERIES} >/dev/null 2>&1 || return 1
  local probe="/tmp/py310probe_$$"
  python${PY_SERIES} -m venv "$probe" >/dev/null 2>&1 || { rm -rf "$probe"; return 1; }
  rm -rf "$probe"
}

echo "=== [1/4] Checking for Python ${PY_SERIES} ==="
if python310_usable; then
  echo "python${PY_SERIES} present and can create venvs: $(python${PY_SERIES} --version)"
  exit 0
fi

echo "=== [2/4] Trying apt ==="
apt_get update
if apt-cache show "python${PY_SERIES}-venv" >/dev/null 2>&1; then
  apt_get install "python${PY_SERIES}" "python${PY_SERIES}-venv" "python${PY_SERIES}-dev"
  if python310_usable; then
    echo "Installed from apt: $(python${PY_SERIES} --version)"
    exit 0
  fi
  echo "apt packages installed but python${PY_SERIES} is still not usable."
else
  echo "python${PY_SERIES}-venv is not in this release's repositories."
fi

echo "=== [3/4] Building Python ${PY_VERSION} from source ==="
echo "This takes several minutes."
apt_get install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
  libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev \
  uuid-dev tk-dev wget

# make altinstall leaves root-owned bytecode in the build tree.
sudo rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
wget -q -O "$BUILD_DIR/Python-${PY_VERSION}.tgz" "$SRC_URL"
tar -xzf "$BUILD_DIR/Python-${PY_VERSION}.tgz" -C "$BUILD_DIR"

# altinstall adds python3.10; it never touches python3 or the system interpreter.
if ! ( cd "$BUILD_DIR/Python-${PY_VERSION}" &&
       ./configure --prefix=/usr/local --with-ensurepip=install &&
       make -j"$(nproc)" &&
       sudo make altinstall ) >"$BUILD_LOG" 2>&1; then
  echo "ERROR: building Python ${PY_VERSION} failed. Last lines of ${BUILD_LOG}:"
  tail -n 30 "$BUILD_LOG"
  exit 1
fi
sudo rm -rf "$BUILD_DIR"

echo "=== [4/4] Verifying ==="
if ! python310_usable; then
  echo "ERROR: python${PY_SERIES} still cannot create a virtual environment."
  exit 1
fi

echo "python${PY_SERIES}: $(command -v python${PY_SERIES})  $(python${PY_SERIES} --version)"
echo "system python3 untouched: $(command -v python3)  $(python3 --version)"
echo "=== Done ==="
