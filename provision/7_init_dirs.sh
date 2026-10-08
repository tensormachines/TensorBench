#!/usr/bin/env bash
# 7_init_dirs.sh — one-time directory setup on any new node.
# Usage: ./7_init_dirs.sh [username]
#   username defaults to the invoking user, or $SUDO_USER when run under sudo
set -euo pipefail

OPT_DIR="${OPT_DIR:-/opt/tensormachines}"
SUITE_DIR="${SUITE_DIR:-/tensormachines/gpu-stress-suite}"
DATA_DIR="$(dirname "$SUITE_DIR")"

TARGET_USER="${1:-${SUDO_USER:-$(id -un)}}"
[ -z "$TARGET_USER" ] && { echo "ERROR: could not determine target user. Pass username as arg: ./7_init_dirs.sh <username>"; exit 1; }

echo "=== Setting up $OPT_DIR for user: $TARGET_USER ==="

# --- boot drive (tooling) --------------------------------------------------
sudo mkdir -p "$OPT_DIR/loggers" "$OPT_DIR/envs" "$OPT_DIR/setup"
sudo chown -R "$TARGET_USER:$TARGET_USER" "$OPT_DIR"
sudo chmod -R u+rwX "$OPT_DIR"

# --- workload drive (if mounted) -------------------------------------------
if mountpoint -q "$DATA_DIR"; then
  sudo mkdir -p "$DATA_DIR/docker" "$DATA_DIR/containerd" "$SUITE_DIR"
  sudo chown -R "$TARGET_USER:$TARGET_USER" "$DATA_DIR"
  sudo chmod -R u+rwX "$DATA_DIR"
  echo "$DATA_DIR (data drive, ext4) — dirs created and owned"
else
  echo "$DATA_DIR (data drive, ext4) — not mounted, skipping"
fi

# --- verify ----------------------------------------------------------------
echo ""
echo "=== Verify ==="
ls -ld "$OPT_DIR" "$OPT_DIR/loggers" "$OPT_DIR/envs" "$OPT_DIR/setup"

if mountpoint -q "$DATA_DIR"; then
  ls -ld "$DATA_DIR" "$DATA_DIR/docker" "$DATA_DIR/containerd" "$SUITE_DIR"
fi

echo ""
echo "=== Done — $TARGET_USER owns $OPT_DIR ==="
