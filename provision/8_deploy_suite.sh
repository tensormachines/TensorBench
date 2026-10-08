#!/usr/bin/env bash
# 8_deploy_suite.sh — copy the suite and loggers from the repo onto the node.
#
# Reads REPO_DIR (repo checkout) and SUITE_DIR (destination for the suite).
set -euo pipefail

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

REPO_DIR="${REPO_DIR:-}"
SUITE_DIR="${SUITE_DIR:-/tensormachines/gpu-stress-suite}"
OPT_DIR="${OPT_DIR:-/opt/tensormachines}"

[ -n "$REPO_DIR" ] || { echo "ERROR: REPO_DIR is not set."; exit 1; }
[ -d "$REPO_DIR/gpu-stress-suite" ] || { echo "ERROR: no $REPO_DIR/gpu-stress-suite"; exit 1; }

mkdir -p "$SUITE_DIR" "$OPT_DIR/loggers" "$OPT_DIR/setup"

echo "=== [1/4] gpu-stress-suite -> $SUITE_DIR ==="
rsync -a --delete \
  --exclude 'suite_results/' --exclude 'cache/' --exclude '__pycache__/' \
  --exclude '/hardware/' \
  "$REPO_DIR/gpu-stress-suite/" "$SUITE_DIR/"

# Hardware profiles are never deleted, a profile edited on the node is kept
# unless the repo copy is newer, and unfinished templates are not copied.
sync_filled_profiles "$REPO_DIR/gpu-stress-suite/hardware" "$SUITE_DIR/hardware"

echo "=== [2/4] loggers -> $OPT_DIR/loggers ==="
rsync -a "$REPO_DIR/loggers/"*.py "$OPT_DIR/loggers/"
sudo install -m 755 "$REPO_DIR/loggers/checklogger" /usr/local/bin/checklogger
echo "installed /usr/local/bin/checklogger"

echo "=== [3/4] logger setup files -> $OPT_DIR/setup ==="
rsync -a "$REPO_DIR/loggers/setup/" "$OPT_DIR/setup/"

echo "=== [4/4] BMC credentials ==="
if [ -n "${BMC_HOST:-}" ]; then
  umask 077
  printf 'BMC_HOST=%s\nBMC_USER=%s\nBMC_PASS=%s\n' \
    "${BMC_HOST:-}" "${BMC_USER:-}" "${BMC_PASS:-}" > "$OPT_DIR/loggers/.env"
  echo "written to $OPT_DIR/loggers/.env"
else
  echo "BMC_HOST unset — skipped (local IPMI needs no credentials)"
fi

echo "=== Done ==="
