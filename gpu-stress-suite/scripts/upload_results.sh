#!/usr/bin/env bash
#
# Usage:
#   ./upload_results.sh    suite_results/assessment_<folder to transfer>/
#   ./upload_results.sh    suite_results/assessment_<folder>/ --dryrun
#   ./upload_results.sh    suite_results/assessment_<folder>/ --platform dgx2
#   ./upload_results.sh    suite_results/assessment_<folder>/ you@example.com
#
# --dryrun builds the archive and prints where it would go, sends nothing.
# --platform sets the platform ID in the upload path; without it, the platform
# profile recorded in the folder's manifest.json is used.
# An email address after the folder is saved to contact.txt in it before upload.
# UPLOAD_URL overrides the upload endpoint.

set -euo pipefail

# Public write-only endpoint; no credentials needed.
UPLOAD_URL="${UPLOAD_URL:-https://d17xy21js5ip99.cloudfront.net}"
UPLOAD_URL="${UPLOAD_URL%/}"
MAX_BYTES=200000000
EMAIL_RE='^[A-Za-z0-9._%+-]+@([A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$'

usage() {
  sed -n '3,13p' "$0" | sed 's/^# \{0,1\}//'
}

RESULTS_DIR=""
PLATFORM=""
CONTACT=""
DRYRUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dryrun) DRYRUN=1; shift ;;
    --platform)
      [[ -n "${2:-}" ]] || { echo "ERROR: --platform needs a value"; exit 1; }
      PLATFORM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "ERROR: unknown argument: $1"; usage; exit 1 ;;
    *)
      if [[ -z "${RESULTS_DIR}" ]]; then
        RESULTS_DIR="$1"
      elif [[ -z "${CONTACT}" ]]; then
        CONTACT="$1"
      else
        echo "ERROR: unexpected argument: $1"; usage; exit 1
      fi
      shift ;;
  esac
done
[[ -n "${RESULTS_DIR}" ]] || { usage; exit 1; }
RESULTS_DIR="${RESULTS_DIR%/}"
[[ -d "${RESULTS_DIR}" ]] || { echo "ERROR: not a folder: ${RESULTS_DIR}"; exit 1; }
if [[ -n "${CONTACT}" && ! "${CONTACT}" =~ ${EMAIL_RE} ]]; then
  echo "ERROR: the contact must be a valid email address, not '${CONTACT}'."
  exit 1
fi

MANIFEST="${RESULTS_DIR}/manifest.json"

# Prints a top-level value from the manifest, or nothing if it is absent.
manifest_value() {
  [[ -f "${MANIFEST}" ]] || return 0
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")' \
    "${MANIFEST}" "$1"
}

PLATFORM="${PLATFORM:-$(manifest_value platform_profile)}"
if [[ -z "${PLATFORM}" ]]; then
  echo "ERROR: no platform ID: ${MANIFEST} has no platform_profile."
  echo "       Pass one with --platform, e.g. --platform dgx2."
  exit 1
fi
if [[ ! "${PLATFORM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: platform ID '${PLATFORM}' may only contain letters, digits, '.', '_' and '-'."
  exit 1
fi

NODE_ID="$(manifest_value node_id)"
NODE_ID="${NODE_ID:-$(hostname -s)}"
RUN_NAME="$(basename "${RESULTS_DIR}")"

# Characters outside the URL-safe set become '-'.
NODE_ID="${NODE_ID//[^A-Za-z0-9._-]/-}"
SAFE_RUN_NAME="${RUN_NAME//[^A-Za-z0-9._-]/-}"

# Random suffix so uploads never overwrite each other.
SUFFIX="$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')"
KEY="uploads/${PLATFORM}/${NODE_ID}/${SAFE_RUN_NAME}-${SUFFIX}.tar.gz"

if [[ -n "${CONTACT}" && "${DRYRUN}" != "1" ]]; then
  echo "${CONTACT}" > "${RESULTS_DIR}/contact.txt"
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

ARCHIVE="${WORK_DIR}/${SAFE_RUN_NAME}.tar.gz"
echo "[upload] Packing ${RESULTS_DIR} → ${SAFE_RUN_NAME}.tar.gz"
tar -czf "${ARCHIVE}" --exclude '*.tmp' -C "$(dirname "${RESULTS_DIR}")" "${RUN_NAME}"
FILE_COUNT=$(tar -tzf "${ARCHIVE}" | grep -cv '/$' || true)
SIZE=$(stat -c %s "${ARCHIVE}")

if (( SIZE > MAX_BYTES )); then
  echo "ERROR: archive is ${SIZE} bytes; the upload limit is ${MAX_BYTES}."
  exit 1
fi

if [[ "${DRYRUN}" == "1" ]]; then
  echo "(dryrun) upload: ${FILE_COUNT} file(s), ${SIZE} bytes → ${UPLOAD_URL}/${KEY}"
  if [[ -n "${CONTACT}" ]]; then
    echo "(dryrun) contact.txt with ${CONTACT} would be added"
  fi
  exit 0
fi

echo "[upload] Uploading ${SIZE} bytes → ${UPLOAD_URL}/${KEY}"
HTTP_CODE=$(curl -sS --retry 3 -o "${WORK_DIR}/response" -w '%{http_code}' \
  -X PUT -H 'Content-Type: application/gzip' -T "${ARCHIVE}" "${UPLOAD_URL}/${KEY}")
if [[ ! "${HTTP_CODE}" =~ ^2 ]]; then
  echo "ERROR: upload failed (HTTP ${HTTP_CODE}): $(head -c 500 "${WORK_DIR}/response")"
  if [[ "${HTTP_CODE}" == "403" ]]; then
    echo "       The endpoint allows 50 requests per 5 minutes per IP; retry later."
  fi
  exit 1
fi

echo ""
echo "============================================================"
echo "[upload] Done: ${FILE_COUNT} file(s) transferred → ${UPLOAD_URL}/${KEY}"
