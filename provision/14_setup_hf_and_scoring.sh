#!/usr/bin/env bash
# 14_setup_hf_and_scoring.sh — Hugging Face token and score inputs.
#
# Asks for HF_TOKEN unless it is already set, checks that it can download every
# gated model the workloads use, then asks for the ownership cost per GPU per
# hour (Oh) of the score; empty uses the GPU class default. Both values are
# saved to the repo's .env.
#
# Reads REPO_DIR (repo checkout). Exits 12 when the token is missing, invalid,
# or lacks access to a model.
set -euo pipefail

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

REPO_DIR="${REPO_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
SUITE_DIR="${SUITE_DIR:-${REPO_DIR}/gpu-stress-suite}"
ENV_FILE="${REPO_DIR}/.env"
HF_URL="https://huggingface.co"

# Gated models the workloads download.
MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "meta-llama/Llama-2-13b-hf"
  "meta-llama/Llama-3.2-3B-Instruct"
)

# Sets KEY=VALUE in the env file, replacing an existing KEY line.
save_env_value() {
  local key="$1" value="$2"
  if [ ! -f "$ENV_FILE" ]; then
    (umask 077; : > "$ENV_FILE")
  fi
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}

# HTTP status of a Hugging Face request made with the token. 000 if unreachable.
hf_status() {
  printf 'Authorization: Bearer %s\n' "$HF_TOKEN" \
    | curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H @- "$@" || true
}

echo "=== [1/3] Hugging Face token ==="
if [ -n "${HF_TOKEN:-}" ]; then
  echo "Using HF_TOKEN from the environment."
else
  HF_TOKEN="$(ask_secret "Hugging Face access token: ")"
  if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: no Hugging Face token given."
    echo "       Create one at ${HF_URL}/settings/tokens, then re-run."
    exit 12
  fi
fi

echo ""
echo "=== [2/3] Checking model access ==="
case "$(hf_status "${HF_URL}/api/whoami-v2")" in
  200) ;;
  401) echo "ERROR: Hugging Face rejected the token. Check it at ${HF_URL}/settings/tokens."
       exit 12 ;;
  000) echo "ERROR: could not reach ${HF_URL}. Check the network, then re-run."
       exit 1 ;;
  *)   echo "ERROR: unexpected answer from ${HF_URL} while checking the token."
       exit 1 ;;
esac

denied=()
for model in "${MODELS[@]}"; do
  status="$(hf_status -I "${HF_URL}/${model}/resolve/main/config.json")"
  case "$status" in
    2??|3??) echo "  OK         ${model}" ;;
    *)       echo "  NO ACCESS  ${model}  (HTTP ${status})"; denied+=("$model") ;;
  esac
done

if [ "${#denied[@]}" -gt 0 ]; then
  echo ""
  echo "ERROR: the token cannot download ${#denied[@]} model(s) the workloads need."
  echo "       For each one, open its page, accept the license and wait for approval:"
  for model in "${denied[@]}"; do
    echo "         ${HF_URL}/${model}"
  done
  echo "       A fine-grained token also needs read access to public gated repos."
  exit 12
fi
save_env_value HF_TOKEN "$HF_TOKEN"

# Name of this node's GPU, empty when nvidia-smi is unavailable.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"

# Prints "CLASS OH" from the pricing table for this GPU, nothing when the lookup fails.
gpu_default_price() {
  [ -n "$GPU_NAME" ] || return 0
  python3 - "${SUITE_DIR}/harness" "$GPU_NAME" <<'PY' 2>/dev/null || true
import sys
sys.path.insert(0, sys.argv[1])
from score import default_prices
gpu_class, oh, _ = default_prices(sys.argv[2])
if gpu_class:
    print(gpu_class, oh)
PY
}

# Asks for a non-negative number, three tries at most. Empty leaves the key
# empty so the default is used, unless there is no default.
ask_cost() {
  local key="$1" label="$2" default="$3" reply try
  for try in 1 2 3; do
    reply="$(ask "${label}${default:+ [default ${default}]}: ")"
    if [ -z "$reply" ] && [ -n "$default" ]; then
      save_env_value "$key" ""
      return 0
    fi
    if [[ "$reply" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      save_env_value "$key" "$reply"
      return 0
    fi
    if [ -z "$reply" ]; then
      echo "  There is no default for ${GPU_NAME:-this GPU}. Enter a number, e.g. 1.25" >&2
    else
      echo "  Enter a number, e.g. 1.25" >&2
    fi
  done
  echo "ERROR: no valid value for ${label,}."
  exit 12
}

echo ""
echo "=== [3/3] Score inputs ==="
read -r pricing_class default_oh <<< "$(gpu_default_price)" || true
if [ "$pricing_class" = "unknown" ]; then
  echo "No pricing data for ${GPU_NAME}; the default is the average across known GPU classes."
fi
echo "Enter the cost in USD. Leave empty to use the default value."
ask_cost SCORE_OH "Ownership cost per GPU per hour" "$default_oh"

echo ""
echo "Saved to ${ENV_FILE}."
echo "=== Done ==="
