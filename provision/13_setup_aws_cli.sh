
#!/bin/bash
# 13_setup_aws_cli.sh — install + configure awscli on the node.
# Idempotent: skips install if already present. Prompts for keys, never hardcodes them.
set -e

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"
 
echo "=== [1/3] Installing awscli ==="
if command -v aws >/dev/null 2>&1; then
  echo "awscli already installed: $(aws --version)"
else
  apt_get update
  # A package can be named in the index yet have no installation candidate.
  CANDIDATE="$(apt-cache policy awscli 2>/dev/null | awk '/Candidate:/ {print $2}')"

  if [ -n "$CANDIDATE" ] && [ "$CANDIDATE" != "(none)" ] && apt_get install awscli; then
    echo "Installed from apt."
  elif command -v snap >/dev/null 2>&1 && sudo snap install aws-cli --classic; then
    echo "Installed the aws-cli snap."
  else
    echo ""
    echo "ERROR: could not install the AWS CLI."
    echo "       apt candidate: ${CANDIDATE:-none}"
    command -v snap >/dev/null 2>&1 && echo "       snap is present but the install failed." \
                                    || echo "       snap is not installed."
    echo "       Install it by hand, then re-run."
    exit 1
  fi
  command -v aws >/dev/null 2>&1 || { echo "ERROR: aws is still not on PATH."; exit 1; }
fi

echo ""
echo "=== [2/3] Verify ==="
aws --version
 
echo ""
echo "=== [3/3] Configure ==="
if aws sts get-caller-identity >/dev/null 2>&1; then
  echo "Credentials already working."
elif [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "Configuring from the environment."
  aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
  aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
  [ -n "${AWS_DEFAULT_REGION:-}" ] && aws configure set region "$AWS_DEFAULT_REGION"
else
  echo "No credentials in the environment. Enter them when prompted."
  aws configure
fi
 
echo ""
echo "=== Done ==="
echo "Test with: aws s3 ls s3://tensormachines-benchmark-weights"