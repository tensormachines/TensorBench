#!/bin/bash
# 6_install_nvidia_container_toolkit.sh
# Install NVIDIA Container Toolkit 1.19.0 and configure the docker runtime.
# Run AFTER docker storage setup.

set -e

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

NCT_VERSION="1.19.0-1"

# ---------------------------------------------------------------------------
echo "=== [1/5] Adding NVIDIA Container Toolkit GPG key ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# ---------------------------------------------------------------------------
echo "=== [2/5] Adding repo ==="
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

apt_get update

# ---------------------------------------------------------------------------
echo "=== [3/5] Installing toolkit (pinned $NCT_VERSION) ==="
if apt-cache madison nvidia-container-toolkit | grep -q "$NCT_VERSION"; then
  echo "Exact version $NCT_VERSION available. Installing."
  # The pinned version may be older than what is installed.
  apt_get install --allow-downgrades \
    nvidia-container-toolkit="$NCT_VERSION" \
    nvidia-container-toolkit-base="$NCT_VERSION" \
    libnvidia-container-tools="$NCT_VERSION" \
    libnvidia-container1="$NCT_VERSION"
else
  echo ""
  echo "WARNING: exact version $NCT_VERSION is NOT available."
  echo "Available versions:"
  apt-cache madison nvidia-container-toolkit
  echo ""
  CONFIRM="yes"
  if [ "${ASSUME_YES:-1}" != "1" ]; then
    read -p "Install latest available instead? Type 'yes' to continue, anything else aborts: " CONFIRM
  fi
  [ "$CONFIRM" = "yes" ] || { echo "Aborted."; exit 1; }
  apt_get install nvidia-container-toolkit
fi

# ---------------------------------------------------------------------------
echo "=== [4/5] Configuring docker runtime ==="
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# ---------------------------------------------------------------------------
echo "=== [5/5] Verifying ==="
nvidia-ctk --version
echo "--- Pulling small CUDA base image for GPU test ---"
sudo docker pull nvidia/cuda:12.6.0-base-ubuntu22.04
echo "--- GPU passthrough test (expect every GPU) ---"
sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "==================================== Done ===================================="