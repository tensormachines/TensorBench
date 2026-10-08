#!/bin/bash
# 4_install_cuda_12_6.sh
# Install the CUDA 12.6 toolkit. Not required by the benchmark — workloads get
# CUDA from the container image. Skipped by run.sh; run it directly if a host
# toolkit is wanted.
set -e

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

echo "=== [1/5] Adding CUDA keyring ==="
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

echo "=== [2/5] Updating apt ==="
apt_get update

echo "=== [3/5] Installing CUDA 12.6 toolkit ==="
apt_get install cuda-toolkit-12-6

echo "=== [4/5] Setting symlink /usr/local/cuda -> cuda-12.6 ==="
sudo ln -sfn /usr/local/cuda-12.6 /usr/local/cuda

echo "=== [5/5] Setting PATH in ~/.bashrc ==="
BASHRC="$HOME/.bashrc"
grep -q "cuda-12.6" "$BASHRC" || cat >> "$BASHRC" << 'ENVEOF'
# CUDA 12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
ENVEOF

echo ""
echo "=== Verification ==="
/usr/local/cuda/bin/nvcc --version
ls -la /usr/local/cuda
echo ""
echo "=== Done — open a new shell to pick up PATH ==="
