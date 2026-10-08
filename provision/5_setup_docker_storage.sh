#!/bin/bash
# 5_setup_docker_storage.sh
# Format a drive you choose (at least 500 GB), mount it at /tensormachines,
# install Docker 29.1.3 and point the docker data-root and containerd root at
# the mounted drive.
#
# A re-run after a failure skips the drive setup once /tensormachines is mounted.
# To format a drive again, first stop docker & containerd, unmount
# /tensormachines and remove its fstab entry:
#  sudo systemctl stop docker containerd docker.socket
#  sudo umount /tensormachines
#  sudo sed -i '/tensormachines/d' /etc/fstab   # remove old entry
#
# WARNING: This FORMATS the chosen drive. All data on it will be lost.

set -e

# shellcheck source=common.sh
. "$(dirname "$(readlink -f "$0")")/common.sh"

DRIVE=""
MOUNT="/tensormachines"
MIN_SIZE_GB=500
DOCKER_VERSION="29.1.3-0ubuntu3"   # docker.io version prefix; the suffix varies by Ubuntu release

# ---------------------------------------------------------------------------
echo "=== [0/8] Pre-flight checks ==="

# A mounted drive skips formatting; Docker on it means nothing is left to do.
SETUP_DRIVE=1
if mountpoint -q "$MOUNT" && grep -q " $MOUNT " /etc/fstab; then
  echo "$MOUNT is mounted and present in /etc/fstab."
  if docker info -f '{{.DockerRootDir}}' 2>/dev/null | grep -q "$MOUNT"; then
    echo "Docker data-root already on $MOUNT. Nothing to do."
    exit 0
  fi
  echo "Docker is not using it yet. Skipping the drive setup."
  SETUP_DRIVE=0
fi

# Unmounted whole disks large enough to hold container storage.
candidates() {
  lsblk -dnb -o NAME,SIZE,TYPE,MOUNTPOINT | awk -v min=$((MIN_SIZE_GB*1024*1024*1024)) \
    '$3=="disk" && $2>=min && $4=="" {print "/dev/"$1}'
}

# Picks a disk, formats it and mounts it at $MOUNT through /etc/fstab.
setup_drive() {
  echo ""
  echo "Current block devices:"
  lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,LABEL
  echo ""

  mapfile -t FOUND < <(candidates)
  if [ "${#FOUND[@]}" -eq 0 ]; then
    echo "No unmounted disk of at least ${MIN_SIZE_GB} GB found."
    echo "Attach or free one, then re-run."
    exit 12
  fi

  echo "Disks that can hold container storage:"
  for i in "${!FOUND[@]}"; do
    printf '  %d) %s  (%s)\n' "$((i+1))" "${FOUND[$i]}" \
      "$(lsblk -dn -o SIZE "${FOUND[$i]}" | tr -d ' ')"
  done
  echo ""

  CHOICE="$(ask "Choose a disk by number (or anything else to abort): ")"
  case "$CHOICE" in
    ''|*[!0-9]*) echo "Aborted."; exit 12 ;;
  esac
  if [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt "${#FOUND[@]}" ]; then
    echo "No such option. Aborted."
    exit 12
  fi
  DRIVE="${FOUND[$((CHOICE-1))]}"

  echo ""
  echo "Target drive: $DRIVE"
  lsblk -o NAME,SIZE,MOUNTPOINT,TYPE,FSTYPE,LABEL "$DRIVE"
  echo ""
  echo "*** This ERASES $DRIVE. Everything on it will be lost. ***"
  # Destructive: always asks.
  CONFIRM="$(ask "Type 'yes' to format $DRIVE: ")"
  [ "$CONFIRM" = "yes" ] || { echo "Aborted."; exit 12; }

  # ---------------------------------------------------------------------------
  echo "=== [1/8] Formatting $DRIVE as ext4 ==="
  sudo mkfs.ext4 -F "$DRIVE"

  # ---------------------------------------------------------------------------
  echo "=== [2/8] Mounting at $MOUNT ==="
  sudo mkdir -p "$MOUNT"
  sudo mount "$DRIVE" "$MOUNT"

  # ---------------------------------------------------------------------------
  echo "=== [3/8] Persisting mount in /etc/fstab (by UUID) ==="
  UUID=$(sudo blkid -s UUID -o value "$DRIVE")
  echo "UUID = $UUID"
  if ! grep -q "$UUID" /etc/fstab; then
    echo "UUID=$UUID  $MOUNT  ext4  defaults  0  2" | sudo tee -a /etc/fstab
  else
    echo "fstab entry already present, skipping."
  fi
  # verify fstab is valid
  sudo mount -a
}

if [ "$SETUP_DRIVE" = "1" ]; then
  setup_drive
fi

# ---------------------------------------------------------------------------
echo "=== [4/8] Creating subdirectories ==="
sudo mkdir -p "$MOUNT/docker"
sudo mkdir -p "$MOUNT/containerd"

# ---------------------------------------------------------------------------
echo "=== [5/8] Installing Docker (docker.io) $DOCKER_VERSION ==="
if command -v dockerd >/dev/null 2>&1; then
  echo "Docker is already installed, keeping it: $(docker --version)"
else
  apt_get update

  # Stop docker before configuring (in case a prior install auto-started it)
  sudo systemctl stop docker 2>/dev/null || true

  # Full package version for this release, e.g. 29.1.3-0ubuntu3~24.04.2
  DOCKER_PACKAGE_VERSION="$(apt-cache madison docker.io |
    awk -v v="$DOCKER_VERSION" '$3 == v || index($3, v "~") == 1 {print $3; exit}')"
  if [ -n "$DOCKER_PACKAGE_VERSION" ]; then
    echo "Version $DOCKER_VERSION available as $DOCKER_PACKAGE_VERSION. Installing."
    apt_get install docker.io="$DOCKER_PACKAGE_VERSION"
  else
    echo ""
    echo "WARNING: version $DOCKER_VERSION is NOT available in the repo."
    echo "Available docker.io versions:"
    apt-cache madison docker.io
    echo ""
    DOCKER_CONFIRM="yes"
    if [ "${ASSUME_YES:-1}" != "1" ]; then
      read -p "Install the latest available version instead? Type 'yes' to continue, anything else aborts: " DOCKER_CONFIRM
    fi
    [ "$DOCKER_CONFIRM" = "yes" ] || { echo "Aborted — install the matching version manually."; exit 1; }
    apt_get install docker.io
  fi
fi

# ---------------------------------------------------------------------------
echo "=== [6/8] Configuring Docker daemon.json (data-root + nvidia runtime) ==="
sudo systemctl stop docker
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<JSON
{
    "data-root": "$MOUNT/docker",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
JSON

# ---------------------------------------------------------------------------
echo "=== [7/8] Configuring containerd root ==="
sudo systemctl stop containerd
sudo mkdir -p /etc/containerd
sudo containerd config default | sudo tee /etc/containerd/config.toml > /dev/null
# change root path
sudo sed -i "s|^root = .*|root = \"$MOUNT/containerd\"|" /etc/containerd/config.toml
# migrate existing containerd data if present
if [ -d /var/lib/containerd ] && [ "$(ls -A /var/lib/containerd 2>/dev/null)" ]; then
  echo "Migrating existing /var/lib/containerd ..."
  sudo rsync -aP /var/lib/containerd/ "$MOUNT/containerd/"
  sudo rm -rf /var/lib/containerd/*
fi
sudo systemctl start containerd

# ---------------------------------------------------------------------------
echo "=== [8/8] Starting Docker & verifying ==="
sudo systemctl start docker
sudo systemctl enable docker containerd
TARGET_USER="${SUDO_USER:-$(id -un)}"
RELOGIN=0
if ! id -nG "$TARGET_USER" | grep -qw docker; then
  sudo usermod -aG docker "$TARGET_USER"
  RELOGIN=1
fi

echo ""
echo "=== Verification ==="
docker --version
echo "--- Docker Root Dir (expect $MOUNT/docker) ---"
sudo docker info 2>/dev/null | grep "Docker Root Dir"
echo "--- containerd root (expect $MOUNT/containerd) ---"
grep "^root" /etc/containerd/config.toml
echo "--- Mount (expect $DRIVE on $MOUNT) ---"
df -h "$MOUNT"

echo ""
echo "==================================== Done ===================================="
echo "Next: install the NVIDIA Container Toolkit."

echo ""
echo "--- Docker access as $TARGET_USER ---"
if docker info >/dev/null 2>&1; then
  echo "Working in this session."
elif sg docker -c 'docker info' >/dev/null 2>&1; then
  echo "Working in a new session; this shell has a stale group list."
  echo "Nothing to do — later steps open their own sessions."
else
  echo ""
  echo "Docker is not usable as $TARGET_USER, in this session or a new one."
  exit 11
fi