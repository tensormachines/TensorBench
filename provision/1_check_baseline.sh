#!/bin/bash
# 1_check_baseline.sh — report whether the node matches the expected baseline.
# Expected: Ubuntu 22.04.5, kernel 5.15.x, driver 580.x
 
# Always continue; don't abort on any failed check
set +e

section() {
  echo ""
  echo "==================  $1  =================="
}

run_check() {
  # run_check "label" command...
  local label="$1"
  shift
  echo "-- $label"
  "$@" 2>/dev/null
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "WARN: $label failed (exit $rc)"
  fi
}

 
section "OS Version"
run_check "os-release" sh -c 'cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID|VERSION_CODENAME"'
 
echo ""
section "Kernel Version"
run_check "kernel" uname -r

echo ""
section "IPMI / BMC"
run_check "ipmitool present" sh -c 'command -v ipmitool || echo "WARN: ipmitool not installed"'

echo ""
section "NVIDIA Driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  run_check "nvidia driver" sh -c 'nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1'
elif [ -f /proc/driver/nvidia/version ]; then
  run_check "nvidia driver (proc fallback)" cat /proc/driver/nvidia/version
else
  echo "-- nvidia driver"
  echo "WARN: nvidia-smi not installed (install nvidia-utils-580 or matching driver utils)"
fi

echo ""
section "NVIDIA Package Holds"
echo "-- apt holds"
HELD=$(apt-mark showhold 2>/dev/null | grep -E 'nvidia|libnvidia' | paste -sd ' ' -)
if [ -n "$HELD" ]; then
  echo "WARN: NVIDIA packages HELD — driver auto-updates (incl. security) are BLOCKED."
  echo "      held: $HELD"
  echo "      Between campaigns: sudo apt-mark unhold \$(apt-mark showhold) && sudo apt upgrade -y && sudo reboot"
else
  echo "no nvidia holds — driver can auto-upgrade and desync from the loaded module mid-run"
fi
run_check "loaded kernel module" sh -c 'grep -o "NVRM version: .*Module *[0-9.]*" /proc/driver/nvidia/version 2>/dev/null || echo "no nvidia module loaded"'
 
echo ""
section "Python"
run_check "python3 version" python3 --version
run_check "python3 location" which python3
run_check "python3 venv" sh -c 'python3 -m venv /tmp/venvtest >/dev/null 2>&1 && rm -rf /tmp/venvtest && echo "python3-venv: OK" || echo "WARN: venv module NOT available (install python3-venv)"'
 
echo ""
section "Docker"
run_check "docker version" sh -c 'docker --version 2>/dev/null || echo "Docker not installed"'

echo ""
section "NVIDIA Container Toolkit"
if command -v nvidia-ctk >/dev/null 2>&1; then
  run_check "nvidia-ctk version" sh -c 'nvidia-ctk --version | head -1'
  run_check "docker nvidia runtime" sh -c 'docker info 2>/dev/null | grep -i "Runtimes:" | grep -q nvidia && echo "nvidia runtime registered" || echo "WARN: nvidia runtime NOT in docker (run install_nvidia_container_toolkit.sh)"'
else
  echo "-- nvidia-ctk"
  echo "WARN: nvidia-ctk not installed (run install_nvidia_container_toolkit.sh)"
fi
 
echo ""
section "Disk Layout"
# lsblk -o NAME,SIZE,MOUNTPOINT,FSTYPE,TYPE --list | grep -E "^sd[a-z]"
lsblk -o NAME,SIZE,MOUNTPOINT,FSTYPE,TYPE | grep -E "^NAME|^sd[a-z]|^nv[a-z]"
echo ""
run_check "root filesystem usage" df -h /
 
echo ""
section "NTP"
sudo timedatectl set-ntp true
run_check "NTP service" sh -c 'timedatectl status | grep -i "NTP service"'

echo ""
section "Done"
 
