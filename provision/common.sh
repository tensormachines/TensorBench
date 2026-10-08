# Shared helpers for the provisioning steps. Source from the step's own directory.

# apt-get that answers its own questions and restarts services itself.
# Set ASSUME_YES=0 to let package prompts reach the terminal instead.
apt_get() {
    if [ "${ASSUME_YES:-1}" = "1" ]; then
        sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
            apt-get -y \
            -o Dpkg::Options::=--force-confdef \
            -o Dpkg::Options::=--force-confold \
            "$@"
    else
        sudo apt-get "$@"
    fi
}

# Reads a reply from the terminal, falling back to stdin when there is no tty.
ask() {
    local reply=""
    if { exec 3< /dev/tty; } 2>/dev/null; then
        read -r -p "$1" reply <&3 || true
        exec 3<&-
    else
        read -r -p "$1" reply || true
    fi
    printf '%s' "$reply"
}

# Like ask, without echoing what is typed.
ask_secret() {
    local reply=""
    if { exec 3< /dev/tty; } 2>/dev/null; then
        read -r -s -p "$1" reply <&3 || true
        exec 3<&-
    else
        read -r -s -p "$1" reply || true
    fi
    echo >&2
    printf '%s' "$reply"
}

# Copies hardware profiles from $1 to $2 where newer, skipping any that still
# hold CHANGEME values.
sync_filled_profiles() {
    (cd "$1" && grep -rL --include='*.json' CHANGEME . || true) \
        | rsync -a --update --files-from=- "$1/" "$2/"
}
