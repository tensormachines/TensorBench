#!/usr/bin/env bash
# run.sh - provision a node and run the GPU stress benchmark on it.
#
#   ./run.sh                      provision, then benchmark
#   ./run.sh --only provision     provision only
#   ./run.sh --clean              discard checkpoints and provision from scratch
#
# Completed provisioning steps are recorded, so a re-run resumes where it
# stopped. Exits 10/11/12/13 when the operator has to act; each case prints
# what to do.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVISION_DIR="${REPO_DIR}/provision"
ENV_FILE="${REPO_DIR}/.env"

OPT_DIR="${OPT_DIR:-/opt/tensormachines}"
SUITE_DST="${SUITE_DST:-/tensormachines/gpu-stress-suite}"
STATE_DIR="${OPT_DIR}/state"
STATE_FILE="${STATE_DIR}/provision.state"

# shellcheck source=provision/common.sh
. "${PROVISION_DIR}/common.sh"

EXIT_REBOOT=10
EXIT_RELOGIN=11
EXIT_OPERATOR=12
EXIT_HARDWARE=13

# LabJack USB vendor id.
DAQ_USB_VENDOR="0cd5"

# Steps left out of an automated run.
SKIP_STEPS="install_cuda setup_aws_cli"

# Accepted --contact values.
EMAIL_RE='^[A-Za-z0-9._%+-]+@([A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$'

PHASES="all"
CLEAN=0
ASSUME_YES=1
WITH_DAQ=0
SUDO_KEEPALIVE_PID=""
BENCHMARK_ARGS=()
UPLOAD=0
CONTACT=""

# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
RULE="────────────────────────────────────────────────────────────────────"

say()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '  WARNING: %s\n' "$*" >&2; }

banner() {
    printf '\n%s\n  %s\n%s\n' "${RULE}" "$1" "${RULE}"
}

# Prints what the operator must do, then exits with the given code.
action_required() {
    local code="$1"; shift
    printf '\n%s\n  ACTION REQUIRED\n%s\n' "${RULE}" "${RULE}" >&2
    printf '  %s\n' "$@" >&2
    printf '\n  Re-run %s when done.\n%s\n\n' "${BASH_SOURCE[0]}" "${RULE}" >&2
    exit "${code}"
}

die() { printf '\nERROR: %s\n\n' "$*" >&2; exit 1; }

require_value() {
    [[ -n "$2" ]] || die "$1 needs a value  (try --help)"
}

# Asks the operator a yes/no question. Always waits for input.
confirm() {
    local reply=""
    # Reads the terminal directly so prompts survive a piped stdout.
    if { exec 3< /dev/tty; } 2>/dev/null; then
        read -r -p "  $1 [y/N] " reply <&3 || true
        exec 3<&-
    else
        read -r -p "  $1 [y/N] " reply || true
    fi
    [[ "${reply}" =~ ^[Yy]$ ]]
}

# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
usage() {
    sed -n "2,10p" "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --only PHASE             provision | benchmark
  --clean                  discard all checkpoints and provision from scratch
  --interactive            confirm each provisioning prompt
  --with-daq               run DAQ setup even if no LabJack is detected
  -h, --help               this message

Benchmark options (ignored with --only provision):
  --skip LIST              workload numbers to skip, e.g. 1,2,4
  --dry-run [LEVEL]        1 skips docker image builds (default), 2 builds them
  --max-vram-gb GB         VRAM cap per GPU, overriding the GPU profile
  --idle-duration SEC      idle baseline length
  --warmup-duration SEC    warmup length
  --cooldown-duration SEC  cooldown length
  --require-multi-gpu      fail the NCCL workload (6) on a single-GPU node instead of skipping it
  --upload                 upload results when the run completes
  --contact EMAIL          save EMAIL as contact.txt in the uploaded results (needs --upload)
  --gpu-profile NAME       force a profile from hardware/gpu/
  --platform-profile NAME  force a profile from hardware/platform/

Local edits to the repo reach the node only on a --clean run, except
filled-in hardware profiles, which are copied whenever they are newer.
Drive formatting and reboots always ask, whatever the flags.

Package installation is fully automated unless --interactive is given.

  ./run.sh --dry-run 1 --skip 1,2,4
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)
            require_value "$1" "${2:-}"
            case "$2" in
                provision|benchmark) PHASES="$2" ;;
                *) die "--only takes 'provision' or 'benchmark', not '$2'" ;;
            esac
            shift 2 ;;
        --interactive) ASSUME_YES=0; shift ;;
        --with-daq) WITH_DAQ=1; shift ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --skip|--max-vram-gb|--idle-duration|--warmup-duration|--cooldown-duration| \
        --gpu-profile|--platform-profile)
            require_value "$1" "${2:-}"
            BENCHMARK_ARGS+=("$1" "$2"); shift 2 ;;
        --contact)
            require_value "$1" "${2:-}"
            CONTACT="$2"
            BENCHMARK_ARGS+=("$1" "$2"); shift 2 ;;
        --upload) UPLOAD=1; BENCHMARK_ARGS+=("$1"); shift ;;
        --require-multi-gpu) BENCHMARK_ARGS+=("$1"); shift ;;
        --dry-run)
            if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
                BENCHMARK_ARGS+=("$1" "$2"); shift 2
            else
                BENCHMARK_ARGS+=("$1"); shift
            fi ;;
        *) die "Unknown argument: $1  (try --help)" ;;
    esac
done

if [[ -n "${CONTACT}" ]]; then
    [[ "${UPLOAD}" == "1" ]] || die "--contact works only together with --upload."
    [[ "${CONTACT}" =~ ${EMAIL_RE} ]] || die "--contact needs a valid email address, not '${CONTACT}'."
fi

if [[ "${CLEAN}" == "1" && "${PHASES}" == "benchmark" ]]; then
    die "--clean discards provisioning checkpoints, so it cannot be used with --only benchmark."
fi

if [[ "${PHASES}" == "provision" && ${#BENCHMARK_ARGS[@]} -gt 0 ]]; then
    warn "Benchmark options ignored with --only provision: ${BENCHMARK_ARGS[*]}"
    BENCHMARK_ARGS=()
fi

wants_phase() {
    [[ "${PHASES}" == "all" || "${PHASES}" == "$1" ]]
}

# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

# Caches sudo credentials and refreshes them until this script exits.
start_sudo_keepalive() {
    if ! sudo -v; then
        die "This needs sudo. Add $(id -un) to the sudoers group and try again."
    fi
    # Detached from this script's output so it never holds a pipe open.
    while sudo -n true 2>/dev/null; do
        sleep 50
        kill -0 "$$" 2>/dev/null || break
    done >/dev/null 2>&1 &
    SUDO_KEEPALIVE_PID=$!
    trap 'pkill -P "${SUDO_KEEPALIVE_PID}" 2>/dev/null; kill "${SUDO_KEEPALIVE_PID}" 2>/dev/null; true' EXIT
}

bootstrap_state_dir() {
    sudo mkdir -p "${STATE_DIR}"
    sudo chown "$(id -un):$(id -gn)" "${STATE_DIR}"
    touch "${STATE_FILE}"
}

load_env_file() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        info "No ${ENV_FILE} — credentials must come from the environment."
        return 0
    fi
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
    info "Loaded credentials from ${ENV_FILE}"
}

# Serials that firmware ships as filler rather than a real value.
serial_is_placeholder() {
    case "$1" in
        ""|"to-be-filled-by-o-e-m"|"system-serial-number"|"not-specified"| \
        "default-string"|"none"|"na"|"n-a"|"0"|"unknown"|"invalid") return 0 ;;
        *) [[ "$1" =~ ^0+$ ]] ;;
    esac
}

# Lowercases and strips anything a hostname may not contain.
sanitize_hostname() {
    local v
    v="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-')"
    v="$(printf '%s' "${v}" | sed 's/-\+/-/g; s/^-//; s/-$//')"
    printf '%.63s' "${v}"
}

# The node is named after its chassis serial number.
check_hostname() {
    local current serial want
    current="$(hostname -s)"

    if ! command -v dmidecode >/dev/null 2>&1; then
        warn "dmidecode not installed — cannot check the hostname. Current: ${current}"
        return 0
    fi

    serial="$(sudo -n dmidecode -s system-serial-number 2>/dev/null | head -1 || true)"
    want="$(sanitize_hostname "${serial}")"

    if serial_is_placeholder "${want}"; then
        warn "Chassis serial is '${serial:-empty}' — not usable as a hostname. Current: ${current}"
        return 0
    fi

    # Hostnames are case-insensitive; an existing one is not renamed over case.
    if [[ "${current,,}" == "${want}" ]]; then
        info "Hostname: ${current} (matches chassis serial)"
        return 0
    fi

    if ! wants_phase provision; then
        warn "Hostname is '${current}' but the chassis serial is '${want}'."
        return 0
    fi

    info "Setting hostname: ${current} -> ${want}"
    sudo hostnamectl set-hostname "${want}"
    info "Hostname set. Results are filed under this name."
}

daq_present() {
    if [[ "${WITH_DAQ}" == "1" ]]; then
        return 0
    fi
    command -v lsusb >/dev/null 2>&1 || return 1
    lsusb -d "${DAQ_USB_VENDOR}:" >/dev/null 2>&1
}

preflight() {
    banner "Preflight"
    [[ -d "${PROVISION_DIR}" ]] || die "No provision/ directory beside ${BASH_SOURCE[0]}"
    start_sudo_keepalive
    bootstrap_state_dir
    if [[ "${CLEAN}" == "1" ]]; then
        : > "${STATE_FILE}"
        info "Checkpoints discarded — every step will run."
    fi
    load_env_file
    check_hostname
    info "State file: ${STATE_FILE}"
}

# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
record_step() {
    local name="$1" hash="$2" tmp="${STATE_FILE}.tmp"
    grep -v -- "^${name}	" "${STATE_FILE}" > "${tmp}" 2>/dev/null || true
    mv "${tmp}" "${STATE_FILE}"
    printf '%s\tsha256:%s\t%s\n' "${name}" "${hash}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STATE_FILE}"
}

# --------------------------------------------------------------------------- #
# provisioning
# --------------------------------------------------------------------------- #
handle_reboot() {
    banner "Reboot required"
    info "$1 completed. The node must reboot before the driver is usable."
    if confirm "Reboot now?"; then
        info "Rebooting."
        sudo reboot
        exit 0
    fi
    action_required "${EXIT_REBOOT}" \
        "$1 completed but the node has not been rebooted." \
        "" \
        "  sudo reboot"
}

run_step() {
    local script="$1"
    local name hash
    name="$(basename "${script}" .sh)"
    hash="$(sha256sum "${script}" | cut -d' ' -f1)"

    # skip this step if it has a recorded checkpoint
    if grep -qs -- "^${name}	sha256:${hash}	" "${STATE_FILE}"; then
        info "[skip] ${name}"
        return 0
    fi

    local skip
    for skip in ${SKIP_STEPS}; do
        if [[ "${name}" == *"${skip}"* ]]; then
            info "[skip] ${name} (not part of an automated run)"
            return 0
        fi
    done

    if [[ "${name}" == *setup_daq* ]] && ! daq_present; then
        info "[skip] ${name} (no LabJack detected; --with-daq forces it)"
        return 0
    fi

    banner "${name}"
    local rc=0
    ASSUME_YES="${ASSUME_YES}" \
    SUITE_DIR="${SUITE_DST}" \
    REPO_DIR="${REPO_DIR}" \
    OPT_DIR="${OPT_DIR}" \
    LOGGER_DIR="${OPT_DIR}/loggers" \
    HW_FINGERPRINT="${OPT_DIR}/hw_fingerprint.json" \
        bash "${script}" || rc=$?

    case "${rc}" in
        0) record_step "${name}" "${hash}" ;;
        "${EXIT_REBOOT}")  record_step "${name}" "${hash}"; handle_reboot "${name}" ;;
        "${EXIT_RELOGIN}")
            record_step "${name}" "${hash}"
            action_required "${EXIT_RELOGIN}" \
                "${name} completed. Your group membership changed and needs a new session." \
                "" \
                "  exec newgrp docker      (or log out and back in)" ;;
        "${EXIT_OPERATOR}")
            action_required "${EXIT_OPERATOR}" \
                "${name} needs something only you can decide." \
                "Its output above says what." ;;
        "${EXIT_HARDWARE}")
            action_required "${EXIT_HARDWARE}" \
                "${name} could not resolve a hardware profile for this node." \
                "Its output above says whether none matched or several tied." ;;
        *) die "${name} failed (exit ${rc}). Fix the cause and re-run." ;;
    esac
}

# Provisioning scripts in step-number order.
provision_scripts() {
    find "${PROVISION_DIR}" -maxdepth 1 -name '[0-9]*.sh' -printf '%f\n' \
        | sort -t_ -k1,1n
}

provision() {
    banner "Provisioning"
    # Read the list up front so steps keep their own stdin.
    local -a scripts=()
    mapfile -t scripts < <(provision_scripts)
    local script
    for script in "${scripts[@]}"; do
        [[ -n "${script}" ]] || continue
        run_step "${PROVISION_DIR}/${script}"
    done
    say ""
    info "Provisioning complete."
}

# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #

benchmark() {
    banner "Running the benchmark"
    [[ -x "${SUITE_DST}/run_all.sh" ]] || die "No ${SUITE_DST}/run_all.sh — provision first."
    sync_filled_profiles "${REPO_DIR}/gpu-stress-suite/hardware" "${SUITE_DST}/hardware"

    local use_sg=0
    if ! docker info >/dev/null 2>&1; then
        if sg docker -c 'docker info' >/dev/null 2>&1; then
            info "This session's group list is stale; running under 'sg docker'."
            use_sg=1
        else
            action_required "${EXIT_RELOGIN}" \
                "Cannot reach the Docker daemon as $(id -un), so no workload can start." \
                "" \
                "If you were just added to the docker group, renew the session:" \
                "  exec newgrp docker      (or log out and back in)" \
                "" \
                "Otherwise check the daemon:  systemctl status docker"
        fi
    fi

    local -a flags=(--score "${BENCHMARK_ARGS[@]+"${BENCHMARK_ARGS[@]}"}")
    if [[ ${#flags[@]} -gt 0 ]]; then
        info "Options: ${flags[*]}"
    fi

    cd "${SUITE_DST}"
    if [[ "${use_sg}" == "1" ]]; then
        local cmd flag
        printf -v cmd 'LOGGER_DIR=%q HW_FINGERPRINT=%q ./run_all.sh' \
            "${OPT_DIR}/loggers" "${OPT_DIR}/hw_fingerprint.json"
        for flag in "${flags[@]+"${flags[@]}"}"; do
            printf -v cmd '%s %q' "${cmd}" "${flag}"
        done
        sg docker -c "${cmd}"
    else
        LOGGER_DIR="${OPT_DIR}/loggers" \
        HW_FINGERPRINT="${OPT_DIR}/hw_fingerprint.json" \
            ./run_all.sh "${flags[@]+"${flags[@]}"}"
    fi
}

# --------------------------------------------------------------------------- #
main() {
    preflight
    if wants_phase provision; then
        provision
        # Provisioning steps can add values to the env file.
        load_env_file
    fi
    if wants_phase benchmark; then benchmark; fi
    say ""
}

main
