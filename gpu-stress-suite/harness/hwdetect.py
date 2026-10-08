#!/usr/bin/env python3
"""
hwdetect.py - hardware detection and profile resolution for the GPU stress suite.

Four separate jobs, one per subcommand:

  detect    probe this node and emit a hardware fingerprint
  verify    check that the current GPUs still match a saved fingerprint
  profile   match a fingerprint against hardware/ and emit shell values
  template  write a profile template for each tree that has no match

Usage:
  hwdetect.py detect [--save PATH]
  hwdetect.py verify FINGERPRINT
  hwdetect.py profile FINGERPRINT [--gpu-profile NAME] [--platform-profile NAME]
  hwdetect.py template FINGERPRINT

Exit codes:
  0 - ok
  1 - error (probe failed, fingerprint unreadable, unknown profile name)
  2 - no GPU profile could be resolved
  3 - detected hardware does not match the fingerprint
  4 - no platform profile could be resolved
  6 - neither could be resolved (2 + 4)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

EXIT_ERROR = 1
EXIT_NO_GPU_PROFILE = 2
EXIT_HARDWARE_CHANGED = 3
EXIT_NO_PLATFORM_PROFILE = 4

# Constraint key -> detected fact key, where the two differ for readability.
CONSTRAINT_ALIASES = {"requires_sensors": "sensors"}

# Marks a template value that still has to be filled in.
PLACEHOLDER = "CHANGEME"

# DMI strings that firmware ships as filler rather than a real value.
DMI_FILLER = {"", "to be filled by o.e.m.", "default string", "system product name",
              "not specified", "none", "unknown"}

# Fields compared by `verify`.
IDENTITY_FIELDS = ("pci_device_id", "pci_sub_device_id", "count", "memory_mib", "compute_cap")

def _run(cmd, timeout=15):
    """Run a command, returning stripped stdout or None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None

def detect_gpu():
    fields = "pci.device_id,pci.sub_device_id,name,memory.total,compute_cap,uuid"
    raw = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not raw:
        sys.exit("ERROR: nvidia-smi produced no output. Is the NVIDIA driver installed?")

    rows = [[c.strip() for c in line.split(",")] for line in raw.splitlines() if line.strip()]
    device_ids = {r[0] for r in rows}
    mem_values = {int(r[3]) for r in rows}

    gpu = {
        "pci_device_id": rows[0][0],
        "pci_sub_device_id": rows[0][1],
        "name": rows[0][2],
        "count": len(rows),
        "memory_mib": int(rows[0][3]),
        "compute_cap": rows[0][4],
        "uuids": [r[5] for r in rows],
        # False when the GPUs are not all the same device id and memory size.
        "homogeneous": len(device_ids) == 1 and len(mem_values) == 1,
    }
    if not gpu["homogeneous"]:
        gpu["heterogeneous_device_ids"] = sorted(device_ids)
        gpu["heterogeneous_memory_mib"] = sorted(mem_values)
    return gpu

def detect_software():
    return {
        "driver_version": _run(["nvidia-smi", "--query-gpu=driver_version",
                                "--format=csv,noheader"]),
        "kernel": _run(["uname", "-r"]),
    }

def _dmi(field):
    """DMI via sysfs (no root); fall back to dmidecode."""
    try:
        value = (Path("/sys/class/dmi/id") / field).read_text().strip()
        if value:
            return value
    except OSError:
        pass
    keyword = {"product_name": "system-product-name",
               "sys_vendor": "system-manufacturer",
               "board_name": "baseboard-product-name"}[field]
    return _run(["sudo", "-n", "dmidecode", "-s", keyword])

def detect_platform_dmi():
    return {field: _dmi(field) for field in ("product_name", "sys_vendor", "board_name")}

def probe_bmc():
    """Find a working IPMI transport and list the sensors it reports."""
    
    attempts = [("open", ["sudo", "-n", "ipmitool", "-I", "open", "sdr", "elist", "-c"])]
    if os.environ.get("BMC_HOST"):
        attempts.append(("lanplus", [
            "ipmitool", "-I", "lanplus", "-H", os.environ["BMC_HOST"],
            "-U", os.environ.get("BMC_USER", ""), "-P", os.environ.get("BMC_PASS", ""),
            "sdr", "elist", "-c",
        ]))

    for transport, cmd in attempts:
        raw = _run(cmd, timeout=60)
        if raw:
            sensors = sorted({line.split(",")[0].strip()
                              for line in raw.splitlines() if "," in line})
            return {"bmc_transport": transport, "sensors": sensors}
    return {"bmc_transport": "none", "sensors": []}

def constraint_holds(key, spec, facts):
    value = facts.get(CONSTRAINT_ALIASES.get(key, key))
    if value is None:
        return False
    if isinstance(spec, dict) and ("min" in spec or "max" in spec):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if "min" in spec and value < spec["min"]:
            return False
        if "max" in spec and value > spec["max"]:
            return False
        return True
    if isinstance(spec, list):
        # Detected list (e.g. sensors): every listed item must be present.
        # Detected scalar (e.g. pci_device_id): the value must be one of them.
        if isinstance(value, list):
            return set(spec).issubset(set(value))
        return value in spec
    return value == spec

def load_profiles(directory):
    profiles = []
    if not directory.is_dir():
        return profiles
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"ERROR: {path} is not readable JSON: {exc}")
        if "match" not in data:
            sys.exit(f"ERROR: {path} has no 'match' block.")
        data["_id"] = path.stem     # filename is the profile ID
        data["_path"] = str(path)
        profiles.append(data)
    return profiles

def unfilled_keys(data, prefix=""):
    """Dotted paths of the values that still hold the placeholder."""
    keys = []
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            keys += unfilled_keys(value, f"{prefix}{key}.")
        elif PLACEHOLDER in json.dumps(value):
            keys.append(f"{prefix}{key}")
    return keys

def matching(profiles, facts):
    return [p for p in profiles
            if p["match"] and all(constraint_holds(k, v, facts) for k, v in p["match"].items())]

def resolve(directory, facts, kind, forced_id):
    """Full-match candidacy, most-constraints-wins.

    Returns the profile, or None after reporting why it could not resolve, so
    both trees can be reported in one pass. Unfilled templates never match.
    """
    profiles = load_profiles(directory)
    drafts = [p for p in profiles if unfilled_keys(p)]
    profiles = [p for p in profiles if not unfilled_keys(p)]

    if forced_id:
        for profile in profiles:
            if profile["_id"] == forced_id:
                return profile
        sys.exit(f"ERROR: no {kind} profile named '{forced_id}' "
                 f"({directory / (forced_id + '.json')} does not exist)")

    candidates = [(p, len(p["match"])) for p in matching(profiles, facts)]
    if not candidates:
        print(no_match_message(kind, directory, drafts), file=sys.stderr)
        return None

    # score is the number of constraints the profile has - the more the better
    best = max(score for _, score in candidates)
    winners = [p for p, score in candidates if score == best]
    if len(winners) > 1:
        names = "\n".join(f"  {p['_path']}" for p in winners)
        print(f"ERROR: ambiguous {kind} profile match - {len(winners)} profiles matched "
              f"{best} constraint(s):\n{names}\n"
              f"Add a distinguishing constraint to one, or remove the overlap.",
              file=sys.stderr)
        return None
    return winners[0]

def gpu_model(facts):
    """Model, e.g. 'Tesla V100-SXM2-32GB' -> 'v100'."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", (facts.get("name") or "").lower())
              if t and t not in ("nvidia", "tesla", "geforce", "quadro")]
    model = []
    for token in tokens:
        model.append(token)
        if any(c.isdigit() for c in token):
            break
    else:
        model = [(facts.get("pci_device_id") or "gpu").lower()]
    return "-".join(model)

def gpu_template_name(fingerprint):
    """Model and memory size, e.g. 'Tesla V100-SXM2-32GB' -> 'v100-32gb'."""
    facts = fingerprint["gpu"]
    gb = round((facts.get("memory_mib") or 0) / 1024)
    return f"{gpu_model(facts)}-{gb}gb"

def platform_label(facts):
    for field in ("product_name", "board_name"):
        value = (facts.get(field) or "").strip()
        if value.lower() not in DMI_FILLER:
            return value
    return None

def platform_template_name(fingerprint):
    """GPU model and chassis name, e.g. V100 in a 'DGX-1' -> 'v100-dgx1'."""
    label = platform_label(fingerprint["platform"]) or "platform"
    name = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    name = re.sub(r"([a-z])-(\d)", r"\1\2", name)
    return f"{gpu_model(fingerprint['gpu'])}-{name}"

def gpu_template(facts, profiles):
    memory = facts.get("memory_mib") or 0
    margin = int(memory * 0.05)
    # Workload and parameter names used by the existing profiles.
    workloads = {}
    for profile in profiles:
        for name, params in profile.get("workloads", {}).items():
            for key, value in params.items():
                if not key.startswith("_"):
                    blank = [PLACEHOLDER] if isinstance(value, list) else PLACEHOLDER
                    workloads.setdefault(name, {})[key] = blank
    return {
        "description": f"{facts.get('name')} boards ({memory} MiB)",
        "match": {
            "pci_device_id": [facts.get("pci_device_id")],
            "memory_mib": {"min": memory - margin, "max": memory + margin},
        },
        "vram": {"reserve_fraction": 0.80},
        "workloads": workloads or PLACEHOLDER,
    }

def platform_template(facts, profiles):
    transport = facts.get("bmc_transport")
    label = platform_label(facts)
    match = {
        "bmc_transport": transport,
        "requires_sensors": [f"{PLACEHOLDER} - sensor names unique to this chassis"],
    }
    if label:
        match["product_name"] = [facts.get("product_name")]
    return {
        "description": f"{label or 'Unknown chassis'} - BMC over IPMI '{transport}'",
        "match": match,
        "bmc": {
            "logger": f"{PLACEHOLDER} - BMC logger file name",
            "requires_env": transport == "lanplus",
            "requires_sudo": transport == "open",
        },
        "nvml": {"logger": "gpu_logger.py"},
    }

def no_match_message(kind, directory, drafts):
    lines = [f"ERROR: no {kind} profile in {directory} matches this system."]
    lines += [f"  Not used until its {PLACEHOLDER} values are filled in: {d['_path']}"
              for d in drafts]
    return "\n".join(lines)

def load_fingerprint(path):
    try:
        return json.loads(path.read_text())
    except OSError:
        sys.exit(f"ERROR: no hardware fingerprint at {path}.")
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: the hardware fingerprint at {path} is not valid JSON: {exc}")

def cmd_detect(args):
    platform = detect_platform_dmi()
    platform.update(probe_bmc())
    fingerprint = {"gpu": detect_gpu(), "platform": platform, "software": detect_software()}

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(fingerprint, indent=2) + "\n")
        print(f"Saved hardware fingerprint to {args.save}", file=sys.stderr)
    print(json.dumps(fingerprint, indent=2))

def cmd_verify(args):
    fingerprint = load_fingerprint(args.fingerprint)
    gpu = detect_gpu()
    changed = {f: (fingerprint["gpu"].get(f), gpu.get(f))
               for f in IDENTITY_FIELDS if fingerprint["gpu"].get(f) != gpu.get(f)}
    if changed:
        lines = [f"  {f:<20} fingerprint={was!r}  detected={now!r}"
                 for f, (was, now) in changed.items()]
        print(f"ERROR: detected GPU hardware does not match the fingerprint at "
              f"{args.fingerprint}.\n" + "\n".join(lines), file=sys.stderr)
        sys.exit(EXIT_HARDWARE_CHANGED)

def cmd_template(args):
    fingerprint = load_fingerprint(args.fingerprint)
    hardware_dir = args.hardware_dir.resolve()
    trees = (("gpu", "GPU", gpu_template_name, gpu_template),
             ("platform", "platform", platform_template_name, platform_template))

    for kind, label, name_of, template_of in trees:
        directory = hardware_dir / kind
        facts = fingerprint[kind]
        profiles = load_profiles(directory)
        finished = [p for p in profiles if not unfilled_keys(p)]
        if matching(finished, facts):
            continue

        drafts = [p for p in profiles if unfilled_keys(p)]
        if drafts:
            heading = f"Unfinished {label} profile template:"
        else:
            name = name_of(fingerprint)
            path = directory / f"{name}.json"
            suffix = 2
            while path.exists():
                path = directory / f"{name}-{suffix}.json"
                suffix += 1
            data = template_of(facts, finished)
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n")
            data["_path"] = str(path)
            drafts = [data]
            heading = f"Created {label} profile template:"

        print(heading)
        for draft in drafts:
            keys = unfilled_keys(draft)
            hardware_keys = [k for k in keys if not k.startswith("workloads")]
            print(f"  {draft['_path']}")
            if hardware_keys:
                print(f"    keys to fill in: {', '.join(hardware_keys)}")
            if len(hardware_keys) < len(keys):
                print(f"    workload parameters: fill in each {PLACEHOLDER}, "
                      f"using the existing profiles in {directory} as reference")

def shell_quote(value):
    return "'" + str(value).replace("'", "'\\''") + "'"


def workload_args(params):
    """Turn one workload's profile block into a shell-quoted argument list."""
    out = []
    for key, value in params.items():
        if key.startswith("_"):
            continue
        out.append("--" + key.replace("_", "-"))
        out.extend(shell_quote(v) for v in (value if isinstance(value, list) else [value]))
    return out


def cmd_profile(args):
    fingerprint = load_fingerprint(args.fingerprint)
    hardware_dir = args.hardware_dir.resolve()

    gpu_profile = resolve(hardware_dir / "gpu", fingerprint["gpu"], "gpu", args.gpu_profile)
    plat_profile = resolve(hardware_dir / "platform", fingerprint["platform"], "platform",
                           args.platform_profile)
    # Codes add, so 6 means both trees failed.
    missing = 0
    if gpu_profile is None:
        missing += EXIT_NO_GPU_PROFILE
    if plat_profile is None:
        missing += EXIT_NO_PLATFORM_PROFILE
    if missing:
        sys.exit(missing)

    # default to 80% of GPU memory if reserve_fraction is missing
    reserve = gpu_profile.get("vram", {}).get("reserve_fraction", 0.80)
    cap = fingerprint["gpu"]["memory_mib"] / 1024.0 * reserve
    pairs = [
        ("HW_GPU_PROFILE_ID", gpu_profile["_id"]),
        ("HW_PLATFORM_PROFILE_ID", plat_profile["_id"]),
        ("HW_MAX_VRAM_GB", f"{cap:.1f}"),
        ("HW_BMC_LOGGER", plat_profile.get("bmc", {}).get("logger", "")),
        ("HW_NVML_LOGGER", plat_profile.get("nvml", {}).get("logger", "gpu_logger.py")),
        ("HW_NODE_DESIGN_POWER_W", plat_profile.get("design_power_w", "")),
    ]
    for key, value in pairs:
        print(f"{key}={shell_quote(value)}")

    # One bash array per workload
    for name, params in sorted(gpu_profile.get("workloads", {}).items()):
        print(f"HW_WORKLOAD_ARGS_{name}=({' '.join(workload_args(params))})")

def main():
    default_hardware = Path(__file__).resolve().parent.parent / "hardware"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect", help="probe this node and emit a hardware fingerprint")
    p.add_argument("--save", type=Path, help="write the fingerprint to this path")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("verify", help="check the current GPUs against a saved fingerprint")
    p.add_argument("fingerprint", type=Path)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("profile", help="resolve profiles for a fingerprint, emit shell values")
    p.add_argument("fingerprint", type=Path)
    p.add_argument("--gpu-profile", metavar="NAME",
                   help="force a GPU profile by name, skipping matching")
    p.add_argument("--platform-profile", metavar="NAME",
                   help="force a platform profile by name, skipping matching")
    p.add_argument("--hardware-dir", type=Path, default=default_hardware)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("template", help="write a profile template for each tree with no match")
    p.add_argument("fingerprint", type=Path)
    p.add_argument("--hardware-dir", type=Path, default=default_hardware)
    p.set_defaults(func=cmd_template)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
