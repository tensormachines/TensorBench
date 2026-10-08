"""
bmc_logger.py
Polls NVML metrics for each GPU once per INTERVAL_S and appends rows to telemetry.csv.

4-17-26
Adding all continous signals

To be done: add full catpure of BMC as metadata 
"""

import csv
import os
import socket
# import re
import subprocess
import time
import traceback # for sudden failures
from datetime import datetime, timedelta, timezone

# import pynvml

def _load_env(path):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Config ───────────────────────────────────────────────────────────────────
INTERVAL_S  = 1.0
OUTPUT_FILE = "bmc_telemetry.csv"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(BASE_DIR, exist_ok=True)

BMC_HOST = os.environ["BMC_HOST"]
BMC_USER = os.environ["BMC_USER"]
BMC_PASS = os.environ["BMC_PASS"]
SDR_CACHE_FILE = os.path.join(BASE_DIR, "sdr_cache.bin")

TIMEZONE = 0  # UTC for stndarization
TZ = timezone(timedelta(hours=TIMEZONE))
# ─────────────────────────────────────────────────────────────────────────────

META_CMD = [
    "ipmitool", "-I", "lanplus",
    "-L", "USER",
    "-H", BMC_HOST,
    "-U", BMC_USER,
    "-P", BMC_PASS,
    "sensor",
]

CMD = [
    "ipmitool", "-I", "lanplus",
    "-N", "1",
    "-R", "1",
    "-L", "OPERATOR",
    "-H", BMC_HOST,
    "-U", BMC_USER,
    "-P", BMC_PASS,
    "-S", SDR_CACHE_FILE,
    "sdr", "elist", "-c",
]

SENSOR_NAMES = [
    "Temp_GPGPU0",
    "Temp_GPGPU1",
    "Temp_GPGPU2",
    "Temp_GPGPU3",
    "Temp_GPGPU4",
    "Temp_GPGPU5",
    "Temp_GPGPU6",
    "Temp_GPGPU7",
    "Temp_GPUB0",       
    "Temp_GPUB1",
    "Temp_CPU0",
    "Temp_CPU1",
    "Temp_VR_CPU0",     
    "Temp_VR_CPU1",
    "Temp_DIMM_AB",   
    "Temp_DIMM_CD",     
    "Temp_DIMM_EF",     
    "Temp_DIMM_GH",
    "Temp_VR_DIMM_AB",  
    "Temp_VR_DIMM_CD", 
    "Temp_VR_DIMM_EF", 
    "Temp_VR_DIMM_GH", 
    "Temp_Inlet_MB", 
    "Temp_Ambient_BP0",
    "Temp_Ambient_BP1",
    "Temp_Ambient_FP",
    "Temp_Ambient_PCI",
    "Temp_EXPB",
    "Temp_OCP_Mezz",
    "Temp_PCH",
    "Temp_PDB",
    "Temp_RaidCard",     
    "Temp_Outlet",
                    
    "Power_GPGPU0",
    "Power_GPGPU1",
    "Power_GPGPU2",
    "Power_GPGPU3",
    "Power_GPGPU4",
    "Power_GPGPU5",
    "Power_GPGPU6",
    "Power_GPGPU7",
    "Pwr_Node",
    "PSU1 Input",
    "PSU2 Input",
    "PSU3 Input",
    "PSU4 Input",
    "HSC0 Input",
    "HSC1 Input",
    "HSC2 Input",
       
    "Volt_P12V",
    "Volt_P5V",
    "Volt_P5V_AUX",
    "Volt_P3V3",
    "Volt_P3V3_AUX",
    "Volt_P3V_BAT",
    "Volt_P1V8_AUX",
    "Volt_P1V05",
    "Volt_VR_CPU0",
    "Volt_VR_CPU1",
    "Volt_VR_DIMM_AB",
    "Volt_VR_DIMM_CD",
    "Volt_VR_DIMM_EF",
    "Volt_VR_DIMM_GH",
    
    "Airflow"
]

FIELDNAMES = [
    "timestamp",
    "timestamp_epoch",
    "Temp_GPGPU0",
    "Temp_GPGPU1",
    "Temp_GPGPU2",
    "Temp_GPGPU3",
    "Temp_GPGPU4",
    "Temp_GPGPU5",
    "Temp_GPGPU6",
    "Temp_GPGPU7",
    "Temp_GPUB0",       
    "Temp_GPUB1",
    "Temp_CPU0",
    "Temp_CPU1",
    "Temp_VR_CPU0",     
    "Temp_VR_CPU1",
    "Temp_DIMM_AB",   
    "Temp_DIMM_CD",     
    "Temp_DIMM_EF",     
    "Temp_DIMM_GH",
    "Temp_VR_DIMM_AB",  
    "Temp_VR_DIMM_CD", 
    "Temp_VR_DIMM_EF", 
    "Temp_VR_DIMM_GH", 
    "Temp_Inlet_MB", 
    "Temp_Ambient_BP0",
    "Temp_Ambient_BP1",
    "Temp_Ambient_FP",
    "Temp_Ambient_PCI",
    "Temp_EXPB",
    "Temp_OCP_Mezz",
    "Temp_PCH",
    "Temp_PDB",
    "Temp_RaidCard",     
    "Temp_Outlet",
    
    "Power_GPGPU0",
    "Power_GPGPU1",
    "Power_GPGPU2",
    "Power_GPGPU3",
    "Power_GPGPU4",
    "Power_GPGPU5",
    "Power_GPGPU6",
    "Power_GPGPU7",
    "Pwr_Node",
    "PSU1_Input",
    "PSU2_Input",
    "PSU3_Input",
    "PSU4_Input",
    "HSC0_Input",
    "HSC1_Input",
    "HSC2_Input",

    "Volt_P12V",
    "Volt_P5V",
    "Volt_P5V_AUX",
    "Volt_P3V3",
    "Volt_P3V3_AUX",
    "Volt_P3V_BAT",
    "Volt_P1V8_AUX",
    "Volt_P1V05",
    "Volt_VR_CPU0",
    "Volt_VR_CPU1",
    "Volt_VR_DIMM_AB",
    "Volt_VR_DIMM_CD",
    "Volt_VR_DIMM_EF",
    "Volt_VR_DIMM_GH",
    
    "Airflow",
    "Fan_SYS0_1",
    "Fan_SYS0_2",
    "Fan_SYS1_1",
    "Fan_SYS1_2",
    "Fan_SYS2_1",
    "Fan_SYS2_2",
    "Fan_SYS3_1",
    "Fan_SYS3_2"
]

def build_sdr_cache():
    """Dump SDR repository to a local cache file (run once at startup)."""
    cache_cmd = [
        "ipmitool", "-I", "lanplus",
        "-L", "OPERATOR",
        "-H", BMC_HOST,
        "-U", BMC_USER,
        "-P", BMC_PASS,
        "sdr", "dump", SDR_CACHE_FILE,
    ]
    print(f"Building SDR cache at {SDR_CACHE_FILE} ...")
    subprocess.check_call(cache_cmd)
    print("SDR cache built.")


def collect_metadata(ts_str):
    meta_file = os.path.join(BASE_DIR, f"{ts_str}_bmc_metadata.txt")
    node_id = socket.gethostname()
    capture_start = datetime.now(TZ).isoformat()
    local_dt = datetime.now().astimezone()
    local_tz = f"{local_dt.tzname()} (UTC{local_dt.strftime('%z')})"
    out = subprocess.check_output(META_CMD).decode()
    with open(meta_file, "w") as f:
        f.write(f"node_id: {node_id}\n")
        f.write(f"capture_start: {capture_start}\n")
        f.write(f"local_timezone: {local_tz}\n\n")
        f.flush()
        f.write(out)
    print(f"BMC metadata saved to {meta_file}")

def parse_value(value):
    value = value.strip()
    if value in ("", "na", "ns"):
        return None
    try:
        num = float(value)
        return int(num) if num.is_integer() else num
    except ValueError:
        return value


def collect():
    out = subprocess.check_output(CMD).decode()
    ts = datetime.now(TZ)
    row = {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "timestamp_epoch": ts.timestamp(),
    }

    for name in FIELDNAMES[2:]:
        row[name] = None

    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue

        sensor_name = parts[0].strip()
        sensor_value = parse_value(parts[1])

        if sensor_name in SENSOR_NAMES:
            key = sensor_name.replace(" ", "_")
            row[key] = sensor_value
        elif sensor_name.startswith("Fan_SYS") and sensor_name in row:
            row[sensor_name] = sensor_value

    return row


def main():
    write_header = True
    run_dt = datetime.now(TZ)
    ts_str = run_dt.strftime("%y-%m-%d_%H-%M")
    output_csv = os.path.join(BASE_DIR, f"{ts_str}_{OUTPUT_FILE}")
    
    if not os.path.exists(SDR_CACHE_FILE):
        build_sdr_cache()
    collect_metadata(ts_str)

    try:
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()

            while True:
                t0 = time.monotonic()
                row = collect()
                writer.writerow(row)
                f.flush()
                print(f"[{row['timestamp']}] wrote BMC sample")
                time.sleep(max(0.0, INTERVAL_S - (time.monotonic() - t0)))

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()