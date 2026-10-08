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

# ── Config ───────────────────────────────────────────────────────────────────
INTERVAL_S  = 1.0
POLL_TIMEOUT = 5.0            # kill a hung ipmitool call so the loop can't freeze
OUTPUT_FILE = "bmc_telemetry.csv"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(BASE_DIR, exist_ok=True)

SDR_CACHE_FILE = os.path.join(BASE_DIR, "sdr_cache.bin")

TIMEZONE = 0  # UTC for stndarization
TZ = timezone(timedelta(hours=TIMEZONE))
# ─────────────────────────────────────────────────────────────────────────────

META_CMD = [
    "sudo", "ipmitool", "-I", "open",
    "sensor",
]

CMD = [
    "sudo", "ipmitool", "-I", "open",
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
    "TEMP_CPU0",
    "TEMP_CPU1",
    "TEMP_VR_CPU0",     
    "TEMP_VR_CPU1",
    "TEMP_DIMM_AB",   
    "TEMP_DIMM_CD",     
    "TEMP_DIMM_EF",     
    "TEMP_DIMM_GH",
    "TEMP_VR_DIMM_AB",  
    "TEMP_VR_DIMM_CD", 
    "TEMP_VR_DIMM_EF", 
    "TEMP_VR_DIMM_GH", 
    "TEMP_Inlet_MB", 
    "TEMP_Ambient_BP0",
    "TEMP_Ambient_BP1",
    "TEMP_Ambient_FP",
    "TEMP_Ambient_PCI",
    "TEMP_EXPB",
    "TEMP_OCP_Mezz",
    "TEMP_PCH",
    "TEMP_PDB",
    "TEMP_RaidCard",     
    "TEMP_Outlet",
    "TEMP_PSU0", 
    "TEMP_PSU1", 
    "TEMP_PSU2",
    "TEMP_PSU3", 
    "TEMP_PSU4", 
    "TEMP_PSU5",
    "TEMP_PDB0", 
    "TEMP_PDB1",
                    
    "Power_GPGPU0",
    "Power_GPGPU1",
    "Power_GPGPU2",
    "Power_GPGPU3",
    "Power_GPGPU4",
    "Power_GPGPU5",
    "Power_GPGPU6",
    "Power_GPGPU7",
    "Pwr_Node",
    "POWER_PSU0", 
    "POWER_PSU1", 
    "POWER_PSU2",
    "POWER_PSU3", 
    "POWER_PSU4", 
    "POWER_PSU5",
    "HSC0 Input",
    "HSC1 Input",
    "HSC2 Input",
       
    "V_12V", 
    "V_5V", 
    "V_5V_STANDBY",
    "V_3.3V",
    "V_3.3V_STANDBY",
    "V_3V_BATTERY",
    "V_1.8V_STBY_PCH", 
    "V_1.05V_STBY_PCH",
    "V_PVCCIN_CPU0", 
    "V_PVCCIN_CPU1",
    "V_PVDDQ_MEM0_ABC", 
    "V_PVDDQ_MEM0_DEF",
    "V_PVDDQ_MEM1_ABC", 
    "V_PVDDQ_MEM1_DEF",
    "Airflow"
]

FIELDNAMES = ["timestamp", "timestamp_epoch"] + SENSOR_NAMES + [
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
        "sudo", "ipmitool", "-I", "open",
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

# def collect():
#     out = subprocess.check_output(CMD).decode()
#     ts = datetime.now(TZ)
#     row = {
#         "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
#         "timestamp_epoch": ts.timestamp(),
#     }

#     for name in FIELDNAMES[2:]:
#         row[name] = None

#     for line in out.splitlines():
#         parts = line.split(",")
#         if len(parts) < 4:
#             continue

#         sensor_name = parts[0].strip()
#         sensor_value = parse_value(parts[1])

#         if sensor_name in SENSOR_NAMES:
#             key = sensor_name.replace(" ", "_")
#             row[key] = sensor_value
#         elif sensor_name.startswith("Fan_SYS") and sensor_name in row:
#             row[sensor_name] = sensor_value

#     return row

def collect():
    """One poll. Captures EVERY sensor the BMC reports (name -> value)."""
    out = subprocess.check_output(CMD, timeout=POLL_TIMEOUT).decode()
    ts = datetime.now(TZ)
    row = {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "timestamp_epoch": ts.timestamp(),
    }
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        name = parts[0].strip().replace(" ", "_")
        if not name:
            continue
        row[name] = parse_value(parts[1])
    return row


# def main():
#     write_header = True
#     run_dt = datetime.now(TZ)
#     ts_str = run_dt.strftime("%y-%m-%d_%H-%M")
#     output_csv = os.path.join(BASE_DIR, f"{ts_str}_{OUTPUT_FILE}")
    
#     if not os.path.exists(SDR_CACHE_FILE):
#         build_sdr_cache()
#     collect_metadata(ts_str)

#     try:
#         with open(output_csv, "a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
#             if write_header:
#                 writer.writeheader()

#             while True:
#                 t0 = time.monotonic()
#                 row = collect()
#                 writer.writerow(row)
#                 f.flush()
#                 print(f"[{row['timestamp']}] wrote BMC sample")
#                 time.sleep(max(0.0, INTERVAL_S - (time.monotonic() - t0)))

#     except KeyboardInterrupt:
#         print("\nStopped.")


def main():
    run_dt = datetime.now(TZ)
    ts_str = run_dt.strftime("%y-%m-%d_%H-%M")
    output_csv = os.path.join(BASE_DIR, f"{ts_str}_{OUTPUT_FILE}")
 
    if not os.path.exists(SDR_CACHE_FILE):
        build_sdr_cache()
    collect_metadata(ts_str)
 
    # Discover schema from the first live poll — capture all sensors, no maintenance.
    first = collect()
    sensor_cols = [k for k in first if k not in ("timestamp", "timestamp_epoch")]
    fieldnames = ["timestamp", "timestamp_epoch"] + sorted(sensor_cols)
    print(f"Discovered {len(sensor_cols)} BMC sensors; logging to {output_csv}")
 
    try:
        with open(output_csv, "a", newline="") as f:
            # extrasaction='ignore' -> a sensor that appears later but wasn't in the
            # first poll is dropped instead of crashing the run (recoverable offline).
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(first)          # don't waste the discovery poll
            f.flush()
 
            while True:
                t0 = time.monotonic()
                try:
                    row = collect()
                    writer.writerow(row)
                    f.flush()
                    print(f"[{row['timestamp']}] wrote sample ({len(row) - 2} sensors)")
                except Exception:
                    # Never let one bad poll kill an unattended multi-hour burn.
                    print(f"[{datetime.now(TZ).isoformat()}] poll failed, continuing:")
                    traceback.print_exc()
                time.sleep(max(0.0, INTERVAL_S - (time.monotonic() - t0)))
 
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()