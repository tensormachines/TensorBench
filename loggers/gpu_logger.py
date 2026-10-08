# import pynvml
# import time
# from datetime import datetime

# pynvml.nvmlInit()
# device_count = pynvml.nvmlDeviceGetCount()

# k = 0
# while (k < 100):
#     for i in range(2,3):
#         ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms precision
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         # name   = pynvml.nvmlDeviceGetName(handle)
#         temp   = pynvml.nvmlDeviceGetTemperatureV(handle, pynvml.NVML_TEMPERATURE_GPU)
#         util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
#         mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         power  = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
#         # all    = pynvml.nvmlDeviceGetAllRunningProcesses(handle)
        

#         print(f"[{ts}] GPU {i} | Temp: {temp}°C | GPU util: {util.gpu}% |"
#             f"Mem: {mem.used//1024**2}/{mem.total//1024**2} MiB | Power: {power:.3f}W")
#         # print(f"Running processes: {all}")
        
#         k += 1
#         time.sleep(0.5)
        
    

# pynvml.nvmlShutdown()






"""
gpu_logger.py
Polls NVML metrics for each GPU once per INTERVAL_S and appends rows to telemetry.csv.
"""

import csv
import os
import socket
import subprocess
import time
import traceback # for sudden failures
from datetime import datetime, timedelta, timezone

import pynvml

# ── Config ───────────────────────────────────────────────────────────────────
INTERVAL_S  = 0.5   # seconds
INTERVAL_slow = 5  # seconds 
ENABLE_SLOW = True
OUTPUT_FILE = "nvml_telemetry.csv"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(BASE_DIR, exist_ok=True)
GPU_INDICES = None   # None = all GPUs; or e.g. [0, 1, 2, 3]

TIMEZONE = 0  # UTC for standardization
TZ = timezone(timedelta(hours=TIMEZONE))
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "timestamp",
    # "node_id", moved to metadata
    "gpu_id",
    
    # Temp & Power 
    "temp_gpu",
    # "temp_mem", # not supported on v100
    "power_gpu_reported",
    "power_limit",
    "energy_consumption", # tatal
    
    # Frequencies
    "clock_core",
    "clock_mem",
    "clock_graphics",
    "clock_video",
    "pstate",
    "throttle_flag",
    
    # Memory
    "mem_free_bytes",
    "mem_used_bytes",
    "ecc_correctable",
    "ecc_uncorrectable",
    
    # Utilizations 
    "utilization_gpu",
    "utilization_mem",
    # "pcie_tx_kb",
    # "pcie_rx_kb",
    # "nvlink_tx",
    # "nvlink_rx",
]


def _safe(fn, *args, default=None):
    """Call fn(*args), return default on any NVMLError."""
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return default

# Only power, power limit and energy have NVML_FI_DEV_* equivalents. Clocks,
# pstate, temperature, memory info and utilization have no field IDs at all and
# stay as individual calls below.
_FAST_FIELDS = [
    pynvml.NVML_FI_DEV_POWER_INSTANT,
    pynvml.NVML_FI_DEV_POWER_AVERAGE,
    pynvml.NVML_FI_DEV_POWER_CURRENT_LIMIT,
    pynvml.NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
]
_SLOW_FIELDS = [
    pynvml.NVML_FI_DEV_ECC_SBE_AGG_TOTAL,
    pynvml.NVML_FI_DEV_ECC_DBE_AGG_TOTAL,
]

_VALUE_READERS = {
    pynvml.NVML_VALUE_TYPE_DOUBLE:             lambda v: v.dVal,
    pynvml.NVML_VALUE_TYPE_UNSIGNED_INT:       lambda v: v.uiVal,
    pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG:      lambda v: v.ulVal,
    pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: lambda v: v.ullVal,
    pynvml.NVML_VALUE_TYPE_SIGNED_LONG_LONG:   lambda v: v.sllVal,
    pynvml.NVML_VALUE_TYPE_SIGNED_INT:         lambda v: v.siVal,
    pynvml.NVML_VALUE_TYPE_UNSIGNED_SHORT:     lambda v: v.usVal,
}


def _field_values(handle, field_ids):
    """fieldId -> value. None for any field the driver did not return."""
    out = dict.fromkeys(field_ids)
    values = _safe(pynvml.nvmlDeviceGetFieldValues, handle, field_ids)
    if values is None:
        return out
    # The call succeeds as a whole even when individual fields fail, so each
    # entry carries its own return code.
    for fv in values:
        if fv.nvmlReturn == pynvml.NVML_SUCCESS:
            reader = _VALUE_READERS.get(fv.valueType)
            if reader is not None:
                out[fv.fieldId] = reader(fv.value)
    return out


def collect_fast_batched(handle, gpu_id):
    ts = datetime.now(TZ)

    fields = _field_values(handle, _FAST_FIELDS)

    # Support for these fields varies across GPU architectures, hence the need for fallbacks
    raw_power = fields[pynvml.NVML_FI_DEV_POWER_INSTANT]
    if raw_power is None:
        raw_power = fields[pynvml.NVML_FI_DEV_POWER_AVERAGE]
    if raw_power is None:
        raw_power = _safe(pynvml.nvmlDeviceGetPowerUsage, handle)

    raw_plimit = fields[pynvml.NVML_FI_DEV_POWER_CURRENT_LIMIT]
    if raw_plimit is None:
        raw_plimit = _safe(pynvml.nvmlDeviceGetPowerManagementLimit, handle)

    energy_j = fields[pynvml.NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION]
    if energy_j is None:
        energy_j = _safe(pynvml.nvmlDeviceGetTotalEnergyConsumption, handle)

    try:
            clk_video = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_VIDEO)
    except pynvml.NVMLError:
            clk_video = None

    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    util       = _safe(pynvml.nvmlDeviceGetUtilizationRates, handle)

    return {
        "timestamp":          ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "gpu_id":             gpu_id,

        # Temp & Power
        "temp_gpu":           _safe(pynvml.nvmlDeviceGetTemperatureV, handle,
                                    pynvml.NVML_TEMPERATURE_GPU),
        "power_gpu_reported": raw_power  / 1000 if raw_power  is not None else None,
        "power_limit":        raw_plimit / 1000 if raw_plimit is not None else None,
        "energy_consumption": energy_j / 1000 if energy_j is not None else None,

        # Frequencies
        "clock_core":         _safe(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM),
        "clock_mem":          _safe(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM),
        "clock_graphics":     _safe(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_GRAPHICS),
        "clock_video": clk_video,
        "pstate": pynvml.nvmlDeviceGetPerformanceState(handle),

        # Memory
        "mem_free_bytes": mem_info.free,
        "mem_used_bytes": mem_info.used,

        # Utilizations
        "utilization_gpu":    util.gpu    if util else None,
        "utilization_mem":    util.memory if util else None,
    }


def collect_slow_batched(handle, gpu_id):
    fields = _field_values(handle, _SLOW_FIELDS)

    ecc_correctable = fields[pynvml.NVML_FI_DEV_ECC_SBE_AGG_TOTAL]
    if ecc_correctable is None:
        ecc_correctable = _safe(pynvml.nvmlDeviceGetTotalEccErrors, handle,
                                pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                pynvml.NVML_AGGREGATE_ECC)

    ecc_uncorrectable = fields[pynvml.NVML_FI_DEV_ECC_DBE_AGG_TOTAL]
    if ecc_uncorrectable is None:
        ecc_uncorrectable = _safe(pynvml.nvmlDeviceGetTotalEccErrors, handle,
                                  pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                  pynvml.NVML_AGGREGATE_ECC)

    return {
        "throttle_flag":      _safe(pynvml.nvmlDeviceGetCurrentClocksEventReasons, handle),
        "ecc_correctable":    ecc_correctable,
        "ecc_uncorrectable":  ecc_uncorrectable,
    }


# def collect(handle, gpu_id, node_id):
def collect_fast(handle, gpu_id):
    ts = datetime.now(TZ)

    # try:
    #         # PYNVML uses 0 for GPU, 1 might map to Memory in newer bindings or fail gracefully
    #         temp_mem = pynvml.nvmlDeviceGetTemperatureV(handle, getattr(pynvml, 'NVML_TEMPERATURE_MEMORY', 1))
    # except pynvml.NVMLError:
    #         temp_mem = None
    raw_power  = _safe(pynvml.nvmlDeviceGetPowerUsage, handle)
    raw_plimit = _safe(pynvml.nvmlDeviceGetPowerManagementLimit, handle)
    energy_j = _safe(pynvml.nvmlDeviceGetTotalEnergyConsumption, handle)
    
    try:
            clk_video = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_VIDEO)
    except pynvml.NVMLError:
            clk_video = None
    
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    util       = _safe(pynvml.nvmlDeviceGetUtilizationRates, handle)

    return {
        "timestamp":          ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        # "node_id":            node_id,
        "gpu_id":             gpu_id,
        
        # Temp & Power 
        "temp_gpu":           _safe(pynvml.nvmlDeviceGetTemperatureV, handle,
                                    pynvml.NVML_TEMPERATURE_GPU),
        # "temp_mem":           _safe(pynvml.nvmlDeviceGetTemperatureV, handle,
        #                             getattr(pynvml, 'NVML_TEMPERATURE_MEMORY', 1)), #not supported on v100
        "power_gpu_reported": raw_power  / 1000 if raw_power  is not None else None,
        "power_limit":        raw_plimit / 1000 if raw_plimit is not None else None,
        "energy_consumption": energy_j / 1000 if energy_j is not None else None,
        
        # Frequencies
        "clock_core":         _safe(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM),
        "clock_mem":          _safe(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM),
        "clock_graphics":     _safe(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_GRAPHICS),
        "clock_video": clk_video,
        "pstate": pynvml.nvmlDeviceGetPerformanceState(handle), # Returns integer (e.g., 0 for P0)
        # "throttle_flag":      _safe(pynvml.nvmlDeviceGetCurrentClocksEventReasons, handle), # moved to slow collect
        
        # Memory
        # "ecc_correctable":    _safe(pynvml.nvmlDeviceGetTotalEccErrors, handle,
        #                             pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
        #                             pynvml.NVML_AGGREGATE_ECC),                   # moved to slow collect
        # "ecc_uncorrectable":  _safe(pynvml.nvmlDeviceGetTotalEccErrors, handle,
        #                             pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
        #                             pynvml.NVML_AGGREGATE_ECC),                   # moved to slow collect
        "mem_free_bytes": mem_info.free,
        "mem_used_bytes": mem_info.used,
        
        # Utilizations 
        "utilization_gpu":    util.gpu    if util else None,
        "utilization_mem":    util.memory if util else None,
    }
    
def collect_slow(handle, gpu_id):
    # nvlink_tx_f = _safe(pynvml.nvmlDeviceGetFieldValues, handle,
    #                     [pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX])
    # nvlink_rx_f = _safe(pynvml.nvmlDeviceGetFieldValues, handle,
    #                     [pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX])
    return {
        "throttle_flag":      _safe(pynvml.nvmlDeviceGetCurrentClocksEventReasons, handle),
        "ecc_correctable":    _safe(pynvml.nvmlDeviceGetTotalEccErrors, handle,
                                    pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    pynvml.NVML_AGGREGATE_ECC),
        "ecc_uncorrectable":  _safe(pynvml.nvmlDeviceGetTotalEccErrors, handle,
                                    pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    pynvml.NVML_AGGREGATE_ECC),
        # "pcie_tx_kb":         _safe(pynvml.nvmlDeviceGetPcieThroughput, handle,
        #                             pynvml.NVML_PCIE_UTIL_TX_BYTES),
        # "pcie_rx_kb":         _safe(pynvml.nvmlDeviceGetPcieThroughput, handle,
        #                             pynvml.NVML_PCIE_UTIL_RX_BYTES),
        # "nvlink_tx":          nvlink_tx_f[0].value.uiVal
        #                       if (nvlink_tx_f and nvlink_tx_f[0].nvmlReturn == 0) else None,
        # "nvlink_rx":          nvlink_rx_f[0].value.uiVal
        #                       if (nvlink_rx_f and nvlink_rx_f[0].nvmlReturn == 0) else None,
        # NVLink: Not Supported on V100 via NVML — logged as None
        # May work with pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, link_id, counter)
    }


def main():
    pynvml.nvmlInit()
    node_id      = socket.gethostname()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_indices  = GPU_INDICES if GPU_INDICES is not None else list(range(device_count))
    
    run_dt = datetime.now(TZ)
    ts_str = run_dt.strftime("%y-%m-%d_%H-%M") 
    
    metadata_file = os.path.join(BASE_DIR, f"{ts_str}_nvml_metadata.txt")
    telemetry_file = os.path.join(BASE_DIR, f"{ts_str}_nvml_telemetry.csv")
    
    # subprocess.run(f"nvidia-smi -q > {metadata_file}", shell=True)
    local_dt = datetime.now().astimezone()
    local_tz = f"{local_dt.tzname()} (UTC{local_dt.strftime('%z')})"

    with open(metadata_file, "w", encoding="utf-8") as mf:
        mf.write(f"node_id: {node_id}\n")
        mf.write(f"capture_start: {run_dt.isoformat()}\n")
        mf.write(f"local_timezone: {local_tz}\n\n")
        mf.flush()
        subprocess.run(
        ["nvidia-smi", "-q"],
        stdout=mf,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
        )    

    write_header = not os.path.exists(telemetry_file)
    print(f"Metadata: {metadata_file}")
    print(f"Logging GPUs {gpu_indices} → {telemetry_file}  (Ctrl+C to stop)")


    try:
        with open(telemetry_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
                
            handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices} # handel dict for handle reuse 
            slow_counter = 0
            
            while True:
                t0 = time.monotonic()
                slow_counter += 1
                
                for gpu_id in gpu_indices:
                    try:
                        handle = handles[gpu_id]

                        # FAST
                        # row    = collect(handle, gpu_id, node_id)
                        # row    = collect_fast(handle, gpu_id)
                        row = collect_fast_batched(handle, gpu_id)


                        # SLOW (NOTE: comment out if consistent time is more important)
                        if ENABLE_SLOW and (slow_counter % int(INTERVAL_slow/INTERVAL_S) == 0):
                            # slow_data = collect_slow(handle, gpu_id)
                            slow_data = collect_slow_batched(handle, gpu_id)
                            row.update(slow_data)
                        
                        writer.writerow(row)
                        # print(
                        #     f"[{row['timestamp']}] GPU {gpu_id} | "
                        #     f"Temp: {row['temp_gpu']}°C | "
                        #     f"Util: {row['utilization_gpu']}% | "
                        #     f"Power: {row['power_gpu_reported']}W | "
                        #     f"Clocks SM/MEM: {row['clock_core']}/{row['clock_mem']} MHz | "
                        #     f"PCIe tx/rx: {row['pcie_tx_kb']}/{row['pcie_rx_kb']} KB/s | "
                        #     f"ECC: {row['ecc_correctable']}/{row['ecc_uncorrectable']} | "
                        #     f"Throttle: {row['throttle_flag']}"
                        # )
                    except Exception as e:
                        print(f"[ERROR] gpu={gpu_id} {type(e).__name__}: {e}", flush=True)
                        print(traceback.format_exc(), flush=True)
                f.flush()
                print(f"[HEARTBEAT] {datetime.now(TZ).isoformat()} wrote cycle", flush=True)
                time.sleep(max(0.0, INTERVAL_S - (time.monotonic() - t0)))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
