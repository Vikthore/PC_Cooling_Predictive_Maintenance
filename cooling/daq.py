import sys, time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from cooling.sim import simulate_stream

def _collect_sim_block(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sim = cfg.get("sim", {})
    df = simulate_stream(duration_s=cfg.get("duration_s", 600),
                         fs=cfg.get("fs", 1),
                         ambient=sim.get("ambient", 28.0),
                         avg_load=sim.get("avg_load", 30),
                         load_style=sim.get("load_style", "idle spikes"),
                         fault=sim.get("fault", "none"),
                         seed=sim.get("seed", 42))
    return df, {"mode": "sim"}

def _collect_windows_live_block(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        import wmi  # type: ignore
    except Exception:
        return None, {"mode": "windows_live", "error": "wmi missing"}

    fs = int(cfg.get("fs", 1)); duration_s = int(cfg.get("duration_s", 600))
    n = fs * duration_s
    t0 = time.time()
    recs = []
    c = wmi.WMI(namespace="root\\OpenHardwareMonitor")
    # Helper: pull sensor value by Type/Name includes
    def get_value(sensors, typ, name_contains):
        for s in sensors:
            try:
                if getattr(s, "SensorType", "") == typ and name_contains.lower() in getattr(s, "Name", "").lower():
                    v = getattr(s, "Value", None)
                    if v is not None:
                        return float(v)
            except Exception:
                pass
        return None

    for i in range(n):
        sensors = c.Sensor()  # snapshot
        cpu_temp = get_value(sensors, "Temperature", "cpu package") or get_value(sensors, "Temperature", "cpu")
        gpu_temp = get_value(sensors, "Temperature", "gpu") or get_value(sensors, "Temperature", "nvidia") or get_value(sensors, "Temperature", "amd")
        fan_rpm = get_value(sensors, "Fan", "cpu") or get_value(sensors, "Fan", "gpu") or get_value(sensors, "Fan", "")
        cpu_load = get_value(sensors, "Load", "cpu total") or get_value(sensors, "Load", "cpu")
        # If something is None, carry forward last or skip
        if len(recs) > 0:
            last = recs[-1]
            cpu_temp = cpu_temp if cpu_temp is not None else last["cpu_temp"]
            gpu_temp = gpu_temp if gpu_temp is not None else last["gpu_temp"]
            fan_rpm = fan_rpm if fan_rpm is not None else last["fan_rpm"]
            cpu_load = cpu_load if cpu_load is not None else last["cpu_load"]
        else:
            if None in (cpu_temp, gpu_temp, fan_rpm, cpu_load):
                # cannot form a row; wait and retry
                time.sleep(1.0 / fs)
                continue
        recs.append({
            "t": i / fs,
            "ambient": np.nan,  # unknown; kept for schema consistency
            "cpu_load": cpu_load,
            "cpu_temp": cpu_temp,
            "gpu_temp": gpu_temp,
            "fan_rpm": fan_rpm
        })
        time.sleep(max(0.0, (i+1)/fs - (time.time()-t0)))
    if len(recs) == 0:
        return None, {"mode": "windows_live", "error": "no sensors"}
    df = pd.DataFrame(recs)
    # Fill ambient with rolling min of temps as a crude proxy (keeps features stable)
    amb_proxy = (df["cpu_temp"].rolling(window=fs*60, min_periods=1).min() - 10).clip(lower=15, upper=40)
    df["ambient"] = amb_proxy
    return df, {"mode": "windows_live"}

def collect_block(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mode = cfg.get("mode", "sim").lower()
    if mode == "windows_live":
        df, meta = _collect_windows_live_block(cfg)
        if df is not None:
            return df, meta
        # fallback to sim if live not available
        return _collect_sim_block(cfg)
    else:
        return _collect_sim_block(cfg)
