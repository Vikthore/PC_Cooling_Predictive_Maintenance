import sys, time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import pythoncom  # <-- add this


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
    """
    Real-time collection from OpenHardwareMonitor WMI for the full duration.
    - Samples at fs for duration_s seconds.
    - Does NOT fabricate sensors: columns are included only if we ever observe a value.
    - Missing samples are recorded as NaN (no forward-fill here).
    - Ambient is estimated from a slow rolling-min proxy if not available.
    """
    try:
        import wmi  # type: ignore
    except Exception:
        return None, {"mode": "windows_live", "error": "wmi missing"}

    fs = int(cfg.get("fs", 1))
    duration_s = int(cfg.get("duration_s", 600))
    n = fs * duration_s
    if n <= 0:
        return None, {"mode": "windows_live", "error": "bad duration/fs"}

    c = wmi.WMI(namespace="root\\OpenHardwareMonitor")

    def get_value(snapshot, sensor_type: str, name_contains: list[str]) -> float | None:
        for s in snapshot:
            try:
                if getattr(s, "SensorType", "").lower() == sensor_type:
                    name = getattr(s, "Name", "").lower()
                    if any(key in name for key in name_contains):
                        v = getattr(s, "Value", None)
                        if v is not None:
                            return float(v)
            except Exception:
                pass
        return None

    recs: list[dict] = []
    t0 = time.time()

    # Track which sensors were EVER observed so we only include real columns
    seen = {"cpu_temp": False, "gpu_temp": False, "fan_rpm": False, "cpu_load": False}

    for i in range(n):
        snap = c.Sensor()  # one WMI snapshot

        cpu_temp = get_value(snap, "temperature", ["cpu package", "cpu"])  # prefer "package"
        gpu_temp = get_value(snap, "temperature", ["gpu", "nvidia", "amd"])
        fan_rpm  = get_value(snap, "fan", ["cpu", "system", "chassis", "gpu"])
        cpu_load = get_value(snap, "load", ["cpu total", "cpu"])

        if cpu_temp is not None: seen["cpu_temp"] = True
        if gpu_temp is not None: seen["gpu_temp"] = True
        if fan_rpm  is not None: seen["fan_rpm"]  = True
        if cpu_load is not None: seen["cpu_load"] = True

        recs.append({
            "t": i / fs,
            "cpu_temp": cpu_temp if cpu_temp is not None else np.nan,
            "gpu_temp": gpu_temp if gpu_temp is not None else np.nan,
            "fan_rpm":  fan_rpm  if fan_rpm  is not None else np.nan,
            "cpu_load": cpu_load if cpu_load is not None else np.nan,
        })

        # precise pacing to maintain fs
        next_tick = t0 + (i + 1) / fs
        sleep_for = max(0.0, next_tick - time.time())
        time.sleep(sleep_for)

    df = pd.DataFrame(recs)
    if df.empty:
        return None, {"mode": "windows_live", "error": "no data"}

    # Drop columns that were NEVER seen (all NaN → remove the column entirely)
    for col, was_seen in seen.items():
        if not was_seen and col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Ambient proxy from a slow rolling-min of CPU temp minus 10°C (bounded)
    if "cpu_temp" in df.columns and df["cpu_temp"].notna().any():
        win = max(1, fs * 300)  # ~5 minutes
        amb_proxy = (df["cpu_temp"].rolling(window=win, min_periods=1).min() - 10).clip(lower=15, upper=40)
        df["ambient"] = amb_proxy.values
    else:
        df["ambient"] = 30.0  # fallback constant if no CPU temp (rare)

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
