import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import platform

IS_WINDOWS = platform.system() == "Windows"

from cooling.sim import simulate_stream


def _collect_sim_block(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sim = cfg.get("sim", {}) or {}
    df = simulate_stream(
        duration_s=int(cfg.get("duration_s", 600)),
        fs=int(cfg.get("fs", 1)),
        ambient=float(sim.get("ambient", 28.0)),
        avg_load=int(sim.get("avg_load", 30)),
        load_style=str(sim.get("load_style", "idle spikes")),
        fault=str(sim.get("fault", "none")),
        seed=int(sim.get("seed", 42)),
    )
    return df, {"mode": "sim"}


def _collect_windows_live_block(cfg: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Real-time collection from OpenHardwareMonitor WMI for the full duration.
    - Samples at fs for duration_s seconds.
    - Includes a column only if a value was ever observed (no phantom sensors).
    - Missing samples are left as NaN (no forward-fill).
    - Ambient is estimated from a slow rolling-min of CPU temp minus 10°C (bounded).
    """
    if not IS_WINDOWS:
        return None, {"mode": "windows_live", "error": "not_windows"}

    # Import WMI lazily and only on Windows
    try:
        import wmi  # type: ignore
    except Exception:
        return None, {"mode": "windows_live", "error": "wmi_missing"}

    fs_int = int(cfg.get("fs", 1))
    duration_s = int(cfg.get("duration_s", 600))
    n = fs_int * duration_s
    if n <= 0:
        return None, {"mode": "windows_live", "error": "bad_duration_or_fs"}

    try:
        c = wmi.WMI(namespace=r"root\OpenHardwareMonitor")
    except Exception as e:
        return None, {"mode": "windows_live", "error": f"wmi_namespace: {e}"}

    def get_value(snapshot, sensor_type: str, name_contains: list[str]) -> Optional[float]:
        typ = sensor_type.lower()
        for s in snapshot:
            try:
                if getattr(s, "SensorType", "").lower() == typ:
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

    # Track which sensors were EVER observed
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
            "t": i / max(1, fs_int),
            "cpu_temp": cpu_temp if cpu_temp is not None else np.nan,
            "gpu_temp": gpu_temp if gpu_temp is not None else np.nan,
            "fan_rpm":  fan_rpm  if fan_rpm  is not None else np.nan,
            "cpu_load": cpu_load if cpu_load is not None else np.nan,
        })

        # precise pacing to maintain fs
        next_tick = t0 + (i + 1) / max(1, fs_int)
        sleep_for = max(0.0, next_tick - time.time())
        time.sleep(sleep_for)

    df = pd.DataFrame(recs)
    if df.empty:
        return None, {"mode": "windows_live", "error": "no_data"}

    # Drop columns that were NEVER seen (all NaN → remove)
    for col, was_seen in seen.items():
        if not was_seen and col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Ambient proxy from a slow rolling-min of CPU temp minus 10°C (bounded)
    if "cpu_temp" in df.columns and df["cpu_temp"].notna().any():
        ambient_roll = max(1, fs_int * 300)  # ~5 minutes, in samples
        amb_proxy = (df["cpu_temp"].rolling(window=ambient_roll, min_periods=1).min() - 10).clip(lower=15, upper=40)
        df["ambient"] = amb_proxy.values
    else:
        df["ambient"] = 30.0  # fallback constant if no CPU temp (rare)

    return df, {"mode": "windows_live"}


def collect_block(cfg: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    mode = str(cfg.get("mode", "sim")).lower()
    if mode == "windows_live":
        df, meta = _collect_windows_live_block(cfg)
        if df is not None and not df.empty:
            return df, meta
        # fallback to sim if live not available
        return _collect_sim_block(cfg)
    else:
        return _collect_sim_block(cfg)
