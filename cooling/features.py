import numpy as np
import pandas as pd


def _roll_stats(x: np.ndarray):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p95": float(np.percentile(x, 95)),
        "p05": float(np.percentile(x, 5)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "p2p": float(np.max(x) - np.min(x)),
    }


def _has(df: pd.DataFrame, col: str) -> bool:
    return (col in df.columns) and df[col].notna().any()


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


def build_feature_table(
    df: pd.DataFrame, fs: int = 1, window_s: int = 120, hop_s: int = 10
):
    """
    Robust feature builder:
    - Works with any subset of {cpu_temp, gpu_temp, fan_rpm, cpu_load, ambient}.
    - Only emits features for sensors that exist.
    """
    assert _has(df, "cpu_temp"), "cpu_temp is required"
    # Optional columns
    have_gpu = _has(df, "gpu_temp")
    have_fan = _has(df, "fan_rpm")
    have_load = _has(df, "cpu_load")
    have_amb = _has(df, "ambient")

    n = len(df)
    win = int(window_s * fs)
    hop = int(hop_s * fs)
    rows = []

    for start in range(0, n - win + 1, hop):
        end = start + win
        w = df.iloc[start:end]
        t0 = float(w["t"].iloc[0])
        t1 = float(w["t"].iloc[-1])

        row = {"t_start": t0, "t_end": t1}

        # ----- CPU temperature dynamics -----
        cpu = w["cpu_temp"].to_numpy()
        d_cpu = np.diff(cpu) * fs
        row["cpu_dTdt_p95"] = float(np.percentile(np.abs(d_cpu), 95))
        if have_amb:
            row["cpu_dT_amb"] = float(np.mean(cpu - w["ambient"].to_numpy()))
        # stats
        row.update({f"cpu_{k}": v for k, v in _roll_stats(cpu).items()})

        # ----- GPU (optional) -----
        if have_gpu:
            gpu = w["gpu_temp"].to_numpy()
            d_gpu = np.diff(gpu) * fs
            row["gpu_dTdt_p95"] = float(np.percentile(np.abs(d_gpu), 95))
            if have_amb:
                row["gpu_dT_amb"] = float(np.mean(gpu - w["ambient"].to_numpy()))
            row.update({f"gpu_{k}": v for k, v in _roll_stats(gpu).items()})

        # ----- Fan RPM (optional) -----
        if have_fan:
            fan = w["fan_rpm"].to_numpy()
            row.update({f"fan_{k}": v for k, v in _roll_stats(fan).items()})

        # ----- CPU Load (optional) -----
        if have_load:
            load = w["cpu_load"].to_numpy()
            row.update({f"load_{k}": v for k, v in _roll_stats(load).items()})
            # correlation load ↔ temp
            row["x_corr_load_cpu"] = _safe_corr(load, cpu)
        else:
            row["x_corr_load_cpu"] = 0.0

        # ----- Temp ↔ Fan correlation and efficiency proxy (optional) -----
        if have_fan:
            row["x_corr_cpu_fan"] = _safe_corr(cpu, w["fan_rpm"].to_numpy())
            if have_amb:
                dta = np.mean(cpu - w["ambient"].to_numpy())
                row["x_rpm_per_deg"] = float(np.mean(w["fan_rpm"])) / max(1e-3, dta)
            else:
                row["x_rpm_per_deg"] = 0.0
        else:
            row["x_corr_cpu_fan"] = 0.0
            row["x_rpm_per_deg"] = 0.0

        rows.append(row)

    feats = pd.DataFrame(rows)

    # Ensure numeric columns exist even if sensors were absent (helps downstream selection by prefixes)
    for base in ["gpu_", "fan_", "load_"]:
        if not any(c.startswith(base) for c in feats.columns):
            feats[base + "stub"] = (
                np.nan
            )  # harmless column to keep prefix selection safe

    meta = {"fs": fs, "window_s": window_s, "hop_s": hop_s}
    return feats, meta
