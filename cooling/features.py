import numpy as np
import pandas as pd

def _roll_stats(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p95": float(np.percentile(x, 95)),
        "p05": float(np.percentile(x, 5)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "p2p": float(np.max(x) - np.min(x)),
    }

def build_feature_table(df: pd.DataFrame, fs: int = 1, window_s: int = 120, hop_s: int = 10):
    n = len(df)
    win = int(window_s * fs); hop = int(hop_s * fs)
    rows = []
    for start in range(0, n - win + 1, hop):
        end = start + win
        w = df.iloc[start:end]
        t0 = float(w["t"].iloc[0]); t1 = float(w["t"].iloc[-1])

        d_cpu = np.diff(w["cpu_temp"].values); d_gpu = np.diff(w["gpu_temp"].values)
        dT_cpu = np.percentile(np.abs(d_cpu*fs), 95); dT_gpu = np.percentile(np.abs(d_gpu*fs), 95)

        dta_cpu = np.mean(w["cpu_temp"] - w["ambient"])
        dta_gpu = np.mean(w["gpu_temp"] - w["ambient"])

        corr_load_cpu = np.corrcoef(w["cpu_load"], w["cpu_temp"])[0,1] if w["cpu_load"].std()>0 and w["cpu_temp"].std()>0 else 0.0
        corr_cpu_fan = np.corrcoef(w["cpu_temp"], w["fan_rpm"])[0,1] if w["cpu_temp"].std()>0 and w["fan_rpm"].std()>0 else 0.0

        rpm_per_deg = (w["fan_rpm"].mean()) / max(1e-3, (w["cpu_temp"].mean() - w["ambient"].mean()))

        cpu_s = _roll_stats(w["cpu_temp"]) ; gpu_s = _roll_stats(w["gpu_temp"]) ; fan_s = _roll_stats(w["fan_rpm"]) ; load_s = _roll_stats(w["cpu_load"])

        rows.append({
            "t_start": t0, "t_end": t1,
            "cpu_dTdt_p95": float(dT_cpu), "gpu_dTdt_p95": float(dT_gpu),
            "cpu_dT_amb": float(dta_cpu), "gpu_dT_amb": float(dta_gpu),
            "x_corr_load_cpu": float(np.nan_to_num(corr_load_cpu)),
            "x_corr_cpu_fan": float(np.nan_to_num(corr_cpu_fan)),
            "x_rpm_per_deg": float(rpm_per_deg),
            **{f"cpu_{k}": v for k,v in cpu_s.items()},
            **{f"gpu_{k}": v for k,v in gpu_s.items()},
            **{f"fan_{k}": v for k,v in fan_s.items()},
            **{f"load_{k}": v for k,v in load_s.items()},
        })
    feats = pd.DataFrame(rows)
    meta = {"fs": fs, "window_s": window_s, "hop_s": hop_s}
    return feats, meta
