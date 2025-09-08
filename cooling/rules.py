import numpy as np
import pandas as pd

def _get(s: pd.Series, key: str, default=np.nan):
    try:
        return float(s.get(key, default))
    except Exception:
        return default

def suggest_actions(last_row: pd.Series, feats: pd.DataFrame, ambient: float):
    s = last_row
    out = []

    cpu_dT_amb = _get(s, "cpu_dT_amb")
    gpu_dT_amb = _get(s, "gpu_dT_amb")
    rpm_per_deg = _get(s, "x_rpm_per_deg")
    corr_cpu_fan = _get(s, "x_corr_cpu_fan")
    corr_load_cpu = _get(s, "x_corr_load_cpu")
    fan_mean = _get(s, "fan_mean")
    cpu_mean = _get(s, "cpu_mean")

    # Dust / airflow issues (only if fan exists; else skip)
    if not np.isnan(rpm_per_deg) and rpm_per_deg > 0 and cpu_dT_amb > 25 and rpm_per_deg < 30:
        out.append("Clean dust filters and heatsink fins; fan isn’t scaling with heat (low RPM per °C).")

    # Paste/contact degradation (requires some fan + correlation info; soft condition if fan missing)
    if cpu_dT_amb > 25 and ( (not np.isnan(corr_cpu_fan) and corr_cpu_fan > 0.5 and (np.isnan(fan_mean) or fan_mean > 1200)) or np.isnan(fan_mean) ):
        out.append("Thermal paste may be degraded or cooler contact may be poor; reapply paste and reseat cooler.")

    # Case airflow / curve misconfig (works even without fan sensors)
    if cpu_dT_amb > 20 and (np.isnan(corr_load_cpu) or corr_load_cpu < 0.2):
        out.append("High temps without matching load changes; check case airflow and fan curve configuration.")

    # GPU/case airflow (only if GPU present)
    if not np.isnan(gpu_dT_amb) and gpu_dT_amb > 25 and (np.isnan(fan_mean) or fan_mean < 1100):
        out.append("GPU or case airflow appears constrained; increase case fan curve or clean intake.")

    # Hot room
    if ambient >= 32 and (not np.isnan(cpu_mean)) and cpu_mean > ambient + 20:
        out.append("Room ambient is high; improve room ventilation or lower ambient temperature.")

    if not out:
        out.append("System appears stable. Continue monitoring.")
    return out
