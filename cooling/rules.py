import numpy as np
import pandas as pd

def suggest_actions(last_row: pd.Series, feats: pd.DataFrame, ambient: float):
    s = last_row; out = []
    if s['cpu_dT_amb'] > 25 and s['x_rpm_per_deg'] < 30:
        out.append("Clean dust filters and heatsink fins; fan isn’t scaling with heat (low RPM per °C).")
    if s['cpu_dT_amb'] > 25 and s['x_corr_cpu_fan'] > 0.5 and s['fan_mean'] > 1200:
        out.append("Thermal paste likely degraded or poor contact; reapply paste and reseat cooler.")
    if s['cpu_dT_amb'] > 20 and s['x_corr_load_cpu'] < 0.2:
        out.append("High temps even without load changes; improve case airflow and fan curve configuration.")
    if s['gpu_dT_amb'] > 25 and s['fan_mean'] < 1100:
        out.append("GPU/Case airflow constrained; increase case fan curve or clean intake.")
    if ambient >= 32 and s['cpu_mean'] > ambient + 20:
        out.append("Room is hot; improve room ventilation or lower ambient temperature.")
    if not out:
        out.append("System appears stable. Continue monitoring.")
    return out
