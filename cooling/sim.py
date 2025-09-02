import numpy as np
import pandas as pd

def simulate_stream(duration_s=600, fs=1, ambient=28.0, avg_load=30, load_style="idle spikes", fault="none", seed=42):
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    if load_style == "idle spikes":
        load = np.clip(avg_load + 30 * (rng.random(n) < 0.08).astype(float) + rng.normal(0, 3, n), 0, 100)
    elif load_style == "ramps":
        load = (avg_load + 40 * (np.sin(2*np.pi*t/120) + np.sin(2*np.pi*t/300))).clip(0,100)
    else:
        load = (avg_load + 30*np.sin(2*np.pi*t/90)).clip(0,100)

    C_th_cpu = 8.0; C_th_gpu = 6.0
    k_fan_cpu = 0.35; k_fan_gpu = 0.30; k_passive = 0.05
    rpm_min, rpm_max = 900, 1800
    slope = 60

    mid = n//2
    amb = np.full(n, ambient)
    k_fan_cpu_arr = np.full(n, k_fan_cpu)
    k_fan_gpu_arr = np.full(n, k_fan_gpu)
    rpm_max_arr = np.full(n, rpm_max)
    slope_arr = np.full(n, slope)
    C_th_cpu_arr = np.full(n, C_th_cpu)

    if fault == "fan_slow":
        rpm_max_arr[mid:] = rpm_max * 0.6
    elif fault == "dust_clog":
        k_fan_cpu_arr[mid:] = k_fan_cpu * 0.6
        k_fan_gpu_arr[mid:] = k_fan_gpu * 0.6
    elif fault == "paste_degraded":
        C_th_cpu_arr[mid:] = C_th_cpu * 1.6
    elif fault == "ambient_hot":
        amb[mid:] = ambient + 6.0
    elif fault == "curve_misconfig":
        slope_arr[mid:] = slope * 0.3

    cpu_temp = np.empty(n); gpu_temp = np.empty(n); fan_rpm = np.empty(n)
    cpu_temp[0] = ambient + 8; gpu_temp[0] = ambient + 6

    for i in range(1, n):
        fan_rpm[i] = np.clip(rpm_min + slope_arr[i]*(cpu_temp[i-1] - amb[i]), rpm_min, rpm_max_arr[i])
        P_cpu = 0.5*load[i] + 3 + rng.normal(0, 0.3)
        P_gpu = 0.35*load[i] + 2 + rng.normal(0, 0.3)
        cool_cpu = (k_fan_cpu_arr[i] * (fan_rpm[i]/1000.0) + k_passive) * (cpu_temp[i-1] - amb[i])
        cool_gpu = (k_fan_gpu_arr[i] * (fan_rpm[i]/1000.0) + k_passive) * (gpu_temp[i-1] - amb[i])
        cpu_temp[i] = cpu_temp[i-1] + (P_cpu - cool_cpu) / C_th_cpu_arr[i]
        gpu_temp[i] = gpu_temp[i-1] + (P_gpu - cool_gpu) / C_th_gpu

    cpu_temp += rng.normal(0, 0.1, n); gpu_temp += rng.normal(0, 0.1, n)
    fan_rpm[0] = fan_rpm[1]
    df = pd.DataFrame({
        "t": t, "ambient": amb, "cpu_load": load,
        "cpu_temp": cpu_temp, "gpu_temp": gpu_temp, "fan_rpm": fan_rpm
    })
    return df
