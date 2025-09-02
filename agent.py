import time, os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from plyer import notification

from cooling.daq import collect_block
from cooling.features import build_feature_table
from cooling.model import train_iforest, score_iforest, severity_from_scores, band_status, load_model, save_model

CFG_PATH = Path("agent_config.json")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "if_model.joblib"

DEFAULT_CFG = {
    "mode": "sim",          # "sim" or "windows_live"
    "fs": 1,
    "window_s": 120,
    "hop_s": 10,
    "warmup_s": 300,
    "green_hi": 30,
    "yellow_hi": 70,
    "notify": True,
    "severity_notify_threshold": 70,
    "duration_s": 900,
    "sim": {
        "ambient": 28.0,
        "avg_load": 30,
        "load_style": "idle spikes",
        "fault": "none",
        "seed": 42
    }
}

def load_cfg():
    if CFG_PATH.exists():
        return json.loads(CFG_PATH.read_text(encoding="utf-8"))
    CFG_PATH.write_text(json.dumps(DEFAULT_CFG, indent=2), encoding="utf-8")
    return DEFAULT_CFG

def notify_desktop(title, message):
    try:
        notification.notify(title=title, message=message, timeout=5)
    except Exception:
        pass

def main():
    cfg = load_cfg()
    fs = int(cfg["fs"]); win_s = int(cfg["window_s"]); hop_s = int(cfg["hop_s"]); warmup_s = int(cfg["warmup_s"])
    green_hi = int(cfg["green_hi"]); yellow_hi = int(cfg["yellow_hi"])
    sev_thr = int(cfg["severity_notify_threshold"]); do_notify = bool(cfg["notify"])
    duration_s = cfg.get("duration_s", None)

    # Try load persisted model (with calibration)
    model_blob = load_model(str(MODEL_PATH)) if MODEL_PATH.exists() else None
    model = model_blob["model"] if model_blob else None
    calib = model_blob["calib"] if model_blob else None

    start = time.time()
    while True:
        df, meta = collect_block(cfg | {"duration_s": win_s + hop_s})  # collect slightly more than a window
        if df is None or df.empty:
            time.sleep(hop_s)
            continue

        feats, _ = build_feature_table(df, fs=fs, window_s=win_s, hop_s=hop_s)
        feature_cols = feats.columns[feats.columns.str.startswith(("cpu_", "gpu_", "fan_", "load_", "x_"))]

        if model is None:
            warm_mask = feats["t_start"] < warmup_s
            if warm_mask.sum() >= 5:
                model = train_iforest(feats.loc[warm_mask, feature_cols])
                # derive and persist calibration
                train_scores = -model.decision_function(feats.loc[warm_mask, feature_cols])
                lo, hi = np.percentile(train_scores, [5,95])
                calib = {"lo": float(lo), "hi": float(hi)}
                save_model(model, calib, str(MODEL_PATH))
            else:
                # wait for enough warm data
                time.sleep(hop_s)
                continue

        scores = -model.decision_function(feats.loc[:, feature_cols])
        if calib:
            lo, hi = calib["lo"], calib["hi"]
            severity = 100 * (scores - lo) / (hi - lo + 1e-9)
            severity = np.clip(severity, 0, 100)
        else:
            severity = severity_from_scores(scores)
        feats["anom_score"] = scores
        feats["severity"] = severity
        feats["status"] = feats["severity"].apply(lambda s: band_status(s, green_hi, yellow_hi))

        feats.to_csv(LOG_DIR / "features_scores.csv", index=False)
        last = feats.iloc[-1]
        if last["severity"] >= sev_thr:
            evt = pd.DataFrame([{"t_start": last["t_start"], "t_end": last["t_end"], "severity": float(last["severity"])}])
            if (LOG_DIR / "events.csv").exists():
                evt.to_csv(LOG_DIR / "events.csv", mode="a", header=False, index=False)
            else:
                evt.to_csv(LOG_DIR / "events.csv", index=False)
            if do_notify:
                notify_desktop("PC Cooling Alert", f"Severity {int(last['severity'])}/100 — Check fans/heatsink/airflow.")

        if duration_s is not None and (time.time() - start) >= duration_s:
            break
        time.sleep(hop_s)

    print("Agent finished. Logs at ./logs")

if __name__ == "__main__":
    main()
