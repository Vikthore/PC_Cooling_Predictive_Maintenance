# agent.py — hysteresis-enabled, calibrated, rate-limited alerts

import time
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from plyer import notification

from cooling.daq import collect_block
from cooling.features import build_feature_table
from cooling.model import (
    train_iforest,
    severity_from_scores,
    band_status,
    load_model,
    save_model,
)
from cooling.hysteresis import apply_hysteresis

# -------------------------- Paths & Defaults --------------------------
CFG_PATH = Path("agent_config.json")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "if_model.joblib"

DEFAULT_CFG = {
    "mode": "sim",              # "sim" or "windows_live"
    "fs": 1,                    # Hz
    "window_s": 180,            # rolling window (s)
    "hop_s": 10,                # hop (s)
    "warmup_s": 900,            # healthy warm-up (s) if model not loaded
    "green_hi": 35,             # thresholds align with app
    "yellow_hi": 80,
    "notify": True,
    "severity_notify_threshold": 80,   # kept for compatibility; hysteresis also rate-limits
    "duration_s": 900,          # set None for continuous
    "sim": {
        "ambient": 28.0,
        "avg_load": 30,
        "load_style": "idle spikes",
        "fault": "none",
        "seed": 42
    }
}

# -------------------------- Helpers --------------------------
def load_cfg():
    if CFG_PATH.exists():
        try:
            return json.loads(CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    CFG_PATH.write_text(json.dumps(DEFAULT_CFG, indent=2), encoding="utf-8")
    return DEFAULT_CFG

def notify_desktop(title, message):
    try:
        notification.notify(title=title, message=message, timeout=5)
    except Exception:
        # non-fatal on headless/blocked environments
        pass

# -------------------------- Main Loop --------------------------
def main():
    cfg = load_cfg()
    mode = cfg.get("mode", "sim")
    fs = int(cfg.get("fs", 1))
    win_s = int(cfg.get("window_s", 180))
    hop_s = int(cfg.get("hop_s", 10))
    warmup_s = int(cfg.get("warmup_s", 900))
    green_hi = int(cfg.get("green_hi", 35))
    yellow_hi = int(cfg.get("yellow_hi", 80))
    do_notify = bool(cfg.get("notify", True))
    sev_thr = int(cfg.get("severity_notify_threshold", 80))  # legacy threshold
    duration_s = cfg.get("duration_s", None)

    # Try to load persisted model + calibration
    model_blob = load_model(str(MODEL_PATH)) if MODEL_PATH.exists() else None
    model = model_blob.get("model") if model_blob else None
    calib = model_blob.get("calib") if model_blob else None

    start_ts = time.time()
    # Collect a little more than one window each cycle to advance the timeline
    block_len = max(win_s + hop_s, win_s * 2)

    while True:
        # Build DAQ request
        daq_cfg = {
            "mode": mode,
            "fs": fs,
            "duration_s": block_len,
            "sim": cfg.get("sim", {}),
        }

        # Collect a block (sim or live)
        df, meta = collect_block(daq_cfg)
        if df is None or df.empty:
            time.sleep(hop_s)
            if duration_s is not None and (time.time() - start_ts) >= duration_s:
                break
            continue

        # Features
        feats, _ = build_feature_table(df, fs=fs, window_s=win_s, hop_s=hop_s)
        feature_cols = feats.columns[feats.columns.str.startswith(("cpu_", "gpu_", "fan_", "load_", "x_"))]
        if len(feats) == 0 or len(feature_cols) == 0:
            time.sleep(hop_s)
            continue

        # Train if needed (on warm-up windows)
        if model is None:
            warm_mask = feats["t_start"] < warmup_s
            if warm_mask.sum() >= 10:
                X_train = feats.loc[warm_mask, feature_cols]
                model = train_iforest(X_train)
                # Conservative calibration from healthy scores (50th..99th pct)
                train_scores = -model.decision_function(X_train)
                lo, hi = np.percentile(train_scores, [50, 99])
                calib = {"lo": float(lo), "hi": float(hi)}
                save_model(model, calib, str(MODEL_PATH))
            else:
                time.sleep(hop_s)
                if duration_s is not None and (time.time() - start_ts) >= duration_s:
                    break
                continue

        # Score all windows in this block
        scores = -model.decision_function(feats.loc[:, feature_cols])
        if calib:
            lo, hi = calib["lo"], calib["hi"]
            severity = 100 * (scores - lo) / (hi - lo + 1e-9)
            severity = np.clip(severity, 0, 100)
        else:
            severity = severity_from_scores(scores)

        # EWMA smooth severity for stability
        feats["anom_score"] = scores
        feats["severity"] = pd.Series(severity).ewm(alpha=0.2).mean().values

        # Raw labels by thresholds
        feats["status_raw"] = feats["severity"].apply(lambda s: band_status(s, green_hi, yellow_hi))

        # Apply hysteresis — single episode per sustained fault + rate limiting
        status_hyst, events = apply_hysteresis(
            feats["status_raw"].tolist(),
            feats["severity"].tolist(),
            feats["t_start"].tolist(),
            feats["t_end"].tolist(),
            red_enter_win=3,
            red_exit_green_win=5,
            yellow_enter_win=2,
            yellow_exit_green_win=3,
            escalate_yellow_to_red_win=6,
            sticky_red=True,
            cooloff_win=max(1, int(60 / max(1, hop_s))),  # ~60s notify cooloff
        )
        feats["status"] = status_hyst

        # Persist rolling features/scores for auditability
        feats.to_csv(LOG_DIR / "features_scores.csv", index=False)

        # Log & notify once per red episode
        for evt in events:
            row = {
                "t_start": evt["start"],
                "t_end": evt.get("end", evt["start"]),
                "severity": float(evt.get("peak", 0.0)),
                "ts": pd.Timestamp.utcnow().isoformat(),
            }
            if (LOG_DIR / "events.csv").exists():
                pd.DataFrame([row]).to_csv(LOG_DIR / "events.csv", mode="a", header=False, index=False)
            else:
                pd.DataFrame([row]).to_csv(LOG_DIR / "events.csv", index=False)

            if do_notify:
                # still respect legacy numeric threshold as an extra guard
                if row["severity"] >= sev_thr or evt.get("notify", False):
                    notify_desktop("PC Cooling Alert", f"Severity {int(row['severity'])}/100 — sustained fault detected.")

        # Exit condition for finite runs
        if duration_s is not None and (time.time() - start_ts) >= duration_s:
            break

        time.sleep(hop_s)

    print("Agent finished. Logs at ./logs")

# -------------------------- Entrypoint --------------------------
if __name__ == "__main__":
    main()
