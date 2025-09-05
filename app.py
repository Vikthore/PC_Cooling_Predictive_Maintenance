import os
import numpy as np
import pandas as pd
import streamlit as st

from cooling.daq import collect_block
from cooling.features import build_feature_table
from cooling.model import (
    train_iforest,
    severity_from_scores,
    band_status,
    save_model,
    load_model,
)
from cooling.rules import suggest_actions
from cooling.report import build_pdf_report
from cooling.hysteresis import apply_hysteresis

# ------------------------- Page -------------------------
st.set_page_config(page_title="PC Cooling Predictive Maintenance", layout="wide")
st.title("PC Cooling Predictive Maintenance")

# ------------------------- Sidebar ----------------------
with st.sidebar:
    st.header("Data Source")
    src = st.selectbox("Source", ["Simulated", "Windows Live"], help="Windows Live requires OpenHardwareMonitor running.")

    st.header("Run Settings")
    dur = st.number_input("Duration (s)", 30, 7200, 600, step=30)
    fs = st.number_input("Sampling rate (Hz)", 1, 5, 1, step=1)

    st.header("Simulation (only if Simulated)")
    amb = st.number_input("Ambient (°C)", 15.0, 45.0, 28.0, step=0.5)
    base_load = st.slider("Avg CPU Load (%)", 0, 100, 30, step=5)
    load_style = st.selectbox("Load profile", ["idle spikes", "ramps", "sine wave"])
    fault = st.selectbox("Fault (starts mid-run)", ["none", "fan_slow", "dust_clog", "paste_degraded", "ambient_hot", "curve_misconfig"])
    seed = st.number_input("Random seed", 0, 1_000_000, 42, step=1)

    st.header("Detection")
    warmup = st.slider("Healthy warm-up (s)", 10, 1800, 300, step=10, help="Used to train & calibrate if no model is loaded.")
    win = st.slider("Window (s)", 20, 600, 180, step=10)
    hop = st.slider("Hop (s)", 5, 120, 10, step=5)

    st.header("Severity Bands")
    green_hi = st.slider("Green < x", 10, 60, 35, step=1)
    yellow_hi = st.slider("Yellow < y (else Red)", 61, 95, 80, step=1)

    st.header("Model")
    model_path = "models/if_model.joblib"
    load_btn = st.button("Load model")
    save_btn = st.button("Save model")

# ------------------------- Action -----------------------
go = st.button("Collect & Detect", type="primary", use_container_width=True)

# Tabs for a tighter UI
tabs = st.tabs(["Status", "Signals", "Timeline", "Downloads"])

if not go:
    with tabs[0]:
        st.info("Choose source, set parameters, then click **Collect & Detect**.")
    st.stop()

# ------------------------- Collect ----------------------
with st.spinner(f"Collecting {dur}s from {src} and analyzing…"):
    if src == "Simulated":
        cfg = {
            "mode": "sim",
            "fs": fs,
            "duration_s": dur,
            "sim": {"ambient": amb, "avg_load": base_load, "load_style": load_style, "fault": fault, "seed": seed},
        }
    else:
        cfg = {"mode": "windows_live", "fs": fs, "duration_s": dur, "sim": {}}

    df, meta = collect_block(cfg)
    if df is None or df.empty:
        st.error("No data collected. If using Windows Live, ensure OpenHardwareMonitor is running.")
        st.stop()

# ------------------------- Features & Model -------------
feats, _ = build_feature_table(df, fs=fs, window_s=win, hop_s=hop)
feature_cols = feats.columns[feats.columns.str.startswith(("cpu_", "gpu_", "fan_", "load_", "x_"))]

model = None
calib = None

if load_btn and os.path.exists(model_path):
    blob = load_model(model_path)
    model = blob.get("model")
    calib = blob.get("calib")
    st.success("Loaded model from disk.")
else:
    healthy_mask = feats["t_start"] < warmup
    X_train = feats.loc[healthy_mask, feature_cols]
    if len(X_train) < 10:
        st.error("Not enough healthy windows to train. Increase duration or reduce warm-up.")
        st.stop()
    model = train_iforest(X_train)
    # Conservative calibration from healthy scores (50th..99th pct)
    train_scores = -model.decision_function(X_train)
    lo, hi = np.percentile(train_scores, [50, 99])
    calib = {"lo": float(lo), "hi": float(hi)}

# Score all windows
scores = -model.decision_function(feats.loc[:, feature_cols])
if calib:
    lo, hi = calib["lo"], calib["hi"]
    severity = 100 * (scores - lo) / (hi - lo + 1e-9)
    severity = np.clip(severity, 0, 100)
else:
    severity = severity_from_scores(scores)

# EWMA smoothing on severity for stability
feats["anom_score"] = scores
feats["severity"] = pd.Series(severity).ewm(alpha=0.2).mean().values

# Raw labels by thresholds
feats["status_raw"] = feats["severity"].apply(lambda s: band_status(s, green_hi, yellow_hi))

# Hysteresis to prevent flapping + one-episode detection
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
    cooloff_win=max(1, int(60 / max(1, hop))),  # ~60s notify cooloff (for parity with agent)
)
feats["status"] = status_hyst

# Persist features/scores for auditability
os.makedirs("logs", exist_ok=True)
feats.to_csv("logs/features_scores.csv", index=False)

# ------------------------- STATUS TAB -------------------
with tabs[0]:
    last = feats.iloc[-1]
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Status", last["status"])
    c2.metric("Severity", f"{int(last['severity'])}/100")
    c3.metric("Windows", f"{len(feats)}")

    st.caption("Top suggestions (based on current window):")
    for s in suggest_actions(last, feats, ambient=df["ambient"].iloc[-1] if "ambient" in df else 25.0)[:3]:
        st.write("• " + s)

    if events:
        st.divider()
        st.subheader("Fault Episodes (hysteresis)")
        st.dataframe(pd.DataFrame(events))

# ------------------------- SIGNALS TAB ------------------
with tabs[1]:
    st.subheader("Signals")
    st.line_chart(df[["cpu_temp", "gpu_temp", "fan_rpm", "cpu_load"]])

# ------------------------- TIMELINE TAB -----------------
with tabs[2]:
    st.subheader("Anomaly Timeline")
    st.dataframe(feats[["t_start", "t_end", "status", "severity", "anom_score"]].round(2))

# ------------------------- DOWNLOADS TAB ----------------
with tabs[3]:
    st.subheader("Exports")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download signals.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="signals.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "Download features_scores.csv",
        data=feats.to_csv(index=False).encode("utf-8"),
        file_name="features_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )
    pdf_bytes = build_pdf_report(feats, df, green_hi, yellow_hi)
    c3.download_button(
        "Download report.pdf",
        data=pdf_bytes,
        file_name="pc_cooling_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# Save model (optional)
if save_btn:
    os.makedirs("models", exist_ok=True)
    save_model(model, calib, model_path)
    st.success(f"Saved model to {model_path}")
