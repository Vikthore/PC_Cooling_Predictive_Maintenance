import streamlit as st
import numpy as np
import pandas as pd
import os, io, time, joblib, sys

from cooling.daq import collect_block
from cooling.features import build_feature_table
from cooling.model import train_iforest, score_iforest, severity_from_scores, band_status, save_model, load_model
from cooling.rules import suggest_actions
from cooling.report import build_pdf_report

st.set_page_config(page_title="PC Cooling Predictive Maintenance", layout="wide")
st.title("Smart Predictive Maintenance — PC Cooling (v3)")

with st.sidebar:
    st.header("Data Source")
    src = st.selectbox("Source", ["Simulated", "Windows Live"])
    st.caption("Windows Live requires OpenHardwareMonitor running.")

    st.header("Run Settings")
    dur = st.number_input("Duration (seconds)", 30, 3600, 600, step=30)
    fs = st.number_input("Sampling rate (Hz)", 1, 5, 1, step=1)

    st.header("Simulation (if Simulated)")
    amb = st.number_input("Ambient Temp (°C)", 15.0, 45.0, 28.0, step=0.5)
    base_load = st.slider("Average CPU Load (%)", 0, 100, 30, step=5)
    load_style = st.selectbox("Load profile", ["idle spikes", "ramps", "sine wave"])
    fault = st.selectbox("Fault (inject at mid-run)", ["none", "fan_slow", "dust_clog", "paste_degraded", "ambient_hot", "curve_misconfig"])
    seed = st.number_input("Random seed", 0, 1_000_000, 42, step=1)

    st.header("Detection")
    warmup = st.slider("Healthy warm-up (seconds)", 10, 900, 300, step=10)
    win = st.slider("Window (seconds)", 20, 300, 120, step=10)
    hop = st.slider("Hop (seconds)", 5, 120, 10, step=5)

    st.header("Severity Bands")
    green_hi = st.slider("Green upper bound", 10, 60, 30, step=1)
    yellow_hi = st.slider("Yellow upper bound", 61, 95, 70, step=1)

    st.header("Model")
    model_path = "models/if_model.joblib"
    do_load = st.button("Load model") 
    do_save = st.button("Save model")

run = st.button("Collect & Detect")

if run:
    with st.spinner(f"Collecting {dur}s from {src} and analyzing…"):
        if src == "Simulated":
            cfg = {"mode": "sim", "fs": fs, "duration_s": dur, "sim": {"ambient": amb, "avg_load": base_load, "load_style": load_style, "fault": fault, "seed": seed}}
        else:
            cfg = {"mode": "windows_live", "fs": fs, "duration_s": dur, "sim": {}}

        df, meta = collect_block(cfg)
        if df is None or df.empty:
            st.error("No data collected. If using Windows Live, ensure OpenHardwareMonitor is running.")
            st.stop()

        st.subheader("Signals")
        st.line_chart(df[["cpu_temp", "gpu_temp", "fan_rpm", "cpu_load"]])

        feats, _ = build_feature_table(df, fs=fs, window_s=win, hop_s=hop)
        feature_cols = feats.columns[feats.columns.str.startswith(("cpu_", "gpu_", "fan_", "load_", "x_"))]

        model_blob = None
        model = None
        calib = None

        if do_load and os.path.exists(model_path):
            model_blob = load_model(model_path)
            model = model_blob["model"]; calib = model_blob["calib"]
            st.success("Loaded model from disk.")
        else:
            healthy_mask = feats["t_start"] < warmup
            X_train = feats.loc[healthy_mask, feature_cols]
            if len(X_train) < 10:
                st.error("Not enough healthy windows to train. Increase duration or reduce warm-up.")
                st.stop()
            model = train_iforest(X_train)
            # derive calibration from training scores
            train_scores = -model.decision_function(X_train)
            lo, hi = np.percentile(train_scores, [5,95])
            calib = {"lo": float(lo), "hi": float(hi)}

        X_all = feats.loc[:, feature_cols]
        scores = -model.decision_function(X_all)  # anomaly score higher => worse
        if calib:
            lo, hi = calib["lo"], calib["hi"]
            severity = 100 * (scores - lo) / (hi - lo + 1e-9)
            severity = np.clip(severity, 0, 100)
        else:
            severity = severity_from_scores(scores)
        feats["anom_score"] = scores
        feats["severity"] = severity
        feats["status"] = feats["severity"].apply(lambda s: band_status(s, green_hi, yellow_hi))

        last = feats.iloc[-1]
        st.subheader("Current Health")
        st.metric("Status", last["status"], delta=f"{int(last['severity'])}/100 severity")

        st.subheader("Top Suggestions")
        for s in suggest_actions(last, feats, ambient=df['ambient'].iloc[-1] if 'ambient' in df else 25.0)[:3]:
            st.write("• " + s)

        st.subheader("Anomaly Timeline")
        st.dataframe(feats[["t_start", "t_end", "status", "severity", "anom_score"]].round(2))

        os.makedirs("logs", exist_ok=True)
        feats.to_csv("logs/features_scores.csv", index=False)

        st.download_button("Download signals (CSV)",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name="signals.csv",
                           mime="text/csv")
        st.download_button("Download features+scores (CSV)",
                           data=feats.to_csv(index=False).encode("utf-8"),
                           file_name="features_scores.csv",
                           mime="text/csv")

        pdf_bytes = build_pdf_report(feats, df, green_hi, yellow_hi)
        st.download_button("Download PDF report",
                           data=pdf_bytes,
                           file_name="pc_cooling_report.pdf",
                           mime="application/pdf")

        if do_save:
            os.makedirs("models", exist_ok=True)
            save_model(model, calib, model_path)
            st.success(f"Saved model to {model_path}")
else:
    st.info("Choose source, set parameters, then click **Collect & Detect**.")
