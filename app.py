import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go_plotly
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import platform

IS_WINDOWS = platform.system() == "Windows"

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


# --- Windows COM/WMI cache fix (prevents com_error: Invalid syntax in Streamlit) ---
def _reset_win32com_cache():
    try:
        import os, site, shutil
        import win32com.client.gencache as gencache
        import win32com.client as win32

        # allow writes
        gencache.is_readonly = False
        # clear user temp cache
        user_tmp = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp", "gen_py")
        shutil.rmtree(user_tmp, ignore_errors=True)
        # clear site-packages cache(s)
        for p in site.getsitepackages():
            shutil.rmtree(os.path.join(p, "win32com", "gen_py"), ignore_errors=True)
        # re-prime COM wrappers
        win32.gencache.EnsureDispatch("WbemScripting.SWbemLocator")
    except Exception:
        # non-fatal; we'll still attempt import below
        pass


# ------------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="PC Cooling Predictive Maintenance",
    page_icon="🖥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    /* Main title styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Status cards styling */
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .status-green {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    
    .status-yellow {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    
    .status-red {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e40af;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    /* Action suggestions */
    .suggestion-item {
        background-color: #f1f5f9;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1e40af;
    }
    
    /* Dark mode support for info boxes */
    @media (prefers-color-scheme: dark) {
        .info-box {
            background-color: #1e293b;
            border: 1px solid #475569;
            color: #e2e8f0;
        }
    }
    
    /* Streamlit dark theme detection */
    .stApp[data-theme="dark"] .info-box {
        background-color: #1e293b;
        border: 1px solid #475569;
        color: #e2e8f0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""",
    unsafe_allow_html=True,
)

# ⇩ append this to the end of your current CSS string
st.markdown(
    """
<style>
/* ---------- Dark theme overrides ---------- */
.stApp[data-theme="dark"] {
  --panel-bg: #0b1220;   /* very dark slate */
  --panel-fg: #e5e7eb;   /* gray-200 */
  --panel-border: #1f2937; /* gray-800 */
}

/* Suggestions: use your .suggestion-item with dark colors */
.stApp[data-theme="dark"] .suggestion-item{
  background-color: var(--panel-bg) !important;
  color: var(--panel-fg) !important;
  border-color: var(--panel-border) !important;
}

/* Optional: a generic dark panel class you can reuse */
.stApp[data-theme="dark"] .dark-panel{
  background-color: var(--panel-bg) !important;
  color: var(--panel-fg) !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 12px;
  padding: 12px 14px;
}

/* DataFrame (timeline) — force dark table + headers + cells */
.stApp[data-theme="dark"] div[data-testid="stDataFrame"] > div{
  background: var(--panel-bg) !important;
}
.stApp[data-theme="dark"] div[data-testid="stDataFrame"] table{
  color: var(--panel-fg) !important;
  background: var(--panel-bg) !important;
}
.stApp[data-theme="dark"] div[data-testid="stDataFrame"] th{
  background: var(--panel-bg) !important;
  color: var(--panel-fg) !important;
  border-color: var(--panel-border) !important;
}
.stApp[data-theme="dark"] div[data-testid="stDataFrame"] tbody td{
  background: var(--panel-bg) !important;
  color: var(--panel-fg) !important;
  border-color: var(--panel-border) !important;
}

/* If you use pandas Styler row-highlighting, ensure text stays visible */
.stApp[data-theme="dark"] .row_heading, 
.stApp[data-theme="dark"] .blank { 
  background: var(--panel-bg) !important; 
  color: var(--panel-fg) !important; 
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------- Header Section -------------------------
st.markdown(
    '<h1 class="main-header">🖥 PC Cooling Predictive Maintenance</h1>'
    '<h2 class="subtitle"><i>Developed by</i></h2>',
    unsafe_allow_html=True,
)

# Team members section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        '<p style="text-align: center; font-size: 0.95rem; color: #666;">Okeke Donald Chisom<br><span style="font-size: 0.85rem; color: #999;">20201208763</span></p>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<p style="text-align: center; font-size: 0.95rem; color: #666;">Iwuchukwu Miracle Chima<br><span style="font-size: 0.85rem; color: #999;">20201233393</span></p>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<p style="text-align: center; font-size: 0.95rem; color: #666;">Nwanne-Udeh Bruno Chinaza<br><span style="font-size: 0.85rem; color: #999;">20201203523</span></p>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="subtitle">Monitor your system health and prevent overheating issues before they occur</p>',
    unsafe_allow_html=True,
)

# ------------------------- Enhanced Sidebar -------------------------
with st.sidebar:
    st.markdown("### ⚙ Configuration Panel")

    # Data Source Section
    st.markdown("#### 📊 Data Source")
    with st.container():
        sources = ["Simulated"] + (["Windows Live"] if IS_WINDOWS else [])
        src = st.selectbox(
            "Source",
            sources,
            help="Windows Live requires Windows + OpenHardwareMonitor.",
        )

        if src == "Windows Live" and not IS_WINDOWS:
            st.error(
                "Windows Live is only supported on Windows. Use Simulated mode here, or run locally on Windows."
            )
            st.stop()

    # Run Settings Section
    st.markdown("#### ⏱ Run Settings")
    col1, col2 = st.columns(2)
    with col1:
        dur = st.number_input("Duration (s)", 30, 7200, 600, step=30)
    with col2:
        fs = st.number_input("Sampling (Hz)", 1, 5, 1, step=1)

    # Simulation Settings (collapsible)
    if src == "Simulated":
        with st.expander("🔧 Simulation Parameters", expanded=True):
            amb = st.number_input("Ambient Temp (°C)", 15.0, 45.0, 28.0, step=0.5)
            base_load = st.slider("Average CPU Load (%)", 0, 100, 30, step=5)
            load_style = st.selectbox(
                "Load Profile", ["idle spikes", "ramps", "sine wave"]
            )
            fault = st.selectbox(
                "Fault Simulation",
                [
                    "none",
                    "fan_slow",
                    "dust_clog",
                    "paste_degraded",
                    "ambient_hot",
                    "curve_misconfig",
                ],
            )
            seed = st.number_input("Random Seed", 0, 1_000_000, 42, step=1)

    # Detection Settings
    with st.expander("🔍 Detection Settings", expanded=False):
        warmup = st.slider(
            "Healthy Warm-up (s)",
            10,
            1800,
            300,
            step=10,
            help="Used for training if no model loaded",
        )
        win = st.slider("Analysis Window (s)", 20, 600, 180, step=10)
        hop = st.slider("Update Interval (s)", 5, 120, 10, step=5)

    # Severity Thresholds
    with st.expander("⚠ Alert Thresholds", expanded=False):
        green_hi = st.slider("🟢 Normal < ", 10, 60, 35, step=1)
        yellow_hi = st.slider("🟡 Warning < ", 61, 95, 80, step=1)
        st.markdown("🔴 *Critical* ≥ " + str(yellow_hi))

    # Model Management
    st.markdown("#### 🤖 Model Management")
    model_path = "models/if_model.joblib"
    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("📥 Load", use_container_width=True)
    with col2:
        save_btn = st.button("💾 Save", use_container_width=True)

# ------------------------- Main Action Button -------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    go = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

# Initial state
if not go:
    # Welcome section with instructions
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
        <div class="info-box">
            <h3>🎯 Ready to Monitor Your System</h3>
            <p>Configure your settings in the sidebar and click <strong>Start Analysis</strong> to begin monitoring your PC's cooling performance.</p>
            <ul>
                <li><strong>Simulated Mode:</strong> Perfect for testing and demonstration</li>
                <li><strong>Windows Live:</strong> Real-time monitoring of your actual system</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Quick stats or system info could go here
    with st.container():
        st.markdown("### 📈 System Overview")
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)

        with info_col1:
            st.metric("🕒 Analysis Duration", f"{dur}s")
        with info_col2:
            st.metric("📊 Sampling Rate", f"{fs} Hz")
        with info_col3:
            st.metric("🔍 Window Size", f"{win}s" if "win" in locals() else "180s")
        with info_col4:
            st.metric("📡 Data Source", src)

    st.stop()

# ------------------------- Data Collection -------------------------
with st.spinner("🔄 Collecting data and analyzing system performance..."):
    if src == "Simulated":
        cfg = {
            "mode": "sim",
            "fs": fs,
            "duration_s": dur,
            "sim": {
                "ambient": amb,
                "avg_load": base_load,
                "load_style": load_style,
                "fault": fault,
                "seed": seed,
            },
        }
        df, meta = collect_block(cfg)
    else:
        cfg = {"mode": "windows_live", "fs": fs, "duration_s": dur, "sim": {}}
        if src == "Windows Live":
            if not IS_WINDOWS:
                st.error(
                    "Windows Live is only supported on Windows. Use Simulated mode here, or run locally on Windows."
                )
                st.stop()

            import time, subprocess, json  # local imports so cloud builds don't see them

            fs_int = int(fs)
            duration_int = int(dur)
            total_steps = fs_int * max(1, duration_int)

            # Try pywin32 WMI first; if that import fails, try cache reset then retry, else CIM
            try:
                import wmi as _wmi

                use_cim = False
            except Exception:
                try:
                    _reset_win32com_cache()
                    import wmi as _wmi

                    use_cim = False
                except Exception:
                    _wmi = None
                    use_cim = True

            # NEW: clamp sampling when using CIM
            if use_cim and fs_int > 1:
                st.warning(
                    "CIM fallback active: limiting sampling to 1 Hz for reliability."
                )
                fs_int = 1
                total_steps = fs_int * max(1, duration_int)

            def _drop_never_seen(_df: pd.DataFrame) -> pd.DataFrame:
                for col in ["cpu_temp", "gpu_temp", "fan_rpm", "cpu_load"]:
                    if col in _df.columns and not _df[col].notna().any():
                        _df.drop(columns=[col], inplace=True)
                return _df

            progress = st.progress(0, text="Collecting Windows Live telemetry…")
            status_txt = st.empty()
            recs = []
            t0 = time.time()

            if IS_WINDOWS and not use_cim:
                # -------- Path A: WMI via pywin32 --------
                try:
                    import pythoncom

                    pythoncom.CoInitialize()
                except ImportError:
                    pass  # Not on Windows, or pythoncom not available
                c = _wmi.WMI(namespace=r"root\OpenHardwareMonitor")

                def get_val_wmi(snapshot, sensor_type: str, keys: list[str]):
                    typ = sensor_type.lower()
                    for s in snapshot:
                        try:
                            if getattr(s, "SensorType", "").lower() == typ:
                                name = getattr(s, "Name", "").lower()
                                if any(k in name for k in keys):
                                    v = getattr(s, "Value", None)
                                    if v is not None:
                                        return float(v)
                        except Exception:
                            pass
                    return None

                for i in range(total_steps):
                    snap = c.Sensor()
                    cpu_temp = get_val_wmi(snap, "temperature", ["cpu package", "cpu"])
                    gpu_temp = get_val_wmi(
                        snap, "temperature", ["gpu", "nvidia", "amd"]
                    )
                    fan_rpm = get_val_wmi(
                        snap, "fan", ["cpu", "system", "chassis", "gpu"]
                    )
                    cpu_load = get_val_wmi(snap, "load", ["cpu total", "cpu"])

                    recs.append(
                        {
                            "t": i / max(1, fs_int),
                            "cpu_temp": cpu_temp if cpu_temp is not None else np.nan,
                            "gpu_temp": gpu_temp if gpu_temp is not None else np.nan,
                            "fan_rpm": fan_rpm if fan_rpm is not None else np.nan,
                            "cpu_load": cpu_load if cpu_load is not None else np.nan,
                        }
                    )

                    pct = int((i + 1) / total_steps * 100)
                    progress.progress(
                        pct, text=f"Collecting Windows Live telemetry… {pct}%"
                    )
                    status_txt.write(
                        f"Elapsed: {int(time.time()-t0)}s / {duration_int}s"
                    )

                    next_tick = t0 + (i + 1) / max(1, fs_int)
                    time.sleep(max(0.0, next_tick - time.time()))
            else:
                # -------- Path B: PowerShell CIM fallback (no pywin32/COM) --------
                def cim_snapshot():
                    ps = r"""
                    Get-CimInstance -Namespace root\OpenHardwareMonitor -ClassName Sensor |
                        Select-Object Name, SensorType, Value |
                        ConvertTo-Json -Compress
                    """
                    out = subprocess.check_output(
                        ["powershell", "-NoProfile", "-Command", ps], text=True
                    )
                    try:
                        data = json.loads(out)
                    except Exception:
                        return []
                    if isinstance(data, dict):
                        data = [data]
                    return data if isinstance(data, list) else []

                def get_val_cim(rows, sensor_type: str, keys: list[str]):
                    typ = sensor_type.lower()
                    for r in rows:
                        try:
                            if str(r.get("SensorType", "")).lower() == typ:
                                name = str(r.get("Name", "")).lower()
                                if any(k in name for k in keys):
                                    v = r.get("Value", None)
                                    if v is not None and str(v).strip() != "":
                                        return float(v)
                        except Exception:
                            pass
                    return None

                for i in range(total_steps):
                    rows = cim_snapshot()
                    cpu_temp = get_val_cim(rows, "temperature", ["cpu package", "cpu"])
                    gpu_temp = get_val_cim(
                        rows, "temperature", ["gpu", "nvidia", "amd"]
                    )
                    fan_rpm = get_val_cim(
                        rows, "fan", ["cpu", "system", "chassis", "gpu"]
                    )
                    cpu_load = get_val_cim(rows, "load", ["cpu total", "cpu"])

                    recs.append(
                        {
                            "t": i / max(1, fs_int),
                            "cpu_temp": cpu_temp if cpu_temp is not None else np.nan,
                            "gpu_temp": gpu_temp if gpu_temp is not None else np.nan,
                            "fan_rpm": fan_rpm if fan_rpm is not None else np.nan,
                            "cpu_load": cpu_load if cpu_load is not None else np.nan,
                        }
                    )

                    pct = int((i + 1) / total_steps * 100)
                    progress.progress(
                        pct, text=f"Collecting Windows Live telemetry… {pct}%"
                    )
                    status_txt.write(
                        f"Elapsed: {int(time.time()-t0)}s / {duration_int}s"
                    )

                    next_tick = t0 + (i + 1) / max(1, fs_int)
                    time.sleep(max(0.0, next_tick - time.time()))

            # Finalize DataFrame, drop never-seen sensors, compute ambient proxy
            df = pd.DataFrame(recs)
            df = _drop_never_seen(df)

            if "cpu_temp" in df.columns and df["cpu_temp"].notna().any():
                ambient_roll = max(1, fs_int * 300)  # ~5 min rolling min (in samples)
                amb_proxy = (
                    df["cpu_temp"].rolling(window=ambient_roll, min_periods=1).min()
                    - 10
                ).clip(lower=15, upper=40)
                df["ambient"] = amb_proxy.values
            else:
                df["ambient"] = 30.0

            progress.empty()
            status_txt.empty()
            meta = {"mode": "windows_live"}
# ------------------------- Model Training/Loading -------------------------
# Build features (fail fast if no windows produced)
feats, _ = build_feature_table(df, fs=fs, window_s=win, hop_s=hop)

if feats is None or feats.empty or ("t_start" not in feats.columns):
    fs_int = int(fs)
    win_s = int(win)
    hop_s = int(hop)
    dur_s = int(dur)
    warmup_s = int(warmup)
    st.error(
        "Not enough data to build feature windows.\n"
        f"- Sampling: {fs_int} Hz, Duration: {dur_s}s, Window: {win_s}s, Hop: {hop_s}s, Warm-up: {warmup_s}s\n"
        f"- To get at least 1 window, set **Duration ≥ Window (≥ {win_s}s)**.\n"
        "- Quick test (live): Duration=90s, Window=60s, Hop=10s, Warm-up=30s."
    )
    st.stop()

st.caption(f"Built {len(feats)} windows (window={win}s, hop={hop}s).")

# Use only real, non-all-NaN feature columns with the expected prefixes
_prefixes = ("cpu_", "gpu_", "fan_", "load_", "x_")
feature_cols = [
    c for c in feats.columns if c.startswith(_prefixes) and feats[c].notna().any()
]
if not feature_cols:
    st.error(
        "No usable feature columns found (all-NaN). Try Simulated mode or ensure CPU temp/load are available."
    )
    st.stop()

model = None
calib = None

# Model loading/training
if load_btn and os.path.exists(model_path):
    blob = load_model(model_path)
    model = blob.get("model")
    calib = blob.get("calib")
    st.success("✅ Model loaded successfully from disk.")
else:
    # Sanity hint if warm-up is less than window (yields zero healthy windows)
    if warmup < win:
        st.warning(
            f"Warm-up ({warmup}s) is less than Window ({win}s) → 0 healthy windows for training. "
            f"Increase Warm-up to ≥ {win}s or Load a saved model."
        )

    healthy_mask = feats["t_start"] < warmup
    X_train = feats.loc[healthy_mask, feature_cols]

    if len(X_train) < 10:
        # How to hit 10 healthy windows: warmup_needed = window + 9*hop
        warmup_needed = int(win + 9 * hop)
        st.error(
            "❌ Insufficient healthy windows to train (need ≥ 10).\n"
            f"- Found: {len(X_train)} healthy windows\n"
            f"- Set **Warm-up ≥ {warmup_needed}s** (for Window={win}s, Hop={hop}s), "
            "or Load a saved model, or extend Duration."
        )
        st.stop()

    model = train_iforest(X_train)
    # Conservative calibration from healthy scores (50th..99th pct)
    train_scores = -model.decision_function(X_train)
    lo, hi = np.percentile(train_scores, [50, 99])
    calib = {"lo": float(lo), "hi": float(hi)}

# Score calculation and processing
scores = -model.decision_function(feats.loc[:, feature_cols])
if calib:
    lo, hi = calib["lo"], calib["hi"]
    severity = 100 * (scores - lo) / (hi - lo + 1e-9)
    severity = np.clip(severity, 0, 100)
else:
    severity = severity_from_scores(scores)

feats["anom_score"] = scores
# EWMA smoothing for stability
feats["severity"] = pd.Series(severity).ewm(alpha=0.2).mean().values
# Raw bands (hysteresis will consume this next)
feats["status_raw"] = feats["severity"].apply(
    lambda s: band_status(s, green_hi, yellow_hi)
)

# Hysteresis processing
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
    cooloff_win=max(1, int(60 / max(1, hop))),
)
feats["status"] = status_hyst


# Normalize status for UI ("green/yellow/red") while keeping human label (e.g., "🟢 Healthy")
def _status_simple(s: str) -> str:
    if "Healthy" in s:
        return "green"
    if "Warning" in s:
        return "yellow"
    if "Faulty" in s:
        return "red"
    return "gray"


feats["status_simple"] = feats["status"].map(_status_simple)

# Save logs
os.makedirs("logs", exist_ok=True)
feats.to_csv("logs/features_scores.csv", index=False)

# ------------------------- Results Dashboard -------------------------
st.markdown("---")

# Current Status Section
st.markdown("## 🎯 Current System Status")

last = feats.iloc[-1]
current_status_label = str(last["status"])  # e.g., "🟢 Healthy"
current_status_simple = str(last["status_simple"])  # "green" | "yellow" | "red"
current_severity = int(last["severity"])

status_emojis = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
status_emoji = status_emojis.get(current_status_simple, "⚪")

# Create 4 columns for the status metrics
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.markdown(
        f"""
    <div class="status-card status-{current_status_simple}">
        <h2>{status_emoji} System Status: {current_status_label}</h2>
        <p>Severity Level: {current_severity}/100</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with status_col2:
    st.metric("🌡 Analysis Windows", f"{len(feats)}")

with status_col3:
    if "cpu_temp" in df.columns and df["cpu_temp"].notna().any():
        st.metric("🔥 Avg CPU Temp", f"{df['cpu_temp'].dropna().mean():.1f}°C")
    else:
        st.metric("🔥 Avg CPU Temp", "N/A")

with status_col4:
    if "fan_rpm" in df.columns and df["fan_rpm"].notna().any():
        st.metric("💨 Avg Fan Speed", f"{df['fan_rpm'].dropna().mean():.0f} RPM")
    else:
        st.metric("💨 Avg Fan Speed", "N/A")

# Action Suggestions
st.markdown("### 💡 Recommended Actions")
suggestions = suggest_actions(
    last,
    feats,
    ambient=df["ambient"].iloc[-1] if "ambient" in df else 25.0,
    green_hi=green_hi,
    yellow_hi=yellow_hi,
)


def get_suggestion_style():
    # Streamlit dark mode detection via st.get_option (works in Streamlit >=1.25)
    theme = st.get_option("theme.base")
    if theme == "dark":
        return "background-color:#0b1220; color:#e5e7eb; border-left:4px solid #3b82f6; padding:0.75rem; border-radius:8px; margin:0.5rem 0;"
    else:
        return "background-color:#f1f5f9; color:#1e40af; border-left:4px solid #3b82f6; padding:0.75rem; border-radius:8px; margin:0.5rem 0;"


if suggestions:
    style = get_suggestion_style()
    for i, suggestion in enumerate(suggestions[:3]):
        st.markdown(
            f"""
        <div style="{style}">
            <strong>{i+1}.</strong> {suggestion}
        </div>
        """,
            unsafe_allow_html=True,
        )
else:
    st.info("✅ System is operating normally. No immediate actions required.")
# --- Episodes table (dark-mode friendly, column-guarded) ---
if events:
    st.markdown("### 🚨 Detected Episodes")
    events_df = pd.DataFrame(events)

    def _styler(df: pd.DataFrame):
        # If no 'status' column present, skip coloring gracefully
        if "status" not in df.columns:
            return df.style

        def _color(v):
            m = str(v).lower()
            if "red" in m:
                return "background-color:#7f1d1d; color:#fde68a;"
            if "yellow" in m:
                return "background-color:#78350f; color:#fde68a;"
            if "green" in m:
                return "background-color:#064e3b; color:#e5e7eb;"
            return ""

        return df.style.applymap(_color, subset=["status"])

    st.dataframe(_styler(events_df), use_container_width=True)

# ------------------------- Visual Analytics Tabs -------------------------
st.markdown("---")
st.markdown("## 📊 System Analytics")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🌡 Temperature Trends",
        "📈 Performance Metrics",
        "⏰ Timeline Analysis",
        "📁 Export Data",
    ]
)

with tab1:
    st.markdown("### Temperature and Fan Monitoring")

    # Create subplots for temperature and fan data
    # Use 't' column if present; otherwise fallback to index
    x_axis = df["t"] if "t" in df.columns else df.index

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Temperature Monitoring", "Fan Performance"),
        vertical_spacing=0.1,
        shared_xaxes=True,
    )

    # CPU temp
    if "cpu_temp" in df.columns and df["cpu_temp"].notna().any():
        fig.add_trace(
            go_plotly.Scatter(
                x=x_axis,
                y=df["cpu_temp"],
                name="CPU Temp",
                line=dict(color="#ef4444", width=2),
            ),
            row=1,
            col=1,
        )

    # GPU temp (if available)
    if "gpu_temp" in df.columns and df["gpu_temp"].notna().any():
        fig.add_trace(
            go_plotly.Scatter(
                x=x_axis,
                y=df["gpu_temp"],
                name="GPU Temp",
                line=dict(color="#f59e0b", width=2),
            ),
            row=1,
            col=1,
        )

    # Fan RPM (if available)
    if "fan_rpm" in df.columns and df["fan_rpm"].notna().any():
        fig.add_trace(
            go_plotly.Scatter(
                x=x_axis,
                y=df["fan_rpm"],
                name="Fan RPM",
                line=dict(color="#3b82f6", width=2),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=600, showlegend=True, title_text="System Thermal Performance"
    )
    fig.update_xaxes(
        title_text="Time (s)" if "t" in df.columns else "Samples", row=2, col=1
    )
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="RPM", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### System Load and Performance")

    # CPU Load chart
    if "cpu_load" in df.columns:
        fig_load = px.area(
            df,
            x=x_axis,
            y="cpu_load",
            title="CPU Load Over Time",
            color_discrete_sequence=["#8b5cf6"],
        )
        fig_load.update_layout(
            xaxis_title="Time (s)" if "t" in df.columns else "Samples",
            yaxis_title="CPU Load (%)",
        )
        st.plotly_chart(fig_load, use_container_width=True)

    # Severity over time
    fig_severity = px.line(
        feats,
        x="t_start",
        y="severity",
        title="Anomaly Severity Timeline",
        color_discrete_sequence=["#ec4899"],
    )

    # Add threshold lines
    fig_severity.add_hline(
        y=green_hi,
        line_dash="dash",
        line_color="green",
        annotation_text="Normal Threshold",
    )
    fig_severity.add_hline(
        y=yellow_hi,
        line_dash="dash",
        line_color="orange",
        annotation_text="Warning Threshold",
    )

    fig_severity.update_layout(xaxis_title="Time (s)", yaxis_title="Severity Score")
    st.plotly_chart(fig_severity, use_container_width=True)

with tab3:
    st.markdown("### Detailed Analysis Timeline")

    # Build timeline table (include normalized status for styling)
    # assumes feats['status_simple'] was created earlier from feats['status']
    # (see earlier patch where we mapped "🟢 Healthy" -> "green", etc.)
    timeline_df = feats[
        ["t_start", "t_end", "status", "status_simple", "severity", "anom_score"]
    ].copy()
    timeline_df = timeline_df.round(2)

    def highlight_status(row):
        s = row["status_simple"]
        if s == "red":
            return ["background-color:#7f1d1d; color:#fde68a;"] * len(row)
        elif s == "yellow":
            return ["background-color:#78350f; color:#fde68a;"] * len(row)
        else:
            return ["background-color:#064e3b; color:#e5e7eb;"] * len(row)

    st.dataframe(
        timeline_df.style.apply(highlight_status, axis=1),
        use_container_width=True,
        height=400,
    )

    # Summary statistics (use normalized labels)
    st.markdown("#### 📋 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        green_count = (feats["status_simple"] == "green").sum()
        st.metric("🟢 Normal Windows", int(green_count))

    with col2:
        yellow_count = (feats["status_simple"] == "yellow").sum()
        st.metric("🟡 Warning Windows", int(yellow_count))

    with col3:
        red_count = (feats["status_simple"] == "red").sum()
        st.metric("🔴 Critical Windows", int(red_count))

    with col4:
        max_severity = float(feats["severity"].max())
        st.metric("⚠ Peak Severity", f"{max_severity:.1f}")

with tab4:
    st.markdown("### 📥 Export Analysis Results")

    # Export section with better layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Raw Data")
        st.download_button(
            label="📄 Download Signals CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"pc_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            label="🔍 Download Analysis CSV",
            data=feats.to_csv(index=False).encode("utf-8"),
            file_name=f"pc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("#### 📋 Reports")
        try:
            pdf_bytes = build_pdf_report(feats, df, green_hi, yellow_hi)
            st.download_button(
                label="📑 Download PDF Report",
                data=pdf_bytes,
                file_name=f"pc_cooling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"⚠ PDF report generation failed: {str(e)}")

    # Export summary
    st.info(
        f"""
    📈 *Export Summary:*
    - Analysis Duration: {dur} seconds
    - Total Data Points: {len(df)} measurements  
    - Analysis Windows: {len(feats)} windows
    - Current Status: {current_status_label}
    """
    )

# ------------------------- Model Management -------------------------
if save_btn:
    try:
        os.makedirs("models", exist_ok=True)
        save_model(model, calib, model_path)
        st.success(f"✅ Model saved successfully to {model_path}")
    except Exception as e:
        st.error(f"❌ Failed to save model: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    🖥 <strong>PC Cooling Predictive Maintenance</strong> | 
    Keep your system cool and running optimally
</div>
""",
    unsafe_allow_html=True,
)
