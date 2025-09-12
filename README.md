# Smart Predictive Maintenance for PC Cooling Systems

A Streamlit web app that applies **machine learning** to monitor PC cooling performance, detect faults, and suggest proactive maintenance.

It works in two modes:

- **Simulation** – synthetic temperature, load, and fan data.
- **Windows Live** – real-time telemetry via [OpenHardwareMonitor](https://openhardwaremonitor.org/) on Windows.

---

## 📁 Project Structure

```
pc_cooling_pm/
│
├── app.py                     # Main Streamlit application
├── agent.py                   # Background monitor/alert agent
├── requirements.txt           # Base dependencies
├── requirements-windows.txt   # Windows-specific dependencies
│
├── cooling/
│   ├── daq.py                 # Data acquisition (sim + live)
│   ├── features.py            # Windowing & feature engineering
│   ├── model.py               # IsolationForest & calibration
│   ├── hysteresis.py          # Status stability
│   ├── rules.py               # Maintenance suggestions
│   └── report.py              # PDF report builder
│
├── tools/
│   ├── OpenHardwareMonitor/   # Portable OHM (auto-downloaded)
│   └── wmi_check.bat          # WMI/CIM sanity check
│
├── run_app.bat                # Launch app with venv
├── run_agent.bat              # Launch agent
└── setup_client.ps1           # One-time setup helper
```

---

## 🚀 Quick Start

### 1️⃣ Clone / download

```bash
git clone https://github.com/your-org/pc_cooling_pm.git
cd pc_cooling_pm
```

### 2️⃣ Windows setup (for Windows Live)

Run once per machine:

1. Right-click **setup_client.ps1** → “Run with PowerShell”  
   - Creates `.venv`, installs dependencies, downloads OHM.
2. Launch `tools/OpenHardwareMonitor/OpenHardwareMonitor.exe` as **Administrator**.
   - Check sensors: “CPU Package Temperature”, “CPU Total Load”.
3. Keep OHM running in the background.

### 3️⃣ Start the app

- Double-click `run_app.bat`, or:
  ```powershell
  .venv\Scripts\activate
  streamlit run app.py
  ```

---

## 🖥️ Using the App

1. **Source** → choose `Simulated` or `Windows Live` (only on Windows).  
2. Configure parameters in the sidebar:  
   - Duration, Sampling, Warm-up, Window, Hop  
   - Severity bands (green / yellow / red)  
   - Model load/save
3. Click **Collect & Detect**.  
   - A spinner shows data capture progress.
4. Review:
   - Signals chart (CPU/GPU temps, fan speed, load)
   - Status card & severity
   - Suggestions
   - Timeline of anomalies
5. Download CSV / PDF or save the model.

---

## 🧠 Under the Hood

1. **Collect** telemetry (Simulated or WMI/CIM).  
2. **Window & extract features**.  
3. Train or load an **Isolation Forest**.  
4. Score windows → map to 0–100 severity.  
5. Apply **EWMA smoothing** and **hysteresis** for stable status.  
6. Suggest maintenance based on severity & trends.

---

## ⚙️ Parameter Hints

| Param | Role |
|-------|------|
| Duration | Total capture time |
| Sampling | Poll rate (Hz) |
| Warm-up | Healthy training period |
| Window / Hop | Feature window size & stride |
| Severity bands | Color thresholds |
| Hysteresis dwell/backoff | Windows before changing state |

> For ≥10 healthy windows: `Duration ≥ Window + 9 × Hop`.

---

## 🪟 Windows Live Mode

- Windows only; requires Admin rights.  
- Needs OpenHardwareMonitor running.  
- WMI is preferred; CIM fallback runs at 1 Hz for reliability.  
- Spinner shows progress during capture.

---

## 🧰 Troubleshooting

| Issue | Solution |
|-------|-----------|
| `No data collected` | OpenHardwareMonitor must run as Admin |
| `com_error` | Restart OHM, rerun setup_client.ps1 |
| CIM too slow | Sampling auto-limited to 1 Hz |
| White rows in dark mode | Keep dark-mode CSS in `app.py` |
| Few healthy windows | Increase Warm-up or Duration, or load a model |

---

## 📜 License

MIT License.  
Telemetry via [OpenHardwareMonitor](https://openhardwaremonitor.org/).  
Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), [Plotly](https://plotly.com/).

---

✅ **Usage summary:**  
Start OHM → run `run_app.bat` → choose *Windows Live* → click **Collect & Detect**.
