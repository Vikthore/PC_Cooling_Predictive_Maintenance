# Smart Predictive Maintenance for PC Cooling — v3 (Ideal/Optimal)

**What’s new in v3**
- **Live Windows telemetry** via **OpenHardwareMonitor** WMI (optional). Simulator remains for dev.
- **Unified DAQ**: choose **Simulated** or **Windows Live** in the UI or agent config.
- **Model persistence**: Save/Load IsolationForest with **severity calibration** so the agent can run without warm-up.
- **Agent modes**: `sim` or `live`, desktop notifications, CSV logging, PDF report (unchanged).

## Quickstart (Windows PowerShell)
```powershell
# unzip, then in this folder:
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```
**Live mode prerequisites (Windows):**
1. Download & run **OpenHardwareMonitor** (or LibreHardwareMonitor with OHM WMI compatibility).
2. Leave it running; it publishes WMI at `root\OpenHardwareMonitor` while active.
3. In the app, select **Windows Live**. If sensors are missing, it falls back to Simulation.

## Background agent
```powershell
# Edit agent_config.json (mode: "live" or "sim"), then:
.\.venv\Scripts\python agent.py
```
- Logs: `logs/features_scores.csv`, `logs/events.csv`
- Model: loads from `models/if_model.joblib` if present; otherwise trains on warm-up windows

## Files
- `app.py` — Streamlit UI (Sim/Live, Save/Load model, PDF report)
- `agent.py` — background monitor (Sim/Live)
- `agent_config.json` — settings (mode, thresholds, windows, notify, DAQ options)
- `cooling/daq.py` — unified data acquisition (Sim + Windows WMI)
- `cooling/sim.py` — simulator used by DAQ
- `cooling/features.py` — rolling window features
- `cooling/model.py` — IF model + severity calibration save/load
- `cooling/rules.py` — suggestion engine
- `cooling/report.py` — PDF report generator
- `requirements.txt` — deps (incl. `wmi`, `pywin32` for Windows live)
