# PC Cooling Predictive Maintenance — Client Guide

## One-time setup
1. Right-click **setup_client.ps1** → *Run with PowerShell*.
2. Open **tools/OpenHardwareMonitor/OpenHardwareMonitor.exe** as **Administrator**.
   - Confirm you see **CPU Package** (Temperature) and **CPU Total** (Load).
3. (Optional) Double-click **tools/wmi_check.bat** to verify WMI output.

## Run the UI (Windows Live)
- Double-click **run_app.bat**.
- In the sidebar:
  - Source: **Windows Live**
  - Duration: **120–300 s** (for a quick test)
  - Sampling: **1 Hz**
  - Window/Hop: **60 / 10**
  - Warm-up: **30–60 s**
- Click **Collect & Detect**. Wait for the progress bar to hit **100%**.
- Click **Save model** after a healthy run to reuse calibration.

## Background agent (optional)
- Double-click **run_agent.bat** (logs to `./logs`, sends Windows notifications on sustained faults).

## Notes
- Keep OpenHardwareMonitor running while using Windows Live.
- If Fan RPM isn’t exposed on your hardware, the app still works with temp/load.
