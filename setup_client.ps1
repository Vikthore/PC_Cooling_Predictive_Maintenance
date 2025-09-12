# setup_client.ps1 — one-time client setup for Windows Live
$ErrorActionPreference = "Stop"
Write-Host "== PC Cooling: Windows setup =="

# Tip: if PowerShell blocks this script, run this once in the same window:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 1) Create venv (.venv) if missing
if (-not (Test-Path ".\.venv")) {
  Write-Host "Creating virtual environment .venv ..."
  py -3 -m venv .venv
}

# 2) Activate venv & upgrade pip
$env:VIRTUAL_ENV = (Resolve-Path ".\.venv").Path
$env:Path = "$($env:VIRTUAL_ENV)\Scripts;$env:Path"
python -m pip install --upgrade pip setuptools wheel

# 3) Install base + Windows overlay
if (Test-Path ".\requirements-windows.txt") {
  python -m pip install -r requirements-windows.txt
} else {
  Write-Host "requirements-windows.txt not found. Aborting." -ForegroundColor Red
  exit 1
}

# 4) pywin32 postinstall (quiet)
try { python -m pywin32_postinstall -install } catch { }

# 5) Fetch OpenHardwareMonitor portable (if not already present)
New-Item -ItemType Directory -Force -Path ".\tools" | Out-Null
$ohmZip = ".\tools\OpenHardwareMonitor.zip"
$ohmDir = ".\tools\OpenHardwareMonitor"

if (-not (Test-Path $ohmDir)) {
  Write-Host "Downloading OpenHardwareMonitor (portable) ..."
  try {
    Invoke-WebRequest -UseBasicParsing -Uri "https://openhardwaremonitor.org/files/openhardwaremonitor-v0.9.6.zip" -OutFile $ohmZip
    Expand-Archive -Path $ohmZip -DestinationPath $ohmDir -Force
  } catch {
    Write-Warning "Auto-download failed (proxy or firewall?)."
    Write-Warning "Manually download the zip from openhardwaremonitor.org and place it at: $ohmZip"
  }
}

# 6) Quick WMI sanity (optional)
Write-Host "`nSanity check (optional): Windows WMI modules installed."
python - << 'PY'
try:
    import wmi, sys
    print("Python:", sys.executable)
    print("wmi OK:", wmi.__file__)
except Exception as e:
    print("WMI import failed:", e)
PY

Write-Host "`nSetup complete."
Write-Host "Next:"
Write-Host "  1) Open '.\\tools\\OpenHardwareMonitor\\OpenHardwareMonitor.exe' (Run as Administrator)."
Write-Host "  2) Confirm CPU Package Temperature & CPU Total Load appear."
Write-Host "  3) Double-click 'run_app.bat' to start the UI (or 'run_agent.bat' for background alerts)."
