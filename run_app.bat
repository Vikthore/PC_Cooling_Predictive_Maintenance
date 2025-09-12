@echo off
setlocal
cd /d %~dp0
set VENV=.venv
if not exist %VENV%\Scripts\python.exe (
  echo Virtual environment not found. Run setup_client.ps1 first.
  pause
  exit /b 1
)
set PATH=%CD%\%VENV%\Scripts;%PATH%
"%VENV%\Scripts\python.exe" -m streamlit run app.py
