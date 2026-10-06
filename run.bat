@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment not found. Run setup.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" -m glowup_local
