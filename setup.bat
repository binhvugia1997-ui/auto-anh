@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install 64-bit Python 3.11 or newer from python.org.
  exit /b 1
)
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo Python 3.11 or newer is required. Install it and run setup.bat again.
  exit /b 1
)
py -3 -m venv .venv
if errorlevel 1 exit /b %errorlevel%
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%
pip install -r requirements.txt
if errorlevel 1 exit /b %errorlevel%
echo.
echo GlowUp Local is ready. Run run.bat to launch the app.
