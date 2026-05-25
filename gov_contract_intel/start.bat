@echo off
REM -----------------------------------------------------------------------------
REM start.bat — one-click setup + launch for Gov Contract Intel (Windows)
REM -----------------------------------------------------------------------------
cd /d "%~dp0"

set VENV=.venv
set REQUIREMENTS=requirements.txt

REM ── 1. Check Python ──────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.10+ from https://python.org and retry.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo Python %PY_VER% detected.

REM ── 2. Create venv if missing ─────────────────────────────────────────────────
if not exist "%VENV%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV%
)

REM ── 3. Activate ───────────────────────────────────────────────────────────────
call %VENV%\Scripts\activate.bat

REM ── 4. Install / update dependencies ─────────────────────────────────────────
if not exist "%VENV%\.deps_installed" (
    echo Installing dependencies (first run may take a few minutes)...
    pip install --upgrade pip --quiet
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    pip install -r %REQUIREMENTS% --quiet
    type nul > "%VENV%\.deps_installed"
    echo Dependencies installed.
)

REM ── 5. Check required model files ────────────────────────────────────────────
if not exist "models\xgb_amount_model.joblib" (
    echo WARNING: Missing models\xgb_amount_model.joblib
)
if not exist "models\xgb_modification_model.joblib" (
    echo WARNING: Missing models\xgb_modification_model.joblib
)
if not exist "models\all-MiniLM-L6-v2" (
    echo NOTE: Embedding model not found locally -- will download ~87 MB on first use.
)

REM ── 6. Ensure data directory exists ──────────────────────────────────────────
if not exist "data" mkdir data

REM ── 7. Launch ─────────────────────────────────────────────────────────────────
echo.
echo Starting Gov Contract Intel at http://localhost:5001
echo Press Ctrl+C to stop.
echo.
python app.py
pause
