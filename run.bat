@echo off
cd /d "%~dp0"

set "PY=%MT_PYTHON%"
if not defined PY set "PY=python"

"%PY%" -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] The current Python has no ultralytics installed:
    "%PY%" -c "import sys; print(sys.executable, sys.version)"
    echo.
    echo Fix it with one of these:
    echo   1. Point MT_PYTHON at a Python that has the deps:
    echo        set MT_PYTHON=C:\path\to\python.exe
    echo   2. Or install the deps into the current Python:
    echo        "%PY%" -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM run the app, keep the window open on failure
"%PY%" main.py %*
if errorlevel 1 pause