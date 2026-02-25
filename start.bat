@echo off
REM CellCount - Hemocytometer Cell Counter
REM Launch script for Windows
REM ==========================================

title CellCount Server

echo.
echo  ============================================
echo   CellCount - Hemocytometer Cell Counter
echo  ============================================
echo.

REM Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Determine project root (where this script lives)
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

REM Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Using virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No .venv found. Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r webapp\backend\requirements.txt
    echo.
)

REM Check that model weights exist
if not exist "checkpoints\yolo\nano\weights\best.pt" (
    echo WARNING: No model weights found in checkpoints\yolo\
    echo The app will fail to start without trained model weights.
    echo Copy the checkpoints\ folder from the development machine.
    pause
    exit /b 1
)

REM Set environment
set PYTHONPATH=%PROJECT_ROOT%webapp\backend

echo Starting CellCount server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir "%PROJECT_ROOT%webapp\backend"

pause
