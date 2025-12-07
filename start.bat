@echo off
setlocal
title WakeGen Starter

REM Check if virtual environment exists
if not exist ".venv" (
    echo [ERROR] Virtual environment not found.
    echo Please run 'install.bat' first.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo ===============================================================================
echo  Starting WakeGen Web UI...
echo ===============================================================================
echo Access the UI at: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server.
echo.

REM Start the server in reload mode for development
python -m uvicorn wakegen.web.app:create_app --factory --reload --host 127.0.0.1 --port 8000

if %errorlevel% neq 0 (
    echo [ERROR] Server crashed or failed to start.
    pause
)

pause
