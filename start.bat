@echo off
setlocal
title WakeGen Starter

REM Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found.
    echo Please run 'install.bat' first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo ===============================================================================
echo  Starting WakeGen Web UI...
echo ===============================================================================
echo Access the UI at: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server.
echo.

REM SYNTAX: 'start ""' launches a new process in the background.
REM WHY empty quotes: The 'start' command interprets the first quoted string as the window title.
REM We give it an empty title ("") so the URL is correctly interpreted as the command to run.
REM HOW: 'timeout /t 2 /nobreak >nul' waits 2 seconds silently, then opens the browser.
REM This gives the server a moment to initialize before the browser connects.
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

REM Start the server directly (not in background)
REM Running uvicorn directly allows Ctrl+C to terminate it properly
python -m uvicorn wakegen.web.app:create_app --factory --reload --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
endlocal
