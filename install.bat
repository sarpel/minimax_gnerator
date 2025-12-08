@echo off
setlocal
title WakeGen Installer

echo ===============================================================================
echo  WakeGen One-Click Installer
echo ===============================================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org and try again.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists.
)

REM Activate virtual environment and install dependencies
echo [INFO] Installing dependencies...
call venv\Scripts\activate.bat
pip install -e .[web]

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo  Installation Complete!
echo ===============================================================================
echo You can now run 'start.bat' to launch the application.
echo.
pause
