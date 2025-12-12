@echo off
REM =============================================================================
REM WakeGen TTS Docker Stack - Start Helper Script (Windows)
REM =============================================================================
REM USAGE:
REM   start.bat              - Start all providers
REM   start.bat piper        - Start only Piper
REM   start.bat gpu          - Start all with GPU support
REM   start.bat piper gpu    - Start Piper with GPU support
REM =============================================================================

setlocal EnableDelayedExpansion

REM Change to docker directory
cd /d "%~dp0"

REM Default values
set PROFILE=full
set USE_GPU=0

REM Parse arguments
:parse_args
if "%~1"=="" goto :run
if /i "%~1"=="gpu" (
    set USE_GPU=1
) else (
    set PROFILE=%~1
)
shift
goto :parse_args

:run
echo.
echo ===============================================
echo  WakeGen TTS Docker Stack
echo ===============================================
echo  Profile: %PROFILE%
echo  GPU: %USE_GPU%
echo ===============================================
echo.

REM Build the docker compose command
if %USE_GPU%==1 (
    echo Starting with GPU support...
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile %PROFILE% up -d
) else (
    echo Starting in CPU mode...
    docker compose --profile %PROFILE% up -d
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo ===============================================
    echo  Containers started successfully!
    echo ===============================================
    echo.
    echo Access the services at:
    echo   Edge TTS:    http://localhost:5000
    echo   Piper:       http://localhost:5001
    echo   Coqui XTTS:  http://localhost:5002
    echo   Bark:        http://localhost:5003
    echo   ChatTTS:     http://localhost:5004
    echo   Kokoro:      http://localhost:5005
    echo   Mimic3:      http://localhost:5006
    echo   F5-TTS:      http://localhost:5007
    echo   StyleTTS2:   http://localhost:5008
    echo   Orpheus:     http://localhost:5009
    echo.
    echo Run 'docker compose ps' to see container status
) else (
    echo.
    echo ERROR: Failed to start containers
    echo Check 'docker compose logs' for details
)

endlocal
