@echo off
REM =============================================================================
REM WakeGen TTS Docker Stack - Stop Helper Script (Windows)
REM =============================================================================
REM USAGE:
REM   stop.bat              - Stop all providers
REM   stop.bat piper        - Stop only Piper
REM   stop.bat clean        - Stop all and remove volumes
REM =============================================================================

setlocal EnableDelayedExpansion

REM Change to docker directory
cd /d "%~dp0"

REM Default values
set PROFILE=full
set CLEAN=0

REM Parse arguments
:parse_args
if "%~1"=="" goto :run
if /i "%~1"=="clean" (
    set CLEAN=1
) else (
    set PROFILE=%~1
)
shift
goto :parse_args

:run
echo.
echo ===============================================
echo  WakeGen TTS Docker Stack - Stopping
echo ===============================================
echo  Profile: %PROFILE%
echo  Clean:   %CLEAN%
echo ===============================================
echo.

if %CLEAN%==1 (
    echo WARNING: This will delete all cached models!
    set /p CONFIRM="Are you sure? (y/N): "
    if /i "!CONFIRM!"=="y" (
        docker compose --profile %PROFILE% down -v
    ) else (
        echo Aborted.
        goto :eof
    )
) else (
    docker compose --profile %PROFILE% down
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo Containers stopped successfully.
) else (
    echo.
    echo ERROR: Failed to stop containers
)

endlocal
