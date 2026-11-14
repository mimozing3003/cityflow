@echo off
setlocal enabledelayedexpansion
title TrajPilot CityFlow - Full City Autonomous Roaming

set "PROJECT=C:\Users\SOMUJIT\Desktop\cityflow"
set "DEMO=%PROJECT%\curla demostrate"
set "PYTHON=C:\Users\SOMUJIT\anaconda3\envs\carla0916\python.exe"
set "CARLA_ROOT=C:\CARLA_0.9.16"
set "CARLA_EXE=%CARLA_ROOT%\CarlaUE4.exe"
set "SCRIPT=%DEMO%\trajpilot_city_roaming.py"

echo ============================================================
echo   TRAJPILOT CITYFLOW - FULL-CITY AUTONOMOUS ROAMING
echo ============================================================
echo.

if not exist "%PYTHON%" (
    echo ERROR: Python environment not found:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo ERROR: Demo script not found:
    echo %SCRIPT%
    pause
    exit /b 1
)

if not exist "%CARLA_EXE%" (
    echo ERROR: CARLA executable not found:
    echo %CARLA_EXE%
    pause
    exit /b 1
)

echo Checking CARLA Simulator on port 2000...
"%PYTHON%" -c "import socket; s=socket.socket(); s.settimeout(1.0); s.connect(('127.0.0.1',2000)); s.close()" 2>nul

if errorlevel 1 (
    echo.
    echo Port 2000 is closed. Starting CARLA Simulator window...
    start "" /d "%CARLA_ROOT%" "%CARLA_EXE%" -quality-level=Low -windowed -ResX=1280 -ResY=720 -carla-rpc-port=2000
    echo Waiting for CARLA to initialize on port 2000...
    set "WAIT_SECONDS=0"
    :wait_carla_loop
    timeout /t 2 /nobreak >nul
    "%PYTHON%" -c "import socket; s=socket.socket(); s.settimeout(1.0); s.connect(('127.0.0.1',2000)); s.close()" 2>nul
    if errorlevel 1 (
        set /a WAIT_SECONDS+=2
        echo   Initializing CARLA... (!WAIT_SECONDS!s / 60s)
        if !WAIT_SECONDS! geq 60 (
            echo.
            echo ERROR: CARLA Simulator failed to respond on port 2000 within 60 seconds.
            pause
            exit /b 1
        )
        goto wait_carla_loop
    )
    echo.
    echo CARLA Simulator window is now active and listening on port 2000!
    timeout /t 2 /nobreak >nul
) else (
    echo CARLA Simulator is already running and connected on port 2000.
)

echo.
echo ============================================================
echo Launching City Roaming: 30+ Vehicles + Autonomous Navigation
echo ============================================================
echo.

cd /d "%DEMO%"
"%PYTHON%" "%SCRIPT%"
set "RESULT=%ERRORLEVEL%"

echo.
echo ============================================================
echo Session finished with exit code: %RESULT%
echo Check summary: %DEMO%\roaming_demo_summary.json
echo ============================================================
echo.
pause
exit /b %RESULT%
