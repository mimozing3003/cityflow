@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo TRAJPILOT CITYFLOW ^| MASTER CARLA 0.9.16 LAUNCHER
echo ==============================================================================

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: 1. Identify Python executable
set "PYTHON_EXE="
if exist "C:\Users\SOMUJIT\anaconda3\envs\carla_py312\python.exe" (
    set "PYTHON_EXE=C:\Users\SOMUJIT\anaconda3\envs\carla_py312\python.exe"
) else if exist "C:\Users\SOMUJIT\anaconda3\envs\carla0916\python.exe" (
    set "PYTHON_EXE=C:\Users\SOMUJIT\anaconda3\envs\carla0916\python.exe"
) else (
    echo [FAIL] Compatible Python environment not found!
    echo [CAUSE] Neither carla_py312 nor carla0916 was located.
    echo [FIX] Run install_carla_0916.ps1 to create the environment.
    pause
    exit /b 1
)

echo [PASS] Active Python: %PYTHON_EXE%

:: 2. Validate Project Assets
if not exist "%PROJECT_ROOT%data\models\gru_baseline\best_gru_baseline.pt" (
    echo [FAIL] Model checkpoint missing: data\models\gru_baseline\best_gru_baseline.pt
    pause
    exit /b 1
)
if not exist "%PROJECT_ROOT%data\processed\i24_sequences\gru_inputs\train_normalization_statistics.csv" (
    echo [FAIL] Normalization stats missing: data\processed\i24_sequences\gru_inputs\train_normalization_statistics.csv
    pause
    exit /b 1
)
echo [PASS] TrajPilot project assets verified.

:: 3. Check and Launch CARLA Server if not already running
echo Checking CARLA Simulator on port 2000...
powershell -Command "$c = Test-NetConnection -ComputerName 127.0.0.1 -Port 2000 -InformationLevel Quiet; if ($c) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo Port 2000 is closed. Starting CARLA Simulator...
    if exist "C:\CARLA_0.9.16\CarlaUE4.exe" (
        start "" "C:\CARLA_0.9.16\CarlaUE4.exe" -quality-level=Epic -RenderOffScreen
        echo Waiting for CARLA server to initialize on port 2000...
        set "WAIT_COUNT=0"
        :wait_loop
        timeout /t 2 /nobreak >nul
        powershell -Command "$c = Test-NetConnection -ComputerName 127.0.0.1 -Port 2000 -InformationLevel Quiet; if ($c) { exit 0 } else { exit 1 }"
        if errorlevel 1 (
            set /a WAIT_COUNT+=2
            if !WAIT_COUNT! geq 45 (
                echo [FAIL] CARLA Simulator failed to open port 2000 within 45 seconds.
                echo [CAUSE] CarlaUE4.exe may have encountered an initialization error.
                echo [COMMAND] Check C:\CARLA_0.9.16\CarlaUE4\Saved\Logs\CarlaUE4.log
                pause
                exit /b 1
            )
            echo Waiting... (!WAIT_COUNT!s)
            goto wait_loop
        )
        echo [PASS] CARLA Simulator is now listening on port 2000!
    ) else (
        echo [FAIL] CARLA installation not found at C:\CARLA_0.9.16\CarlaUE4.exe
        echo [FIX] Run install_carla_0916.ps1 to complete CARLA installation.
        pause
        exit /b 1
    )
) else (
    echo [PASS] CARLA Simulator already running on port 2000.
)

:: 4. Run Stage 00 Environment & Asset Preflight
"%PYTHON_EXE%" carla_demo\00_environment_check.py
if errorlevel 1 (
    echo [FAIL] Environment preflight check failed.
    pause
    exit /b 1
)

:: 5. Launch Interactive TrajPilot CARLA Demonstration
echo ==============================================================================
echo Launching Live Demonstration...
echo ==============================================================================
"%PYTHON_EXE%" carla_demo\08_run_demo.py
if errorlevel 1 (
    echo [FAIL] Demo exited with error code %errorlevel%.
    pause
    exit /b 1
)

echo ==============================================================================
echo TRAJPILOT CARLA RUN COMPLETED SUCCESSFULLY
echo ==============================================================================
pause
