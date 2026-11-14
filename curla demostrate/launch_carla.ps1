$ErrorActionPreference = "Stop"

$Root = "C:\CARLA_0.9.16"
$ShippingExe = Join-Path $Root "CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe"
$LauncherExe = Join-Path $Root "CarlaUE4.exe"
$LogDir = Join-Path $Root "TrajPilotLogs"

Write-Host ""
Write-Host "=== TRAJPILOT: LAUNCH CARLA 0.9.16 ===" -ForegroundColor Cyan

if (!(Test-Path $ShippingExe)) {
    throw "Shipping executable missing: $ShippingExe"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Existing = Get-NetTCPConnection -State Listen -LocalPort 2000 -ErrorAction SilentlyContinue

if ($Existing) {
    Write-Host "Port 2000 is already listening. Check the server before launching another instance." -ForegroundColor Yellow
    $Existing | Select-Object LocalAddress, LocalPort, OwningProcess | Format-Table
    exit 0
}

Write-Host "Launching CARLA in off-screen rendering mode..."
Write-Host "Log directory: $LogDir"

$Arguments = @(
    "-RenderOffScreen",
    "-quality-level=Low",
    "-carla-rpc-port=2000",
    "-nosound",
    "-log"
)

$Process = Start-Process `
    -FilePath $ShippingExe `
    -ArgumentList $Arguments `
    -WorkingDirectory (Join-Path $Root "CarlaUE4") `
    -RedirectStandardOutput (Join-Path $LogDir "stdout.txt") `
    -RedirectStandardError (Join-Path $LogDir "stderr.txt") `
    -PassThru

Write-Host "Started process ID: $($Process.Id)"
Write-Host "Waiting up to 180 seconds for CARLA to initialize..."

$Deadline = (Get-Date).AddSeconds(180)

do {
    Start-Sleep -Seconds 3

    $Process.Refresh()

    if ($Process.HasExited) {
        Write-Host ""
        Write-Host "CARLA exited with code $($Process.ExitCode)" -ForegroundColor Red
        Write-Host "Inspect logs in: $LogDir"
        exit 1
    }

    $Listener = Get-NetTCPConnection -State Listen -LocalPort 2000 -ErrorAction SilentlyContinue

    if ($Listener) {
        Write-Host ""
        Write-Host "Port 2000 is listening." -ForegroundColor Green
        break
    }
} while ((Get-Date) -lt $Deadline)

if (!$Listener) {
    Write-Host ""
    Write-Host "CARLA has not opened port 2000 yet." -ForegroundColor Yellow
    Write-Host "Do not launch another instance. Inspect the logs and process."
    Get-Process -Id $Process.Id -ErrorAction SilentlyContinue |
        Select-Object Id, ProcessName, CPU, WorkingSet64 |
        Format-Table
    exit 2
}

Write-Host ""
Write-Host "Launch check passed: CARLA process is alive and port 2000 is listening." -ForegroundColor Green
Write-Host "This does not yet prove the Python client can connect."
