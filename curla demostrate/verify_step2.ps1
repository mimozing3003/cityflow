$ErrorActionPreference = "Stop"

$CarlaRoot = "C:\CARLA_0.9.16"
$ProjectRoot = "C:\Users\SOMUJIT\Desktop\cityflow"
$DemoRoot = Join-Path $ProjectRoot "curla demostrate"

Write-Host ""
Write-Host "=== STEP 2: VERIFY CARLA 0.9.16 ===" -ForegroundColor Cyan

Write-Host ""
Write-Host "[1] Simulator executable"
$Exe = Join-Path $CarlaRoot "CarlaUE4.exe"

if (!(Test-Path $Exe)) {
    throw "CarlaUE4.exe not found at $Exe"
}

$File = Get-Item $Exe
Write-Host "Executable: $($File.FullName)"
Write-Host "Size MB: $([math]::Round($File.Length / 1MB, 2))"
Write-Host "File version: $($File.VersionInfo.FileVersion)"

Write-Host ""
Write-Host "[2] Python API directory"
$ApiRoot = Join-Path $CarlaRoot "PythonAPI"
if (!(Test-Path $ApiRoot)) {
    throw "PythonAPI directory not found"
}

Get-ChildItem $ApiRoot -Filter "*.whl" -Recurse -ErrorAction SilentlyContinue |
    Select-Object FullName, Length |
    Format-Table -AutoSize

Write-Host ""
Write-Host "[3] CARLA Python client"
conda run -n carla0916 python -c "import sys, carla; print('Python:', sys.version); print('Client module:', carla.__file__); print('Client import: PASS')"

if ($LASTEXITCODE -ne 0) {
    throw "CARLA Python client import failed in carla0916"
}

Write-Host ""
Write-Host "[4] Installed package metadata"
conda run -n carla0916 python -m pip show carla

Write-Host ""
Write-Host "[5] Existing archive"
$Zip = Join-Path $env:USERPROFILE "Downloads\CARLA_0.9.16.zip"

if (Test-Path $Zip) {
    $ZipInfo = Get-Item $Zip
    Write-Host "Archive: $($ZipInfo.FullName)"
    Write-Host "Size GB: $([math]::Round($ZipInfo.Length / 1GB, 2))"
} else {
    Write-Host "Archive not found at expected path"
}

Write-Host ""
Write-Host "[6] Demo workspace"
Write-Host "Path: $DemoRoot"
Write-Host "Exists: $(Test-Path $DemoRoot)"

Write-Host ""
Write-Host "=== STEP 2 CHECKS FINISHED ===" -ForegroundColor Green
