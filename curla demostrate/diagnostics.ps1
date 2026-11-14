$Project = "C:\Users\SOMUJIT\Desktop\cityflow"
Write-Host ""
Write-Host "=== TRAJPILOT / CARLA READ-ONLY DIAGNOSTICS ===" -ForegroundColor Cyan

Write-Host ""
Write-Host "[1] Windows"
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture

Write-Host ""
Write-Host "[2] Disk space"
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | Select-Object DeviceID, @{Name="FreeGB";Expression={[math]::Round($_.FreeSpace / 1GB, 2)}}, @{Name="SizeGB";Expression={[math]::Round($_.Size / 1GB, 2)}}

Write-Host ""
Write-Host "[3] GPU"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} else {
    Write-Host "nvidia-smi not found"
}

Write-Host ""
Write-Host "[4] Python"
Get-Command python -ErrorAction SilentlyContinue | Select-Object Source
python --version

Write-Host ""
Write-Host "[5] Conda"
if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda --version
    conda env list
} else {
    Write-Host "Conda not found in PATH"
}

Write-Host ""
Write-Host "[6] Existing project files"
@(
    "data\models\gru_baseline\best_gru_baseline.pt",
    "data\processed\i24_sequences\gru_inputs\train_normalization_statistics.csv",
    "data\processed\i24_sequences\feature_names.npy"
) | ForEach-Object {
    $Path = Join-Path $Project $_
    [PSCustomObject]@{
        File   = $_
        Exists = Test-Path $Path
        SizeMB = if (Test-Path $Path) { [math]::Round((Get-Item $Path).Length / 1MB, 2) } else { $null }
    }
} | Format-Table -AutoSize

Write-Host ""
Write-Host "[7] Existing CARLA folders"
@(
    "C:\CARLA_0.9.16",
    "C:\CARLA_Install",
    "$env:USERPROFILE\Downloads"
) | ForEach-Object {
    [PSCustomObject]@{
        Path   = $_
        Exists = Test-Path $_
    }
} | Format-Table -AutoSize

Write-Host ""
Write-Host "[8] CARLA default ports"
foreach ($Port in 2000, 2001) {
    $Listening = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    $status = [bool]$Listening
    Write-Host "Port $Port listening: $status"
}

Write-Host ""
Write-Host "=== DIAGNOSTICS FINISHED - NOTHING INSTALLED ===" -ForegroundColor Green
