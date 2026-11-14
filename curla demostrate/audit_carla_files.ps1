$ErrorActionPreference = "Stop"

$Root = "C:\CARLA_0.9.16"
$Zip = "$env:USERPROFILE\Downloads\CARLA_0.9.16.zip"

Write-Host ""
Write-Host "=== STEP 3A: SIMULATOR FILE AUDIT ===" -ForegroundColor Cyan

$Required = @(
    "CarlaUE4.exe",
    "Engine\Binaries\Win64",
    "Engine\Binaries\Win64\CarlaUE4-Win64-Shipping.exe",
    "CarlaUE4\Content",
    "PythonAPI\carla\dist"
)

foreach ($RelativePath in $Required) {
    $Path = Join-Path $Root $RelativePath
    $Exists = Test-Path $Path
    Write-Host ("{0,-60} {1}" -f $RelativePath, $(if ($Exists) {"FOUND"} else {"MISSING"}))
    
    if ($Exists -and (Test-Path $Path -PathType Leaf)) {
        $Item = Get-Item $Path
        Write-Host ("  Size: {0:N2} MB" -f ($Item.Length / 1MB))
    }
}

Write-Host ""
Write-Host "[Directory overview]"
Get-ChildItem $Root -Force |
    Select-Object Mode, Name,
        @{Name="SizeMB";Expression={
            if (!$_.PSIsContainer) {
                [math]::Round($_.Length / 1MB, 2)
            } else {
                $null
            }
        }} |
    Format-Table -AutoSize

Write-Host ""
Write-Host "[Archive integrity]"
if (Test-Path $Zip) {
    & tar.exe -tf $Zip | Select-Object -First 15
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Archive listing failed. Archive may be damaged or unsupported."
    }
} else {
    Write-Host "Archive not found: $Zip"
}

Write-Host ""
Write-Host "=== AUDIT COMPLETE; NO FILES MODIFIED ===" -ForegroundColor Green
