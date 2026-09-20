# Final-system run (configs\final.yaml: N+B channels): archive the four-channel
# artifacts, then train -> calibrate -> evaluate -> backtest -> forecast.
# Logs: results\logs\final_*.log   Progress: results\logs\final_progress.log
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force (Join-Path $root "results\logs") | Out-Null

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$progress = Join-Path $root "results\logs\final_progress.log"
function Note($msg) {
    $line = "[final] $msg  $(Get-Date -Format s)"
    Write-Host $line
    Add-Content -Path $progress -Value $line -Encoding utf8
}

# Archive the four-channel headline artifacts once (needed as the "Full
# (four channels)" reference in the paper).
if (-not (Test-Path "results\full_4ch")) {
    Note "ARCHIVE four-channel artifacts -> results\full_4ch"
    New-Item -ItemType Directory -Force results\full_4ch | Out-Null
    foreach ($p in "idata.pkl", "idata.nc") {
        if (Test-Path "results\$p") { Move-Item "results\$p" "results\full_4ch\$p" }
    }
    foreach ($d in "evaluation", "calibration", "backtest", "forecasts") {
        if (Test-Path "results\$d") { Move-Item "results\$d" "results\full_4ch\$d" }
    }
}

$steps = @(
    @{ name = "final_1_train";     cmd = "scripts\train.py --config configs\final.yaml" },
    @{ name = "final_2_calibrate"; cmd = "scripts\calibrate_damage.py --config configs\final.yaml" },
    @{ name = "final_3_evaluate";  cmd = "scripts\evaluate.py --config configs\final.yaml" },
    @{ name = "final_4_backtest";  cmd = "scripts\backtest.py --config configs\final.yaml --extend-to none --tail-months 12" },
    @{ name = "final_5_forecast";  cmd = "scripts\forecast.py --config configs\final.yaml --horizon 12" }
)

Note "PIPELINE START (configs\final.yaml)"
foreach ($s in $steps) {
    $log = Join-Path $root ("results\logs\" + $s.name + ".log")
    Note ("START " + $s.name)
    cmd /s /c "`"$py`" $($s.cmd) > `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Note ("FAILED " + $s.name + " (exit $LASTEXITCODE) - see $log")
        exit $LASTEXITCODE
    }
    Note ("DONE " + $s.name)
}
Note "ALL DONE"
