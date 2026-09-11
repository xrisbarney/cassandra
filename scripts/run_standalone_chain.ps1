# End-to-end standalone-branch run: train -> calibrate -> evaluate ->
# backtest -> forecast -> ablation.  Stops at the first failing step.
# Logs: results\logs\<step>.log   Progress: results\logs\chain_progress.log
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force (Join-Path $root "results\logs") | Out-Null

# Prefer the project venv's interpreter when present.
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$progress = Join-Path $root "results\logs\chain_progress.log"
function Note($msg) {
    $line = "[chain] $msg  $(Get-Date -Format s)"
    Write-Host $line
    Add-Content -Path $progress -Value $line -Encoding utf8
}

$steps = @(
    @{ name = "1_train";     cmd = "scripts\train.py" },
    @{ name = "2_calibrate"; cmd = "scripts\calibrate_damage.py" },
    @{ name = "3_evaluate";  cmd = "scripts\evaluate.py" },
    @{ name = "4_backtest";  cmd = "scripts\backtest.py --extend-to none --tail-months 12" },
    @{ name = "5_forecast";  cmd = "scripts\forecast.py --horizon 12" },
    @{ name = "6_ablation";  cmd = "scripts\ablation.py" }
)

Note "PIPELINE START"
foreach ($s in $steps) {
    $log = Join-Path $root ("results\logs\" + $s.name + ".log")
    Note ("START " + $s.name)
    # cmd /c handles native stderr (progress bars) without PowerShell 5.1's
    # NativeCommandError wrapping under ErrorActionPreference = Stop.
    cmd /s /c "`"$py`" $($s.cmd) > `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Note ("FAILED " + $s.name + " (exit $LASTEXITCODE) - see $log")
        exit $LASTEXITCODE
    }
    Note ("DONE " + $s.name)
}
Note "ALL DONE"
