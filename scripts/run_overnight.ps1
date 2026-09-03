# Overnight conformance chain: evaluation -> ablation -> full population.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

"$PID" | Out-File -Encoding ascii results\overnight_chain.pid

"[chain] START evaluate  $(Get-Date -Format s)" | Out-File -Encoding utf8 results\overnight_chain.log
& $py scripts\evaluate.py *> results\overnight_1_evaluate.log
"[chain] DONE evaluate (exit $LASTEXITCODE)  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\overnight_chain.log

"[chain] START ablation  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\overnight_chain.log
& $py scripts\ablation.py --num-warmup 300 --num-samples 300 *> results\overnight_2_ablation.log
"[chain] DONE ablation (exit $LASTEXITCODE)  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\overnight_chain.log

"[chain] START full_population  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\overnight_chain.log
& $py scripts\full_population.py --k-full 1024 --vi-steps 6000 *> results\overnight_3_full_population.log
"[chain] DONE full_population (exit $LASTEXITCODE)  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\overnight_chain.log

"[chain] ALL DONE  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\overnight_chain.log
Remove-Item results\overnight_chain.pid -ErrorAction SilentlyContinue
