# Dot-source this file to configure an already-open PowerShell session.
$compilerProjectRoot = Split-Path -Parent $PSScriptRoot
$compilerRecordPath = Join-Path $compilerProjectRoot '.tools/compiler/installation.json'
if (-not (Test-Path -LiteralPath $compilerRecordPath)) {
    throw 'Run .venv\Scripts\python.exe scripts/setup_compiler.py first.'
}
$compilerRecord = Get-Content -LiteralPath $compilerRecordPath -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $compilerRecord.compiler_path)) {
    throw 'The recorded compiler is missing; rerun setup_compiler.py.'
}
$env:PATH = "$($compilerRecord.compiler_bin);$env:PATH"
# Preserve other flags; PyTensor gives the rightmost cxx definition precedence.
$compilerFlag = 'cxx=' + $compilerRecord.compiler_path.Replace('\', '/')
$env:PYTENSOR_FLAGS = if ($env:PYTENSOR_FLAGS) { "$env:PYTENSOR_FLAGS,$compilerFlag" } else { $compilerFlag }
Write-Output "PyTensor compiler configured: $($compilerRecord.version_output)"
