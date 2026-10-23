# Phase F staged search (pre-registered order): S1, S2a -> select -> S2b -> select -> S3 -> select -> S4 -> select.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
foreach ($step in @("search S1", "search S2a", "select", "search S2b", "select", "search S3", "select", "search S4", "select")) {
    Write-Output ("{0} >>> {1}" -f (Get-Date -Format "HH:mm:ss"), $step)
    & $py scripts/run_physics_ablation.py $step.Split(" ")
    if ($LASTEXITCODE -ne 0) { throw "step failed: $step" }
}
Write-Output "SEARCH DONE"
