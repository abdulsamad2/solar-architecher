# Phase F final ablations after the staged search: waits for the search chain, then trains every selected arm
# (3 seeds x 5 folds) and runs the development out-of-fold evaluation. Arms come from configs/physics/*.yaml
# written by the pre-registered selection step.
param([int]$WaitPid = 0)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($WaitPid -gt 0) { try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch {} }
if (-not (Test-Path configs/physics/selection.json)) { throw "selection.json missing: search did not finish" }
$py = ".venv\Scripts\python.exe"
foreach ($arm in @("A1", "R", "A3", "A4", "A5")) {
    if (-not (Test-Path "configs/physics/$arm.yaml")) { Write-Output "skip $arm (not selected)"; continue }
    Write-Output ("{0} >>> final {1}" -f (Get-Date -Format "HH:mm:ss"), $arm)
    & $py scripts/run_physics_ablation.py final $arm --seeds 42 123 456
    if ($LASTEXITCODE -ne 0) { throw "final $arm failed" }
}
Write-Output ("{0} >>> evaluate" -f (Get-Date -Format "HH:mm:ss"))
& $py -W ignore scripts/evaluate_physics_ablation.py
Write-Output "FINAL DONE"
