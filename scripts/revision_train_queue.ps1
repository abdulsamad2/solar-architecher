# Revision GPU training queue (protocols: revision/protocols/R-E1, R-E4, R-E3).
# Usage: powershell -File scripts/revision_train_queue.ps1 -Stream A|B
# Stream A: development CV arms (E1 P15, E1 P60, E4 3D@H15), seeds 42/123/456 x 5 blocked folds.
# Stream B: E3 forward refits (1 chronological fold), seeds 42/123/456/789/1011.
param([Parameter(Mandatory = $true)][ValidateSet("A", "B")][string]$Stream)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force revision\logs | Out-Null
$status = "revision\logs\queue_$Stream.status"
$common = @("--target", "csi", "--input", "csi", "--max-epochs", "15", "--patience", "5")

$jobs = @()
if ($Stream -eq "A") {
    foreach ($s in 42, 123, 456) {
        $jobs += , @("R-E1 P15 s$s", @("--head", "quantile", "--calib-holdout", "--encoder", "pv_gru", "--seed", "$s", "--label", "R-E1 P15", "--out", "experiments/R_E1/P15"))
    }
    foreach ($s in 42, 123, 456) {
        $jobs += , @("R-E4 A2-H15 s$s", @("--head", "quantile", "--calib-holdout", "--e8-window", "H15", "--encoder", "cnn3d_gru", "--e8-features", "none", "--seed", "$s", "--label", "R-E4 3D-H15", "--out", "experiments/R_E4/A2_H15"))
    }
    foreach ($s in 42, 123, 456) {
        $jobs += , @("R-E1 P60 s$s", @("--head", "quantile", "--calib-holdout", "--e8-window", "H60", "--encoder", "pv_gru", "--e8-features", "none", "--seed", "$s", "--label", "R-E1 P60", "--out", "experiments/R_E1/P60"))
    }
}
else {
    $fw = @("--folds", "revision/manifests/folds_forward_v1.json", "--fold-ids", "0")
    foreach ($s in 42, 123, 456, 789, 1011) {
        $jobs += , @("R-E3 E6-S s$s", (@("--head", "event", "--seed", "$s", "--label", "R-E3 E6-S", "--out", "experiments/R_E3/E6-S") + $fw))
        $jobs += , @("R-E3 A2 s$s", (@("--head", "quantile", "--e8-window", "H60", "--encoder", "cnn3d_gru", "--e8-features", "none", "--seed", "$s", "--label", "R-E3 A2", "--out", "experiments/R_E3/A2") + $fw))
        $jobs += , @("R-E3 E4-B-fit s$s", (@("--head", "quantile", "--seed", "$s", "--label", "R-E3 E4-B-fit", "--out", "experiments/R_E3/E4-B-fit") + $fw))
        $jobs += , @("R-E3 P15 s$s", (@("--head", "quantile", "--encoder", "pv_gru", "--seed", "$s", "--label", "R-E3 P15", "--out", "experiments/R_E3/P15") + $fw))
    }
}

foreach ($j in $jobs) {
    $name, $args_ = $j
    $log = "revision\logs\" + ($name -replace '[^A-Za-z0-9_-]', '_') + ".log"
    $t0 = Get-Date
    "$(Get-Date -Format s) START $name" | Add-Content $status
    & $py -m src.train @common @args_ *> $log
    $rc = $LASTEXITCODE
    "$(Get-Date -Format s) END $name rc=$rc min=$([math]::Round(((Get-Date) - $t0).TotalMinutes, 1))" | Add-Content $status
}
"$(Get-Date -Format s) QUEUE $Stream DONE" | Add-Content $status
