# Revision optional-experiment GPU queue (protocols: revision/protocols/R-X3, R-X4).
# Usage: powershell -File scripts/revision_x_queue.ps1 -Stream I|I60|F [-Seeds 42,123]
# Stream I:   R-X3 I15 images-only (2D, 15 min), seeds 42/123/456 x 5 blocked folds.
# Stream I60: R-X3 I60 images-only (3D, 60 min), seeds 42/123/456 x 5 blocked folds.
# Stream F:   R-X4 forward refits with separate early-stopping / gate days, seeds 42/123/456/789/1011.
param([Parameter(Mandatory = $true)][ValidateSet("I", "I60", "F")][string]$Stream, [int[]]$Seeds = @())
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force revision\logs | Out-Null
$status = "revision\logs\queue_X_$Stream.status"
$common = @("--target", "csi", "--max-epochs", "15", "--patience", "5")

$jobs = @()
if ($Stream -eq "I") {
    if (-not $Seeds) { $Seeds = 42, 123, 456 }
    foreach ($s in $Seeds) {
        $jobs += , @("R-X3 I15 s$s", @("--input", "none", "--head", "quantile", "--calib-holdout", "--seed", "$s", "--label", "R-X3 I15 images-only", "--out", "experiments/R_X3/I15"))
    }
}
elseif ($Stream -eq "I60") {
    if (-not $Seeds) { $Seeds = 42, 123, 456 }
    foreach ($s in $Seeds) {
        $jobs += , @("R-X3 I60 s$s", @("--input", "none", "--head", "quantile", "--calib-holdout", "--e8-window", "H60", "--encoder", "cnn3d_gru", "--e8-features", "none", "--seed", "$s", "--label", "R-X3 I60 images-only", "--out", "experiments/R_X3/I60"))
    }
}
else {
    if (-not $Seeds) { $Seeds = 42, 123, 456, 789, 1011 }
    $fw = @("--input", "csi", "--folds", "revision/manifests/folds_forward_v2.json", "--fold-ids", "0")
    foreach ($s in $Seeds) {
        $jobs += , @("R-X4 E6-S s$s", (@("--head", "event", "--seed", "$s", "--label", "R-X4 E6-S", "--out", "experiments/R_X4/E6-S") + $fw))
        $jobs += , @("R-X4 A2 s$s", (@("--head", "quantile", "--e8-window", "H60", "--encoder", "cnn3d_gru", "--e8-features", "none", "--seed", "$s", "--label", "R-X4 A2", "--out", "experiments/R_X4/A2") + $fw))
        $jobs += , @("R-X4 E4-B-fit s$s", (@("--head", "quantile", "--seed", "$s", "--label", "R-X4 E4-B-fit", "--out", "experiments/R_X4/E4-B-fit") + $fw))
        $jobs += , @("R-X4 P15 s$s", (@("--head", "quantile", "--encoder", "pv_gru", "--seed", "$s", "--label", "R-X4 P15", "--out", "experiments/R_X4/P15") + $fw))
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
"$(Get-Date -Format s) QUEUE X_$Stream DONE" | Add-Content $status
