# Phase E training (reports/phaseE_gatton_prereg.md sec. 4): F1 fine-tuned from SKIPP'D members, then F0 from scratch.
# 3 experts x 3 seeds x 5 folds each; frozen SKIPP'D recipe; development days only. Resumable (skips finished seeds).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
$common = @("--target", "csi", "--input", "csi", "--calib-holdout", "--max-epochs", "15", "--patience", "5", "--site", "gatton",
            "--min-clear-kw", "108.8", "--h5", "D:/solar_data/gatton/processed/gatton_skippd.hdf5",
            "--manifest", "data/manifests/gatton_forecast_v1.npz", "--folds", "data/manifests/gatton_folds_v1.json")
$experts = @(
    @{n = "E4-B-fit"; a = @("--head", "quantile"); init = "experiments/E4_calib/v1/csi" },
    @{n = "E6-S"; a = @("--head", "event"); init = "experiments/E6/v1/E6-S/csi" },
    @{n = "A2"; a = @("--head", "quantile", "--e8-window", "H60", "--encoder", "cnn3d_gru", "--e8-dir", "data/features_gatton"); init = "experiments/E9/v1/A2/csi" }
)
foreach ($arm in @("F1", "F0")) {
    foreach ($e in $experts) {
        foreach ($s in @(42, 123, 456)) {
            $out = "experiments/E_gatton/$arm/$($e.n)"
            if (Test-Path "$out/csi/seed$s/summary.json") { Write-Output "skip $arm $($e.n) seed $s"; continue }
            $init = if ($arm -eq "F1") { @("--init-from", $e.init) } else { @() }
            Write-Output ("{0} >>> {1} {2} seed {3}" -f (Get-Date -Format "HH:mm:ss"), $arm, $e.n, $s)
            & $py -W ignore -m src.train @common @($e.a) @init --seed $s --out $out
            if ($LASTEXITCODE -ne 0) { throw "failed: $arm $($e.n) seed $s" }
        }
    }
}
Write-Output "GATTON TRAINING DONE"
