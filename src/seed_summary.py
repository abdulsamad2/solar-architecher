"""Gate 1 replication summary: per-seed paired E3b-vs-E2 comparison, across-seed variation, closing rule.

Seed variation (spread of pooled scores across seeds) is reported separately from the within-seed
paired day-bootstrap CI. No seed is selected or dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.compare import GROUPS, compare


def summarise(e2_root: Path, e3b_root: Path, seeds) -> dict:
    per_seed = {}
    for s in seeds:
        r = compare({"E2": e2_root / f"seed{s}", "E3b": e3b_root / f"seed{s}"})
        per_seed[s] = {"pooled": r["pooled"], "diff": r["paired_rmse_diff_ci95"]["E3b - E2"],
                       "fold_wins_E3b": r["fold_wins"]["E3b < E2"],
                       "best_epochs": {m: [r["per_fold"][k][m]["best_epoch"] for k in r["per_fold"]] for m in ("E2", "E3b")}}
    across = {}
    for m in ("E2", "E3b", "persistence"):
        across[m] = {g: _stats([per_seed[s]["pooled"][m][g] for s in seeds]) for g in GROUPS}
    across["E3b - E2"] = {g: _stats([per_seed[s]["diff"][g][0] for s in seeds]) for g in GROUPS}

    improve = sum(per_seed[s]["diff"]["cloudy"][0] < 0 and per_seed[s]["diff"]["cloudy"][2] < 0 for s in seeds)
    sunny_regress = sum(per_seed[s]["diff"]["sunny"][1] > 0 for s in seeds)
    verdict = ("IMPROVES" if improve >= 2 and sunny_regress < 2 else "NO DETECTABLE BENEFIT")
    return {"seeds": list(seeds), "per_seed": per_seed, "across_seeds": across,
            "closing_rule": {"seeds_with_cloudy_improvement_ci_below_0": int(improve),
                             "seeds_with_sunny_regression_ci_above_0": int(sunny_regress), "verdict": verdict}}


def _stats(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "values": v.round(4).tolist()}


def to_markdown(r) -> str:
    seeds = r["seeds"]
    L = ["", "## Gate 1 replication results (seeds " + ", ".join(map(str, seeds)) + ")", "",
         "### Pooled validation RMSE (kW) per seed", "",
         "| Seed | E2 overall / sunny / cloudy | E3b overall / sunny / cloudy | Best epochs E2 | Best epochs E3b |",
         "|---|---|---|---|---|"]
    for s in seeds:
        p = r["per_seed"][s]
        L.append(f"| {s} | " + " / ".join(f"{p['pooled']['E2'][g]:.3f}" for g in GROUPS) + " | "
                 + " / ".join(f"{p['pooled']['E3b'][g]:.3f}" for g in GROUPS)
                 + f" | {p['best_epochs']['E2']} | {p['best_epochs']['E3b']} |")
    L += ["", "### Seed variation (across seeds; separate from bootstrap uncertainty)", "",
          "| Quantity | Overall mean ± SD [min, max] | Sunny | Cloudy |", "|---|---|---|---|"]
    for m, d in r["across_seeds"].items():
        L.append(f"| {m} | " + " | ".join(f"{d[g]['mean']:.3f} ± {d[g]['sd']:.3f} [{d[g]['min']:.3f}, {d[g]['max']:.3f}]"
                                          for g in GROUPS) + " |")
    L += ["", "### Paired E3b − E2 per seed: 95% day-block bootstrap CI (within-seed sampling uncertainty)", "",
          "| Seed | Overall | Sunny | Cloudy | Folds where E3b < E2 (overall / sunny / cloudy) |", "|---|---|---|---|---|"]
    for s in seeds:
        d, w = r["per_seed"][s]["diff"], r["per_seed"][s]["fold_wins_E3b"]
        L.append(f"| {s} | " + " | ".join(f"{d[g][0]:+.3f} [{d[g][1]:+.3f}, {d[g][2]:+.3f}]" for g in GROUPS)
                 + f" | {w['overall']}/5, {w['sunny']}/5, {w['cloudy']}/5 |")
    c = r["closing_rule"]
    L += ["", f"**Closing rule applied:** {c['seeds_with_cloudy_improvement_ci_below_0']}/{len(seeds)} seeds show a cloudy "
          f"improvement with CI below 0 (required ≥ 2); {c['seeds_with_sunny_regression_ci_above_0']}/{len(seeds)} seeds "
          f"show a sunny regression with CI above 0 (allowed < 2). **Verdict: {c['verdict']}.**", ""]
    return "\n".join(L)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--e2", default="experiments/E2E3_cnn_gru/v2/kw")
    p.add_argument("--e3b", default="experiments/E3b_csi_inputs/v2/csi")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--append-to", default="reports/E2_E3_E3b_comparison.md")
    p.add_argument("--json", default="reports/gate1_replication.json")
    a = p.parse_args()
    r = summarise(Path(a.e2), Path(a.e3b), a.seeds)
    Path(a.json).write_text(json.dumps(r, indent=2, default=str))
    md = to_markdown(r)
    with open(a.append_to, "a", encoding="utf-8") as f:
        f.write(md)
    print(md)
