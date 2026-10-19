"""Paired comparison of point forecasters on the blocked dev folds (validation predictions only).

Asserts identical validation samples across experiments, reports per-fold RMSE (overall / clear / cloudy)
and best epoch, and day-block bootstrap 95% CIs (stratified by clear/cloudy day) for pooled RMSE
differences. Never reads the official test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GROUPS = ("overall", "sunny", "cloudy")


def load(exp_dir: Path, folds) -> dict:
    out = {}
    for k in folds:
        d = exp_dir / f"fold_{k}"
        z = np.load(d / "preds_val.npz")
        out[k] = {key: z[key] for key in z.files} | {"metrics": json.loads((d / "metrics.json").read_text())}
    return out


def rmse(e) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def masks(clear):
    return {"overall": np.ones_like(clear, bool), "sunny": clear, "cloudy": ~clear}


def bootstrap_diff(err_a, err_b, day, clear, n_boot=2000, seed=0):
    """CI of RMSE(a) - RMSE(b) over resampled days, stratified by day label."""
    days, inv = np.unique(day, return_inverse=True)
    sa = np.bincount(inv, weights=err_a ** 2)
    sb = np.bincount(inv, weights=err_b ** 2)
    cnt = np.bincount(inv)
    day_clear = np.zeros(len(days), bool)
    day_clear[inv] = clear
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(day_clear), np.flatnonzero(~day_clear)]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups if len(g)])
        boots[b] = np.sqrt(sa[idx].sum() / cnt[idx].sum()) - np.sqrt(sb[idx].sum() / cnt[idx].sum())
    point = np.sqrt(sa.sum() / cnt.sum()) - np.sqrt(sb.sum() / cnt.sum())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def compare(exps: dict[str, Path], folds=(0, 1, 2, 3, 4), out_md: Path | None = None) -> dict:
    data = {name: load(p, folds) for name, p in exps.items()}
    names = list(exps)
    ref = data[names[0]]
    for name in names[1:]:
        for k in folds:
            assert np.array_equal(data[name][k]["sample_ids"], ref[k]["sample_ids"]), f"{name} fold {k}: samples differ"
            assert np.allclose(data[name][k]["target"], ref[k]["target"])

    per_fold = {}
    for k in folds:
        r = ref[k]
        ms = masks(r["clear_day"])
        row = {"persistence": {g: rmse(r["persistence"][ms[g]] - r["target"][ms[g]]) for g in GROUPS}}
        for name in names:
            d = data[name][k]
            row[name] = {g: rmse(d["pred"][ms[g]] - d["target"][ms[g]]) for g in GROUPS}
            row[name]["best_epoch"] = d["metrics"]["best_epoch"]
        per_fold[k] = row

    target = np.concatenate([ref[k]["target"] for k in folds])
    day = np.concatenate([np.char.add(f"f{k}_", ref[k]["day"].astype(str)) for k in folds])
    clear = np.concatenate([ref[k]["clear_day"] for k in folds])
    err = {name: np.concatenate([data[name][k]["pred"] for k in folds]) - target for name in names}
    err["persistence"] = np.concatenate([ref[k]["persistence"] for k in folds]) - target
    ms = masks(clear)
    pooled = {name: {g: rmse(e[ms[g]]) for g in GROUPS} for name, e in err.items()}
    pairs = [(a, b) for i, a in enumerate(names) for b in names[:i]] + [(n, "persistence") for n in names]
    diffs = {f"{a} - {b}": {g: bootstrap_diff(err[a][ms[g]], err[b][ms[g]], day[ms[g]], clear[ms[g]]) for g in GROUPS}
             for a, b in pairs}
    wins = {f"{a} < {b}": {g: int(sum(per_fold[k][a][g] < per_fold[k][b][g] for k in folds)) for g in GROUPS}
            for a, b in pairs}
    result = {"experiments": {n: str(p) for n, p in exps.items()}, "per_fold": per_fold, "pooled": pooled,
              "paired_rmse_diff_ci95": diffs, "fold_wins": wins, "n_samples": int(len(target)),
              "n_days": int(len(np.unique(day)))}
    if out_md:
        out_md.write_text(to_markdown(result, names, folds))
        out_md.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str))
    return result


def to_markdown(r, names, folds) -> str:
    cols = names + ["persistence"]
    lines = ["# Paired comparison on blocked dev folds (validation only)", "",
             f"{r['n_samples']:,} validation samples over {r['n_days']} fold-days; identical samples across models.", ""]
    for g in GROUPS:
        lines += [f"## {g} RMSE (kW) per fold", "", "| Fold | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for k in folds:
            cells = []
            vals = {c: r["per_fold"][k][c][g] for c in cols}
            best = min(vals, key=vals.get)
            for c in cols:
                v = f"{vals[c]:.3f}"
                if c != "persistence":
                    v += f" (ep {r['per_fold'][k][c]['best_epoch']})"
                cells.append(f"**{v}**" if c == best else v)
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("| pooled | " + " | ".join(f"{r['pooled'][c][g]:.3f}" for c in cols) + " |")
        lines.append("")
    lines += ["## Paired pooled RMSE differences, 95% day-block bootstrap CI (stratified clear/cloudy)", "",
              "| Comparison | Overall | Clear days | Cloudy days |", "|---|---|---|---|"]
    for pair, d in r["paired_rmse_diff_ci95"].items():
        lines.append(f"| {pair} | " + " | ".join(f"{d[g][0]:+.3f} [{d[g][1]:+.3f}, {d[g][2]:+.3f}]" for g in GROUPS) + " |")
    lines += ["", "Negative = first model better. Fold wins (first model lower RMSE):", ""]
    for pair, w in r["fold_wins"].items():
        lines.append(f"- {pair}: overall {w['overall']}/{len(folds)}, clear {w['sunny']}/{len(folds)}, cloudy {w['cloudy']}/{len(folds)}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="+", required=True, help="name=path/to/seed_dir")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="reports/E2_E3_E3b_comparison.md")
    a = p.parse_args()
    exps = dict(x.split("=", 1) for x in a.exp)
    r = compare({k: Path(v) for k, v in exps.items()}, a.folds, Path(a.out))
    print(Path(a.out).read_text())
