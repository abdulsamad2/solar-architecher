"""Revision R-X1: image shuffling diagnostic on the official test set (protocol revision/protocols/R-X1_image_shuffling.md).

Frozen 15-member E4-B-fit, E6-S and A2 are run with the inputs of src/revision_posthoc_dump.run_member, except that the
image stack is perturbed (PV/K history, anchors and targets unchanged). G3''-F is recomputed from the perturbed experts
with the frozen gates. POST HOC; nothing is refitted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.clear_sky import clear_sky_index
from src.final_eval import FEAT, H5, MAN, SEEDS, Split, blend, gate_params, member_anchor, record_table_unsorted
from src.metrics import crps_quantile_interp
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_posthoc_dump import EXPERTS
from src.train import build_model, predict_raw

MODELS = ("E4-B-fit", "E6-S", "A2")
PERM_SEEDS = (0, 1, 2, 3, 4)
CONDS = ("original",) + tuple(f"S-perm{s}" for s in PERM_SEEDS) + ("T-rev", "T-rand", "T-static")
STRATA = ("overall", "sunny", "cloudy", "ramp", "non-ramp")
TAU = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())


class Perturbed:
    """GpuSeq view whose image rows are rewritten by img_rows(idx, rows); history/extra rows are untouched."""

    def __init__(self, data, img_rows):
        self.d, self.img_rows = data, img_rows

    def __getattr__(self, k):  # k_now, residual, ...
        return getattr(self.d, k)

    def __call__(self, idx):
        rows = self.d.hist[idx]
        irows = self.img_rows(idx, rows)
        x = self.d.images[irows].permute(0, 1, 4, 2, 3).float().div_(255.0)
        if self.d.extra_table is None:
            return x, self.d.hist_feat[rows]
        return x, self.d.hist_feat[rows], self.d.extra_table[rows]


def perturbations(n, eligible, device):
    """Image-row maps per condition; sample permutations only among samples the expert can predict."""
    out = {"original": lambda idx, rows: rows,
           "T-rev": lambda idx, rows: rows.flip(1),
           "T-static": lambda idx, rows: rows[:, -1:].expand(-1, rows.shape[1])}
    fr = np.argsort(np.random.default_rng(0).random((n, 16)), axis=1)  # per-sample random frame order (seed 0)
    frt = torch.as_tensor(fr, device=device)
    out["T-rand"] = lambda idx, rows: torch.gather(rows, 1, frt[idx])
    el = np.flatnonzero(eligible)
    for s in PERM_SEEDS:
        p = np.arange(n)
        p[el] = el[np.random.default_rng(1000 + s).permutation(len(el))]
        out[f"S-perm{s}"] = make_perm(torch.as_tensor(p, device=device))
    return out


def make_perm(pt):
    def f(idx, rows, _pt=pt):
        return PERM_HIST[0][_pt[idx]]
    return f


PERM_HIST = [None]  # the current window table (hist rows of the expert being run), set before each member


def run_members(split, name):
    root, kind = EXPERTS[name]
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    data, n = split.data, len(split.ds.sample_ids)
    idx = np.arange(n)
    if kind == "H60":
        idx = idx[split.h60_ok]
    pert = perturbations(n, split.h60_ok if kind == "H60" else np.ones(n, bool), split.device)
    res = {c: [] for c in CONDS}
    for s in SEEDS:
        run = root / f"seed{s}"
        cfg = json.loads((run / "config.json").read_text())
        for k in range(5):
            fold = folds_doc["folds"][k]
            fdir = run / f"fold_{k}"
            pc = member_anchor(fold)(split.rec_t)
            k_rec = clear_sky_index(split.pv, pc, 1.0)
            data.set_history_feature(k_rec)
            data.residual = False
            data.k_now = torch.tensor(k_rec[split.ds.hist_rows[:, -1]], dtype=torch.float32, device=split.device)
            if kind == "H60":
                data.set_window(split.h60_rows)
                std = json.loads((fdir / "e8_standardisation.json").read_text())
                mask = np.array(std["mask"], np.float32)
                table = split.table if mask.any() else np.zeros((len(split.pv), 11), np.float32)
                data.set_extra(((table - np.array(std["mu"])) / np.array(std["sd"]) * mask).astype(np.float32))
            else:
                data.set_window(split.ds.hist_rows)
                data.set_extra(None)
            PERM_HIST[0] = data.hist
            net = build_model(cfg).to(split.device)
            net.encoder.cnn_bf16 = cfg["cnn_bf16"]
            net.load_state_dict(torch.load(fdir / "best.pt", map_location=split.device))
            net.eval()
            it = torch.as_tensor(idx, device=split.device)
            scale = np.maximum(pc[split.ds.target_rows[idx]], 1.0)[:, None]
            for c in CONDS:
                q = np.full((n, 9), np.nan)
                q[idx] = predict_raw(net, Perturbed(data, pert[c]), it).cpu().numpy().astype(float) * scale
                res[c].append({"pred_q": q, "pclear_target_kw": pc[split.ds.target_rows], "pclear_now_kw": pc[split.ds.hist_rows[:, -1]]})
            del net
            torch.cuda.empty_cache()
        print(name, "seed", s, "done", flush=True)
    return res


def main():
    import h5py
    with h5py.File(H5, "r") as f:
        gray = np.round(f["test/images_log"][...].astype(np.float32).mean(-1)).astype(np.uint8)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 1
    t_test = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    split = Split("test", torch.device("cuda"), record_table_unsorted(t_test, gray, FEAT / "gray_u8_test.npy"))
    c = test_context()
    assert np.array_equal(c["sample_ids"], split.ds.sample_ids), "STOP: sample order differs"
    pv_now = split.pv[split.ds.hist_rows[:, -1]]
    h60 = split.h60_ok
    assert np.array_equal(h60, c["h60"])
    mem = {nm: run_members(split, nm) for nm in MODELS}
    members = [(s, k) for s in SEEDS for k in range(5)]
    md = lambda d: {**d, "pv_now_kw": pv_now, "quantile_levels": LEVELS}
    sub = lambda d, msk: {kk: (v[msk] if np.ndim(v) and len(v) == len(msk) else v) for kk, v in d.items()}
    ens = {}
    for cnd in CONDS:
        g3 = []
        for i, (s, k) in enumerate(members):
            tau = TAU[str(k)]["training_labels_fit_days"]
            d, r, a = md(mem["E4-B-fit"][cnd][i]), md(mem["E6-S"][cnd][i]), md(mem["A2"][cnd][i])
            q = blend(d, r, gate_params("G3", s, k), tau)
            q[h60] = blend(sub(a, h60), sub(r, h60), gate_params("G3''", s, k), tau)
            g3.append(q)
        ens[cnd] = {"E4-B-fit": np.mean([x["pred_q"] for x in mem["E4-B-fit"][cnd]], 0),
                    "A2": np.mean([x["pred_q"] for x in mem["A2"][cnd]], 0), "G3''-F": np.mean(g3, 0)}
    frozen = frozen_test_predictions(c["sample_ids"])
    repro = {"E4-B-fit": float(np.abs(ens["original"]["E4-B-fit"] - frozen["E4-B-fit"]).max()),
             "G3''-F": float(np.abs(ens["original"]["G3''-F"] - frozen["G3''-F"]).max())}
    print("reproduction max |diff| kW:", repro)
    assert max(repro.values()) < 1e-6, "STOP: original run does not reproduce the frozen test predictions"
    y, day, sunny = c["y"], c["day"], c["sunny"]
    res = {"protocol": "revision/protocols/R-X1_image_shuffling.md", "label": "POST HOC (test; frozen models, perturbed images)",
           "reproduction_max_abs_kw": repro, "models": {}}
    for mdl, pop in (("E4-B-fit", np.ones(len(y), bool)), ("A2", h60), ("G3''-F", np.ones(len(y), bool))):
        masks = {g: c["masks"][g][pop] for g in STRATA}
        yy, dd, ss = y[pop], day[pop], sunny[pop]
        crps = {cnd: crps_quantile_interp(ens[cnd][mdl][pop], yy, LEVELS) for cnd in CONDS}
        crps["S-perm (mean of 5)"] = np.mean([crps[f"S-perm{s}"] for s in PERM_SEEDS], 0)
        out = {"n": int(pop.sum()), "crps": {}, "rmse_median": {}, "change_vs_original": {}}
        for cnd, cc in crps.items():
            out["crps"][cnd] = {g: float(cc[v].mean()) for g, v in masks.items()}
            if cnd in ens:
                med = ens[cnd][mdl][pop][:, 4]
                out["rmse_median"][cnd] = {g: float(np.sqrt(np.mean((med[v] - yy[v]) ** 2))) for g, v in masks.items()}
        for cnd in ("S-perm (mean of 5)", "T-rev", "T-rand", "T-static"):
            diff = crps[cnd] - crps["original"]
            out["change_vs_original"][cnd] = {g: day_bootstrap(diff[v], dd[v], ss[v]) for g, v in masks.items()}
        out["S-perm_seed_range_overall"] = [min(out["crps"][f"S-perm{s}"]["overall"] for s in PERM_SEEDS),
                                            max(out["crps"][f"S-perm{s}"]["overall"] for s in PERM_SEEDS)]
        res["models"][mdl] = out
        print(mdl, {cnd: (round(v["overall"], 3), round(v["ramp"], 3)) for cnd, v in out["crps"].items()})
        print("  change:", {cnd: tuple(round(x, 3) for x in v["overall"]) for cnd, v in out["change_vs_original"].items()})
    p = save("x1_image_shuffling.json", res)
    print("saved", p)


if __name__ == "__main__":
    main()
