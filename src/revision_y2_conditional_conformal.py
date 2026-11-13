"""Revision R-Y2: conditional (Mondrian) conformal calibration on forecast-time state (protocol
revision/protocols/R-Y2_conditional_conformal.md). Development selection, post hoc test. CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, conformity, interval_score
from src.calibrate import base_interval
from src.final_eval import EXPERTS, blend, gate_params, rolling_conformal_daily
from src.gate2_report import ramp_labels
from src.revision_c07 import MATURE_US, interval_table, past_only_init
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_x2_linear_pool import TAU

SEEDS, FOLDS, K = (42, 123, 456), range(5), 3
WB_GRID = (990, 1540, 2310, 4620)
PP = "G3''-F"


def _val(model, s, k):
    return dict(np.load(EXPERTS[model][0] / f"seed{s}" / f"fold_{k}" / "preds_val.npz"))


def _sub(d, m):
    return {kk: (v[m] if np.ndim(v) and len(v) == len(m) else v) for kk, v in d.items()}


def route(e4, e6, a2, s, k, tau):
    """G3''-F member quantiles and the disagreement d of the expert pair used (timestamp routing = A2 coverage)."""
    h = np.isin(e6["sample_ids"], a2["sample_ids"])
    q, d = np.empty((len(h), 9)), np.empty(len(h))
    q[~h] = blend(_sub(e4, ~h), _sub(e6, ~h), gate_params("G3", s, k), tau)
    d[~h] = np.abs(e4["pred_q"][~h].mean(1) - e6["pred_q"][~h].mean(1))
    r60 = _sub(e6, h)
    assert np.array_equal(r60["sample_ids"], a2["sample_ids"]), "STOP: A2 order differs from E6-S"
    q[h] = blend(a2, r60, gate_params("G3''", s, k), tau)
    d[h] = np.abs(a2["pred_q"].mean(1) - r60["pred_q"].mean(1))
    return q, d


def development():
    """Out-of-fold G3''-F quantiles (3 members per sample) and features; reproduces the frozen dev scores."""
    acc_q, acc_d, cnt, meta = {}, {}, {}, {}
    for s in SEEDS:
        for k in FOLDS:
            tau = TAU[str(k)]["training_labels_fit_days"]
            e4, e6, a2 = _val("E4-B-fit", s, k), _val("E6-S", s, k), _val("A2", s, k)
            assert np.array_equal(e4["sample_ids"], e6["sample_ids"])
            q, d = route(e4, e6, a2, s, k, tau)
            for i, sid in enumerate(e6["sample_ids"]):
                if sid not in acc_q:
                    acc_q[sid], acc_d[sid], cnt[sid] = np.zeros(9), 0.0, 0
                    meta[sid] = (e6["target"][i], e6["day"][i], e6["clear_day"][i], e6["origin_time_us"][i])
                acc_q[sid] += q[i]
                acc_d[sid] += d[i]
                cnt[sid] += 1
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    # Deviation 2 (recorded in the results): the frozen routing is the manifest 60-min rule; 140 development samples
    # are 60-min eligible but have no stored A2 prediction (the frozen dry run predicted them afresh), so they are left
    # out of the development features and initial-score pools.
    from src.e8_audit import history_rows
    from src.skippd import SkippdForecastDataset
    man = np.load("data/manifests/forecast_tol30_v2.npz")
    ds = SkippdForecastDataset("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", "dev", preload=False)
    h60 = dict(zip(ds.sample_ids.tolist(), (history_rows(man["times_us"], man["origin"][~man["is_test"].astype(bool)], 4) >= 0).all(1)))
    a2ids = set().union(*(set(_val("A2", s, k)["sample_ids"].tolist()) for s in SEEDS for k in FOLDS))
    keep = np.array([not (h60[x] and x not in a2ids) for x in cal["sample_ids"].tolist()])
    sid = cal["sample_ids"][keep]
    assert all(cnt[x] == 3 for x in sid), "STOP: not 3 out-of-fold members per development sample"
    q = np.stack([acc_q[x] / 3 for x in sid])
    d = np.array([acc_d[x] / 3 for x in sid])
    y = np.array([meta[x][0] for x in sid], float)
    day = np.array([str(meta[x][1]) for x in sid])
    clear = np.array([bool(meta[x][2]) for x in sid])
    org = cal["origin_us"].astype(np.int64)[keep]
    assert np.array_equal(org, np.array([meta[x][3] for x in sid], np.int64)), "STOP: origin times differ"
    worst, scores = {}, {}
    for nm in ALPHA:
        sc = conformity(*base_interval(q, LEVELS, nm), y)
        scores[nm] = cal[f"{PP}|{nm}"][keep]  # frozen scores are used; the reconstruction only checks them
        diff = np.abs(sc - scores[nm])
        worst[nm] = {"max": float(diff.max()), "share_within_1e-4": float((diff <= 1e-4).mean())}
        # Deviation 1: the project's frozen reproduction criterion (amendment 2), not the protocol's 1e-6
        assert (diff <= 1e-4).mean() >= 0.999 and diff.max() <= 5e-3, f"STOP: dev {PP} scores not reproduced ({nm}: {diff.max()})"
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    ramp = np.array([rl["is_ramp"][x] for x in sid])
    w90 = np.diff(np.stack(base_interval(q, LEVELS, "90"), 1), axis=1)[:, 0]
    return {"q": q, "y": y, "day": day, "clear": clear, "org": org, "tgt": org + 15 * 60_000_000, "ramp": ramp,
            "feat": {"d": d, "w": w90}, "repro_max_abs_kw": worst, "scores": scores, "n_excluded": int((~keep).sum())}


def test_features(c, q):
    from src.revision_posthoc_test import mdict, members, sub
    h60 = c["h60"]
    e4, e6, a2 = members("E4-B-fit"), members("E6-S"), members("A2")
    acc = []
    for i, (s, k) in enumerate(e6["members"]):
        md, mr, ma = mdict(e4, i), mdict(e6, i), mdict(a2, i)
        d = np.abs(md["pred_q"].mean(1) - mr["pred_q"].mean(1))
        d[h60] = np.abs(sub(ma, h60)["pred_q"].mean(1) - sub(mr, h60)["pred_q"].mean(1))
        acc.append(d)
    w90 = np.diff(np.stack(base_interval(q, LEVELS, "90"), 1), axis=1)[:, 0]
    return {"d": np.mean(acc, 0), "w": w90}


def bins(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right"), 0, K - 1)


def mondrian(org, tgt, day, lo, hi, y, a, b, init_scores_fn, Wb):
    """Rolling conformal run separately inside each feature bin (daily reset, past-only initial scores)."""
    clo, chi = np.empty(len(y)), np.empty(len(y))
    for j in range(K):
        m = b == j
        if not m.any():
            continue
        days = np.unique(day[m])
        init = {d: init_scores_fn(j, org[m & (day == d)].min()) for d in days}
        cl, ch = rolling_conformal_daily(org[m], tgt[m], day[m], lo[m], hi[m], y[m], a, Wb, init)
        clo[m], chi[m] = cl, ch
    return clo, chi


def select_wb(dev, fname, edges):
    """Chronological development rehearsal (past-only matured initial scores) choosing W_b by the 90% interval score."""
    b = bins(dev["feat"][fname], edges)
    lo, hi = base_interval(dev["q"], LEVELS, "90")
    sc = dev["scores"]["90"]
    first = np.sort(np.unique(dev["day"]))[20]  # rehearsal days need some history; first 20 development days only seed
    keep = dev["day"] >= first
    out = {}
    for Wb in WB_GRID:
        fn = lambda j, start: past_only_init(dev["org"][b == j], sc[b == j], start, Wb)[0]
        clo, chi = mondrian(dev["org"][keep], dev["tgt"][keep], dev["day"][keep], lo[keep], hi[keep], dev["y"][keep], 0.1, b[keep], fn, Wb)
        out[Wb] = float(interval_score(clo, chi, dev["y"][keep], 0.1).mean())
    best = min(out, key=out.get)
    return best, out


def main():
    dev = development()
    print("dev G3''-F out-of-fold scores reproduced, max |diff| kW:", dev["repro_max_abs_kw"], flush=True)
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])[PP]
    y, day, org, tgt, masks = c["y"], c["day"], c["org"], c["tgt"], c["masks"]
    feats_t = test_features(c, q)
    c07 = json.loads(Path("revision/results/c07_conformal_pastonly.json").read_text())
    W = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())["W"][PP]
    days = np.unique(day)
    start = {d: org[day == d].min() for d in days}
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    cal_org = cal["origin_us"].astype(np.int64)
    res = {"protocol": "revision/protocols/R-Y2_conditional_conformal.md", "dev_reproduction_max_abs_kw": dev["repro_max_abs_kw"],
           "deviations": ["1: reconstruction check uses the project's frozen tolerance (>= 99.9% within 1e-4 kW, max 5e-3 kW; amendment 2) "
                          "instead of the protocol's 1e-6, because stored member predictions differ from re-inference by up to ~1e-3 kW",
                          f"2: {dev['n_excluded']} development samples that are 60-min eligible but have no stored A2 prediction are left out of the "
                          "development features and Mondrian initial-score pools; the frozen scores are used for all other samples",
                          "3: the W_b rehearsal uses past-only matured initial scores (stricter than 'other days only')",
                          "all three were fixed before any test quantity was computed"],
           "edges": {}, "W_b": {}, "W_b_rehearsal_interval_score": {}, "U": {}, "M-d": {}, "M-w": {}, "bin_coverage90": {},
           "bootstrap_IS90_M_minus_U": {}, "decision": {}}
    store = {}
    # U: the C07 past-only rule, reproduced
    for nm, a in ALPHA.items():
        lo, hi = base_interval(q, LEVELS, nm)
        init = {d: past_only_init(cal_org, cal[f"{PP}|{nm}"], start[d], W)[0] for d in days}
        clo, chi = rolling_conformal_daily(org, tgt, day, lo, hi, y, a, W, init)
        tab = interval_table(clo, chi, y, a, masks)
        want = c07["past_only"][PP][nm]["overall"]["coverage"]
        assert abs(tab["overall"]["coverage"] - want) < 1e-9, f"STOP: C07 past-only not reproduced ({nm})"
        res["U"][nm] = tab
        store[("U", nm)] = (clo, chi)
    # Mondrian variants
    for fname, key in (("d", "M-d"), ("w", "M-w")):
        edges = np.quantile(dev["feat"][fname], [1 / 3, 2 / 3])
        res["edges"][fname] = edges.tolist()
        Wb, curve = select_wb(dev, fname, edges)
        res["W_b"][key], res["W_b_rehearsal_interval_score"][key] = Wb, curve
        print(f"{key}: dev terciles {np.round(edges, 3).tolist()}, W_b = {Wb} (rehearsal IS90 {curve})", flush=True)
        bd, bt = bins(dev["feat"][fname], edges), bins(feats_t[fname], edges)
        for nm, a in ALPHA.items():
            lo, hi = base_interval(q, LEVELS, nm)
            dsc = dev["scores"][nm]
            fn = lambda j, st: past_only_init(dev["org"][bd == j], dsc[bd == j], st, Wb)[0]
            clo, chi = mondrian(org, tgt, day, lo, hi, y, a, bt, fn, Wb)
            res[key][nm] = interval_table(clo, chi, y, a, masks)
            store[(key, nm)] = (clo, chi)
        res["bin_coverage90"][key] = {meth: [float(np.mean((y[bt == j] >= store[(meth, "90")][0][bt == j]) & (y[bt == j] <= store[(meth, "90")][1][bt == j])))
                                             for j in range(K)] for meth in ("U", key)}
        res["bin_coverage90"][key]["n"] = [int((bt == j).sum()) for j in range(K)]
        res["bin_coverage90"][key]["test_ramp_share"] = [float(c["ramp"][bt == j].mean()) for j in range(K)]
        isu = interval_score(*store[("U", "90")], y, 0.1)
        ism = interval_score(*store[(key, "90")], y, 0.1)
        res["bootstrap_IS90_M_minus_U"][key] = {g: day_bootstrap((ism - isu)[masks[g]], day[masks[g]], c["sunny"][masks[g]])
                                                for g in ("overall", "ramp", "non-ramp")}
        u, m = res["U"]["90"], res[key]["90"]
        ok = (m["ramp"]["coverage"] - u["ramp"]["coverage"] >= 0.05 and 0.87 <= m["overall"]["coverage"] <= 0.93
              and m["overall"]["interval_score"] <= 1.02 * u["overall"]["interval_score"])
        res["decision"][key] = "improves conditional reliability" if ok else "does not"
        print(f"{key} 90%: overall cov {u['overall']['coverage']:.3f}->{m['overall']['coverage']:.3f}  ramp cov {u['ramp']['coverage']:.3f}->{m['ramp']['coverage']:.3f}"
              f"  IS {u['overall']['interval_score']:.2f}->{m['overall']['interval_score']:.2f}  width {u['overall']['width_kw']:.2f}->{m['overall']['width_kw']:.2f}"
              f"  ramp IS {u['ramp']['interval_score']:.2f}->{m['ramp']['interval_score']:.2f}  => {res['decision'][key]}", flush=True)
    p = save("y2_conditional_conformal.json", res)
    print("saved", p)


if __name__ == "__main__":
    main()
