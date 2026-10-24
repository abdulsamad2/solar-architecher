"""E2/E3: CNN-GRU point forecaster on temporally blocked dev folds.

E2 target: PV(t+15) in kW (trained on kW / 30).
E3 target: clear-sky index K(t+15) = PV / max(P_clear, min_clear); reconstructed to kW with the
           (known in advance) clear-sky PV at the target time. P_clear is fitted on the fold's
           TRAINING days only.
Both: identical inputs (16 chronological 64x64 frames + PV history / 30), identical model, optimiser and
budget; model selection on validation RMSE in kW.

E3b (post-hoc diagnostic extension, --target csi --input csi): as E3, but the PV-history input is
replaced by the clear-sky-index history K(t-j) computed with the same fold-fitted anchor.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines import split_scores
from src.clear_sky import ClearSkyPV, clear_sky_index, from_clear_sky_index, smart_persistence
from src.folds import calibration_split
from src.metrics import (crps_quantile_interp, interval_coverage, interval_width, pinball_loss, quantile_function,
                         quantile_reliability)
from src.events import event_labels, future_rows
from src.models import (Conv3dGRUEncoder, ConvGRUEncoder, DiffSkyEncoder, EventMixtureHead, Forecaster,
                        NonCrossingQuantileHead, PointHead, SmallSkyEncoder, TemporalTransformerEncoder)
from src.physics_losses import PhysicsConfig, total_loss
from src.skippd import SkippdForecastDataset

PV_SCALE = 30.0  # kW, ~nameplate; input/target scaling only
# E8 per-frame feature columns: 5 solar-position (P) + 6 motion (M); unused columns are zero-filled
E8_MASKS = {"none": [0] * 11, "P": [1] * 5 + [0] * 6, "M": [0] * 5 + [1] * 6, "PM": [1] * 11}
QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]  # roadmap Step 5.1


class GpuSeq:
    """GPU-resident chronological stacks: images [B,16,3,64,64], history feature [B,16].

    The per-record history feature defaults to PV / 30; ``set_history_feature`` swaps it (e.g. to K)
    and must be called per fold when it depends on fold-fitted quantities.
    """

    def __init__(self, ds: SkippdForecastDataset, device):
        self.images = ds.images.to(device)
        self.pv = ds.pv.to(device)
        self.hist = torch.from_numpy(ds.hist_rows).to(device)
        self.tgt = torch.from_numpy(ds.target_rows).to(device)
        self.hist_feat = self.pv / PV_SCALE
        self.extra_table = None  # E8: [R, 11] standardised, arm-masked per-record features

    def set_history_feature(self, per_record: np.ndarray | None) -> None:
        self.hist_feat = (self.pv / PV_SCALE if per_record is None
                          else torch.as_tensor(per_record, dtype=torch.float32, device=self.pv.device))

    def set_window(self, rows_chrono: np.ndarray) -> None:
        """E8: replace the 16 history rows per sample (chronological; -1 where the window is ineligible)."""
        self.hist = torch.from_numpy(np.ascontiguousarray(rows_chrono)).to(self.pv.device)

    def set_extra(self, table: np.ndarray | None) -> None:
        self.extra_table = None if table is None else torch.as_tensor(table, dtype=torch.float32, device=self.pv.device)

    def __call__(self, idx):
        rows = self.hist[idx]
        x = self.images[rows].permute(0, 1, 4, 2, 3).float().div_(255.0)
        if self.extra_table is None:
            return x, self.hist_feat[rows]
        return x, self.hist_feat[rows], self.extra_table[rows]


def build_encoder(cfg):
    extra = 11 if cfg.get("e8_window") else 0
    kind = cfg.get("encoder", "cnn_gru")
    if kind == "cnn_gru":
        return SmallSkyEncoder(image_features=cfg["image_features"], hidden_size=cfg["hidden_size"],
                               pool=cfg.get("pool", "avg"), extra_features=extra)
    if kind == "cnn_gru_wide":  # E9 A1
        return SmallSkyEncoder(image_features=128, hidden_size=128, pool="avg", extra_features=extra,
                               channels=(32, 64, 128, 128))
    if kind == "cnn3d_gru":  # E9 A2
        return Conv3dGRUEncoder(cfg["image_features"], cfg["hidden_size"], extra)
    if kind == "convgru":  # E9 A3
        return ConvGRUEncoder(cfg["hidden_size"], extra)
    if kind == "transformer":  # E9 A4
        return TemporalTransformerEncoder(cfg["image_features"], cfg["hidden_size"], extra)
    if kind == "diff_cnn_gru":  # E9 A5
        return DiffSkyEncoder(image_features=cfg["image_features"], hidden_size=cfg["hidden_size"],
                              pool=cfg.get("pool", "avg"), extra_features=extra)
    raise ValueError(kind)


def build_model(cfg) -> torch.nn.Module:
    enc = build_encoder(cfg)
    hs = enc.hidden_size
    head = cfg.get("head", "point")
    if head == "quantile":
        return Forecaster(enc, NonCrossingQuantileHead(hs, 1, QUANTILES))
    if head == "quantile_mlp":  # Phase 6 capacity-matched direct-quantile comparator (~ EventMixtureHead size)
        mlp = torch.nn.Sequential(torch.nn.Linear(hs, 24), torch.nn.ReLU(),
                                  NonCrossingQuantileHead(24, 1, QUANTILES))
        return Forecaster(enc, mlp)
    if head == "event":
        return Forecaster(enc, EventMixtureHead(hs, n_bins=3, n_comp=cfg.get("event_components", 2)))
    return Forecaster(enc, PointHead(hs, 1))


def _qhead(model):
    """The NonCrossingQuantileHead inside a (possibly MLP) quantile head."""
    return model.head[-1] if isinstance(model.head, torch.nn.Sequential) else model.head


def _out(pred: torch.Tensor) -> torch.Tensor:
    """Point head [B,1] -> [B]; quantile head [B,1,Q] -> [B,Q]."""
    return pred[:, 0, :] if pred.ndim == 3 else pred.squeeze(-1)


class Ema:
    """EMA of float params and buffers with warm-up d_n = min(decay, (1+n)/(10+n)); fused foreach update."""

    def __init__(self, model: torch.nn.Module, decay: float):
        import copy

        self.module = copy.deepcopy(model).eval()
        self.decay, self.n = decay, 0
        self._pairs = [(e, m) for e, m in zip(itertools.chain(self.module.parameters(), self.module.buffers()),
                                              itertools.chain(model.parameters(), model.buffers()))]
        self._float = [(e, m) for e, m in self._pairs if e.dtype.is_floating_point]
        self._other = [(e, m) for e, m in self._pairs if not e.dtype.is_floating_point]

    @torch.no_grad()
    def update_parameters(self, model: torch.nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        torch._foreach_lerp_([e for e, _ in self._float], [m.detach() for _, m in self._float], 1.0 - d)
        for e, m in self._other:
            e.copy_(m)


@torch.no_grad()
def predict_raw(model, data, idx, batch=1024):
    """Point [N], quantile [N,Q], or (event head) K(t+15) quantiles [N,Q] = K(t) + dK15 mixture quantiles."""
    model.eval()
    outs = []
    for s in range(0, len(idx), batch):
        ib = idx[s:s + batch]
        o = model(*data(ib))
        if isinstance(o, dict):
            o = model.head.marginal_quantiles(o, QUANTILES) + data.k_now[ib][:, None]
        else:
            o = _out(o)
            if getattr(data, "residual", False):  # residual quantile head: dK15 quantiles -> K(t+15) quantiles
                o = o + data.k_now[ib][:, None]
        outs.append(o)
    return torch.cat(outs)


@torch.no_grad()
def event_outputs(model, data, idx, tau: float, batch=1024) -> dict:
    model.eval()
    probs, p_abs = [], []
    for s in range(0, len(idx), batch):
        o = model(*data(idx[s:s + batch]))
        probs.append(o["time_logits"].softmax(-1))
        p_abs.append(model.head.prob_abs_change_at_least(o, tau))
    return {"time_probs": torch.cat(probs).cpu().numpy(), "p_abs_change_ge_tau": torch.cat(p_abs).cpu().numpy()}


def probabilistic_scores(pred_q: np.ndarray, y: np.ndarray, persistence: np.ndarray, clear: np.ndarray) -> dict:
    """kW-space scores for a quantile forecast, overall / sunny (clear day) / cloudy.

    CRPS: interpolated-quantile CRPS (src.metrics.crps_quantile_interp); persistence is a point mass, so
    its CRPS equals its MAE. Central intervals: 90% = [Q.05, Q.95], 80% = [Q.10, Q.90],
    50% = [Q.25, Q.75] from the interpolated quantile function.
    """
    qf = lambda tau: quantile_function(pred_q, QUANTILES, [tau])[:, 0]
    bounds = {"50": (qf(0.25), qf(0.75)), "80": (qf(0.10), qf(0.90)), "90": (qf(0.05), qf(0.95))}
    mean_fc = quantile_function(pred_q, QUANTILES, (np.arange(200) + 0.5) / 200).mean(axis=1)
    crps = crps_quantile_interp(pred_q, y, QUANTILES)
    out = {}
    for g, m in (("overall", np.ones_like(clear, bool)), ("sunny", clear), ("cloudy", ~clear)):
        if not m.any():
            continue
        out[g] = {
            "n": int(m.sum()),
            "crps_kw": float(crps[m].mean()),
            "persistence_crps_kw": float(np.abs(persistence[m] - y[m]).mean()),
            "median_mae_kw": float(np.abs(pred_q[m, QUANTILES.index(0.5)] - y[m]).mean()),
            "mean_rmse_kw": float(np.sqrt(np.mean((mean_fc[m] - y[m]) ** 2))),
            "coverage": {k: interval_coverage(lo[m], hi[m], y[m]) for k, (lo, hi) in bounds.items()},
            "width_kw": {k: interval_width(lo[m], hi[m]) for k, (lo, hi) in bounds.items()},
            "reliability": dict(zip(map(str, QUANTILES), quantile_reliability(pred_q[m], y[m]).round(4).tolist())),
        }
        out[g]["crps_skill_vs_persistence"] = 1 - out[g]["crps_kw"] / out[g]["persistence_crps_kw"]
    return out


def run_fold(ds, data, fold, target_kind, cfg, out_dir: Path, log) -> dict:
    device = data.pv.device
    day_str = ds.day.astype(str)
    train_days, val_days = set(fold["train_days"]), set(fold["val_days"])
    cal_days, cal_purged = set(), set()
    if cfg.get("calib_holdout"):
        # Gate 2 calibration: hold a calibration-day set out of the fold's training days (never fitted on)
        fit_l, cal_l, pur_l = calibration_split(fold["train_days"])
        cal_days, cal_purged = set(cal_l), set(pur_l)
        train_days = set(fit_l)
        # STOP conditions: any overlap between fit / calibration / validation / purged days
        assert not (train_days & cal_days), "STOP: fit and calibration days overlap"
        assert not (train_days & val_days) and not (cal_days & val_days), "STOP: overlap with validation days"
        assert not (cal_purged & (train_days | cal_days | val_days)), "STOP: purged days reused"
        (out_dir / "calibration_split.json").write_text(json.dumps(
            {"fit_days": sorted(train_days), "cal_days": sorted(cal_days), "purged_days": sorted(cal_purged)}, indent=1))
    tr = np.flatnonzero(np.isin(day_str, list(train_days)))
    va = np.flatnonzero(np.isin(day_str, list(val_days)))
    ca = np.flatnonzero(np.isin(day_str, list(cal_days)))
    assert not (set(tr) & set(ca)) and not (set(tr) & set(va)) and not (set(ca) & set(va)), "STOP: sample overlap"
    va_extra = np.array([], dtype=np.int64)
    if cfg.get("e8_window"):
        # E8: train / calibrate / validate on the COMMON population only (identical samples in every arm);
        # window-eligible but non-common validation samples are predicted separately for the fallback analysis
        common, elig = ds.e8_common, ds.e8_eligible
        va_extra = va[elig[va] & ~common[va]]
        tr, va, ca = tr[common[tr]], va[common[va]], ca[common[ca]]
        # per-frame features: z-score with statistics from FIT-day samples' window frames only, then arm mask
        fit_rows = np.unique(ds.e8_rows[tr].ravel())
        mu, sd = ds.e8_table[fit_rows].mean(0), ds.e8_table[fit_rows].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        mask = np.array(E8_MASKS[cfg["e8_features"]], np.float32)
        data.set_extra(((ds.e8_table - mu) / sd * mask).astype(np.float32))
        (out_dir / "e8_standardisation.json").write_text(json.dumps(
            {"mu": mu.tolist(), "sd": sd.tolist(), "mask": mask.tolist(), "n_fit_rows": int(len(fit_rows)),
             "n_train": int(len(tr)), "n_val": int(len(va)), "n_cal": int(len(ca)), "n_val_extra": int(len(va_extra))}))

    # clear-sky anchor fitted on training-day records only (fit days only in calibration mode)
    rec_times = ds.record_times
    rec_days = rec_times.astype("datetime64[D]").astype(str)
    train_rec = np.isin(rec_days, list(train_days))
    if cfg.get("site", "skippd") == "gatton":  # Phase E: Gatton clear-sky model (frozen geometry, scale fitted here)
        from src.gatton_clearsky import GattonClearSky
        cs = GattonClearSky().fit(rec_times[train_rec], ds.pv.numpy()[train_rec])
    else:
        cs = ClearSkyPV().fit(rec_times[train_rec], ds.pv.numpy()[train_rec])
    pclear_rec = cs(rec_times)
    pclear_tgt = pclear_rec[ds.target_rows]
    pclear_now = pclear_rec[ds.hist_rows[:, -1]]
    y_kw = ds.pv.numpy()[ds.target_rows].astype(float)
    if target_kind == "kw":
        y = y_kw / PV_SCALE
    else:
        y = clear_sky_index(y_kw, pclear_tgt, cfg["min_clear_kw"])
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    # history input: PV/30 (E2, E3) or K from the same fold-fitted anchor (E3b); only records <= origin are used
    data.set_history_feature(
        clear_sky_index(ds.pv.numpy().astype(float), pclear_rec, cfg["min_clear_kw"]) if cfg["input"] == "csi" else None)

    def to_kw(raw: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Model output -> kW. For quantiles, multiplying by a positive P_clear preserves their order."""
        if target_kind == "kw":
            return raw * PV_SCALE
        pc = pclear_tgt[idx][:, None] if raw.ndim == 2 else pclear_tgt[idx]
        return from_clear_sky_index(raw, pc, cfg["min_clear_kw"])

    event = cfg.get("head", "point") == "event"
    quantile = cfg.get("head", "point") in ("quantile", "quantile_mlp", "event")
    sel_key = "val_crps_kw" if quantile else "val_rmse_kw"
    data.residual = bool(cfg.get("residual"))
    if data.residual:
        # persistence-residual quantiles (E6-QR): fit dK15 = K(t+15) - K(t); K(t) is an input-time quantity
        assert quantile and not event and target_kind == "csi" and cfg["input"] == "csi"
        k_rec_r = clear_sky_index(ds.pv.numpy().astype(float), pclear_rec, cfg["min_clear_kw"])
        k_now_r = k_rec_r[ds.hist_rows[:, -1]]
        data.k_now = torch.tensor(k_now_r, dtype=torch.float32, device=device)
        y_t = torch.tensor(y - k_now_r, dtype=torch.float32, device=device)
    if event:
        # Phase 6 labels (never inputs): K trajectory t+1..t+15 with this fold's anchor; tau from TRAINING samples
        assert target_kind == "csi" and cfg["input"] == "csi", "event head runs in the K representation"
        k_rec = clear_sky_index(ds.pv.numpy().astype(float), pclear_rec, cfg["min_clear_kw"])
        k_now = k_rec[ds.hist_rows[:, -1]]
        k_fut = np.where(ds.future_rows >= 0, k_rec[np.maximum(ds.future_rows, 0)], np.nan)
        tau = float(np.quantile(np.abs(y - k_now)[tr], 0.9))
        if cfg.get("frozen_thresholds"):
            frozen = json.loads(Path(cfg["frozen_thresholds"]).read_text())[str(fold["fold"])]["training_labels_fit_days"]
            assert abs(tau - frozen) < 1e-5, f"STOP: event threshold {tau:.6f} != frozen {frozen:.6f} (fold {fold['fold']})"
        ev = event_labels(k_now, k_fut, y, tau)
        dk_t = torch.tensor(ev["severity"], dtype=torch.float32, device=device)
        cls_t = torch.tensor(ev["time_class"], dtype=torch.long, device=device)
        valid_t = torch.tensor(ev["event_valid"], dtype=torch.bool, device=device)
        data.k_now = torch.tensor(k_now, dtype=torch.float32, device=device)

    def val_score(kw: np.ndarray) -> float:
        if quantile:
            return float(crps_quantile_interp(kw, y_kw[va], QUANTILES).mean())
        return float(np.sqrt(np.mean((kw - y_kw[va]) ** 2)))

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    model.encoder.cnn_bf16 = cfg["cnn_bf16"]
    tr_dev = torch.as_tensor(tr, device=device)
    w_t = None
    if cfg.get("loss_weight_alpha"):
        # E9 L1: input-only weights; v = mean |K_j - K_{j-1}| over the window frames, median over FIT-day samples
        assert cfg["input"] == "csi" and quantile and not event
        kh = data.hist_feat[data.hist].detach().cpu().numpy()          # [N, 16] K history per sample (inputs only)
        v = np.abs(np.diff(kh, axis=1)).mean(1)
        med = float(np.median(v[tr]))
        w = 1.0 + cfg["loss_weight_alpha"] * np.minimum(v / max(med, 1e-6), 5.0)
        w_t = torch.tensor(w, dtype=torch.float32, device=device)
        (out_dir / "loss_weights.json").write_text(json.dumps({"alpha": cfg["loss_weight_alpha"], "median_v_fit": med,
                                                               "w_mean_train": float(w[tr].mean())}))
    phys = PhysicsConfig.from_dict(cfg.get("physics"))
    if phys.any_enabled:
        # physics-informed auxiliary terms (src/physics_losses.py); quantile heads in the K representation only
        assert quantile and not event and target_kind == "csi" and cfg["input"] == "csi" and w_t is None, \
            "physics losses apply to K-quantile heads only"
        k_rec_p = clear_sky_index(ds.pv.numpy().astype(float), pclear_rec, cfg["min_clear_kw"])
        k_now_p = k_rec_p[ds.hist_rows[:, -1]]
        fit_all = np.flatnonzero(np.isin(day_str, list(train_days)))  # all fit-day samples (before any E8 restriction)
        tau_p = float(np.quantile(np.abs(y - k_now_p)[fit_all], 0.9))  # existing ramp/event threshold rule (fit days)
        if cfg.get("frozen_thresholds"):
            frozen_p = json.loads(Path(cfg["frozen_thresholds"]).read_text())[str(fold["fold"])]["training_labels_fit_days"]
            assert abs(tau_p - frozen_p) < 1e-5, f"STOP: ramp threshold {tau_p:.6f} != frozen {frozen_p:.6f}"
        phys_t = {"k_now": torch.tensor(k_now_p, dtype=torch.float32, device=device),
                  "p_true": torch.tensor(y_kw, dtype=torch.float32, device=device),
                  "p_clear": torch.tensor(pclear_tgt, dtype=torch.float32, device=device),
                  "is_ramp": torch.tensor(np.abs(y - k_now_p) >= tau_p, device=device),
                  "k_true": torch.tensor(y, dtype=torch.float32, device=device)}
        (out_dir / "physics.json").write_text(json.dumps(
            {**phys.__dict__, "tau_ramp_fit_days": tau_p, "ramp_share_train": float((np.abs(y - k_now_p) >= tau_p)[tr].mean()),
             "k_target_p999_train": float(np.quantile(y[tr], 0.999))}, indent=1))
    if event:  # train-only climatology init (class frequencies, per-class dK spread)
        keep = tr_dev[valid_t[tr_dev]]
        model.head.init_from_targets(dk_t[keep], cls_t[keep])
    elif quantile:  # same train-only procedure for every target representation
        _qhead(model).init_from_targets(y_t[tr_dev])
    if cfg.get("init_from"):
        # Phase E fine-tuning: start from the SKIPP'D member with the same seed and fold index (after head init)
        init_ckpt = Path(cfg["init_from"]) / f"seed{cfg['seed']}" / f"fold_{fold['fold']}" / "best.pt"
        model.load_state_dict(torch.load(init_ckpt, map_location=device))
        log(f"  initialised from {init_ckpt}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = -(-len(tr) // cfg["batch_size"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"] * steps_per_epoch)
             if cfg["schedule"] == "cosine" else None)
    # EMA of weights and BN buffers; the EMA copy is what gets validated and checkpointed
    ema = Ema(model, cfg["ema_decay"]) if cfg["ema_decay"] else None
    eval_model = ema.module if ema is not None else model
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    tr_t, va_t = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)

    def val_score_of(m) -> float:
        return val_score(to_kw(predict_raw(m, data, va_t).cpu().numpy().astype(float), va))

    best, wait = float("inf"), 0
    hist = {"train_loss": [], sel_key: [], f"{sel_key}_raw_weights": [], "grad_norm_p99": []}
    if phys.any_enabled:
        hist["loss_components"] = []
    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=device)]
        tot, n, gnorms = 0.0, 0, []
        comp_sum = {}
        for s in range(0, len(perm), cfg["batch_size"]):
            b = perm[s:s + cfg["batch_size"]]
            if event:
                loss = model.head.loss(model(*data(b)), dk_t[b], cls_t[b], valid_t[b],
                                       supervised=cfg["event_supervised"])
            elif phys.any_enabled:
                pred = _out(model(*data(b)))
                k_q = pred + phys_t["k_now"][b].unsqueeze(-1) if data.residual else pred
                loss, parts = total_loss(k_q, phys_t["k_true"][b], QUANTILES, phys, k_now=phys_t["k_now"][b],
                                         p_true_kw=phys_t["p_true"][b], p_clear_kw=phys_t["p_clear"][b],
                                         is_ramp=phys_t["is_ramp"][b])
                for kk, vv in parts.items():
                    comp_sum[kk] = comp_sum.get(kk, 0.0) + vv * len(b)
            else:
                pred = _out(model(*data(b)))
                if w_t is not None:
                    qs = torch.as_tensor(QUANTILES, device=device, dtype=pred.dtype)
                    err = y_t[b].unsqueeze(-1) - pred
                    per = torch.maximum(qs * err, (qs - 1) * err).mean(-1)
                    loss = (w_t[b] * per).sum() / w_t[b].sum()
                else:
                    loss = pinball_loss(pred, y_t[b], QUANTILES) if quantile else F.mse_loss(pred, y_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"] or float("inf")))
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += loss.item() * len(b)
            n += len(b)
        val_s = val_score_of(eval_model)
        raw_s = val_score_of(model) if ema is not None else val_s
        hist["train_loss"].append(tot / n)
        hist[sel_key].append(val_s)
        hist[f"{sel_key}_raw_weights"].append(raw_s)
        hist["grad_norm_p99"].append(float(torch.quantile(torch.stack(gnorms).float(), 0.99)))
        if phys.any_enabled:
            hist["loss_components"].append({kk: vv / n for kk, vv in comp_sum.items()})
            log("    components " + "  ".join(f"{kk} {vv / n:.5f}" for kk, vv in comp_sum.items()))
        improved = val_s < best
        if improved:
            best, wait = val_s, 0
            torch.save(eval_model.state_dict(), out_dir / "best.pt")
        else:
            wait += 1
        log(f"  epoch {epoch + 1:3d}  train loss {tot / n:.5f}  val {'CRPS' if quantile else 'RMSE'} {val_s:.3f} kW"
            f" (raw {raw_s:.3f})"
            f"{'  *' if improved else ''}  lr {opt.param_groups[0]['lr']:.1e}  |g|p99 {hist['grad_norm_p99'][-1]:.2f}"
            f"  ({time.time() - t0:.0f}s)")
        if wait >= cfg["patience"]:
            break

    model = eval_model
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    out_kw = to_kw(predict_raw(model, data, va_t).cpu().numpy().astype(float), va)
    pv_now = ds.pv.numpy()[ds.hist_rows[va, -1]].astype(float)
    pers = smart_persistence(pv_now, pclear_now[va], pclear_tgt[va], cfg["min_clear_kw"])
    labels = np.array([fold["day_labels"][d] for d in day_str[va]])
    extra = {}
    if quantile:
        pred_q = out_kw
        pred_kw = pred_q[:, QUANTILES.index(0.5)]  # median as the point forecast
        extra["pred_q"] = pred_q
    else:
        pred_kw = out_kw
    metrics = {
        "best_epoch": int(np.argmin(hist[sel_key])) + 1, "epochs_run": len(hist[sel_key]),
        "selection_metric": sel_key, "clear_sky_coef": cs.coef.tolist(),
        "model": split_scores(pred_kw, y_kw[va], labels),
        "smart_persistence": split_scores(pers, y_kw[va], labels),
    }
    metrics["skill_vs_smart_persistence"] = {
        g: 1 - metrics["model"][g]["rmse_kw"] / metrics["smart_persistence"][g]["rmse_kw"] for g in metrics["model"]}
    if quantile:
        metrics["probabilistic"] = probabilistic_scores(pred_q, y_kw[va], pers, labels)
    if event:
        eo = event_outputs(model, data, va_t, tau)
        extra.update(time_probs=eo["time_probs"], p_abs_change_ge_tau=eo["p_abs_change_ge_tau"],
                     time_class=ev["time_class"][va], event_valid=ev["event_valid"][va],
                     severity_dk=ev["severity"][va], arrival_min=ev["arrival_min"][va], tau=np.array(tau))
        v_ok = ev["event_valid"][va]
        onehot = np.eye(4)[ev["time_class"][va]]
        metrics["event"] = {"tau": tau, "time_class_brier": float(((eo["time_probs"] - onehot) ** 2).sum(1)[v_ok].mean()),
                            "time_class_logloss": float(-np.log(np.clip(eo["time_probs"][v_ok, ev["time_class"][va][v_ok]], 1e-12, 1)).mean()),
                            "supervised": cfg["event_supervised"]}
    def meta(idx):  # per-sample context saved alongside predictions
        return {"origin_time_us": ds.origin_time[idx].astype("datetime64[us]").astype(np.int64),
                "target_time_us": ds.target_time[idx].astype("datetime64[us]").astype(np.int64),
                "pclear_target_kw": pclear_tgt[idx], "pclear_now_kw": pclear_now[idx],
                "pv_now_kw": ds.pv.numpy()[ds.hist_rows[idx, -1]].astype(float),
                "fold": np.full(len(idx), fold["fold"], np.int8)}

    np.savez_compressed(out_dir / "preds_val.npz", sample_ids=ds.sample_ids[va], target=y_kw[va], pred=pred_kw,
                        persistence=pers, day=day_str[va], clear_day=labels, quantile_levels=np.array(QUANTILES),
                        **extra, **meta(va))
    if len(va_extra):
        # E8 fallback population: window-eligible validation samples outside the common population
        vx = torch.as_tensor(va_extra, device=device)
        qx = to_kw(predict_raw(model, data, vx).cpu().numpy().astype(float), va_extra)
        np.savez_compressed(out_dir / "preds_val_extra.npz", sample_ids=ds.sample_ids[va_extra], target=y_kw[va_extra],
                            pred_q=qx, pred=qx[:, QUANTILES.index(0.5)], day=day_str[va_extra],
                            clear_day=np.array([fold["day_labels"][d] for d in day_str[va_extra]]),
                            quantile_levels=np.array(QUANTILES), **meta(va_extra))
    if len(ca):
        # calibration-day predictions from the SAME checkpoint (trained on fit days only)
        ca_t = torch.as_tensor(ca, device=device)
        cal_out = to_kw(predict_raw(model, data, ca_t).cpu().numpy().astype(float), ca)
        cal_pers = smart_persistence(ds.pv.numpy()[ds.hist_rows[ca, -1]].astype(float), pclear_now[ca],
                                     pclear_tgt[ca], cfg["min_clear_kw"])
        cal_extra = {"pred_q": cal_out, "pred": cal_out[:, QUANTILES.index(0.5)]} if quantile else {"pred": cal_out}
        np.savez_compressed(out_dir / "preds_cal.npz", sample_ids=ds.sample_ids[ca], target=y_kw[ca],
                            persistence=cal_pers, day=day_str[ca], quantile_levels=np.array(QUANTILES),
                            **cal_extra, **meta(ca))
        metrics["calibration_split"] = {"n_fit_samples": int(len(tr)), "n_cal_samples": int(len(ca)),
                                        "n_val_samples": int(len(va)), "n_fit_days": len(train_days),
                                        "n_cal_days": len(cal_days), "n_purged_days": len(cal_purged)}
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(a):
    cfg = {"seed": a.seed, "lr": a.lr, "weight_decay": 1e-4, "batch_size": a.batch_size, "max_epochs": a.max_epochs,
           "patience": a.patience, "image_features": 64, "hidden_size": 64, "min_clear_kw": 1.0, "cnn_bf16": not a.fp32_cnn,
           "schedule": a.schedule, "clip": a.clip, "ema_decay": a.ema_decay, "pool": a.pool, "input": a.input,
           "label": a.label, "head": a.head, "quantiles": QUANTILES if a.head == "quantile" else None,
           "calib_holdout": a.calib_holdout, "event_supervised": not a.event_unsupervised, "residual": a.residual,
           "frozen_thresholds": a.frozen_thresholds,
           "event_components": a.event_components, "e8_window": a.e8_window, "e8_features": a.e8_features,
           "encoder": a.encoder, "loss_weight_alpha": a.loss_weight_alpha}
    if a.site != "skippd":
        cfg["site"] = a.site
    if a.init_from:
        cfg["init_from"] = a.init_from
    cfg["min_clear_kw"] = a.min_clear_kw
    if a.physics_config:
        import yaml
        cfg["physics"] = yaml.safe_load(Path(a.physics_config).read_text())
        cfg["physics_config_path"] = a.physics_config
    if a.e8_window:
        from src.e8_features import FEAT, verify_frozen

        if a.e8_dir:
            FEAT = Path(a.e8_dir)  # Phase E: site-specific windows (no frozen SKIPP'D geometry check)
        else:
            verify_frozen()  # STOP if the frozen geometry / eligibility manifests changed
        cfg["e8_feature_hashes"] = json.loads((FEAT / "e8_features_v1.json").read_text())
    out_root = Path(a.out) / a.target / f"seed{a.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_f = open(out_root / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    folds_doc = json.loads(Path(a.folds).read_text())
    import subprocess

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())
    (out_root / "config.json").write_text(json.dumps({**cfg, "git_commit": commit, "src_dirty": dirty,
                                                      "target": a.target, "manifest": a.manifest,
                                                      "folds": a.folds, "folds_sha256": folds_doc["content_sha256"]}, indent=2))
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ds = SkippdForecastDataset(a.h5, a.manifest, "dev", preload=True)
    m = np.load(a.manifest)
    ds.record_times = m["times_us"].astype("datetime64[us]")[m["hdf5_group"] == 0][np.argsort(m["hdf5_row"][m["hdf5_group"] == 0])]
    if a.head == "event":  # label-only lookups of the records at t+1..t+15 min (never model inputs)
        ds.future_rows = future_rows(ds.record_times.astype(np.int64), ds.origin_time.astype("datetime64[us]").astype(np.int64))
    if a.subset_days:
        for fd in folds_doc["folds"]:
            fd["train_days"], fd["val_days"] = fd["train_days"][:a.subset_days], fd["val_days"][:a.subset_days]
    data = GpuSeq(ds, device)
    if a.e8_window:
        win = np.load(FEAT / "e8_windows_v1.npz")
        rec = np.load(FEAT / "e8_record_features_v1.npz")
        assert np.array_equal(win["sample_ids"], ds.sample_ids), "STOP: E8 window order differs from dataset"
        rows = win[a.e8_window][:, ::-1]                     # chronological (oldest first), as GpuSeq expects
        ds.e8_eligible = (rows >= 0).all(1)
        ds.e8_common = win["eligible_common"]
        assert (ds.e8_eligible[ds.e8_common]).all()
        ds.e8_rows = np.where(ds.e8_eligible[:, None], rows, ds.hist_rows)  # placeholder rows are never sampled
        assert (ds.e8_rows[:, -1] == ds.hist_rows[:, -1]).all(), "STOP: window origin differs from forecast origin"
        ds.e8_table = np.concatenate([rec["P"], rec["M"]], 1).astype(np.float32)
        data.set_window(ds.e8_rows)
    summary = {}
    for k in a.fold_ids:
        fold = {**folds_doc["folds"][k], "day_labels": folds_doc["day_labels"]}
        fold_dir = out_root / f"fold_{k}"
        fold_dir.mkdir(exist_ok=True)
        log(f"[{a.target}] fold {k}: val {fold['val_range'] if 'val_range' in fold else ''}")
        summary[k] = run_fold(ds, data, fold, a.target, cfg, fold_dir, log)
        s = summary[k]
        log(f"[{a.target}] fold {k} done: best epoch {s['best_epoch']}  RMSE "
            + "  ".join(f"{g} {s['model'][g]['rmse_kw']:.3f} (pers {s['smart_persistence'][g]['rmse_kw']:.3f})" for g in s["model"]))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    log_f.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="E2 (kW) / E3 (clear-sky index) CNN-GRU on blocked folds")
    p.add_argument("--target", choices=["kw", "csi"], required=True)
    p.add_argument("--input", choices=["pv", "csi"], default="pv", help="history input: PV/30 or clear-sky index K")
    p.add_argument("--label", default="", help="free-text experiment label stored in config.json")
    p.add_argument("--calib-holdout", action="store_true",
                   help="hold ~20%% of training days out as calibration days (src.folds.calibration_split)")
    p.add_argument("--event-unsupervised", action="store_true",
                   help="event head trained on marginal NLL only (capacity-matched ablation of event supervision)")
    p.add_argument("--event-components", type=int, default=2)
    p.add_argument("--frozen-thresholds", default="", help="JSON of frozen per-fold event thresholds; mismatch = STOP")
    p.add_argument("--e8-window", choices=["H15", "H30", "H60"], default=None,
                   help="E8 history window (16 frames, stride 1/2/4 min); restricts to the common population")
    p.add_argument("--encoder", default="cnn_gru",
                   choices=["cnn_gru", "cnn_gru_wide", "cnn3d_gru", "convgru", "transformer", "diff_cnn_gru"])
    p.add_argument("--loss-weight-alpha", type=float, default=0.0, help="E9 L1 input-weighted pinball (0 = off)")
    p.add_argument("--physics-config", default="", help="YAML with physics_loss / ramp_loss / bound_loss / direction_loss")
    p.add_argument("--site", choices=["skippd", "gatton"], default="skippd", help="clear-sky model (Phase E: gatton)")
    p.add_argument("--min-clear-kw", type=float, default=1.0, help="clear-sky floor (Gatton: 108.8 kW, prereg 3.6)")
    p.add_argument("--init-from", default="", help="Phase E fine-tuning: SKIPP'D run root with seed*/fold_*/best.pt")
    p.add_argument("--e8-dir", default="", help="Phase E: directory with site-specific e8 windows/features")
    p.add_argument("--e8-features", choices=list(E8_MASKS), default="none",
                   help="E8 per-frame features: none / P (solar) / M (motion) / PM; unused columns zero-filled")
    p.add_argument("--residual", action="store_true",
                   help="quantile heads predict dK15 = K(t+15) - K(t) (persistence-residual, E6-QR)")
    p.add_argument("--head", choices=["point", "quantile", "quantile_mlp", "event"], default="point",
                   help="point (MSE) or non-crossing quantile head (pinball); quantile selects on val CRPS in kW")
    p.add_argument("--fold-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--fp32-cnn", action="store_true", help="disable bf16 for the CNN (7x slower); GRU/head are always fp32")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip; 0 disables")
    p.add_argument("--ema-decay", type=float, default=0.999, help="0 disables")
    p.add_argument("--pool", choices=["avg", "spatial"], default="avg")
    p.add_argument("--subset-days", type=int, default=0, help="debug: limit train/val days per fold")
    p.add_argument("--h5", default="data/raw/2017_2019_images_pv_processed.hdf5")
    p.add_argument("--manifest", default="data/manifests/forecast_tol30_v2.npz")
    p.add_argument("--folds", default="data/manifests/folds_blocked5_v1.json")
    p.add_argument("--out", default="experiments/E2E3_cnn_gru/v1")
    main(p.parse_args())
