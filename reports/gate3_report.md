# Gate 3 report: event-time / attenuation-severity decomposition (Phase 6)

2026-10-21.
- **Pre-registration:** `reports/phase6_prereg.md` (frozen at 4e2a2d0 before any run). No architecture, training or rule changes were made between freezing and this report.
- **Raw numbers:** `reports/gate3_results.json`, `reports/gate3_severity.json`.
- **Scope:**
  - Development folds only; the official 20-day test set remains sealed.
  - Gate 2 (NOT MET) and E4-CAL2 are preserved unchanged.
  - Seeds 42/123/456; 164,362 identical fold-validation samples across all four arms (asserted).
  - No stop condition triggered: frozen thresholds matched in every fold/run, no overlap, no crossing quantiles, causal conformal updates only.

## Verdict (pre-registered rule)
**The event-time/severity decomposition is NOT retained: 0 of 3 seeds pass, and E6-S beats E6-QR on ramp CRPS in 0 of 5 folds (seed mean).**

| Seed | Ramp CRPS E6-S − E6-QR | Ramp CRPS E6-S − E6-U | Overall E6-S − E6-QR (δ = 0.02) | Pass |
|---|---|---|---|---|
| 42 | +0.119 [+0.085, +0.155] | −0.051 [−0.072, −0.031] | −0.043 | no (condition 1) |
| 123 | +0.118 [+0.088, +0.150] | −0.075 [−0.095, −0.054] | −0.037 | no (condition 1) |
| 456 | +0.242 [+0.212, +0.276] | +0.057 [+0.033, +0.081] | −0.026 | no (conditions 1 and 2) |

## 1. Raw predictive distributions: CRPS in kW (mean ± SD over 3 seeds)
| Stratum | E4-B-fit (direct K) | E6-QR (residual quantiles) | E6-U (mixture, unsupervised) | E6-S (mixture + event supervision) | Persistence |
|---|---|---|---|---|---|
| Overall | 0.751 ± 0.002 | 0.646 ± 0.009 | 0.612 ± 0.003 | **0.610 ± 0.000** | 0.877 |
| Sunny | 0.305 | 0.149 | **0.123** | **0.123** | 0.223 |
| Cloudy | 1.068 | 0.999 | 0.961 | **0.958** | 1.343 |
| **Ramp (primary)** | **2.448 ± 0.010** | 2.771 ± 0.052 | 2.954 ± 0.052 | 2.931 ± 0.019 | 4.672 |
| Non-ramp | 0.560 | 0.408 | **0.350** | 0.351 | – |
| Fold 1 | 1.315 | 1.157 | 1.105 | **1.095** | – |
| Fold 1, Sep–Oct 2017 | 0.997 | 0.759 | **0.737** | 0.743 | – |

## 2. Attribution (paired day-bootstrap 95% CIs; per seed 42 / 123 / 456)
| Effect | Overall | Sunny | Cloudy | **Ramp** | Non-ramp |
|---|---|---|---|---|---|
| **Residual target** (E6-QR − E4-B-fit) | −0.100 / −0.100 / −0.114, all CIs < 0 | −0.146 / −0.157 / −0.166 | −0.067 / −0.060 / −0.077 | **+0.344 / +0.351 / +0.274, all CIs > 0** | −0.149 / −0.151 / −0.158 |
| **Mixture architecture** (E6-U − E6-QR) | −0.041 / −0.032 / −0.027, all < 0 | −0.032 / −0.029 / −0.017 | −0.047 / −0.035 / −0.034 | **+0.170 / +0.193 / +0.185, all > 0** | −0.064 / −0.057 / −0.051 |
| **Event supervision** (E6-S − E6-U) | −0.002 / −0.005 / +0.002 | ≈ 0 | −0.004 / −0.008 / +0.002 | −0.051 / −0.075 / **+0.057** (sign flips) | +0.003 / +0.003 / −0.005 |

**Interpretation:**
1. **Residual target (persistence-residual forecasting, the requested follow-up).**
   - **Large gain** overall (−13 to −15%), and especially on sunny and non-ramp samples.
   - It **closes the sunny-day gap to persistence for the first time**: E6-QR 0.149, E6-U/E6-S 0.123, against persistence 0.223.
   - The cost is **substantially worse ramp CRPS (+0.27 to +0.35 kW)**. Anchoring the forecast on K(t) concentrates mass near "no change", and ramps are where that is wrong.
2. **Mixture form:** a further overall gain (−0.03 to −0.04) with the **same trade-off**, a further ramp loss (+0.17 to +0.19).
3. **Event supervision:** no consistent effect on CRPS. Ramp gains in 2 seeds reverse in the third, and overall/sunny/cloudy changes are ≤ 0.008 kW.

## 3. What event supervision does learn
**Arrival classes** (4-class; mean of 3 seeds):

| | Brier | Log loss | Bin accuracy on true events | Mean P(event) on true events | Mean P(event) on non-events |
|---|---|---|---|---|---|
| Training climatology | 0.277 | 0.598 | – | – | – |
| E6-U (unsupervised; classes not tied to meaning) | 0.238 | 0.475 | 0.367 | 0.670 | 0.170 |
| **E6-S** | **0.202** | **0.401** | **0.455** | 0.507 | **0.085** |

**Event occurrence** P(|ΔK₁₅| ≥ τ): Brier 0.0663 to 0.0671 for **all four arms**, against 0.0906 for climatology. The explicit event head gives **no advantage** in deciding *whether* a ramp occurs over probabilities implied by the direct quantile baseline.

**Severity on true ramps** (median ΔK₁₅, each arm's K space):

| Arm | MAE | Sign accuracy | 80% coverage of ΔK |
|---|---|---|---|
| E4-B-fit | **0.376** | 0.791 | 0.432 |
| E6-QR | 0.388 | **0.817** | 0.358 |
| E6-U | 0.436 | 0.793 | 0.421 |
| E6-S | 0.438 | 0.797 | 0.444 |

**E6-S read-only checkpoint pass:** given the **true** arrival class, the class-conditional mean severity has MAE **0.287–0.292**, against 0.315–0.321 for the marginal mean (all seeds). The severity heads do encode timing-specific attenuation. The bottleneck is **detecting arrival from the input window**: true events receive only about 0.51 probability, and arrival-bin accuracy is 0.45.

## 4. Calibrated intervals (E4-CAL2 rolling conformal, causal) and conditional coverage
90% intervals, seed mean (coverage / width kW / interval score):

| Stratum | E4-B-fit | E6-QR | E6-U | E6-S |
|---|---|---|---|---|
| Overall | 0.890 / 5.94 / 7.28 | 0.897 / 5.22 / 7.00 | 0.898 / 3.72 / 5.48 | 0.897 / 3.83 / **5.44** |
| **Ramp** | 0.613 / 8.36 / **17.72** | 0.545 / 8.06 / 22.05 | 0.642 / 8.72 / 22.00 | **0.663** / 8.98 / 20.93 |
| Non-ramp | 0.921 / 5.67 / 6.11 | 0.936 / 4.90 / 5.31 | 0.927 / 3.16 / **3.63** | 0.924 / 3.25 / 3.71 |
| Fold 1 | 0.900 / 8.84 / 11.19 | 0.894 / 7.29 / 10.98 | 0.890 / 6.45 / 9.99 | 0.890 / 6.44 / **9.79** |

- **All arms after rolling conformal:** 50/80/90% overall coverage is 0.50/0.80/0.90.
- **E6-S intervals:** **35% narrower** than E4-B-fit at equal marginal coverage (interval score −25%).
- **Ramp-conditioned coverage stays poor for every arm (0.55–0.66 at 90%).** E6-S has the highest ramp coverage but a worse ramp interval score than E4-B-fit.
- **No conditional coverage guarantee is claimed.**

## 5. Computational cost (RTX 5000 Ada)
| Arm | Head / total params | Batch-1 latency | Throughput | Epochs, 5 folds (seed mean) | Training wall time (s, 2 runs sharing GPU) |
|---|---|---|---|---|---|
| E4-B-fit | 585 / 90,761 | 2.8 ms | 10,600/s | 46 | ≈ 3,500 |
| E6-QR | 1,785 / 91,961 | 2.8 ms | 10,600/s | 52 | 1,765–4,111 |
| E6-U | 1,820 / 91,996 | **19.7 ms** | 8,600/s | 80 | 6,822–7,222 |
| E6-S | 1,820 / 91,996 | **19.7 ms** | 8,600/s | 79 | 7,213–7,893 |

- **Latency:** the mixture's 60-step CDF bisection makes predictive quantiles about 7× slower at batch 1, though still far under real-time needs.
- **Training:** the event arms trained about 2× longer. They **mostly ran to the 15-epoch cap** (best epochs often 11–15), while the quantile arms early-stopped.
- **Caveat on the matched budget:** the event arms may be budget-limited under the shared clip of 1.0 (raw gradient p99 ≈ 100–700). This was disclosed before launch and is not a basis for re-running under this gate.

## 6. Conclusions
1. **The SHADOW-MoE core hypothesis is not supported as specified.** The event-time/severity decomposition does not improve ramp CRPS over residual quantiles at matched capacity, and event supervision has no consistent CRPS effect. **Gate 3: NOT RETAINED.**
2. **The mechanism is partly validated:**
   - supervision produces genuinely informative arrival probabilities (Brier −27% vs climatology, −15% vs unsupervised);
   - severity conditional on the true arrival class is more accurate than marginal severity.
   - The limiting factor is **arrival detection from a 15-minute, 64×64 image history**, not the decomposition itself.
3. **A strong, replicated positive result came from the controls:** persistence-residual forecasting (+ mixture form) cuts overall CRPS by 19% vs E4-B-fit and 30% vs persistence, and **beats persistence on sunny days** (0.123 vs 0.223), the weakness of every earlier model including SUNSET. Its cost is a large, replicated **ramp penalty**: a calm-vs-ramp trade-off, not a uniform gain.
4. **Ramp forecasting is the open problem.** The best ramp model remains direct K quantiles (E4-B-fit). No arm reaches acceptable ramp coverage (≤ 0.66 at 90%).

## 7. Recommendations for the next step (each needs its own pre-registration; nothing here changes Gate 3)
- **Report the residual-target result** as the main validated finding: a large overall and sunny-day gain, with an explicit ramp trade-off.
- **Combine the strengths of the two families:** a pre-registered blend or gate between a ramp-robust direct-K forecaster and a calm-condition residual forecaster, e.g. a learned or rule-based switch on the arrival probability, which *is* informative.
  - This is the natural "mixture-of-experts" reading of SHADOW-MoE.
  - It must be evaluated against both components at matched data and budget.
- **If the event decomposition is revisited,** the evidence points to **better arrival detection** (higher-resolution or longer image history, sun-region features, cloud-motion cues), not to the severity side.
  - A training-budget sensitivity check for the NLL arms is also warranted, as a new, separately pre-registered experiment.
- **Data issues to carry forward** (`sepoct2017_audit.md`):
  - an inter-annual PV/clear-sky level decline of about 2.5–3%/yr;
  - an evidenced spring-2019 camera colour shift (Phase 8).
