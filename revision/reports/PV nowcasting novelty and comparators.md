# Disagreement gating is prior art; position SHADOW-MoE as a configuration

The closest prior work to SHADOW-MoE is **GateFormer (Hamlyn et al., *Energy and AI* 2026)**. It is a sky-image irradiance forecaster whose per-sample sigmoid gate weights two candidate forecasts and is explicitly given "the disagreement between the two decoder outputs" ([arXiv PDF](https://arxiv.org/pdf/2610.30942)). Because of it, any claim that gating on expert disagreement is new must be dropped. The revision's own experiments make this more pressing. In development cross-validation over 3 seeds, a **disagreement-only gate (CRPS 0.590 kW, ramp CRPS 2.90 kW) matches the proposed two-feature gate (0.587 / 2.92 kW)**, so the event-probability feature adds little. The gate itself clearly helps, though: a CRPS-optimised constant weight (0.616 / 3.25) and a 50/50 blend (0.638 / 3.07) are both worse. What remains defensible is a specific configuration. Two *separately trained probabilistic* experts with different PV-history lengths are combined by a *low-capacity, CRPS-fitted logistic gate* on distribution-derived features. The output is a *weighted quantile average* (Vincentization, not a mixture), followed by *causal rolling conformal calibration*. Each ingredient has a precedent; no found work combines them for sky-image PV. In terminology, "mixture" and "linear pool" must be reserved for F = Σ wᵢFᵢ ([Gneiting & Ranjan 2013](https://arxiv.org/abs/1106.1638)). The SKIPP'D comparison should be framed as **contextual and unpaired**. Every published probabilistic number uses SkyGPT's 2,582-sample, 5-cloudy-day split rather than the official SKIPP'D test. The papers use different sample counts m and the non-fair empirical CRPS, and they report raw, uncalibrated intervals ([SkyGPT arXiv](https://arxiv.org/html/2306.11682v1)).

## GateFormer, not classic MoE, is the nearest neighbour

Published methods that combine sky-image forecasts set their weights in one of four ways. **Lead-time-specific regression weights** are fitted offline and do not depend on current conditions. Straub et al. combine all-sky-imager, satellite and persistence nowcasts this way and gain 5–13% RMSE over the best single method ([Straub et al. 2024](https://publica.fraunhofer.de/bitstreams/bcb5b879-996d-4308-8744-8cf7fcf6699a/download)). **Recent-performance weights** set each method's weight from its inverse-squared RMSD over the past 5 minutes ([Nouri et al. 2021](https://elib.dlr.de/145707/1/EUPVSEC2021_5BO.7.4.pdf)), or update ensemble weights online from past CRPS ([Thorey et al. 2018](https://inria.hal.science/hal-01909827/)). **Hard regime routing** includes, on the Stanford data itself, a two-stage framework that classifies images into sunny, cloudy and overcast and sends each to a condition-specific CNN ([Nie et al. 2020](https://eao.stanford.edu/publications/journal-articles/pv-power-output-prediction-sky-images-using-convolutional-neural)). A 2026 *Solar Energy* paper does the same with pseudo-labelled sky regimes ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0038092X26007930)). Finally, **learned per-sample gates** sit inside one network. Hategan et al. (2025) "dynamically aggregate" machine-learning and sky-image models ([ScienceDirect abstract](https://www.sciencedirect.com/science/article/pii/S0360544225002373)), but their weighting rule could not be verified and must be checked manually before it is characterised in the manuscript.

GateFormer belongs to the last type and is the closest match. Its gate is a "Linear 256→1 followed by sigmoid", fed a context vector containing "the encoded sky-image sequence representation, recent GHI-state information, the Trajectory Head forecast, the Delta-derived final-horizon forecast, and the disagreement between the two decoder outputs" ([arXiv PDF](https://arxiv.org/pdf/2610.30942)). It differs from SHADOW-MoE on four points. It is **deterministic**, giving point GHI. Its two candidates come from **one jointly trained backbone**. Its gate has **high capacity** and sees visual features. It uses **no event probability**. Gating on output dispersion also appears in a 2025 IEEE conference PV ensemble, which uses "real-time input features, local prediction variance, and historical residuals" ([OpenAlex](https://api.openalex.org/works/doi:10.1109/icemce68156.2025.11466687)). In general forecasting, forecast-pool diversity is a known feature for learning combination weights ([Kang et al. 2022](https://arxiv.org/pdf/2012.01643v2)).

The recent probabilistic PV mixture-of-experts papers are architectural rather than combination comparators. STD-MoE uses a sparse MoE over decomposed series components with multi-quantile heads and conformal calibration ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306261926011992)). TD-M²oE uses a multi-gate MoE that allocates experts across quantile tasks ([OpenAlex](https://api.openalex.org/works/doi:10.1109/tste.2026.3681486)). Both gate hidden representations inside a single network, and neither uses sky images. On the mixture side, generalized Gaussian mixtures ([Cherni et al. 2026](https://www.sciencedirect.com/science/article/pii/S3051052X25000221)) and Markov-chain mixture distributions ([Munkhammar & Widén 2024](https://api.openalex.org/works/doi:10.1016/j.seja.2024.100057)) already give mixture-type probabilistic solar forecasts. The residual Gaussian-mixture expert is therefore an *application* of mixture-density modelling. What no source used does is turn its output into an event probability that drives a gate.

| Work | Combined objects | Weight rule | Forecast-time dependent? | Probabilistic? | Difference from SHADOW-MoE |
|---|---|---|---|---|---|
| GateFormer 2026 | 2 heads, shared ViT | Sigmoid on 256-d context incl. disagreement | Yes | No | Deterministic; joint training; high-capacity gate |
| ICEMCE 2025 (conf.) | LSTM/GRU/Transformer/trees | Gate on inputs, prediction variance, residuals | Yes | Intervals by fusion | No images; conference-tier |
| Nie et al. 2020 | Sky-condition CNNs | Hard routing by cloudiness | Yes (image) | No | Discrete; contemporaneous nowcast |
| Nouri et al. 2021 | ASI + persistence | Inverse RMSD, past 5 min | Past errors | No | Performance-based |
| Straub et al. 2024 | ASI + satellite + persistence | Lead-time regression | No | No | Static |
| Thorey et al. 2018 | NWP ensemble PV | Online past-CRPS weights | Past CRPS | Yes | Not output-dependent |
| STD-MoE / TD-M²oE 2026 | Internal experts | Learned gates on hidden features | Yes | Yes | Single network; no images |

## The experiments relocate the contribution from feature (g) to the probabilistic configuration

The three controls answer the reviewer's question about static and CRPS-optimal weights directly. A single constant weight fitted by CRPS minimisation is the static analogue of CRPS-weighted combination ([Thorey et al. 2018](https://inria.hal.science/hal-01909827/); [Berrisch & Ziel 2023](https://arxiv.org/abs/2102.00968)) and of an optimal convex reference combination in solar forecasting ([Yang 2019](https://doi.org/10.1016/j.solener.2019.10.006)). The 50/50 blend is the equal-weight benchmark, which forecast-combination studies show is often hard to beat ([Taylor & Taylor 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC8236414/)). The fitted gate beats the constant weight by **0.029 kW overall CRPS (≈4.7%)** and by **0.33 kW ramp CRPS (≈10%)**. Most of the value of state dependence therefore shows up in ramps, which supports a forecast-time-covariate gate over past-performance weights.

The disagreement-only result is decisive for the wording. At 0.590 versus 0.587 overall and 2.90 versus 2.92 on ramps, the two gates are indistinguishable at the precision reported. Seed-level dispersion was not supplied, so "no material difference" is the strongest claim that can be supported. Two consequences follow. First, the gate's skill comes mainly from expert disagreement, the same signal GateFormer uses, so disagreement-based gating cannot carry the novelty. Second, the event-probability feature (g) is not a demonstrated contributor. It can be kept for interpretability, or because it never hurts, but only if the manuscript states that it adds little beyond disagreement. Alternatively the authors could promote the one-feature gate as the parsimonious final model. Either way, the contribution moves to the combination of four things. The experts are separately trained and probabilistic. The combination is a quantile-function blend rather than a point blend. The gate is low-capacity and interpretable. Conformal calibration follows. No source in this search combines these for sky-image PV.

One further control is worth adding if time allows: a CRPS-fitted **linear pool (mixture)** with the same gate. It directly tests quantile averaging against probability pooling, and it is the control a combination-theory reviewer is likely to expect.

## Call it a gated quantile blend, not a mixture

The output Q(τ|x) = w(x)Q_D(τ|x) + (1−w(x))Q_R(τ|x) is a **covariate-dependent weighted quantile average**, or weighted Vincentization ([Ratcliff 1979](https://doi.org/10.1037/0033-2909.86.3.446); [Genest 1992](https://doi.org/10.1214/aos/1176348676)). It is not a probability mixture. The difference matters for the result. Averaging quantiles and averaging probabilities give the same mean, but "the average quantile forecast is always sharper: it has lower variance than the average probability forecast" ([Lichtendahl et al. 2013](https://econpapers.repec.org/RePEc:inm:ormnsc:v:59:y:2013:i:7:p:1594-1611)). A linear pool, in contrast, "is at least as dispersed as the least dispersed of the components", and is overdispersed when its components are calibrated ([Gneiting & Ranjan 2013, Thm 3.1](https://arxiv.org/html/1106.1638v1)). This generalises Hora's result that a nontrivial convex combination of calibrated densities is uncalibrated ([Hora 2004](https://doi.org/10.1287/mnsc.1040.0205)).

Two properties follow and should be stated. Because w lies in [0,1] and does not vary with τ, the blend of two non-decreasing quantile functions cannot cross. And when the experts disagree in location, the quantile blend does *not* widen the predictive distribution as a mixture would. That is a principled reason for the conformal step, and it rules out wording such as "the blend reflects model uncertainty across experts". Recent deep-ensemble work finds quantile aggregation often better than linear density combination ([Schulz, Köhler & Lerch 2026](https://publikationen.bibliothek.kit.edu/1000193481)). The quantile-combination precedent in solar is Yang, Yang & Liu's combination of calibrated NWP-ensemble quantiles ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0960148123008996)).

The name "SHADOW-MoE" can stay as a label if the manuscript defines it once. Classic mixture-of-experts gates a *probability mixture* ([Jacobs et al. 1991](https://doi.org/10.1162/neco.1991.3.1.79)), so the text should say "MoE-style gating" and "gated quantile blend" and never call the output a mixture. The audit also credits Gneiting & Ranjan 2013 with "covariate-dependent pooling", but their pools have **constant** weights fitted by log score ([arXiv HTML](https://arxiv.org/html/1106.1638v1)). Cite them for pool dispersion. For covariate- or regime-dependent weights, cite Elliott & Timmermann's regime-switching weights ([OpenAlex](https://api.openalex.org/works/doi:10.1111/j.1468-2354.2005.00361.x)), hierarchical stacking (Yao et al. 2022), or Jacobs et al.

The conformal step should be credited, not claimed. Adaptive conformal inference ([Gibbs & Candès 2021](https://arxiv.org/abs/2106.00170)), its aggregated variant applied to electricity prices ([Zaffran et al. 2022](https://arxiv.org/abs/2202.07282)), nonexchangeable weighted conformal ([Barber et al. 2023](https://arxiv.org/abs/2202.13415)) and conformal PID ([Angelopoulos et al. 2023](https://arxiv.org/abs/2307.16895)) are the foundations. In solar, conformal prediction has been applied to day-ahead PV ([Solar Energy Advances 2024](https://www.sciencedirect.com/science/article/pii/S2667113124000093)) and to 5–15-min PV, where an LSTM showed "significant undercoverage" ([Energy and AI 2025](https://www.sciencedirect.com/science/article/pii/S2666546825001053)). STD-MoE also adds conformal calibration to quantile heads ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306261926011992)). No adaptive-conformal application to sky-image nowcasting on SKIPP'D was found, but that is a limited-search negative. The revision's own Phase-A check found the frozen causal rolling conformal method reached **77% coverage at 90% nominal** on the SkyGPT winter days. This is the shift that ACI-type methods target, so the manuscript should present it as a limitation rather than imply guaranteed coverage.

## SKIPP'D comparisons are contextual, unpaired and estimator-dependent

No published probabilistic result uses the official SKIPP'D 20-day test. SkyGPT was run "before the official release of our curated dataset SKIPP'D". Its split gives **53,336 / 4,467 / 2,582** training, validation and test samples ([SkyGPT arXiv §4](https://arxiv.org/html/2306.11682v1)), and the test set is 5 cloudy days in Nov–Dec 2019. The SKIPP'D paper's own SUNSET benchmark is a point forecast ([Nie et al. 2023](https://doi.org/10.1016/j.solener.2023.03.043)). SUNSET's probabilistic numbers come from SkyGPT's split, using m = 10 cross-validation folds.

| Method (SkyGPT test split) | m | Test CRPS (kW) | WS90 | Source |
|---|---|---|---|---|
| SkyGPT → U-Net | 100 | 2.81 | 26.70 | [SkyGPT Table 1](https://arxiv.org/html/2306.11682v1) |
| PhyDNet+GAN → U-Net | 100 | 3.02 | 39.54 | same |
| VideoGPT → U-Net | 100 | 3.34 | 36.22 | same |
| SUNSET | 10 | 3.31 | 56.95 | same |
| Smart persistence | 1 | 3.67 | – | same |
| PV-MM-Diffusion | 30 | 2.63 | 21.46 | [Huang et al. 2026](https://doi.org/10.1016/j.apenergy.2026.127816) (author-supplied full text) |

The estimator differences are not cosmetic. SkyGPT computes CRPS from the empirical CDF of the m predictions and gives WS90 from raw 5th/95th sample percentiles, with no recalibration ([SkyGPT Eqs. 13–14](https://arxiv.org/html/2306.11682v1)). The empirical CRPS exceeds the CRPS of the generating distribution by about E|X−X′|/(2m) in expectation. At equal spread, an m = 10 row is therefore penalised ten times more than an m = 100 row. The fair estimator replaces the 1/(2m²) spread normalisation with 1/(2m(m−1)) ([Ferro 2014](https://doi.org/10.1002/qj.2270)). Zamo & Naveau compare quantile-based and ensemble-based CRPS estimators ([Zamo & Naveau 2018](https://doi.org/10.1007/s11004-017-9709-7)), and SHADOW-MoE's 9-level quantile-interpolation CRPS is of that quantile-based kind. The revision's internal check shows the effect: 2.49 at m = 30 versus 2.59 at m = 10, and the quantile estimator was 1.2% above the empirical one on SUNSET's members. Appendix C should therefore report SHADOW-MoE at m ∈ {10, 30, 100} and with the fair estimator. m = 30 is the headline value because it matches the strongest comparator. Calibrated (conformal) intervals should go in a separate row from raw intervals.

Two numbers must not be used until verified. Xiong et al. 2025's open-access abstract confirms a generative-video plus Transformer design and a 19.4% CRPS reduction at 15 min against a PV-only model ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666792425000447)). But the 2.44 / 19.62 values, the dataset split and m were not visible. PV-MM-Diffusion's Table 3 rests on author-supplied full text, and whether its oracle row uses m = 10 or m = 30 was not established.

## Recommended manuscript wording

**Contribution list (replace any "first" or "novel gating signal" claim):**

> "We propose SHADOW-MoE, a gated quantile blend of two separately trained probabilistic experts for 15-min-ahead sky-image PV forecasting: a direct image + PV quantile model and a residual Gaussian-mixture event expert that uses a longer PV history. A two-feature logistic gate, fitted by CRPS minimisation, sets a forecast-time weight from quantities derived from the experts' predictive distributions—the disagreement between their predictive means and the event expert's implied probability of a large 15-min change—and applies it to the experts' quantile functions, after which causal rolling conformal calibration adjusts the intervals."

> "In development cross-validation (3 seeds), the fitted gate outperforms both a CRPS-optimised constant weight and an equal-weight blend (overall CRPS 0.587 vs 0.616 and 0.638 kW; ramp-period CRPS 2.92 vs 3.25 and 3.07 kW), showing that state-dependent weighting, rather than combination per se, drives the gain on ramps. A gate using expert disagreement alone performs equivalently (0.590 / 2.90 kW), indicating that the benefit derives primarily from disagreement and that the event-probability feature adds little beyond it."

**Terminology / methods:**

> "The combined forecast is Q(τ|x) = w(x)Q_D(τ|x) + (1 − w(x))Q_R(τ|x), a covariate-dependent weighted quantile average (Vincentization; Ratcliff, 1979; Genest, 1992). It is not a probability mixture or linear pool (Gneiting and Ranjan, 2013): it has the same mean as the corresponding mixture but lower variance (Lichtendahl et al., 2013), cannot produce crossing quantiles because w is constant across τ, and does not widen when the experts disagree in location—one motivation for the subsequent conformal step. We use 'MoE' to denote input-dependent gating (Jacobs et al., 1991), not a mixture-density output."

**Related work:**

> "Sky-image forecasts have been combined with satellite, persistence and statistical forecasts using lead-time-specific regression weights (Straub et al., 2024), weights derived from recent errors (Nouri et al., 2021) or past CRPS (Thorey et al., 2018), dynamically weighted ensembles (Hategan et al., 2025), and hard routing to sky-condition sub-models, including on the Stanford data (Nie et al., 2020). Learned per-sample gates have recently been placed over candidate sky-image forecasts; GateFormer's gate receives the candidate forecasts and their disagreement (Hamlyn et al., 2026), but its output is deterministic and its candidates share a jointly trained backbone. Probabilistic PV mixture-of-experts models gate internal representations within a single network (Liu et al., 2026; Su et al., 2026). More broadly, our gate is an instance of covariate-dependent forecast combination (Elliott and Timmermann, 2005; Yao et al., 2022) with weights estimated by a proper score (Thorey et al., 2018; Berrisch and Ziel, 2023), applied to quantiles rather than densities (Lichtendahl et al., 2013; Yang et al., 2023; Schulz et al., 2026). Our contribution is the specific configuration—separately trained probabilistic experts, a low-capacity gate on distribution-derived features, quantile blending, and causal conformal calibration—evaluated probabilistically on SKIPP'D."

> "Conformal and adaptive conformal methods (Gibbs and Candès, 2021; Zaffran et al., 2022; Barber et al., 2023; Angelopoulos et al., 2023) have been applied to PV forecasting with tabular inputs; we apply a causal rolling variant to sky-image nowcasts and report its coverage under seasonal shift as a limitation."

**Appendix C framing:**

> "No published probabilistic result exists on the official SKIPP'D 20-day test set. We therefore additionally evaluate on the SkyGPT test population (2,582 samples from five cloudy days, Nov–Dec 2019; Nie et al., 2024) used by SkyGPT and PV-MM-Diffusion. The comparison is contextual and unpaired, because competitors' per-sample predictions are not public, and it is estimator-dependent: published values use the empirical sample CRPS with m = 100 (SkyGPT), m = 30 (PV-MM-Diffusion) or m = 10 (SUNSET) predictions, whose bias grows as 1/m (Ferro, 2014), and Winkler scores from raw 5–95% sample percentiles without recalibration. We report SHADOW-MoE's CRPS from m ∈ {10, 30, 100} samples drawn from its predictive quantiles, with both the empirical and the fair estimator (Ferro, 2014; Zamo and Naveau, 2018), and we report Winkler scores for raw and conformally calibrated intervals in separate rows. Differences smaller than the estimator-induced gap should not be interpreted as ranking."

**Sentences to delete:** "we are not aware of earlier work that uses the experts' disagreement as the gating signal" (contradicted by GateFormer); any description of the output as a "mixture" or as "reflecting model uncertainty across experts"; any claim that the event-probability feature is essential.

## Conclusion

The revision data change the paper's story. The evidence supports a narrower and stronger claim: a cheap, interpretable gate on expert disagreement, applied to the quantile functions of two independently trained probabilistic experts, captures nearly all the gain from combination, and most of that gain appears in ramps. Framed this way, GateFormer's disagreement input corroborates the design instead of threatening it, and the parsimonious one-feature gate becomes a candidate final model rather than an embarrassing ablation. The remaining risks are verification gaps: Hategan et al.'s weighting rule, Xiong et al.'s protocol, PV-MM-Diffusion's oracle m, and seed-level variability for the 0.587 vs 0.590 contrast. Close these before submission, and add a gated linear-pool control. That control turns the quantile-versus-mixture argument from theory into evidence.

## References

All DOIs were checked on Crossref or OpenAlex in the underlying notes unless marked. MDPI and Chinese-published journals are excluded, as is Tech Science Press (CMES) as a likely exclusion.

**Closest prior art and sky-image combination**
- Hamlyn, Chaudhary, Hare, Rahman (2026). GateFormer. *Energy and AI* 26:100917. doi:10.1016/j.egyai.2026.100917 (arXiv:2610.30942)
- Hategan, Stefu, Petreuș et al. (2025). *Energy* 316:134595. doi:10.1016/j.energy.2025.134595
- Nie, Sun, Chen, Orsini, Brandt (2020). PV power output prediction from sky images using convolutional neural network. *J. Renew. Sustain. Energy* 12(4):046101. doi:10.1063/5.0014016
- Nouri, Blum, Wilbert, Zarzalejo (2021). *Solar RRL* 6:2100442. doi:10.1002/solr.202100442
- Nouri, Lezaca, Fabel, Hammer, Blum, Wilbert (2025). *Solar RRL* 9(24):e202500486. doi:10.1002/solr.202500486
- Nouri et al. (2023). Probabilistic solar nowcasting based on all-sky imagers. *Solar Energy* 253:285–307. doi:10.1016/j.solener.2023.01.060
- Straub, Herzberg, Dittmann, Lorenz (2024). *Solar Energy* 269:112319. doi:10.1016/j.solener.2024.112319
- Two-stage weather-aware unlabeled sky image classification and solar irradiance forecasting (2026). *Solar Energy* 319:115104. doi:10.1016/j.solener.2026.115104
- Zhang, Wilson, Sumner, Wu (2023). *Renewable Energy* 216:118952. doi:10.1016/j.renene.2023.118952 (DOI not independently resolved)

**Probabilistic PV MoE and mixture models**
- Liu et al. (2026). STD-MoE. *Applied Energy* 426:128543. doi:10.1016/j.apenergy.2026.128543
- Su et al. (2026). TD-M²oE. *IEEE Trans. Sustain. Energy*. doi:10.1109/TSTE.2026.3681486
- Wang et al. (2025). Context-Gated Dynamic Fusion. ICEMCE 2025. doi:10.1109/ICEMCE68156.2025.11466687
- Cherni et al. (2026). *Sustainable Cities and Society: Advances* 2:100022. doi:10.1016/j.scsadv.2025.100022
- Munkhammar, Widén (2024). *Solar Energy Advances*. doi:10.1016/j.seja.2024.100057
- Jacobs, Jordan, Nowlan, Hinton (1991). Adaptive mixtures of local experts. *Neural Computation* 3(1):79–87. doi:10.1162/neco.1991.3.1.79

**Forecast combination theory**
- Lichtendahl, Grushka-Cockayne, Winkler (2013). *Management Science* 59(7):1594–1611. doi:10.1287/mnsc.1120.1667
- Gneiting, Ranjan (2013). Combining predictive distributions. *Electron. J. Stat.* 7:1747–1782. doi:10.1214/13-EJS823
- Ranjan, Gneiting (2010). *JRSS-B* 72(1):71–91. doi:10.1111/j.1467-9868.2009.00726.x
- Hora (2004). *Management Science* 50(5):597–604. doi:10.1287/mnsc.1040.0205
- Ratcliff (1979). *Psychological Bulletin* 86(3):446–461. doi:10.1037/0033-2909.86.3.446
- Genest (1992). Vincentization revisited. *Ann. Stat.* 20(2):1137–1142. doi:10.1214/aos/1176348676
- Busetti (2017). Quantile aggregation of density forecasts. *Oxf. Bull. Econ. Stat.* 79(4):495–512. doi:10.1111/obes.12163
- Schulz, Köhler, Lerch (2026). *Machine Learning* 115(5):120. doi:10.1007/s10994-025-06946-3
- Berrisch, Ziel (2023). CRPS learning. *J. Econometrics* 237(2):105221. doi:10.1016/j.jeconom.2021.11.008
- Thorey, Chaussin, Mallet (2018). *Int. J. Forecast.* 34(4):762–773. doi:10.1016/j.ijforecast.2018.05.007
- Taylor, Taylor (2023). *EJOR* 304(1):25–41. doi:10.1016/j.ejor.2021.06.044
- Wang, Hyndman, Li, Kang (2023). Forecast combinations: an over 50-year review. *Int. J. Forecast.* 39(4):1518–1547. doi:10.1016/j.ijforecast.2022.11.005
- Wang, Zhang, Tan, Hong, Kirschen et al. (2019). Combining probabilistic load forecasts. *IEEE Trans. Smart Grid* 10(4):3664–3674. doi:10.1109/TSG.2018.2833869
- Elliott, Timmermann (2005). *Int. Econ. Rev.* 46(4):1081–1102. doi:10.1111/j.1468-2354.2005.00361.x
- Kapetanios, Mitchell, Price, Fawcett (2015). *J. Econometrics* 188(1):150–165. doi:10.1016/j.jeconom.2015.02.047
- Yao, Pirš, Vehtari, Gelman (2022). Bayesian hierarchical stacking. *Bayesian Analysis* 17(4):1043–1071. doi:10.1214/21-BA1287
- Kang, Cao, Petropoulos, Li (2022). Forecast with forecasts: diversity matters. *EJOR* 301:180–190. doi:10.1016/j.ejor.2021.10.024
- Gneiting, Raftery (2007). *JASA* 102(477):359–378. doi:10.1198/016214506000001437
- Yang (2019). *Solar Energy* 193:981–985. doi:10.1016/j.solener.2019.10.006
- Yang, Yang, Liu (2023). *Renewable Energy* 215:118993. doi:10.1016/j.renene.2023.118993
- Yang, van der Meer (2021). *RSER* 140:110735. doi:10.1016/j.rser.2021.110735

**SKIPP'D comparators and scoring**
- Nie, Li, Scott, Sun, Venugopal, Brandt (2023). SKIPP'D. *Solar Energy* 255:171–179. doi:10.1016/j.solener.2023.03.043
- Nie, Zelikman, Scott, Paletta, Brandt (2024). SkyGPT. *Advances in Applied Energy*. doi:10.1016/j.adapen.2024.100172 (arXiv:2306.11682; check article number)
- Huang, Shao, Ke, Gao, Mazzoni (2026). PV-MM-Diffusion. *Applied Energy* 413:127816. doi:10.1016/j.apenergy.2026.127816
- Xiong, Chen, Zhao, Su, Fu, Chen (2025). *Advances in Applied Energy* 20:100250. doi:10.1016/j.adapen.2025.100250
- Ferro (2014). Fair scores for ensemble forecasts. *QJRMS* 140:1917–1923. doi:10.1002/qj.2270
- Zamo, Naveau (2018). *Math. Geosci.* 50:209–234. doi:10.1007/s11004-017-9709-7

**Conformal calibration**
- Gibbs, Candès (2021). Adaptive conformal inference under distribution shift. NeurIPS 34. arXiv:2106.00170 (no DOI)
- Zaffran, Dieuleveut, Féron, Goude, Josse (2022). Adaptive conformal predictions for time series. ICML (PMLR 162). arXiv:2202.07282 (no DOI)
- Barber, Candès, Ramdas, Tibshirani (2023). *Ann. Stat.* 51(2):816–845. doi:10.1214/23-AOS2276
- Angelopoulos, Candès, Tibshirani (2023). Conformal PID control. NeurIPS 36:23047–23074. doi:10.52202/075280-1000
- Enhancing the reliability of probabilistic PV power forecasts using conformal prediction (2024). *Solar Energy Advances* 4:100059. doi:10.1016/j.seja.2024.100059
- Comparative analysis of conformal prediction techniques… very short-term solar power forecasting (2025). *Energy and AI* 21:100573. doi:10.1016/j.egyai.2025.100573
