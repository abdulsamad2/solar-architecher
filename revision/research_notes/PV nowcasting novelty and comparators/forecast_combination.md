# Forecast combination theory for probabilistic (quantile) forecasts

Scope: the manuscript combines two experts' 9 quantiles as w·Q_D(τ) + (1−w)·Q_R(τ), with w from a logistic gate. Controls are a single CRPS-optimised constant w and w = 0.5.

Verification method: every DOI below was checked on 2026-11-12 against the Crossref REST API (`api.crossref.org/works/{doi}`), which returned matching metadata. The bibliographic strings below are taken from those records. Abstracts and full-text passages come from the linked open copies (arXiv, HAL, PMC, RePEc). No MDPI or Chinese-published journals are listed. One MDPI hit (Energies 12(6):1011, 2019, weighted quantile combination for PV) was excluded on purpose.

**Status against `paper/qc/v4_reference_audit.md`** (CITED = already in the v4 bibliography, with its number; AUDIT-PROPOSED = suggested in the audit but not in the bibliography; NEW = not in either):

| Work | DOI | Status |
|---|---|---|
| Lichtendahl, Grushka-Cockayne & Winkler 2013, Manag. Sci. 59(7):1594–1611 | 10.1287/mnsc.1120.1667 | CITED [31] |
| Thorey, Chaussin & Mallet 2018, IJF 34(4):762–773 | 10.1016/j.ijforecast.2018.05.007 | CITED [28] |
| Jacobs, Jordan, Nowlan & Hinton 1991, Neural Comput. 3(1):79–87 | 10.1162/neco.1991.3.1.79 | CITED [11] |
| Shazeer et al. 2017 (sparsely-gated MoE, ICLR) | arXiv | CITED [12] |
| Lakshminarayanan et al. 2017 (deep ensembles) | arXiv | CITED [30] |
| Jordan, Krüger & Lerch 2019 (scoringRules, JSS) | – | CITED [54] |
| Gneiting & Ranjan 2013, Electron. J. Stat. 7:1747–1782 | 10.1214/13-EJS823 | AUDIT-PROPOSED |
| Busetti 2017, Oxf. Bull. Econ. Stat. 79(4):495–512 | 10.1111/obes.12163 | AUDIT-PROPOSED |
| Sill et al. 2009, feature-weighted linear stacking | arXiv:0911.0460 | AUDIT-PROPOSED |
| Ranjan & Gneiting 2010, JRSS-B 72(1):71–91 | 10.1111/j.1467-9868.2009.00726.x | NEW |
| Hora 2004, Manag. Sci. 50(5):597–604 | 10.1287/mnsc.1040.0205 | NEW |
| Berrisch & Ziel 2023, J. Econometrics 237(2):105221 | 10.1016/j.jeconom.2021.11.008 | NEW |
| Taylor & Taylor 2023, EJOR 304(1):25–41 | 10.1016/j.ejor.2021.06.044 | NEW |
| Wang, Hyndman, Li & Kang 2023, IJF 39(4):1518–1547 (review) | 10.1016/j.ijforecast.2022.11.005 | NEW |
| Wang, Zhang, Tan, Hong, Kirschen et al. 2019, IEEE TSG 10(4):3664–3674 | 10.1109/TSG.2018.2833869 | NEW |
| Schulz, Köhler & Lerch 2026, Mach. Learn. 115(5):120 | 10.1007/s10994-025-06946-3 | NEW |
| Ratcliff 1979, Psychol. Bull. 86(3):446–461 | 10.1037/0033-2909.86.3.446 | NEW |
| Genest 1992, Ann. Stat. 20(2):1137–1142 | 10.1214/aos/1176348676 | NEW |
| Hall & Mitchell 2007, IJF 23(1):1–13 | 10.1016/j.ijforecast.2006.08.001 | NEW |
| Geweke & Amisano 2011, J. Econometrics 164(1):130–141 | 10.1016/j.jeconom.2011.02.017 | NEW |
| Gneiting & Raftery 2007, JASA 102(477):359–378 | 10.1198/016214506000001437 | NEW (check the bib; not in the audit table) |
| Gneiting 2011, IJF 27(2):197–207 (quantiles as optimal point forecasts) | 10.1016/j.ijforecast.2009.12.015 | NEW |
| Elliott & Timmermann 2005, Int. Econ. Rev. 46(4):1081–1102 | 10.1111/j.1468-2354.2005.00361.x | NEW |
| Billio, Casarin, Ravazzolo & van Dijk 2013, J. Econometrics 177(2):213–232 | 10.1016/j.jeconom.2013.04.009 | NEW |
| Del Negro, Hasegawa & Schorfheide 2016, J. Econometrics 192(2):391–405 | 10.1016/j.jeconom.2016.02.006 | NEW |
| Kapetanios, Mitchell, Price & Fawcett 2015, J. Econometrics 188(1):150–165 | 10.1016/j.jeconom.2015.02.047 | NEW |
| Bassetti, Casarin & Ravazzolo 2018, JASA 113(522):675–685 | 10.1080/01621459.2016.1273117 | NEW |
| Opschoor, van Dijk & van der Wel 2017, J. Appl. Econometrics 32(7):1298–1313 | 10.1002/jae.2575 | NEW |
| Yao, Vehtari, Simpson & Gelman 2018, Bayesian Anal. 13(3):917–1007 | 10.1214/17-BA1091 | NEW |
| Yao, Pirš, Vehtari & Gelman 2022, Bayesian Anal. 17(4):1043–1071 | 10.1214/21-BA1287 | NEW |
| Montero-Manso, Athanasopoulos, Hyndman & Talagala 2020 (FFORMA), IJF 36(1):86–92 | 10.1016/j.ijforecast.2019.02.011 | NEW |
| Wolpert 1992, Neural Netw. 5(2):241–259 (stacking) | 10.1016/S0893-6080(05)80023-1 | NEW |
| Jose, Grushka-Cockayne & Lichtendahl 2014, Manag. Sci. 60(2):463–475 | 10.1287/mnsc.2013.1781 | NEW |
| Grushka-Cockayne, Jose & Lichtendahl 2017, Manag. Sci. 63(4):1110–1130 | 10.1287/mnsc.2015.2389 | NEW |
| Yang (Dazhi) 2019, Solar Energy 193:981–985 (optimal convex combination of climatology and persistence) | 10.1016/j.solener.2019.10.006 | NEW |
| Yang & van der Meer 2021, RSER 140:110735 | 10.1016/j.rser.2021.110735 | NEW |
| Yang, Yang & Liu 2023, Renew. Energy 215:118993 | 10.1016/j.renene.2023.118993 | NEW |
| van der Meer, Widén & Munkhammar 2018, RSER 81:1484–1512 | 10.1016/j.rser.2017.05.212 | NEW |
| Liu, Nowotarski, Hong & Weron 2017 (online 2015), IEEE TSG (QRA on sister forecasts) | 10.1109/TSG.2015.2437877 | NEW (Crossref gives no volume or pages; check the final issue) |
| Raza, Mithulananthan & Summerfield 2018, Solar Energy 166:226–241 | 10.1016/j.solener.2018.03.066 | NEW (point forecasts) |

Corrections to the brief's citation keys:
- "Thorey et al. 2017" is the 2018 IJF paper above. The manuscript already cites it as [28].
- "Wang et al. 2019 forecast combination review": no 2019 review was found. The review is **Wang, Hyndman, Li & Kang 2023** (IJF). The 2019 Wang et al. paper is **Y. Wang et al., "Combining Probabilistic Load Forecasts", IEEE TSG 2019**.
- "Yang & van der Meer" is the 2021 RSER paper on post-processing tools. "Yang 2020" could be several papers: Yang 2020, *Solar Energy* 210:49–58 (probabilistic reconciliation, doi:10.1016/j.solener.2020.03.095), or two 2020 JRSE papers on EMOS. None of these is primarily about combination. The most relevant Yang combination papers are Yang 2019 (Solar Energy) and Yang, Yang & Liu 2023 (Renew. Energy).
- "Taylor & Taylor 2023" was verified (EJOR, COVID-19 mortality).

---

## Q1. How quantile averaging (Vincentization) differs from linear pooling, and their sharpness and calibration properties

### Takeaway
A weighted average of corresponding quantiles, Q(τ) = Σ w_i Q_i(τ), is quantile averaging (Vincentization). It is not a probability mixture. The linear pool, F(y) = Σ w_i F_i(y), is the mixture. Both give the same mean. The quantile average always has lower variance (it is sharper), while the linear pool is always at least as dispersed as its least dispersed member. A linear pool of calibrated, distinct members is therefore overdispersed and uncalibrated.

### Cited Findings
- Lichtendahl, Grushka-Cockayne & Winkler (2013) compare "two ways to aggregate expert opinions using simple averages: averaging probabilities and averaging quantiles". Their abstract states: "In terms of location, the two average forecasts have the same mean. The average quantile forecast is always sharper: it has lower variance than the average probability forecast." It also states: "Even when the average probability forecast is overconfident, the shape of the average quantile forecast still offers the possibility of a better forecast." On Survey of Professional Forecasters data, the probability average "is outperformed by the average quantile forecast" both when it is overconfident and when it is underconfident. — [EconPapers abstract, Manag. Sci. 59(7):1594–1611, doi:10.1287/mnsc.1120.1667](https://econpapers.repec.org/RePEc:inm:ormnsc:v:59:y:2013:i:7:p:1594-1611)
- In the same paper: "On four scoring rules, we show analytically that the score of the average probability forecast can be no worse than the average score of the individual experts' forecasts. We also identify conditions under which the same result is true for the average quantile forecast on the quadratic scoring rule." — [JSTOR first page snippet](https://www.jstor.org/stable/23443872)
- Hora (2004) "proved that any nontrivial convex combination of two calibrated density forecasts is uncalibrated", as Gneiting & Ranjan report it. — [Gneiting & Ranjan, arXiv:1106.1638 (= EJS 7:1747–1782)](https://arxiv.org/abs/1106.1638). Primary record: Hora SC, "Probability Judgments for Continuous Quantities: Linear Combinations and Calibration", Manag. Sci. 50(5):597–604, 2004, doi:10.1287/mnsc.1040.0205 (Crossref verified; full text not read).
- Gneiting & Ranjan 2013, Theorem 3.1. Take any linear pool F = Σ w_i F_i with strictly positive weights summing to 1, whose distinct components differ with positive probability. Then: "(a) The linearly combined forecast is at least as dispersed as the least dispersed of the components. (b) If the component forecasts are regular, then is more dispersed than the least dispersed of the components. (c) If the components are neutrally dispersed and regular, then is overdispersed." The authors call Hora's result "an immediate consequence of part (c)". Gist: "dispersion tends to increase under linear aggregation." — [arXiv HTML of 1106.1638](https://arxiv.org/html/1106.1638v1)
- Gneiting & Ranjan 2013, Discussion: "any linear combination formula with strictly positive coefficients fails to be coherent". With underdispersed components, "the tendency of linear combination formulas to increase dispersion can be beneficial". "If the components are neutrally dispersed, the failure of the traditional linear pool to be flexibly dispersive is a serious limitation." They propose the spread-adjusted linear pool and the beta-transformed linear pool (BLP), and show that the BLP is "exchangeably flexibly dispersive". — [arXiv HTML](https://arxiv.org/html/1106.1638v1)
- Ranjan & Gneiting (2010), "Combining Probability Forecasts", JRSS-B 72(1):71–91, doi:10.1111/j.1467-9868.2009.00726.x, introduced the beta-transformed linear pool. As Gneiting & Ranjan 2013 summarise it, they showed that in the discrete (binary) case "linear combination formulas with strictly positive coefficients fail to be coherent". — [arXiv HTML of 1106.1638](https://arxiv.org/html/1106.1638v1); Crossref verified.
- Busetti (2017), "Quantile Aggregation of Density Forecasts", Oxf. Bull. Econ. Stat. 79(4):495–512, doi:10.1111/obes.12163, is the econometric reference for quantile aggregation of density forecasts (Crossref verified; full text not read this session; the audit lists a green copy as Banca d'Italia WP 979).
- Schulz, Köhler & Lerch (2026, Machine Learning 115(5):120, doi:10.1007/s10994-025-06946-3, open access) "systematically compare probability- and quantile-based aggregation methods" for deep ensembles on twelve benchmark datasets. They "propose a general quantile aggregation framework for deep ensembles that allows for corrections of systematic deficiencies", which "performs well in a variety of settings, often superior compared to a linear combination of the forecast densities." — [Crossref abstract](https://api.crossref.org/works/10.1007/s10994-025-06946-3); [KITopen record](https://publikationen.bibliothek.kit.edu/1000193481)
- "Vincentization" (averaging quantile functions) comes from Ratcliff (1979), Psychol. Bull. 86(3):446–461, doi:10.1037/0033-2909.86.3.446. Genest (1992), "Vincentization Revisited", Ann. Stat. 20(2):1137–1142, doi:10.1214/aos/1176348676, studied it formally as a pooling rule. Both Crossref verified; full texts not read.
- Trimmed and robust variants: Jose, Grushka-Cockayne & Lichtendahl (2014) on trimmed opinion pools and "the crowd's calibration problem" (Manag. Sci. 60(2):463–475, doi:10.1287/mnsc.2013.1781), and Grushka-Cockayne, Jose & Lichtendahl (2017) on ensembles of overfit and overconfident forecasts (Manag. Sci. 63(4):1110–1130, doi:10.1287/mnsc.2015.2389). Both Crossref verified; only titles were read.

### Inferences
- For two members the manuscript's rule is Q(τ) = w·Q_D(τ) + (1−w)·Q_R(τ), with w ∈ (0,1) depending on covariates but not on τ. It is therefore a **weighted, covariate-dependent quantile average (weighted Vincentization)**, not a mixture. The reviewer is correct. Under location disagreement the mixture would be bimodal or wider. The quantile average stays unimodal-like: for location–scale members of the same family it returns the same family, with averaged location and averaged scale. This follows from linearity of Q, F⁻¹(τ) = μ + σ·Z⁻¹(τ), applied to Lichtendahl et al.'s setting.
- Because w does not depend on τ and w ∈ [0,1], the blend of two non-decreasing quantile functions is non-decreasing. No quantile crossing can occur (elementary).
- On CRPS: CRPS(F, y) = 2∫₀¹ QS_τ(F⁻¹(τ), y) dτ, and the pinball loss QS_τ(q, y) is convex in q. Hence the quantile average satisfies CRPS(Vincentized, y) ≤ w·CRPS(F_D, y) + (1−w)·CRPS(F_R, y) for every outcome y. The same bound holds for the mixture, because CRPS is convex in F. With the 9-quantile pinball/CRPS approximation, the same inequality holds level by level. Neither bound shows that one combination beats the other; this has to be tested empirically. The quantile representation is consistent with Berrisch & Ziel's "aggregation across quantiles that optimize with respect to the CRPS" (Q2). The 2∫QS identity itself is standard but was not re-read here (see Gaps).
- Suggested wording for the manuscript: "Vincentization yields a forecast with the same mean as, and lower variance than, the corresponding probability mixture [Lichtendahl 2013]. Linear pools, by contrast, are at least as dispersed as their least dispersed member and overdispersed when the members are calibrated [Hora 2004; Gneiting & Ranjan 2013, Thm 3.1]. When members disagree in location, the quantile average therefore does not widen the distribution; the conformal step compensates for this." This replaces any claim that quantile averaging "reflects model uncertainty across members".

### Gaps
- Lichtendahl 2013, Hora 2004 and Busetti 2017 full texts were not accessible this session (INFORMS 403). Statements rely on the abstract, the JSTOR first page, and Gneiting & Ranjan's summary. The exact conditions in Lichtendahl's quadratic-score result are not quoted.
- The CRPS = 2∫QS_τ dτ identity is widely used (e.g., Gneiting & Raftery 2007, JASA 102(477):359–378, doi:10.1198/016214506000001437, Crossref verified; Gneiting 2011, IJF 27(2):197–207, doi:10.1016/j.ijforecast.2009.12.015, Crossref verified). The exact equation and page in those papers was not re-read this session.

---

## Q2. CRPS- and proper-score-optimal combination weights (static and covariate-dependent)

### Takeaway
Fitting combination weights by minimising a proper score on training data is standard. Log score is used for density pools (Hall & Mitchell 2007; Geweke & Amisano 2011; Gneiting & Ranjan 2013). CRPS or quantile loss is used for quantile and ensemble combinations (Thorey 2018; Berrisch & Ziel 2023; Taylor & Taylor 2023; Y. Wang et al. 2019). Weights that vary over time, regime, or covariates are also established: regime switching (Elliott & Timmermann 2005), time-varying and dynamic pools (Billio 2013; Del Negro 2016), input-dependent stacking (Yao 2022), and mixtures of experts (Jacobs 1991). A logistic gate fitted by exact CRPS minimisation is a covariate-dependent instance of these. It is not a new principle.

### Cited Findings
- **Berrisch & Ziel 2023, "CRPS learning"**, J. Econometrics 237(2):105221, doi:10.1016/j.jeconom.2021.11.008 (arXiv:2102.00968). Abstract: "the quality of different forecasts may vary not only over time but also within the distribution … we introduce a new weighting method that considers the differences in performance over time and within the distribution. We discuss pointwise combination algorithms based on aggregation across quantiles that optimize with respect to the continuous ranked probability score (CRPS)." Estimation uses "B- and P-Spline-based estimation techniques for batch and online learning, based on quantile regression and prediction with expert advice". They prove that "fully adaptive Bernstein online aggregation (BOA) … has optimal convergence properties." — [arXiv](https://arxiv.org/abs/2102.00968)
- **Thorey, Chaussin & Mallet 2018**, IJF 34(4):762–773, doi:10.1016/j.ijforecast.2018.05.007 (already [28]). PV plants in France, up to 6 days ahead, 30-min step. "our parameter-free online learning technique generates a weighted combination of the production forecasts for each PV plant. The weights are computed sequentially before each forecast using only past information. Our strategy is to minimize the Continuous Ranked Probability Score (CRPS)." — [HAL hal-01909827 abstract](https://inria.hal.science/hal-01909827/)
- **Taylor & Taylor 2023**, EJOR 304(1):25–41, doi:10.1016/j.ejor.2021.06.044. They combine quantile/interval forecasts from many teams (COVID-19 Forecast Hub). They evaluate "the simple average, the median, and trimming methods" and "propose several new weighted combining methods". Findings: "the median was very useful for the early weeks … the simple average was preferable thereafter, and … as a history of forecast accuracy accumulates, the best results can be produced by a weighted combining method that uses weights that are inversely proportional to the historical accuracy of the individual forecasting teams." — [PMC8236414](https://pmc.ncbi.nlm.nih.gov/articles/PMC8236414/)
- **Gneiting & Ranjan 2013** fit linear, generalized, spread-adjusted and beta-transformed pools "by maximizing the mean logarithmic score over training data", which they call maximum likelihood estimates. The weights are constant, not covariate-dependent. — [arXiv HTML](https://arxiv.org/html/1106.1638v1)
- **Hall & Mitchell 2007**, "Combining density forecasts", IJF 23(1):1–13, doi:10.1016/j.ijforecast.2006.08.001, and **Geweke & Amisano 2011**, "Optimal prediction pools", J. Econometrics 164(1):130–141, doi:10.1016/j.jeconom.2011.02.017, are the standard references for log-score-optimal linear-pool weights. Both Crossref verified; content as commonly cited, abstracts not re-read.
- **Opschoor, van Dijk & van der Wel 2017**, "Combining density forecasts using focused scoring rules", J. Appl. Econometrics 32(7):1298–1313, doi:10.1002/jae.2575, fits weights with region-focused scores (Crossref verified; title only).
- **Y. Wang, Zhang, Tan, Hong, Kirschen et al. 2019**, "Combining Probabilistic Load Forecasts", IEEE TSG 10(4):3664–3674, doi:10.1109/TSG.2018.2833869, is a quantile-combination paper in energy (Crossref verified; method details not read). Related: **Liu, Nowotarski, Hong & Weron**, "Probabilistic Load Forecasting via Quantile Regression Averaging on Sister Forecasts", IEEE TSG, doi:10.1109/TSG.2015.2437877.
- **Review:** Wang, Hyndman, Li & Kang 2023, "Forecast combinations: An over 50-year review", IJF 39(4):1518–1547, doi:10.1016/j.ijforecast.2022.11.005 (Crossref verified). It covers both point and probabilistic combination.

**Time-varying, regime- or covariate-dependent weights:**
- **Elliott & Timmermann 2005**, "Optimal forecast combination under regime switching", Int. Econ. Rev. 46(4):1081–1102, doi:10.1111/j.1468-2354.2005.00361.x. Abstract: "a new forecast combination method that lets the combination weights be driven by regime switching in a latent state variable … time variations in the combination weights arise when the target variable and the predictors share a common factor structure driven by a hidden Markov process." — [OpenAlex abstract via Crossref DOI](https://api.openalex.org/works/doi:10.1111/j.1468-2354.2005.00361.x)
- **Billio, Casarin, Ravazzolo & van Dijk 2013**, "Time-varying combinations of predictive densities using nonlinear filtering", J. Econometrics 177(2):213–232, doi:10.1016/j.jeconom.2013.04.009. **Del Negro, Hasegawa & Schorfheide 2016**, "Dynamic prediction pools", J. Econometrics 192(2):391–405, doi:10.1016/j.jeconom.2016.02.006. **Bassetti, Casarin & Ravazzolo 2018**, "Bayesian Nonparametric Calibration and Combination of Predictive Distributions", JASA 113(522):675–685, doi:10.1080/01621459.2016.1273117. All Crossref verified; titles only.
- **Kapetanios, Mitchell, Price & Fawcett 2015**, "Generalised density forecast combinations", J. Econometrics 188(1):150–165, doi:10.1016/j.jeconom.2015.02.047. Crossref verified; content not read. From memory it lets weights vary over the support of the density; treat as unverified.
- **Yao, Pirš, Vehtari & Gelman 2022**, "Bayesian Hierarchical Stacking: Some Models Are (Somewhere) Useful", Bayesian Anal. 17(4):1043–1071, doi:10.1214/21-BA1287. This is input-dependent stacking of predictive distributions. Crossref verified; the "input-varying weights" description follows the title and is not quoted from the abstract. Base stacking: Yao, Vehtari, Simpson & Gelman 2018, Bayesian Anal. 13(3):917–1007, doi:10.1214/17-BA1091. Original stacking: Wolpert 1992, Neural Netw. 5(2):241–259, doi:10.1016/S0893-6080(05)80023-1.
- **Feature/meta-learned weights:** Montero-Manso et al. 2020, FFORMA, IJF 36(1):86–92, doi:10.1016/j.ijforecast.2019.02.011 (weights from time-series features); Sill et al. 2009, feature-weighted linear stacking, arXiv:0911.0460 (audit-proposed).
- **Mixture of experts:** Jacobs, Jordan, Nowlan & Hinton 1991, "Adaptive Mixtures of Local Experts", Neural Comput. 3(1):79–87, doi:10.1162/neco.1991.3.1.79 (already [11]). A softmax gate on the inputs weights the experts. In classic MoE the output is a probability mixture of expert densities.

### Inferences
- **How the manuscript's gate fits in:** w(x) = σ(βᵀx), with x = expert disagreement and the calm expert's event probability. It is fitted by exact CRPS minimisation and applied to quantiles. It combines three established ideas:
  1. covariate-dependent (gating) weights, as in MoE [11], feature-weighted stacking, and hierarchical stacking (Yao 2022);
  2. proper-score (CRPS) estimation of the weights, as in Thorey [28] and Berrisch & Ziel;
  3. quantile averaging rather than pooling (Lichtendahl [31]; Busetti; Schulz et al. 2026).
- Differences from the closest works:
  - Thorey's and Berrisch & Ziel's weights adapt only to past performance (online). Berrisch & Ziel's also vary with τ. The manuscript's weights use current forecast-time covariates and are constant in τ.
  - Classic MoE gates a probability mixture. The manuscript gates a quantile average.
  - Elliott & Timmermann use a latent Markov regime. The manuscript uses an observed, continuous gate.
- **Controls:** a CRPS-optimised constant w is the static analogue (Thorey/Berrisch-style without time variation). w = 0.5 is the equal-weight Vincentization benchmark. Taylor & Taylor 2023 support including the simple average and the median, since equal weights are often hard to beat. A natural extra control the reviewer may expect is a CRPS-fitted **linear pool** (mixture) with the same gate. It directly tests quantile averaging against pooling.
- **Audit discrepancy:** `v4_reference_audit.md` (items 9 and 143) credits Gneiting & Ranjan 2013 with "covariate-dependent pooling". The full text read this session fits **constant** weights by log score. The word "covariate" appears only in the simulation's data-generating process. Cite Gneiting & Ranjan for linear-pool dispersion and BLP. Cite Elliott & Timmermann 2005, Yao et al. 2022, Jacobs et al. 1991 or Sill 2009 for covariate/regime-dependent weights.

### Gaps
- Abstracts of Billio 2013, Del Negro 2016, Kapetanios 2015, Bassetti 2018, Yao 2022, Hall & Mitchell 2007, Geweke & Amisano 2011 and Y. Wang 2019 were not retrieved: OpenAlex has no abstract and Semantic Scholar was rate-limited (HTTP 429). Their descriptions above are limited to titles.
- No paper was found that fits a **logistic, covariate-dependent weight on a quantile average by exact CRPS minimisation**. This is a negative finding from a limited search (about 15 queries), not proof of absence.

---

## Q3. Solar-specific combination literature

### Takeaway
In solar and PV forecasting, combination of probabilistic forecasts is established mainly through ensemble/NWP post-processing. Examples are online CRPS-weighted combination (Thorey 2018), quantile combination of calibrated NWP-ensemble forecasts (Yang, Yang & Liu 2023), and optimal convex combination of reference forecasts (Yang 2019). No refereed, non-MDPI solar paper with a forecast-time-covariate gate on quantiles from sky-image experts was found.

### Cited Findings
- Thorey et al. 2018 (IJF; [28]): online CRPS-minimising weights for PV ensembles derived from ECMWF and Météo-France NWP, with weights "computed sequentially before each forecast using only past information". — [HAL](https://inria.hal.science/hal-01909827/)
- **Yang, Yang & Liu 2023**, "Combining quantiles of calibrated solar forecasts from ensemble numerical weather prediction", Renewable Energy 215:118993, doi:10.1016/j.renene.2023.118993. Authors per Crossref: Dazhi Yang, Guoming Yang, Bai Liu (Elsevier). An aggregator snippet says the quantiles are "combined through a linear program". — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0960148123008996); [X-MOL snippet, secondary](https://www.x-mol.com/paper/1676973287860322304). The full abstract was not retrieved.
- **Yang (Dazhi) 2019**, "Making reference solar forecasts with climatology, persistence, and their optimal convex combination", Solar Energy 193:981–985, doi:10.1016/j.solener.2019.10.006 (Crossref verified). A convex combination with an optimal weight serves as a reference forecast. This is relevant to the constant-weight control.
- **Yang & van der Meer 2021**, "Post-processing in solar forecasting: Ten overarching thinking tools", RSER 140:110735, doi:10.1016/j.rser.2021.110735 (Crossref verified). It frames combination as a post-processing tool. Content not re-read.
- **van der Meer, Widén & Munkhammar 2018**, "Review on probabilistic forecasting of photovoltaic power production and electricity consumption", RSER 81:1484–1512, doi:10.1016/j.rser.2017.05.212 (Crossref verified).
- Point-forecast PV combination: Raza, Mithulananthan & Summerfield 2018, Solar Energy 166:226–241, doi:10.1016/j.solener.2018.03.066 ("Bayesian adaptive combination"); Yang & Dong 2018, Solar Energy 166:529–541, doi:10.1016/j.solener.2018.02.011 (seasonal time-series ensemble). Both Crossref verified; titles only.
- The audit also lists Hategan et al. 2025, *Energy* 316:134595, doi:10.1016/j.energy.2025.134595, which "dynamically aggregates" ML and sky-image models; its weighting rule was not read (from `v4_reference_audit.md`).

### Inferences
- In the solar literature, combination weights are mostly static or updated online from past scores. A gate driven by forecast-time inputs (expert disagreement and event probability) is the manuscript's distinctive element for sky-image nowcasting, though the principle is generic (Q2).
- Yang, Yang & Liu 2023 is the most direct solar precedent for *combining quantiles* (rather than CDFs) and should be cited alongside Thorey [28].

### Gaps
- Full text and abstract of Yang, Yang & Liu 2023 were not accessible, so whether its weights are τ-specific, pinball-optimised or covariate-dependent is unverified.
- Searches did not surface a peer-reviewed solar paper using covariate-dependent (gated) weights on probabilistic forecasts in a non-MDPI venue. The only hits were MDPI or SSRN preprints, which were excluded.

---

## Q4. Correct terminology

### Takeaway
Call the manuscript's operation a **covariate-dependent (gated) weighted quantile average** or "gated quantile blend". It is a weighted Vincentization. Reserve "mixture", "linear pool" and "probability averaging" for F = Σ w_i F_i.

### Cited Findings
- "Averaging quantiles" vs "averaging probabilities" are distinct aggregation methods with different variance. — [Lichtendahl 2013 abstract](https://econpapers.repec.org/RePEc:inm:ormnsc:v:59:y:2013:i:7:p:1594-1611)
- The "linear pool" is F = Σ w_i F_i. Generalized, spread-adjusted and beta-transformed linear pools are its nonlinear variants. — [Gneiting & Ranjan 2013](https://arxiv.org/html/1106.1638v1)
- "Quantile aggregation" is used for combining density forecasts by their quantiles (Busetti 2017, title; doi:10.1111/obes.12163). "Probability- and quantile-based aggregation methods" is used by Schulz, Köhler & Lerch 2026 ([abstract](https://api.crossref.org/works/10.1007/s10994-025-06946-3)).
- "Vincentization": Ratcliff 1979 (doi:10.1037/0033-2909.86.3.446); Genest 1992 (doi:10.1214/aos/1176348676).
- "Pointwise" or "quantile-wise" CRPS learning, with weights varying in τ and time: Berrisch & Ziel 2023 ([arXiv](https://arxiv.org/abs/2102.00968)).

### Inferences
- Recommended phrasing: "We combine the experts by a gated quantile blend, i.e. a covariate-dependent weighted quantile average (Vincentization; Ratcliff 1979; Genest 1992; Lichtendahl et al. 2013): Q(τ|x) = w(x)Q_D(τ|x) + (1−w(x))Q_R(τ|x). This is not a probability mixture (linear pool; Gneiting & Ranjan 2013). The gate weight w(x) is fitted by minimising the CRPS, as in CRPS-based forecast combination (Thorey et al. 2018; Berrisch & Ziel 2023). It differs from those works in depending on forecast-time covariates rather than on past performance, as in mixture-of-experts gating (Jacobs et al. 1991) and covariate-dependent stacking (Yao et al. 2022)."
- Avoid "mixture of experts" for the output, since MoE output is a mixture density. "Gated ensemble" is acceptable if it is defined as a quantile average.

### Gaps
- No authoritative source prescribes the exact phrase "gated quantile blend". It is the reviewer's suggestion and is consistent with the terminology above.
