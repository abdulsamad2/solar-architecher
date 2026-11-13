# Prior art: mixture-of-experts, regime-dependent combination and gated expert blending for probabilistic solar/PV forecasting (for the SHADOW-MoE novelty statement)

Compiled 2026-11-12. "Already cited" status was checked against `paper/qc/v4_reference_audit.md` (60 refs) and `paper/qc/citation_verification.md`. Note that `paper/manuscript/citation_verification.md` does not exist; the file is at `paper/qc/`. Every DOI below was resolved on Crossref on 2026-11-12, unless it is marked "DOI not independently resolved". Depth of reading is given per item: FT = full text read, Abs = abstract or highlights only.

**The SHADOW-MoE design these works are compared against:** two separately trained probabilistic experts are combined. The first is a direct image+PV quantile model. The second is a residual Gaussian-mixture "event" expert that uses a longer PV history. A 2-feature logistic gate sets the weights. Its features are (g) the event expert's quantile-implied P(large 15-min change) and (d) the disagreement between the experts' predictive means. The output is a gated weighted average of quantiles, followed by causal rolling conformal calibration. The data are SKIPP'D (Stanford), 15 min ahead, PV power.

**Excluded by author preference (MDPI or Chinese-published), listed only for awareness:**
- López-Cuesta et al. 2023, *Remote Sensing* 15(9): blends ASI, satellite and data-driven models with ML.
- "An Ensemble Approach for Intra-Hour Forecasting of Solar Resource", *Energies* 16(18):6608 (2023). Statistical, ML and all-sky ensemble "ENw" ([mdpi](https://www.mdpi.com/1996-1073/16/18/6608)); authorship not checked.
- "Probabilistic Intraday PV Power Forecast Using Ensembles of Deep Gaussian Mixture Density Networks", *Energies* 2023, doi:10.3390/en16020646.
- Song et al. 2025, *Sensors*, "Dynamic Model Selection in a Hybrid Ensemble Framework…", doi:10.3390/s25144489.
- "Weather-Transition-Aware Dynamic MoE for Day-Ahead Probabilistic Wind Power Forecasting", *Processes* 2026, doi:10.3390/pr14213400.
- Bayasgalan et al. 2025, *Energies*, doi:10.3390/en18092300.

---

## Q1. Which published works use MoE or gated/regime-switching combinations for solar irradiance or PV forecasting?

### Takeaway
Combining sky-image forecasts with other models is well established. On the Stanford data itself, Nie et al. (2020) route images to sky-condition-specific sub-models, which is a hard classifier with no blending. Most sky-image blending work uses weights that depend on lead time or on recent error (Straub 2024; Nouri 2021, 2025), not on forecast-time outputs. The 2025–2026 probabilistic PV MoE papers (STD-MoE, TD-M²oE, seasonal and context-gated MoE) put gates on internal representations or input context, inside one jointly trained network, and none uses sky images. The closest sky-image work is GateFormer (Energy and AI, 2026). It gates two candidate irradiance forecasts per sample, and its gate is given the disagreement between them. GateFormer is deterministic.

### Cited Findings

**Sky-image / nowcasting combination and regime work**

- **Nie, Sun, Chen, Orsini & Brandt 2020**, *J. Renew. Sustain. Energy* 12(4):046101, doi:[10.1063/5.0014016](https://doi.org/10.1063/5.0014016). Status: **NOT currently cited.** It was removed in the OA-swap pass; the v4 audit (issue 7, MAJOR) recommends restoring it.
  - Design: a "two-stage classification-prediction framework". Input images are first classified into sky conditions, then sent to condition-specific CNN sub-models. It is compared with an end-to-end CNN.
  - Classifiers: a CSI-labelled CNN classifier versus a non-parametric fractional-cloudiness threshold. The cloudiness classifier with 3 classes (sunny, cloudy, overcast) worked best.
  - Results: RMSE 2.20 kW (30 kW array) on 18 test days. The task is a contemporaneous PV *nowcast* (image → concurrent PV), not 15 min ahead.
  - Source: [Stanford EAO abstract](https://eao.stanford.edu/publications/journal-articles/pv-power-output-prediction-sky-images-using-convolutional-neural).
  - Weights: hard routing on a regime label derived from the current image (current-information dependent, but discrete). Deterministic. Same site (Stanford).
  - Difference from SHADOW: no soft blending, no probabilistic outputs, and the routing signal is an input-image statistic rather than an expert output.
- **Straub, Herzberg, Dittmann & Lorenz 2024**, *Solar Energy* 269:112319, doi:[10.1016/j.solener.2024.112319](https://doi.org/10.1016/j.solener.2024.112319). Status: **already cited [27].** Read in full (Fraunhofer repository PDF).
  - Combines an ML all-sky-imager (ASI) nowcast, a satellite nowcast and persistence of GHI / clear-sky index. Data: 8 stations within ~10 km in Freiburg, lead times up to 15 min.
  - Weights: "LT-dependent linear regression … FChybrid(LT,x) = FCASI·cASI(LT) + FCSAT·cSAT(LT) + FCPST·cPST(LT) … coefficients c depend on the LT, but not on the location … 16 sets of c-coefficients" ([Fraunhofer full text](https://publica.fraunhofer.de/bitstreams/bcb5b879-996d-4308-8744-8cf7fcf6699a/download)).
  - **The weights do not depend on current conditions**, only on lead time. They were fitted offline. Variability classes are used only for evaluation. Deterministic.
  - Hybrid RMSE improvement: 5–13% over the best individual method ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0038092X24000136)).
  - Non-linear ML blending appears in a parallel EU PVSEC 2023 paper (doi:10.4229/EUPVSEC2023/4CO.8.3; DOI not independently resolved), which gained "SRMSE = 0–2% alongside SMAE = 5–7.5%" over the linear hybrid (same full text).
- **Hategan, Stefu, Petreuș, et al. 2025**, *Energy* 316:134595, doi:[10.1016/j.energy.2025.134595](https://doi.org/10.1016/j.energy.2025.134595). Status: **NOT cited** (the audit proposes it). Abstract only; CC BY-NC open access, but ScienceDirect blocked automated full-text retrieval.
  - "A new weighted ensemble model … dynamically aggregates machine learning and models based on sky images". Data: Solar Platform, West University of Timișoara, Romania; intra-hour PV power.
  - The ensemble "penalizes large errors at the cost of average precision". Gains are reported at horizons >15 min and in high sky variability ([ScienceDirect abstract](https://www.sciencedirect.com/science/article/pii/S0360544225002373)).
  - **The weighting rule (whether the weights depend on current conditions or on past errors) could not be verified.** See Gaps. Output appears deterministic: the highlights mention only skill score over persistence.
- **Nouri, Blum, Wilbert & Zarzalejo 2021**, *Solar RRL* 6:2100442, doi:[10.1002/solr.202100442](https://doi.org/10.1002/solr.202100442). Status: NOT cited. FT read of the EU PVSEC 2021 conference version.
  - Combines ASI and persistence GHI nowcasts. Data: Plataforma Solar de Almería, 8 stations, 62 days.
  - Weights: inverse-squared RMSD of each method, computed in real time over the "past 5 minutes" of pyranometer data. This is "a sliding validation of recent historical nowcast" ([DLR elib PDF](https://elib.dlr.de/145707/1/EUPVSEC2021_5BO.7.4.pdf)).
  - Time-varying weights driven by recent *performance* (causal), not by forecast-time outputs. Deterministic. RMSD reduction about 12%.
- **Nouri, Lezaca, Fabel, Hammer, Blum & Wilbert 2025**, *Solar RRL* 9(24):e202500486, doi:10.1002/solr.202500486 (resolved via OpenAlex). Status: NOT cited. Abstract only.
  - A transformer ASI model (to 1 h) is blended with Heliosat-3 satellite and persistence "via linear regression as well as with distinct advanced machine learning algorithms". RMSE −13.6% and MAE −17.0% versus satellite alone. Trained in southern Spain, validated in northern Germany ([OpenAlex record](https://api.openalex.org/works/doi:10.1002/solr.202500486)).
  - Deterministic. Whether the ML blenders take condition features as inputs was not verified.
- **Hamlyn, Chaudhary, Hare & Rahman 2026, "GateFormer"**, *Energy and AI* 26:100917, doi:[10.1016/j.egyai.2026.100917](https://doi.org/10.1016/j.egyai.2026.100917); also arXiv:2610.30942. Status: **NOT cited. This is the closest prior art for the gate.** FT read from arXiv.
  - Architecture: a GHI-conditioned dual-decoder ViT. A Trajectory Head and a Delta Head give two final-horizon GHI forecasts, which are "adaptively weighted in prediction space", followed by a supervised residual correction.
  - Data: NREL-TSI, Chilbolton and SIRTA. +30 min RMSE 125.01 W/m² on NREL-TSI ([arXiv abs](https://arxiv.org/abs/2610.30942)).
  - Gate: "Linear 256→1 followed by sigmoid ω∈[0,1]". Its weight "is predicted from a context vector containing the encoded sky-image sequence representation, recent GHI-state information, the Trajectory Head forecast, the Delta-derived final-horizon forecast, and the disagreement between the two decoder outputs" ([arXiv PDF](https://arxiv.org/pdf/2610.30942)).
  - The authors contrast this with "conventional ensembles that combine independently trained models" and with "MoE architectures that route inputs" (same PDF).
  - Weights depend on current forecast-time information, including output disagreement. Deterministic (point GHI). Both candidates come from one shared, jointly trained backbone.
- **Zhang, Wilson, Sumner & Wu 2023**, *Renewable Energy* 216:118952, doi:10.1016/j.renene.2023.118952. Status: NOT cited; DOI taken from the GateFormer reference list and not independently resolved.
  - Compares early, late and *gated* fusion of sky-image and meteorological features. Ramp-event balanced accuracy improves by 9.43% and 3.91% at 2 and 6 min (as summarised in [GateFormer arXiv](https://arxiv.org/pdf/2610.30942)).
  - This is feature-level gating, not output-level gating.
- **"Two-stage weather-aware unlabeled sky image classification and solar irradiance forecasting…"**, *Solar Energy* 319:115104 (Nov 2026), doi:[10.1016/j.solener.2026.115104](https://doi.org/10.1016/j.solener.2026.115104). Status: NOT cited. Abstract only.
  - Unsupervised clustering with pseudo-labels gives sky regimes, which then condition a "regime-aware solar irradiance forecasting model" ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0038092X26007930)).
  - Regime classification from the current image. Deterministic GHI.
- **McCandless et al. 2016** (k-means regimes; already cited [26]) and **Thorey et al. 2018**, *Int. J. Forecast.*, "Ensemble forecast of photovoltaic power with online CRPS learning", doi:10.1016/j.ijforecast.2018.05.007 (found by Crossref title search; already cited [28]). Both are established regime and CRPS-weighting precedents ([ScienceDirect Thorey](https://www.sciencedirect.com/science/article/pii/S016920701830089X)).

**2022–2026 probabilistic PV MoE papers (the reviewer's "recent probabilistic PV MoE methods")**

- **Liu et al. 2026, "STD-MoE"**, *Applied Energy* 426:128543, doi:[10.1016/j.apenergy.2026.128543](https://doi.org/10.1016/j.apenergy.2026.128543). Status: NOT cited. Abstract only.
  - Decomposes the PV series into structural components. A "sparse Mixture-of-Experts mechanism … adaptively select[s] periodic patterns". Multi-quantile heads follow, and "conformal calibration is applied" ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306261926011992)).
  - Probabilistic (quantiles plus conformal), like SHADOW, but the MoE sits on internal feature branches of a single end-to-end network. PV time series and meteorology only; no sky images. Dataset not stated in the abstract.
- **Su et al. 2026, "TD-M²oE"**, *IEEE Trans. Sustain. Energy*, doi:10.1109/TSTE.2026.3681486 (resolved via Crossref and OpenAlex). Status: NOT cited. Abstract only.
  - A "task-difficulty-guided multi-gate mixture-of-experts … within paired-quantile multi-task learning … a novel masking gate that adaptively allocates experts across varied quantile tasks". The loss enforces quantile monotonicity and expert diversity ([OpenAlex](https://api.openalex.org/works/doi:10.1109/tste.2026.3681486); [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11475548)).
  - Probabilistic (intervals). The gates allocate experts per *quantile task*, not per weather or event state. No sky images.
- **Ur Rehman et al. 2025, "Seasonal MoE"**, IEEE PES APPEEC 2025, doi:10.1109/APPEEC66370.2025.11382130. Status: NOT cited.
  - Season-specialised LSTM experts. "The gating network dynamically weights their contributions based on current context" ([OpenAlex](https://api.openalex.org/works/doi:10.1109/appeec66370.2025.11382130)).
  - Deterministic. Conference paper.
- **Wang et al. 2025, "Context-Gated Dynamic Fusion"**, ICEMCE 2025, doi:10.1109/ICEMCE68156.2025.11466687. Status: NOT cited.
  - A "Context-Gated MoE … lightweight gating network to dynamically assign weights to diverse base learners (LSTM, GRU, Transformer, tree-based) based on real-time input features, **local prediction variance**, and historical residuals". Prediction intervals come from fusing MC-dropout, LightGBM quantile regression and conformal prediction ([OpenAlex](https://api.openalex.org/works/doi:10.1109/icemce68156.2025.11466687)).
  - This is a low-tier conference paper, but it does give the gate an output-dispersion feature ("local prediction variance").
- A KSII TIIS (Korea) 2026 paper, "A Multimodal Fusion Framework for Solar Forecasting using Dynamic GNNs and Mixture of Experts", doi:10.3837/tiis.2026.03.012, was found only by Crossref title search. Not read.

### Inferences
- Within sky-image PV and irradiance work, the published combination weights fall into four types:
  - static or lead-time-dependent: Straub 2024; the Nouri 2025 linear variant;
  - driven by recent past errors: Nouri 2021; Thorey 2018;
  - hard regime routing from image or observation statistics: Nie 2020; McCandless 2016; Solar Energy 2026 two-stage;
  - learned per-sample gates inside one network: GateFormer.
- SHADOW-MoE differs from each of these. Its weights are a per-forecast function of *probabilistic expert outputs* only. It blends full quantile sets of *separately trained* experts with different input histories, and it is evaluated probabilistically with conformal calibration on SKIPP'D.
- None of the probabilistic PV MoE papers found (STD-MoE, TD-M²oE) uses sky images. Their gates operate on hidden representations or quantile tasks, so they are architectural comparators rather than forecast-combination comparators.

### Gaps
- **Hategan 2025:** the weighting rule could not be read (ScienceDirect blocks scripts; the paper is CC BY-NC open access). A manual browser read is needed to state whether the weights depend on current conditions, such as state-of-the-sky or variability, or on past errors.
- **Nouri 2025:** the inputs to the ML blenders are unknown.
- **STD-MoE and TD-M²oE:** datasets, horizons and gating inputs are unknown (abstracts only).
- No refereed (Elsevier, IEEE journal, AIP, Wiley) sky-image MoE with *probabilistic* outputs was found.

---

## Q2. Which works use expert disagreement, ensemble spread or predicted event probability as gating or weighting features?

### Takeaway
Using the disagreement between candidate forecasts as a gate input is **not new**. GateFormer (Energy and AI 2026) feeds "the disagreement between the two decoder outputs" to its sigmoid gate on sky-image GHI forecasts. A 2025 IEEE conference PV ensemble uses "local prediction variance" in its gate. In general forecasting, forecast diversity is a meta-learning feature for combination weights (Kang et al. 2022), and density-combination weights can vary across the predictive distribution (Kapetanios et al. 2015). No work was found that gates experts on one expert's **predicted probability of a large change (event or ramp probability)**.

### Cited Findings
- **GateFormer** (doi:10.1016/j.egyai.2026.100917). The gate context includes "the Trajectory Head forecast, the Delta-derived final-horizon forecast, and the disagreement between the two decoder outputs". Retaining both candidates "exposes their values and disagreement to the gate before the final prediction is formed" ([arXiv PDF](https://arxiv.org/pdf/2610.30942)). Deterministic, so the inputs are point forecasts.
- **Wang et al. 2025 (ICEMCE)** gate inputs: "real-time input features, local prediction variance, and historical residuals" ([OpenAlex](https://api.openalex.org/works/doi:10.1109/icemce68156.2025.11466687)).
- **Kang, Cao, Petropoulos & Li 2022**, "Forecast with forecasts: Diversity matters", *Eur. J. Oper. Res.* 301:180–190, doi:[10.1016/j.ejor.2021.10.024](https://doi.org/10.1016/j.ejor.2021.10.024). Status: NOT cited.
  - Builds on the importance of "the forecast method pool's diversity when constructing forecast combination models" ([arXiv 2012.01643](https://arxiv.org/pdf/2012.01643v2)).
  - Uses diversity of the pool's forecasts as features to learn combination weights. This is series-level meta-learning, not per-time-step gating (from abstract and introduction; details not re-read).
- **Montero-Manso, Athanasopoulos, Hyndman & Talagala 2020, "FFORMA"**, *Int. J. Forecast.* 36:86–92, doi:[10.1016/j.ijforecast.2019.02.011](https://doi.org/10.1016/j.ijforecast.2019.02.011). Feature-based meta-learned combination weights; the features are of the input series. Status: NOT cited.
- **Kapetanios, Mitchell, Price & Fawcett 2015**, "Generalised density forecast combinations", *J. Econometrics* 188:150–165, doi:[10.1016/j.jeconom.2015.02.047](https://doi.org/10.1016/j.jeconom.2015.02.047). Status: NOT cited.
  - Combination weights that vary over regions of the outcome or forecast density. This is a precedent for weights that depend on the forecast distribution itself (Crossref-verified metadata; content from prior knowledge, not re-read this session).
- **Gneiting & Ranjan 2013**, "Combining predictive distributions", *Electron. J. Stat.* 7:1747–1782, doi:[10.1214/13-EJS823](https://doi.org/10.1214/13-EJS823). Status: NOT cited (proposed in the audit). Linear, spread-adjusted and beta-transformed pools; the audit notes covariate-dependent pooling.
- **Nouri et al. 2023**, "Probabilistic solar nowcasting based on all-sky imagers", *Solar Energy* 253:285–307, doi:[10.1016/j.solener.2023.01.060](https://doi.org/10.1016/j.solener.2023.01.060). Status: NOT cited.
  - Not a combination method. Its "nonparametric probabilistic quantile nowcasting method based on deterministic ASI nowcast … takes irradiance variabilities as main predictor of nowcast uncertainties" ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0038092X23000683)).
  - Precedent for conditioning the predictive distribution on a variability state. That state comes from observations, not from an expert's event probability.
- **Wang, Zhang, Chen, Hong, Kang & Xia 2019**, "Combining Probabilistic Load Forecasts", *IEEE Trans. Smart Grid* 10:3664–3674, doi:[10.1109/TSG.2018.2833869](https://doi.org/10.1109/TSG.2018.2833869). Status: NOT cited.
  - Quantile-level combination of probabilistic forecasts with weights fitted to minimise pinball loss (constrained quantile regression averaging). Static weights. Comparator for "CRPS/pinball-optimised quantile combination" (content from prior knowledge, metadata Crossref-verified).

### Inferences
- The claim proposed in the audit (P0034: "we are not aware of earlier work that uses the experts' disagreement as the gating signal") is **contradicted by GateFormer** and should be dropped.
- What remains distinctive, with no counter-example found in this search:
  - (g) is a *probabilistic* output: an event probability implied by one expert's predictive quantiles. It is not a point-forecast difference.
  - The gate blends whole *quantile functions* of two *independently trained probabilistic* experts with different PV-history lengths.
  - The gate is deliberately low-capacity (2 features, logistic) and interpretable.
- GateFormer gives its gate a 256-dimensional context that includes visual features. Its candidates share one backbone and the model is deterministic.

### Gaps
- I did not find any solar or energy paper that gates on a *predicted event or ramp probability* from one component. Absence of evidence is limited by search depth: about 6 targeted web, Crossref and OpenAlex queries; Google Scholar was not queried directly.
- Kapetanios 2015 and Kang 2022 were not re-read in full this session; their descriptions rest on abstracts and prior knowledge.

---

## Q3. Which works use mixture-density or Gaussian-mixture heads for solar nowcasting?

### Takeaway
Mixture-distribution PV and irradiance forecasters exist: GGMM-FNN, Markov-chain mixture and CVAE continuous–discrete mixtures. None found uses sky images or acts as a gated "event" expert. The only deep Gaussian-mixture density network found for PV is in an excluded MDPI journal. Within the reviewer's preferred novelty focus, the "event-mixture expert" is the least contested component, but it should be described as an application of mixture-density modelling.

### Cited Findings
- **Cherni et al. 2026**, "Interpretable generalized Gaussian mixture modeling for risk-aware solar power forecasting", *Sustainable Cities and Society: Advances* 2:100022, doi:[10.1016/j.scsadv.2025.100022](https://doi.org/10.1016/j.scsadv.2025.100022). Status: NOT cited.
  - A generalized Gaussian mixture parameterised by an FNN and trained by NLL. The mixture weights and shape parameters are read as indicators of "regime changes, tail behavior". Data: GEFCom2014 and UNISOLAR; no sky images ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S3051052X25000221)).
- **Munkhammar & Widén 2024**, "Very short-term probabilistic and scenario-based forecasting of solar irradiance using Markov-chain mixture distribution modeling", *Solar Energy Advances*, doi:10.1016/j.seja.2024.100057 (Crossref/OpenAlex). Status: NOT cited.
  - Markov-chain mixture distributions for minute-resolution clear-sky index (Oahu; Norrköping). They are the most accurate in CRPS against PeEn and climatology ([OpenAlex](https://api.openalex.org/works/doi:10.1016/j.seja.2024.100057)).
  - Earlier version: Munkhammar et al. 2019, *Solar Energy*, doi:10.1016/j.solener.2019.04.014 (found by Crossref search).
- **Yang et al. 2026**, "Improved probabilistic forecasting for solar photovoltaic generation via CVAE with continuous-discrete mixture distributions", *Solar Energy*, doi:10.1016/j.solener.2025.114209 (found by Crossref title search; abstract not retrieved). Status: NOT cited.
- **Nouri et al. 2023** (above): a nonparametric quantile ASI nowcast, not a mixture.
- **Huang et al. 2026 PV-MM-Diffusion [22]** and **Xiong et al. 2025 [25]**, both already cited. The GateFormer review summarises them as image and PV pathways that "jointly generate future sky states and probabilistic PV trajectories" and that "aggregate the resulting distributions" ([arXiv](https://arxiv.org/pdf/2610.30942)). These are sample or mixture-type uncertainty without a parametric mixture head.
- Excluded MDPI: *Energies* 2023, doi:10.3390/en16020646 (ensembles of deep Gaussian mixture density networks for intraday PV).

### Inferences
- A residual Gaussian-mixture expert that uses a longer PV history and whose mixture output is turned into an *event probability that drives a gate* was not found in the sources searched. This is a defensible point of specificity.
- The mixture-density head itself is not new; it should be credited generically, for example to Bishop's MDN (not verified this session) or to the PV mixture works above.

### Gaps
- No refereed sky-image + MDN nowcasting paper was found. The Yang 2026 CVAE abstract was not read.

---

## Q4. What is the closest prior art to "output-derived gate features", and how should the novelty be worded?

### Takeaway
The closest prior art is **GateFormer (2026)**: a sky-image GHI forecaster with a per-sample sigmoid gate over two candidate forecasts. Its gate is given the candidates' values and their disagreement. Next closest are Gneiting & Ranjan (2013), Kapetanios et al. (2015) and Kang et al. (2022) for weights that depend on forecasts or covariates, and Thorey (2018) and Wang et al. (2019) for CRPS- or pinball-optimal combination. Novelty should therefore be phrased as a specific configuration, not as a new mechanism.

### Cited Findings (closest prior art, ranked by similarity to SHADOW's gate)

| # | Work | What is combined | How weights are set | Depend on current forecast-time info? | Probabilistic? | Data | Key difference from SHADOW-MoE | Cited? |
|---|---|---|---|---|---|---|---|---|
| 1 | GateFormer, Energy & AI 2026, [doi](https://doi.org/10.1016/j.egyai.2026.100917) | 2 heads (trajectory, delta) from one shared ViT | Learned sigmoid gate on a 256-d context (visual + GHI state + both forecasts + their disagreement) | Yes | No (point GHI) | NREL-TSI, Chilbolton, SIRTA | Deterministic; joint training; high-capacity gate; no event probability | No |
| 2 | Wang et al. ICEMCE 2025, [doi](https://doi.org/10.1109/ICEMCE68156.2025.11466687) | LSTM, GRU, Transformer, tree learners | Lightweight gating network on inputs, local prediction variance and historical residuals | Yes | Point MoE; intervals by fusion of MC-dropout, QR and CP | PV (multi-site, unnamed) | No images; conference paper; dispersion feature, not event probability | No |
| 3 | Nie et al. 2020, [doi](https://doi.org/10.1063/5.0014016) | Sky-condition CNN sub-models | Hard routing by cloudiness threshold or CSI-CNN classifier | Yes (current image) | No | Stanford (SKIPP'D predecessor) | Discrete routing from an input statistic; nowcast | No (audit: add) |
| 4 | Hategan et al. 2025, [doi](https://doi.org/10.1016/j.energy.2025.134595) | ML + sky-image models | "Dynamically aggregates" (rule unverified) | Unknown | No (apparently) | Timișoara | Rule unverified | No (audit: add) |
| 5 | Nouri et al. 2021, [doi](https://doi.org/10.1002/solr.202100442) | ASI + persistence | Inverse-squared RMSD over the past 5 min | Past errors only | No | PSA, Spain | Performance-based, not output-based | No |
| 6 | Straub et al. 2024, [doi](https://doi.org/10.1016/j.solener.2024.112319) | ASI + satellite + persistence | Lead-time-specific linear regression | No | No | Freiburg | Static per lead time | Yes [27] |
| 7 | Thorey et al. 2018, [doi](https://doi.org/10.1016/j.ijforecast.2018.05.007) | NWP ensemble members | Online weights from past CRPS | Past CRPS only | Yes | French PV plants | Weights not a function of current outputs | Yes [28] |
| 8 | Gneiting & Ranjan 2013, [doi](https://doi.org/10.1214/13-EJS823) | Predictive distributions | Linear, spread-adjusted and beta-transformed pools | Covariate-dependent variants | Yes | General | General theory | No (audit: add) |
| 9 | Kapetanios et al. 2015, [doi](https://doi.org/10.1016/j.jeconom.2015.02.047) | Density forecasts | Weights varying across density regions | Yes (outcome-region dependent) | Yes | Macro | Not solar; not gated on an event probability | No |
| 10 | Kang et al. 2022, [doi](https://doi.org/10.1016/j.ejor.2021.10.024) | Forecast pool | Meta-learned from forecast diversity | Series-level | Point | M4 etc. | Diversity as a feature, but series-level | No |
| 11 | Wang et al. 2019 IEEE TSG, [doi](https://doi.org/10.1109/TSG.2018.2833869) | Quantile load forecasts | Pinball-optimal (constrained QRA) | No | Yes | Load | Static CRPS/pinball-optimal comparator | No |
| 12 | STD-MoE 2026, [doi](https://doi.org/10.1016/j.apenergy.2026.128543); TD-M²oE 2026, [doi](https://doi.org/10.1109/TSTE.2026.3681486) | Internal experts | Learned sparse or multi-gate on hidden features | Yes | Yes (quantiles; STD-MoE adds conformal) | PV series | Internal MoE, jointly trained, no images | No |

### Inferences: suggested wording (no first-of-its-kind claims)

**Drop** the audit-proposed sentence "we are not aware of earlier work that uses the experts' disagreement as the gating signal". GateFormer contradicts it.

**Suggested positioning sentence (Related work, P0034):**
> "Sky-image forecasts have been combined with persistence, satellite and statistical models using lead-time-specific regression weights [Straub 2024], weights from recent errors [Nouri 2021; Thorey 2018], dynamically weighted ensembles [Hategan 2025], or hard routing to sky-condition sub-models, including on the Stanford data [Nie 2020]. Learned per-sample gates have recently been placed over candidate sky-image forecasts. In GateFormer, for example, the gate sees the candidate forecasts and their disagreement [Hamlyn 2026]. Probabilistic PV mixture-of-experts models gate internal representations within one network [STD-MoE; TD-M²oE]. Our design follows covariate- and forecast-dependent pooling [Gneiting & Ranjan 2013; Kapetanios 2015]."

**Suggested contribution sentence:**
> "We combine two separately trained probabilistic experts that use different PV-history lengths: a direct sky-image + PV quantile model and a residual Gaussian-mixture event expert. The combination uses a two-feature logistic gate fitted by CRPS minimisation. The gate's inputs are derived from the experts' predictive distributions, namely the event expert's implied probability of a large 15-min change and the gap between the experts' predictive means. Its weight blends the experts' quantile functions and is followed by causal conformal calibration. Unlike gated sky-image forecasters with deterministic outputs [Hamlyn 2026], the gate operates on probabilistic quantities and outputs a calibrated predictive distribution."

**Comparators the reviewer requested:** add an ablation against static and CRPS-optimal combination weights (Thorey 2018 [28]; Wang 2019 constrained QRA; Gneiting & Ranjan linear pool). If feasible, also add a gate that uses (d) only, which mimics GateFormer's disagreement input. Together these isolate the value of the event-probability feature (g).

### Gaps
- Hategan 2025 weighting rule: unverified.
- GateFormer's journal version (Energy and AI 26:100917) was not compared line by line with arXiv:2610.30942. The gate description was taken from the arXiv version (stated as "submitted to Elsevier Sep 22, 2026").
- Google Scholar and Semantic Scholar were not queried directly; discovery used web search, Crossref and OpenAlex. Some 2026 IEEE or Elsevier items may have been missed.
