# Phase B: related work and novelty positioning

2026-10-23. Compiled from four literature searches (sky-image forecasting; calibration and PV decline; ramps, mixtures and rigour; competitor check).
- **Verified:** a page (abstract or full text) was actually opened.
- **[UNVERIFIED]:** search snippet only. Check against the publisher before citing.

## 1. What each of our claims can and cannot say

| Our claim | Closest prior work | Verdict | How to phrase it |
|---|---|---|---|
| **(a)** Soft logistic gate between a calm-condition expert and a 60-min 3D-CNN ramp expert, driven by **between-expert disagreement** and a forecast-time event probability, fitted by CRPS on held-out days | **Hard routing:** Nie et al. 2020 (sky-condition sub-models vs end-to-end, Stanford); McCandless et al. 2016 (cloud-regime ANNs). **Imager and persistence blending:** Straub et al. 2024. **Situation-dependent blending:** Lu et al. 2015. **CRPS-fitted, unconditional weights:** Thorey et al. 2018. **Gates on within-expert variance:** MoGU 2025 | Likely novel; medium–high confidence | "a soft, CRPS-fitted gate driven by expert disagreement". Contrast it with hard sky-class routing (Nie 2020) and with static blending (Straub 2024). **Read Straub 2024's full text before submission.** |
| **(b)** Quantile forecasts with **causal rolling conformal** on sky-image models, plus coverage reported on SKIPP'D and on out-of-period cloudy days | PV conformal without images: Renkema et al. 2024 (Mondrian/kNN, day-ahead); Suresh 2025 (ACI, hour-ahead); CACP 2025. Sky-image intervals without conformal: Chu et al. 2015. Static Mondrian conformal on SKIPP'D: solar-sde (GitHub, not peer-reviewed) | Likely novel in published work; medium confidence | "To our knowledge, the first causal online conformal calibration of sky-image PV forecasts, with regime- and out-of-period coverage analysis." |
| **(c)** Better than published **SUNSET** (SKIPP'D 20-day test) and **SkyGPT** (its 5 cloudy Nov–Dec 2019 days) | SUNSET: RMSE 3.03, FS 16.44% (Nie 2023). SkyGPT: CRPS 2.81, Winkler score 26.70. **PV-MM-Diffusion** (Applied Energy 2026): CRPS **2.63**, Winkler score **21.46**, "18% below SkyGPT→U-Net", trained on SKIPP'D; its test set is **not verified**. Others that do not beat SkyGPT: I-GPT (Pan & Lim 2025, CRPS 2.912 / WS 33.10, test set unclear) and solar-sde repo (3.18 on SkyGPT's test) | Our 2.455 / 18.16 is below *every* published number we found. We cannot claim state of the art until we confirm PV-MM-Diffusion's test set and check "Multimodal ultra-short-term probabilistic solar power forecasting with generative AI and Transformer" (Adv. Appl. Energy 2025) [UNVERIFIED, paywall/403] | "lower CRPS and Winkler score than published SkyGPT, SUNSET and PV-MM-Diffusion numbers on SkyGPT's test samples", only after the test sets are confirmed. Otherwise compare with SkyGPT and SUNSET only. |
| **(d)** SKIPP'D data issues: ±30 s timestamp drift, a **camera change on 2019-05-22**, a ~1-min image/PV offset, and a non-chronological test file | The SKIPP'D paper/repo documents 2017–18 DST failures, gaps, FFmpeg duplicate frames, forward-averaged PV, and a camera replacement on **2022-04-29** (outside the release) | Our four findings appear undocumented; high confidence for the camera date and test order | Report them as new findings, and offer forward-averaged PV as a plausible cause of the 1-min offset. Possible extra: the Hugging Face mirror may store UTC while labelling it Pacific [UNVERIFIED] |
| Forecasts **lag ramps** | Paletta et al. 2021 ("very smart persistence", "always late", TDM > 0.30); ECLIPSE; SkyGPT on SUNSET; SolarBench 2026 (ramp recall < 5%); Schmidt et al. 2016 | **Known**, not novel | Present it as confirmation on SKIPP'D, quantified, and as the motivation for (a) |
| **PV/clear-sky ratio decline** breaks static calibration | Degradation medians 0.5–0.8%/yr (Jordan & Kurtz 2013; fleet about 0.75 [UNVERIFIED]); plant-level 1.2%/yr including soiling (Bolinger 2022). **3-year clear-sky-normalised trends are highly uncertain** (Marion 2021). Clear-sky index is non-stationary (Yang 2020; Voyant 2025) | The link between ratio drift and *conformal coverage* appears unreported; medium confidence | **Do not call it degradation.** Write: "an apparent decline of about 2.2%/yr (season-adjusted) in the PV/clear-sky-model ratio, exceeding typical degradation rates and likely combining degradation, soiling, clear-sky (aerosol) model error and data artefacts (Marion 2021)." |
| Pre-registration and a sealed one-shot test | Pre-registration in ML: NeurIPS workshops (Bertinetto et al. 2021). Holdout reuse: Dwork et al. 2015; Recht et al. 2019. Leakage: Kapoor & Narayanan 2023. Blocked CV: Bergmeir & Benítez 2012; Roberts et al. 2017 [UNVERIFIED]; Cerqueira et al. 2020 | Rare in solar forecasting; high confidence | Make it a strength: "the first pre-registered sky-image solar forecasting study we are aware of" |

## 2. Key references by topic (verified unless marked)

**SKIPP'D and the Stanford group**
- Nie, Li, Scott, Sun, Venugopal, Brandt (2023). SKIPP'D. *Solar Energy* 255:171–179. arXiv:2207.00913.
- Sun, Venugopal, Brandt (2019). SUNSET forecast. *Solar Energy* 188:730–741.
- Venugopal, Sun, Brandt (2019). *J. Renew. Sustain. Energy* 11.
- Nie, Sun, Chen, Orsini, Brandt (2020). Sky-condition-specific sub-models vs end-to-end. *JRSE* 12:046101. This is the closest prior to (a). Volume and results were not opened.
- Nie, Zelikman, Scott, Paletta, Brandt (2024). SkyGPT. *Advances in Applied Energy*. doi:10.1016/j.adapen.2024.100172. arXiv:2306.11682.
- Nie, Paletta, et al. (2024). Heterogeneous multi-location data. *Applied Energy*. arXiv:2211.02108.
- Nie, Li, Paletta, Aragon, Scott, Brandt (2024). Open-source sky image datasets survey. *RSER* 189. arXiv:2211.14709.
- Nie, Campbell, Paletta, …, Brandt, Wang (2026). SolarBench. arXiv:2609.06187. It covers 11 sites, finds ramp recall < 5%, and is deterministic only.

**Other sky-image forecasting**
- Paletta, Arbod, Lasenby (2021). *Solar Energy*. arXiv:2102.00721.
- Paletta, Hu, Arbod, Lasenby (2022). ECLIPSE. *Applied Energy*. arXiv:2104.12419.
- Schmidt et al. (2016). *ACP* 16:3399.
- Siddiqui et al. (2019). WACV.
- Feng et al. (2022). *Applied Energy* 310.
- Zhang et al. (2018). *Solar Energy* 176.
- Wen et al. (2021). *IEEE TII* 17(2) (ramp-rate control).
- Chow et al. (2011). *Solar Energy* 85 (cloud-motion baseline).
- Chu et al. (2015). *Renewable Energy* 83 (sky-image prediction intervals).
- Mercier et al. (2023). arXiv:2305.18487.
- Lanzilao & Meyer (2026). arXiv:2603.26596 (ramp-stratified probabilistic scores; cite it, do not claim the idea).
- Wang & You (2026). arXiv:2605.03165 (ramp CSI on SKIPP'D).

**Probabilistic competitors**
- PV-MM-Diffusion. *Applied Energy* (2026), ScienceDirect S030626192600468X. Trained on SKIPP'D; test set **to verify**.
- Pan & Lim (2025). I-GPT. *CMES* 145(3).
- *Multimodal ultra-short-term probabilistic solar power forecasting with generative AI and Transformer*. Adv. Appl. Energy (2025) [UNVERIFIED].
- SolCloudLLM (2026). arXiv:2609.11135. Uses the SKIPP'D 20-day test, but with normalised MSE only.

**Calibration**
- Gneiting, Balabdaoui, Raftery (2007). *JRSS-B*.
- Gneiting & Raftery (2007). *JASA* [UNVERIFIED page].
- Lauret, David, Pinson (2019). *Solar Energy* 194.
- van der Meer et al. (2018). *RSER* 81.
- Hong et al. (2016). GEFCom2014. *IJF* 32.
- Romano et al. (2019). CQR.
- Tibshirani et al. (2019). Weighted conformal prediction.
- Gibbs & Candès (2021). ACI, NeurIPS.
- Gibbs & Candès (2024). DtACI, JMLR 25.
- Zaffran et al. (2022). AgACI, ICML.
- Xu & Xie. EnbPI.
- Barber et al. (2023). NexCP.
- Angelopoulos et al. (2023). Conformal PID.
- Bhatnagar et al. (2023). SAOCP.
- Renkema et al. (2024). *Solar Energy Advances* 4 (Mondrian PV).
- Suresh (2025). *Energies* 18 (ACI, PV).
- Moradi et al. (2025). CACP, arXiv:2510.15780.

**PV decline and clear-sky modelling**
- Jordan & Kurtz (2013). *Prog. Photovolt.* 21.
- Jordan et al. (2018). RdTools.
- Bolinger et al. (2022). LBNL.
- Marion (2021). *Solar Energy* 223.
- Deceglie et al. (2018). Soiling ratio, *IEEE JPV*.
- Mejia & Kleissl (2013). *Solar Energy* (via a secondary page).
- Ineichen & Perez (2002). *Solar Energy* 73.
- Hay & Davies (1980).
- Holmgren et al. (2018). pvlib, *JOSS*.
- Yang (2020). *JRSE*.
- Voyant et al. (2025). *Solar Energy* 289.

**Combination and rigour**
- Lichtendahl et al. (2013). *Management Science* 59 (quantile averaging is sharper).
- Busetti (2017). Vincentization.
- Gneiting & Ranjan (2013). Linear pools are overdispersed.
- Smith & Wallis (2009). The forecast combination puzzle.
- Thorey et al. (2018). *IJF* 34 (online CRPS weights for PV).
- Dwork et al. (2015).
- Recht et al. (2019). Note: they attribute the drop mostly to distribution shift.
- Kapoor & Narayanan (2023). *Patterns*.
- Bergmeir & Benítez (2012).
- Cerqueira et al. (2020).
- Yang et al. (2020). Verification of deterministic solar forecasts.
- Still needed: a block-bootstrap citation (Künsch 1989; Politis & Romano 1994).

## 3. Actions arising
1. **Before any state-of-the-art wording:** obtain the full texts of PV-MM-Diffusion and the Adv. Appl. Energy 2025 multimodal paper, and record their test sets. If they used SkyGPT's 5 days, add them to the Phase A comparison table (ours 2.455 / 18.16 vs 2.63 / 21.46).
2. **Read Straub et al. 2024** (blending rule) and **Nie et al. 2020** (sky-condition sub-models), the two closest priors to the gate.
3. **Re-word the PV decline everywhere** as an *apparent PV/clear-sky ratio decline*, not degradation.
4. **Phase C comparators already pre-registered** (rolling, ACI, DtACI-style, Mondrian) cover the main families. Conformal PID, NexCP and weighted/similarity conformal are possible later extensions; they are not added now, to keep the pre-registration intact.
5. **Quantify lag on SKIPP'D** (TDM or lag cross-correlation, as in Paletta 2021) as a figure that motivates the gate.
6. **External validation candidates (Phase E):** SolarBench Gatton (the only other PV site), Folsom (CC BY 4.0, GHI), SIRTA.
