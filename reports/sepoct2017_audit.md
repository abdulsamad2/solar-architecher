# Audit: the Sep–Oct 2017 coverage collapse (fold 1)

2026-10-20.
- **Nature of the audit:** descriptive, CPU-only, development data only. It changes **no** Phase 6 training, selection or decision rule.
- **Code:** `src/audit_sepoct2017.py`. Numbers: `reports/sepoct2017_audit.json`.
- **Reference anchor:** the clear-sky anchor used here is fitted on all dev days except September/October of any year. It is a descriptive reference only, not a model input.

## 1. Direction of the misses (E4-B-fit, 3 seeds, fold-validation samples)
| Month | P(y < Q.05) | P(y > Q.95) | Median error y − Q.50 (kW) | Median 90% width (kW) |
|---|---|---|---|---|
| 2017-08 | 0.017 | 0.016 | −0.31 | 6.0 |
| **2017-09** | 0.016 | **0.257** | **+0.23** | 7.0 |
| **2017-10** | 0.022 | **0.406** | **+0.47** | 5.4 |
| 2017-11 | 0.061 | 0.120 | −0.26 | 7.6 |
| 2018-10 | 0.027 | 0.212 | +0.26 | 4.0 |
| 2019-09 | 0.025 | 0.098 | −0.25 | 4.1 |
| 2019-10 | 0.023 | 0.103 | −0.04 | 4.1 |

**The collapse is one-sided: realised PV is above the forecast distribution.** In October 2017, 41% of outcomes exceed the 95% quantile, and the median is biased low by about 0.5 kW. This is a **level bias**, not a lack of spread.

## 2. Inter-annual decline in PV relative to clear sky
Midday clear-day K (descriptive anchor):

| Month | 2017 | 2018 | 2019 |
|---|---|---|---|
| October | **1.008** | 0.971 | 0.948 |
| June | 0.997 | 0.970 | 0.944 |
| May | 1.003 | 0.971 | 0.997 |

- **Trend:** clear-sky-normalised output falls by roughly 2.5–3% per year in most months, consistent with the 1.001 → 0.972 drift between early and late dev days seen when the anchor was first validated.
- **Why it hits fold 1:** fold 1's model is trained mostly on 2018–2019 (plus early 2017). Its clear-sky anchor and the K-space model therefore expect a lower PV/clear-sky ratio than autumn 2017 delivered. That produces the under-prediction in §1.
- **Same pattern, smaller:** October 2018 shows the same under-prediction (21% above Q.95) when the model is trained largely on 2017 and 2019 data.
- **Likely causes, not determinable from this data:** module degradation, soiling, or a change in system or measurement conventions.

## 3. September 2017 weather was atypical
The clear-day share in September was 17% in 2017, against 86% (2018, 7 days only) and 57% (2019). The model saw comparatively few cloudy-September regimes in training. This plausibly adds to the September misses on top of the level bias.

## 4. Wildfire-smoke hypothesis: not supported as the main driver
During the 2017 North Bay fires (Oct 8–31), midday sky-image red/blue ratios dip (Oct 9–20: 1.01 to 1.12, against 1.27 to 1.34 on early-October clear days), and clear-day K is slightly lower (0.96 to 0.97).
- **Direction mismatch:** smoke attenuation would make outcomes fall **below** the forecast. The observed misses are overwhelmingly **above** Q.95.
- **Conclusion:** smoke may affect individual days, but it does not explain the collapse.

## 5. Incidental finding: sky-camera colour change in spring 2019 (bears on Phase 8)
Midday red/blue ratio by month:
- 1.03–1.14 from June 2018 to February 2019;
- **1.76 in May 2019**;
- 1.91–2.15 from June to September 2019.

There are no dev data in March–April 2019 (test or missing days). This is consistent with the suspected camera change around **2019-05-22**, the Phase 8 hypothesis. It now has direct image-statistics support, and the exact date should be pinned down in Phase 8.

**Exposure:** fold 4 (Jul–Oct 2019) and about half of fold 3 lie after the change. Every model so far was trained across both camera regimes.

## 6. Implications (no Phase 6 rule changed)
1. **Mechanism of the fold-1 failure:** an inter-annual level shift in the PV/clear-sky ratio, plus an atypical cloudy September. This explains why **adaptive conformal fixes it** (rolling or ACI widening absorbs a persistent bias) while split conformal fixed on other periods cannot.
2. **Candidate model-level remedies, for after Gate 3** (each needs its own pre-registration):
   - a time-varying (rolling) clear-sky scale;
   - a slow bias-correction term;
   - feeding recent clear-sky-index statistics as inputs.
3. **The 2019 camera colour shift** is now evidenced. Phase 8 robustness work should treat it as a confirmed regime boundary, pending exact dating.
