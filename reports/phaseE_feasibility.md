# Phase E: second-site validation, feasibility (2026-10-23)

Marked items are UNVERIFIED (search snippet only, or the page was not readable).

## Candidates

| Rank | Dataset | Sky images | PV power | Main issues |
|---|---|---|---|---|
| 1 | **Gatton, Univ. of Queensland, Australia** (Xu & Ma 2023, DOI 10.48610/bd7e108; also a SolarBench site) | 2 × Mobotix Q25, 2408×1536, every 10 s [UNVERIFIED] | 1-min, 3.275 MW research plant [array/orientation UNVERIFIED] | Only 2020-01-23 to 2020-05-06 (≈104 days, one season). Southern hemisphere. MW-scale plant with possibly mixed fixed and tracking arrays. Licence "commercial use restricted" [UNVERIFIED]. The SolarBench copy is still private (HTTP 401) until publication. |
| 2 | SIPPMIF, Wollongong (Mendeley cb8t8np9z3) | 1024×768, 10 s, 08:00–16:45 only | 1-min, two rooftops (no metadata) | CC BY-NC 3.0; empty file listing; unknown time span |
| 3 | SIRTA images + SIRTA PV1 bench | 1–2 min (separate product) | IV curves of single modules, 2019–2022 | Manual join; module-scale, not system-scale |
| 4 | SIPM, PUC-Rio | ? | PV V/I/P | Almost no metadata |
| – | UCSD Folsom (Zenodo 2826939) | 1-min, 2014–2016, ~50 GB | **None** (irradiance only) | Not a PV test; same region as Stanford |

**Only Gatton is a verified second site with both 1-min sky images and co-located PV.**

## Proposed Gatton protocol (to be pre-registered before any Gatton outcome is read)
1. **Confirm access and metadata.** Licence, files, and which array the PV series measures (capacity, tilt, azimuth, tracking). Needs the UQ RDM page in a browser, or contact with the SolarBench / UQ authors.
2. **Preprocess like SKIPP'D:**
   - crop the fisheye circle, resize to 64×64, take 1-min frames aligned with the PV;
   - clear-sky model from the array geometry, or an empirical clear-day envelope fitted on non-test days only.
3. **Fix a day-level split before looking:** use SolarBench's own test days if released, else a pre-registered late block. Keep about 20% of days sealed.
4. **Arms:**
   - (a) zero-shot frozen G3″-F, with an image-flip ablation for the southern hemisphere;
   - (b) fine-tuned on Gatton training days (expected to be necessary; SolarBench reports > 50% data savings from pre-training);
   - persistence and SUNSET-style baselines.
5. **Calibration:** continuous causal rolling conformal or ACI (Phase C), pre-registered.
6. **One-shot evaluation** on the sealed Gatton days.

## Effort estimate
- Data access and preprocessing: 1–3 days, depending on the raw download size.
- Fine-tuning 15 members: about 1 day of GPU time.
- Evaluation and report: 1 day.
- **Main blocker:** confirming the Gatton data access, licence and array metadata.
