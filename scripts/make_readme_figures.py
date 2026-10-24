"""README figures (docs/figures/*.svg), built only from committed result files. No model is run; no other file is written.

results_summary.svg      official SKIPP'D test: paired CRPS differences of G3''-F (95% day-bootstrap CIs, as % of the
                         comparator's test CRPS) and calibrated 90% coverage by stratum   <- reports/test_results.json
published_comparison.svg SkyGPT's public external test: our one-shot G3''-F scored with the competitors' 30-sample
                         estimator vs published CRPS and Winkler scores                  <- reports/estimator_check.json
                         (published values as recorded in reports/phaseA_report.md)
ablation_ladder.svg      development out-of-fold CRPS as ingredients are added (post hoc) <- reports/ablation_ladder.json
Usage: python scripts/make_readme_figures.py
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

S = json.loads(Path("reports/test_results.json").read_text())
EST = json.loads(Path("reports/estimator_check.json").read_text())
LAD = json.loads(Path("reports/ablation_ladder.json").read_text())
OUTDIR = Path("docs/figures")
W = 960

CSS = """
.bg{fill:#ffffff}.title{fill:#0b0b0b;font-size:15px;font-weight:600}.sub{fill:#52514e;font-size:12px}
.lab{fill:#0b0b0b;font-size:13.5px;font-weight:500}.labsm{fill:#0b0b0b;font-size:13px}
.tick{fill:#6e6d68;font-size:11.5px;font-variant-numeric:tabular-nums}
.axis{fill:#52514e;font-size:12px}.colhead{fill:#6e6d68;font-size:11.5px}.ptitle{fill:#0b0b0b;font-size:13px;font-weight:600}
.val{fill:#0b0b0b;font-size:12.5px;font-variant-numeric:tabular-nums}.valmuted{fill:#6e6d68;font-size:12.5px;font-variant-numeric:tabular-nums}
.grid{stroke:#e1e0d9;stroke-width:1}.zero{stroke:#898781;stroke-width:1.5}.base{stroke:#c3c2b7;stroke-width:1}
.sep{stroke:#eeede8;stroke-width:1}.band{fill:#1baf7a;fill-opacity:.16}
.s1{fill:#2a78d6}.l1{stroke:#2a78d6;stroke-width:2;stroke-linecap:round}.ring{stroke:#ffffff;stroke-width:2}
.bar-hi{fill:#2a78d6}.bar-mid{fill:#86b6ef}.bar-lo{fill:#c3c2b7}
@media (prefers-color-scheme:dark){
.bg{fill:#0d1117}.title,.lab,.labsm,.val,.ptitle{fill:#f0efec}.sub,.axis{fill:#c3c2b7}.tick,.colhead,.valmuted{fill:#9a9890}
.grid{stroke:#2c2c2a}.zero{stroke:#898781}.base{stroke:#383835}.sep{stroke:#21262d}.band{fill:#199e70;fill-opacity:.22}
.s1{fill:#3987e5}.l1{stroke:#3987e5}.ring{stroke:#0d1117}.bar-hi{fill:#3987e5}.bar-mid{fill:#1c5cab}.bar-lo{fill:#52514e}}
"""


class Fig:
    def __init__(self):
        self.el = []

    def text(self, x, y, s, cls, anchor="start"):
        self.el.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, cls):
        self.el.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>')

    def dot(self, x, y):
        self.el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" class="s1 ring"/>')

    def bar(self, x0, x1, y, h, cls):
        """Horizontal bar anchored at x0 with a 4px rounded data end at x1."""
        r = min(4.0, (x1 - x0) / 2)
        self.el.append(f'<path class="{cls}" d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
                       f'V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z"/>')

    def save(self, name, height, title, desc):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height:.0f}" width="{W}" height="{height:.0f}" '
               f'role="img" aria-labelledby="t d" font-family="system-ui, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">\n'
               f'<title id="t">{escape(title)}</title>\n<desc id="d">{escape(desc)}</desc>\n'
               f'<style>{CSS}</style>\n<rect class="bg" width="{W}" height="{height:.0f}"/>\n' + "\n".join(self.el) + "\n</svg>\n")
        (OUTDIR / name).write_text(svg, encoding="utf-8")
        print("wrote", OUTDIR / name)


def minus(s):
    return s.replace("-", "−")


# ------------------------------------------------------------------ results_summary.svg
def results_summary():
    H = S["hypotheses"]

    def rel(diff_ci, model, stratum):
        base = S["scores"][model][stratum]["crps_kw"]
        return [100 * v / base for v in diff_ci]

    rows = [("vs direct expert E4-B-fit", "overall · H1, primary", rel(H["H1 overall CRPS G3''-F - E4-B-fit"], "E4-B-fit", "overall")),
            ("vs smart persistence", "overall · H2", rel(H["H2 overall CRPS G3''-F - persistence"], "persistence", "overall")),
            ("vs smart persistence", "cloudy days · H2", rel(H["H2 cloudy CRPS G3''-F - persistence"], "persistence", "cloudy")),
            ("vs direct expert E4-B-fit", "ramp samples · H3 (non-inferiority)",
             rel(H["H3 ramp CRPS G3''-F - E4-B-fit (non-inferiority margin 0.02)"], "E4-B-fit", "ramp"))]
    cov = [(lab, S["intervals"]["G3''-F"]["90"][key]["coverage"])
           for lab, key in (("Overall", "overall"), ("Sunny days", "sunny"), ("Cloudy days", "cloudy"), ("Ramp samples", "ramp"))]
    f, LX, PX0, PX1, VX = Fig(), 24, 300, 700, 724
    f.text(LX, 34, "A · Change in CRPS of G3″-F on the official SKIPP'D test (lower is better)", "title")
    f.text(LX, 54, "Paired difference and 95% day-bootstrap CI, as % of the comparator's test CRPS · 20 days, 13,266 forecasts", "sub")
    X = lambda v: PX0 + (v + 50.0) / 60.0 * (PX1 - PX0)  # noqa: E731
    top, row_h = 82, 56
    bottom = top + row_h * len(rows)
    for t in range(-50, 11, 10):
        f.line(X(t), top - 6, X(t), bottom, "zero" if t == 0 else "grid")
        f.text(X(t), bottom + 18, ("−" if t < 0 else "") + f"{abs(t)}%", "tick", "middle")
    f.text(X(-25), bottom + 38, "← G3″-F better", "axis", "middle")
    f.text(X(5), bottom + 38, "worse →", "axis", "middle")
    f.text(VX, top - 12, "Estimate [95% CI]", "colhead")
    for i, (lab, sub, (est, lo, hi)) in enumerate(rows):
        yc = top + i * row_h + row_h / 2
        if i:
            f.line(LX, top + i * row_h, W - 24, top + i * row_h, "sep")
        f.text(LX, yc - 3, lab, "lab")
        f.text(LX, yc + 15, sub, "sub")
        f.line(X(lo), yc, X(hi), yc, "l1")
        f.dot(X(est), yc)
        excl = hi < 0 or lo > 0
        f.text(VX, yc + 4.5, minus(f"{est:+.1f}%  [{lo:+.1f}, {hi:+.1f}]") + ("" if excl else "  CI spans 0"), "val" if excl else "valmuted")
    y2 = bottom + 84
    f.text(LX, y2, "B · Calibrated 90% prediction-interval coverage of G3″-F", "title")
    f.text(LX, y2 + 20, "Shaded band: pre-specified target 0.90 ± 0.03 (H5 not met: overall coverage too high)", "sub")
    C = lambda v: PX0 + (v - 0.6) / 0.4 * (PX1 - PX0)  # noqa: E731
    ctop = y2 + 44
    cbot = ctop + 40 * len(cov)
    f.text(VX, ctop - 12, "Coverage", "colhead")
    f.el.append(f'<rect x="{C(0.87):.1f}" y="{ctop - 6}" width="{C(0.93) - C(0.87):.1f}" height="{cbot - ctop + 6}" class="band"/>')
    for t in (0.6, 0.7, 0.8, 0.9, 1.0):
        f.line(C(t), ctop - 6, C(t), cbot, "grid")
        f.text(C(t), cbot + 18, f"{t:.1f}", "tick", "middle")
    for i, (lab, v) in enumerate(cov):
        yc = ctop + i * 40 + 20
        f.text(LX, yc + 4.5, lab, "lab")
        f.dot(C(v), yc)
        f.text(VX, yc + 4.5, f"{v:.3f}", "val")
    f.save("results_summary.svg", cbot + 34, "SHADOW-MoE official SKIPP'D test results",
           "Panel A: CRPS change of G3″-F versus E4-B-fit (overall and ramps) and versus smart persistence (overall and cloudy "
           "days), with 95% CIs. Panel B: calibrated 90% interval coverage by stratum versus the 0.87-0.93 target.")
    for lab, sub, v in rows:
        print(f"  {lab} ({sub}): {[round(x, 1) for x in v]}")


# ------------------------------------------------------------------ published_comparison.svg
def published_comparison():
    g = EST["models"]["G3''-F"]
    ours_crps = g["m=30|tail=linear|clip0=False|V-stat (as competitors)"]["mean"]
    ours_wink = EST["winkler90_from_30_samples_percentiles"]["G3''-F"]["mean"]
    pub = EST["published"]
    # (label, sublabel, CRPS, Winkler 90%, ours?) - published values as recorded in reports/phaseA_report.md
    rows = [("G3″-F (ours)", "frozen, run once · scored like competitors", ours_crps, ours_wink, True),
            ("Real future images → U-Net", "oracle reported in the PV-MM-Diffusion paper", 2.52, 30.72, False),
            ("PV-MM-Diffusion", "Huang et al., Applied Energy 2026", pub["PV-MM-Diffusion (m=30)"], EST["winkler90_from_30_samples_percentiles"]["PV-MM-Diffusion published"], False),
            ("SkyGPT", "Nie et al., Adv. Appl. Energy 2024", pub["SkyGPT (m=100)"], EST["winkler90_from_30_samples_percentiles"]["SkyGPT published"], False),
            ("SUNSET", "Sun et al., Solar Energy 2019 (as published)", 3.31, 56.95, False),
            ("Smart persistence", "no-skill reference", 3.67, None, False)]
    f, LX = Fig(), 24
    f.text(LX, 34, "Comparison on SkyGPT's public external test set (5 cloudy days, Nov–Dec 2019, 2,582 forecasts)", "title")
    f.text(LX, 54, "Lower is better in both panels. Ours: frozen model run once. Others: published numbers (unpaired; protocols differ).", "sub")
    A0, A1, B0, B1 = 330, 600, 650, 900
    top, row_h, bh = 104, 44, 18
    bottom = top + row_h * len(rows)
    f.text(A0, top - 20, "CRPS (kW)", "ptitle")
    f.text(B0, top - 20, "Winkler score, 90% interval", "ptitle")
    XA = lambda v: A0 + v / 4.0 * (A1 - A0)  # noqa: E731
    XB = lambda v: B0 + v / 60.0 * (B1 - B0)  # noqa: E731
    for t in range(0, 5):
        f.line(XA(t), top - 8, XA(t), bottom, "grid" if t else "base")
        f.text(XA(t), bottom + 18, str(t), "tick", "middle")
    for t in range(0, 61, 20):
        f.line(XB(t), top - 8, XB(t), bottom, "grid" if t else "base")
        f.text(XB(t), bottom + 18, str(t), "tick", "middle")
    for i, (lab, sub, c, w, ours) in enumerate(rows):
        y = top + i * row_h
        yc = y + row_h / 2
        f.text(LX, yc - 2, lab, "lab")
        f.text(LX, yc + 14, sub, "sub")
        cls = "bar-hi" if ours else "bar-lo"
        f.bar(XA(0), XA(c), yc - bh / 2, bh, cls)
        f.text(XA(c) + 6, yc + 4.5, f"{c:.2f}", "val")
        if w is None:
            f.text(XB(0) + 6, yc + 4.5, "not reported", "valmuted")
        else:
            f.bar(XB(0), XB(w), yc - bh / 2, bh, cls)
            f.text(XB(w) + 6, yc + 4.5, f"{w:.1f}", "val")
    f.save("published_comparison.svg", bottom + 40, "Comparison with published models on SkyGPT's external test",
           "Horizontal bars of CRPS and 90% Winkler score for G3″-F (ours, scored with the competitors' 30-sample estimator), "
           "an oracle using real future images, PV-MM-Diffusion, SkyGPT, SUNSET and smart persistence.")
    print(f"  ours: CRPS {ours_crps:.3f}, Winkler {ours_wink:.2f}")


# ------------------------------------------------------------------ ablation_ladder.svg
def ablation_ladder():
    c = LAD["crps"]
    rows = [("Smart persistence", "reference: cloudiness stays as it is now", c["smart persistence"], "bar-lo"),
            ("Forecast power in kW", "single CNN-GRU network", c["kW single (E4-A): 1 member"], "bar-mid"),
            ("Forecast clear-sky index K", "same network, physics-normalised target", c["K single (E4-B): 1 member"], "bar-mid"),
            ("Direct expert E4-B-fit", "3-member ensemble", c["E4-B-fit: 3-member"], "bar-mid"),
            ("Calm expert E6-S alone", "residual event-mixture head", c["E6-S: 3-member"], "bar-mid"),
            ("50/50 average of both experts", "no gate", c["avg50 E4+E6: 3-member"], "bar-mid"),
            ("Disagreement gate G3", "learned blend of the two experts", c["G3 (gated, 15-min): 3-member"], "bar-mid"),
            ("+ 60-min 3D-CNN expert = G3″-F", "final model (routed)", c["G3''-F (+60-min 3D-CNN expert, routed): 3-member"], "bar-hi")]
    f, LX = Fig(), 24
    f.text(LX, 34, "Where the improvement comes from: one ingredient at a time", "title")
    f.text(LX, 54, f"Mean CRPS (kW, lower is better) on {LAD['n']:,} held-out development forecasts; post hoc, descriptive; "
           "no test data", "sub")
    A0, A1, B0, B1 = 330, 590, 640, 900
    top, row_h, bh = 104, 40, 18
    bottom = top + row_h * len(rows)
    f.text(A0, top - 20, "All forecasts", "ptitle")
    f.text(B0, top - 20, "Ramp forecasts only (fast changes)", "ptitle")
    XA = lambda v: A0 + v / 1.0 * (A1 - A0)  # noqa: E731
    XB = lambda v: B0 + v / 5.0 * (B1 - B0)  # noqa: E731
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        f.line(XA(t), top - 8, XA(t), bottom, "grid" if t else "base")
        f.text(XA(t), bottom + 18, f"{t:g}", "tick", "middle")
    for t in range(0, 6):
        f.line(XB(t), top - 8, XB(t), bottom, "grid" if t else "base")
        f.text(XB(t), bottom + 18, str(t), "tick", "middle")
    for i, (lab, sub, v, cls) in enumerate(rows):
        yc = top + i * row_h + row_h / 2
        f.text(LX, yc - 2, lab, "labsm")
        f.text(LX, yc + 13, sub, "sub")
        f.bar(XA(0), XA(v["overall"]), yc - bh / 2, bh, cls)
        f.text(XA(v["overall"]) + 6, yc + 4.5, f"{v['overall']:.3f}", "val")
        f.bar(XB(0), XB(v["ramp"]), yc - bh / 2, bh, cls)
        f.text(XB(v["ramp"]) + 6, yc + 4.5, f"{v['ramp']:.2f}", "val")
    f.save("ablation_ladder.svg", bottom + 40, "SHADOW-MoE ablation ladder on development data",
           "Horizontal bars of development out-of-fold CRPS, overall and on ramp forecasts, from smart persistence through the "
           "clear-sky index, the two experts, a 50/50 average, the disagreement gate and the final G3″-F model.")


if __name__ == "__main__":
    results_summary()
    published_comparison()
    ablation_ladder()
