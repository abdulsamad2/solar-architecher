"""README results figure (docs/figures/results_summary.svg), built only from the committed sealed-test result file.

Reads reports/test_results.json (Stanford SKIPP'D official test); writes no other file.
Panel A: paired CRPS differences of G3''-F and their 95% day-bootstrap CIs, each divided by the comparator's test CRPS.
Panel B: calibrated 90% prediction-interval coverage by stratum against the pre-specified 0.87-0.93 target.
Usage: python scripts/make_readme_figures.py
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

S = json.loads(Path("reports/test_results.json").read_text())
OUT = Path("docs/figures/results_summary.svg")
H = S["hypotheses"]


def rel(diff_ci, model, stratum):
    base = S["scores"][model][stratum]["crps_kw"]
    return [100 * v / base for v in diff_ci]


ROWS = [  # (label, sublabel, [est, lo, hi] %)
    ("vs direct expert E4-B-fit", "overall · H1, primary",
     rel(H["H1 overall CRPS G3''-F - E4-B-fit"], "E4-B-fit", "overall")),
    ("vs smart persistence", "overall · H2",
     rel(H["H2 overall CRPS G3''-F - persistence"], "persistence", "overall")),
    ("vs smart persistence", "cloudy days · H2",
     rel(H["H2 cloudy CRPS G3''-F - persistence"], "persistence", "cloudy")),
    ("vs direct expert E4-B-fit", "ramp samples · H3 (non-inferiority)",
     rel(H["H3 ramp CRPS G3''-F - E4-B-fit (non-inferiority margin 0.02)"], "E4-B-fit", "ramp")),
]
COV = [(lab, S["intervals"]["G3''-F"]["90"][key]["coverage"])
       for lab, key in (("Overall", "overall"), ("Sunny days", "sunny"), ("Cloudy days", "cloudy"), ("Ramp samples", "ramp"))]

W, LX, PX0, PX1, VX = 960, 24, 300, 700, 724
el = []


def text(x, y, s, cls, anchor="start"):
    el.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>')


def dot(x, y):
    el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" class="s1 ring"/>')


def fmt(v):
    return f"{v:+.1f}".replace("-", "−")


# ---- Panel A ----
text(LX, 34, "A · Change in CRPS of G3″-F on the official SKIPP'D test (lower is better)", "title")
text(LX, 54, "Paired difference and 95% day-bootstrap CI, as % of the comparator's test CRPS · 20 days, 13,266 forecasts", "sub")
ax0, ax1 = -50.0, 10.0
X = lambda v: PX0 + (v - ax0) / (ax1 - ax0) * (PX1 - PX0)  # noqa: E731
top, row_h = 82, 56
bottom = top + row_h * len(ROWS)
for t in range(-50, 11, 10):
    el.append(f'<line x1="{X(t):.1f}" y1="{top - 6}" x2="{X(t):.1f}" y2="{bottom}" class="{"zero" if t == 0 else "grid"}"/>')
    text(X(t), bottom + 18, ("−" if t < 0 else "") + f"{abs(t)}%", "tick", "middle")
text(X(-25), bottom + 38, "← G3″-F better", "axis", "middle")
text(X(5), bottom + 38, "worse →", "axis", "middle")
text(VX, top - 12, "Estimate [95% CI]", "colhead")
for i, (lab, sub, (est, lo, hi)) in enumerate(ROWS):
    yc = top + i * row_h + row_h / 2
    if i:
        el.append(f'<line x1="{LX}" y1="{top + i * row_h:.1f}" x2="{W - 24}" y2="{top + i * row_h:.1f}" class="sep"/>')
    text(LX, yc - 3, lab, "lab")
    text(LX, yc + 15, sub, "sub")
    el.append(f'<line x1="{X(lo):.1f}" y1="{yc:.1f}" x2="{X(hi):.1f}" y2="{yc:.1f}" class="l1"/>')
    dot(X(est), yc)
    excl = hi < 0 or lo > 0
    text(VX, yc + 4.5, f"{fmt(est)}%  [{fmt(lo)}, {fmt(hi)}]" + ("" if excl else "  CI spans 0"), "val" if excl else "valmuted")

# ---- Panel B ----
y2 = bottom + 84
text(LX, y2, "B · Calibrated 90% prediction-interval coverage of G3″-F", "title")
text(LX, y2 + 20, "Shaded band: pre-specified target 0.90 ± 0.03 (H5 not met: overall coverage too high)", "sub")
c0, c1 = 0.6, 1.0
C = lambda v: PX0 + (v - c0) / (c1 - c0) * (PX1 - PX0)  # noqa: E731
ctop = y2 + 44
cbot = ctop + 40 * len(COV)
text(VX, ctop - 12, "Coverage", "colhead")
el.append(f'<rect x="{C(0.87):.1f}" y="{ctop - 6}" width="{C(0.93) - C(0.87):.1f}" height="{cbot - ctop + 6}" class="band"/>')
for t in (0.6, 0.7, 0.8, 0.9, 1.0):
    el.append(f'<line x1="{C(t):.1f}" y1="{ctop - 6}" x2="{C(t):.1f}" y2="{cbot}" class="grid"/>')
    text(C(t), cbot + 18, f"{t:.1f}", "tick", "middle")
for i, (lab, v) in enumerate(COV):
    yc = ctop + i * 40 + 20
    text(LX, yc + 4.5, lab, "lab")
    dot(C(v), yc)
    text(VX, yc + 4.5, f"{v:.3f}", "val")
HT = cbot + 34

CSS = """
.bg{fill:#ffffff}.title{fill:#0b0b0b;font-size:15px;font-weight:600}.sub{fill:#52514e;font-size:12px}
.lab{fill:#0b0b0b;font-size:13.5px;font-weight:500}.tick{fill:#6e6d68;font-size:11.5px;font-variant-numeric:tabular-nums}
.axis{fill:#52514e;font-size:12px}.colhead{fill:#6e6d68;font-size:11.5px}
.val{fill:#0b0b0b;font-size:12.5px;font-variant-numeric:tabular-nums}.valmuted{fill:#6e6d68;font-size:12.5px;font-variant-numeric:tabular-nums}
.grid{stroke:#e1e0d9;stroke-width:1}.zero{stroke:#898781;stroke-width:1.5}
.sep{stroke:#eeede8;stroke-width:1}.band{fill:#1baf7a;fill-opacity:.16}
.s1{fill:#2a78d6}.l1{stroke:#2a78d6;stroke-width:2;stroke-linecap:round}.ring{stroke:#ffffff;stroke-width:2}
@media (prefers-color-scheme:dark){
.bg{fill:#0d1117}.title,.lab,.val{fill:#f0efec}.sub,.axis{fill:#c3c2b7}.tick,.colhead,.valmuted{fill:#9a9890}
.grid{stroke:#2c2c2a}.zero{stroke:#898781}.sep{stroke:#21262d}.band{fill:#199e70;fill-opacity:.22}
.s1{fill:#3987e5}.l1{stroke:#3987e5}.ring{stroke:#0d1117}}
"""
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT:.0f}" width="{W}" height="{HT:.0f}" role="img" '
       f'aria-labelledby="t d" font-family="system-ui, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">\n'
       '<title id="t">SHADOW-MoE official SKIPP\'D test results</title>\n'
       '<desc id="d">Panel A: CRPS change of G3″-F versus the direct expert E4-B-fit (overall and ramps) and versus smart '
       'persistence (overall and cloudy days), with 95% CIs. Panel B: calibrated 90% interval coverage by stratum versus the '
       '0.87-0.93 target.</desc>\n'
       f'<style>{CSS}</style>\n<rect class="bg" width="{W}" height="{HT:.0f}"/>\n' + "\n".join(el) + "\n</svg>\n")
OUT.write_text(svg, encoding="utf-8")
for lab, sub, v in ROWS:
    print(f"{lab} ({sub}): {[round(x, 1) for x in v]}")
print("coverage:", [(c[0], round(c[1], 3)) for c in COV])
print("wrote", OUT)
