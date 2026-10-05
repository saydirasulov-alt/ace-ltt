"""TikZ / pgfplots figure style and the PRE-REGISTERED figure specification (part of analysis_code_sha256).

Every figure is written as <name>.tex: one self-contained tikzpicture (pgfplots), data inline, no external files,
to be \\input{} in the IEEE paper; the plotted numbers are also written to <name>.csv. When pdflatex is available
a preview (preview/<name>.pdf, .png) is compiled from exactly that .tex; it is only a check, the .tex is the figure.

Paper preamble (also written to figure_preamble.tex):
    \\usepackage{pgfplots} \\usepgfplotslibrary{groupplots} \\pgfplotsset{compat=1.17}
    \\usetikzlibrary{arrows.meta,shapes.geometric,positioning,fit,patterns,calc}
Widths are relative to \\textwidth (double column) and \\columnwidth (single); text is \\footnotesize (8 pt in
IEEEtran). Colours: Okabe-Ito (colour-blind safe); heat map: cividis. Certified points are filled, uncertified
hollow, so the figures also read in greyscale.

DEFAULT_SPEC is used unless protocol.yaml has a `figures:` block (then locked by freeze). Axis limits, method order
and the main / supplement split are fixed BEFORE freeze; a value beyond an axis limit is drawn AT the limit with a
distinct marker and flagged in the CSV (column `clipped`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

OKABE_ITO = {"oiOrange": "E69F00", "oiSky": "56B4E9", "oiGreen": "009E73", "oiYellow": "F0E442",
             "oiBlue": "0072B2", "oiVerm": "D55E00", "oiPurple": "CC79A7", "oiBlack": "000000", "oiGrey": "999999"}

# method -> (display label, colour name, pgfplots mark, extra mark options)
METHODS = {
    "edge_fixed":               ("Edge, fixed",                 "oiGrey",   "square*",      ""),
    "edge_LTT_image":           ("Edge LTT (image)",            "oiSky",    "triangle*",    "rotate=180"),
    "edge_LTT":                 ("Edge LTT (unit)",             "oiBlue",   "*",            ""),
    "detector_gated_agent":     ("Detector-gated VLM",          "oiYellow", "pentagon*",    ""),
    "cascade_fixed":            ("Cascade, fixed",              "oiBlack",  "otimes*",      ""),
    "cascade_LTT_image":        ("Cascade LTT (image)",         "oiPurple", "triangle*",    ""),
    "cascade_LTT_bonf":         ("Cascade LTT (Bonferroni)",    "oiOrange", "diamond*",     ""),
    "cascade_LTT":              ("Cascade LTT (unit)",          "oiVerm",   "*",            ""),
    "cascade_LTT+human":        ("Cascade LTT + human",         "oiGreen",  "oplus*",       ""),
    "all_frames_cloud_overlay": ("All frames to VLM (overlay)", "oiPurple", "halfcircle*",  ""),
    "pure_cloud_plain":         ("Pure cloud VLM (plain)",      "oiGrey",   "halfsquare*",  ""),
}

DEFAULT_SPEC = {
    "method_order": ["edge_fixed", "edge_LTT_image", "edge_LTT", "detector_gated_agent", "cascade_fixed",
                     "cascade_LTT_image", "cascade_LTT_bonf", "cascade_LTT", "cascade_LTT+human",
                     "all_frames_cloud_overlay", "pure_cloud_plain"],
    "cost_methods": ["pure_cloud_plain", "all_frames_cloud_overlay", "detector_gated_agent", "cascade_fixed",
                     "cascade_LTT", "cascade_LTT+human", "edge_LTT"],
    "source_methods": ["edge_LTT", "cascade_LTT", "cascade_LTT+human"],
    "main": ["fig1_architecture", "fig2_dataset_audit", "fig3_risk_cost", "fig4_forest", "fig6_by_source"],
    "supplement": ["fig5_edge_cloud_cost", "fig7_head_selection"],
    "risk_max": 0.20,             # event miss risk axis (alpha = 0.05)
    "calls_max": 1.0,             # share of frames sent to the cloud
    "uplink_kb_max": 400.0,       # KB per frame
    "latency_ms_max": 5000.0,     # server-measured mean end-to-end ms per frame (NOT the edge device)
    "preview": "auto",            # auto = compile a PDF/PNG preview if pdflatex exists; off = .tex/.csv only
    "preview_dpi": 300,
}

PREAMBLE = r"""% packages needed by the cascade figures (\input{figX.tex})
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.17}
\usetikzlibrary{arrows.meta,shapes.geometric,positioning,fit,patterns,calc}
"""

# cividis (5 stops), used for the heat map; colours are computed here so the .tex needs no colormap package
CIVIDIS = [(0.00, (0x00, 0x22, 0x4E)), (0.25, (0x35, 0x45, 0x6C)), (0.50, (0x7C, 0x7B, 0x78)),
           (0.75, (0xBC, 0xAF, 0x6F)), (1.00, (0xFE, 0xE8, 0x38))]


def spec_from(P):
    S = dict(DEFAULT_SPEC)
    src = "figure_style.DEFAULT_SPEC"
    if P and isinstance(P.get("figures"), dict):
        S.update(P["figures"])
        src = "protocol.yaml figures: (over DEFAULT_SPEC)"
    return S, src


def label(m):
    return METHODS.get(m, (m, "oiGrey", "*", ""))[0]


def colour(m):
    return METHODS.get(m, (m, "oiGrey", "*", ""))[1]


def ordered(names, spec, key="method_order"):
    order = spec.get(key) or spec["method_order"]
    return [m for m in order if m in names]


def tex(s):
    """Escape plain text for LaTeX (method / source names)."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in str(s))


def f(x, nd=5):
    """Deterministic number formatting for inline pgfplots data."""
    return f"{float(x):.{nd}f}".rstrip("0").rstrip(".") if abs(float(x)) < 1e15 else "0"


def mark_style(m, filled=True, clipped=False):
    c = colour(m)
    _, _, mk, extra = METHODS.get(m, (m, "oiGrey", "*", ""))
    if clipped:
        mk, extra = "triangle*", "rotate=-90"
    opts = [f"draw={c}", f"fill={c if filled else 'white'}", "line width=0.6pt", "solid"]
    if extra:
        opts.append(extra)
    return f"mark={mk}, mark size={'2.6' if mk in ('oplus*', 'otimes*', 'pentagon*') else '2.2'}pt, " \
           f"mark options={{{', '.join(opts)}}}"


def colour_defs():
    return "\n".join(rf"\definecolor{{{n}}}{{HTML}}{{{h}}}" for n, h in OKABE_ITO.items())


def cividis(v, vmax, reverse=True):
    """HTML colour for v in [0, vmax] (reverse: 0 -> light yellow, vmax -> dark blue)."""
    t = 0.0 if vmax <= 0 else min(max(v / vmax, 0.0), 1.0)
    if reverse:
        t = 1.0 - t
    for (t0, c0), (t1, c1) in zip(CIVIDIS, CIVIDIS[1:]):
        if t <= t1:
            w = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return "".join(f"{round(a + w * (b - a)):02X}" for a, b in zip(c0, c1))
    return "FEE838"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


WRAPPER = r"""\documentclass[border=2pt]{standalone}
\pdfinfoomitdate=1 \pdftrailerid{} \pdfsuppressptexinfo=-1
%s
\usepackage{xcolor}
%s
\setlength{\textwidth}{7.16in}
\setlength{\columnwidth}{3.5in}
\begin{document}
\input{%s}
\end{document}
"""


def compile_preview(out, name, spec, man):
    """pdflatex the figure's .tex exactly as written -> preview/<name>.pdf (+ .png). Fails only if the .tex itself
    does not compile (a missing font package falls back to the default fonts)."""
    if spec.get("preview", "auto") == "off":
        return None
    exe = shutil.which("pdflatex")
    if not exe:
        man.setdefault("preview_notes", []).append("pdflatex not found: .tex/.csv only (compile in the paper)")
        return None
    out = Path(out)
    b = out / "preview" / "_build"
    b.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1"}
    src = (out / f"{name}.tex").resolve().as_posix()
    # preview fonts: Times text if the installation has it; a TeX install with missing font files (e.g. no
    # rsfs) must not block the check, so retry once with the default fonts. The figure .tex is identical.
    errs = []
    for fonts in (r"\usepackage{times}", ""):
        (b / f"{name}_wrap.tex").write_text(WRAPPER % (fonts, PREAMBLE, src))
        ok = True
        for _ in range(2):                               # 2 passes: legend to name / \ref
            p = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", f"{name}_wrap.tex"], cwd=b,
                               capture_output=True, text=True, env=env)
            if p.returncode != 0:
                log = (b / f"{name}_wrap.log").read_text(errors="replace") if (b / f"{name}_wrap.log").exists() \
                    else ""
                errs.append([ln for ln in log.splitlines() if ln.startswith("!")][:3])
                ok = False
                break
        if ok:
            if fonts == "":
                man.setdefault("preview_notes", []).append(f"{name}: preview compiled with default fonts "
                                                           f"(Times failed: {errs[0]})")
            break
    else:
        man.setdefault("preview_notes", []).append(f"{name}: pdflatex failed {errs}")
        raise RuntimeError(f"pdflatex failed for {name} (the .tex itself does not compile): {errs[-1]}")
    pdf = out / "preview" / f"{name}.pdf"
    shutil.copyfile(b / f"{name}_wrap.pdf", pdf)
    files = [pdf]
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", str(spec.get("preview_dpi", 300)), "-singlefile", str(pdf),
                        str(pdf.with_suffix(""))], check=False, capture_output=True)
        if pdf.with_suffix(".png").exists():
            files.append(pdf.with_suffix(".png"))
    return files


def save(out, name, tikz, spec, rows, cols, manifest):
    """Write <name>.tex (the figure) and <name>.csv (plotted numbers); compile a preview if possible."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    head = (f"% {name}: generated by cascade/figures.py (do not edit by hand; regenerate)\n"
            "% needs: pgfplots (groupplots), tikz libraries arrows.meta, shapes.geometric, positioning, fit,"
            " patterns, calc\n")
    t = out / f"{name}.tex"
    t.write_text(head + colour_defs() + "\n" + tikz.strip() + "\n")
    c = out / f"{name}.csv"
    with open(c, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    files = [t, c]
    prev = compile_preview(out, name, spec, manifest) or []
    base = name.split("_dev")[0]
    role = "main" if base in spec["main"] else "supplement" if base in spec["supplement"] else "other"
    manifest["figures"][name] = {"role": role, "files": {p.name: sha256(p) for p in files},
                                 "preview": {p.name: sha256(p) for p in prev}}
    return files


def write_manifest(out, manifest):
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "figure_preamble.tex").write_text(PREAMBLE)
    Path(out, "figures_manifest.json").write_text(json.dumps(manifest, indent=1))
