"""Paper figures as TikZ / pgfplots (.tex, data inline) + the plotted numbers (.csv). They never re-open or re-run
anything; with pdflatex available a PDF/PNG preview is compiled from exactly the written .tex.

    python -m cascade.figures dev   --protocol <P> --dev-dir <RES> [--head-json H] [--dataset-audit A] --out <dir>
    python -m cascade.figures final --protocol <P> [--out <dir>] [--post-freeze-figure-fix]

dev    reads ONLY dev outputs: <RES>/dev_summary.json, dev_trials.csv, dev_by_source.json (validity_* and final/*
       are never opened), head_report.json (dev role) and dataset_audit.json (label-free). Result figures carry a
       "DEV (val role)" stamp: they choose nothing and are not paper results. fig1 / fig2 / fig7 are the paper
       versions (architecture, label-free dataset audit, dev head selection).
final  reads ONLY final.json of the finished final, located through the protocol receipt:
         * receipt state must be 'done'; sha256(final.json) and sha256(final.md) must equal the receipt;
         * protocol.yaml / splits.csv / locked score files must still match the lock;
         * the analysis code must match analysis_code_sha256. If ONLY figure code (figure_style.py, figures.py)
           changed after freeze, --post-freeze-figure-fix allows it and records the changed files in
           figures_manifest.json (to be disclosed); results cannot change because only final.json is read.
       No records, labels or scores are read, the receipt is not modified, final is never re-run.
Every run writes figures_manifest.json (input hashes, code hash, spec source, clipped points, file hashes) and
figure_preamble.tex (the packages the paper needs).

Figures: fig1_architecture, fig2_dataset_audit, fig3_risk_cost, fig4_forest, fig5_edge_cloud_cost,
fig6_by_source, fig7_head_selection. Main / supplement split, method order and axis limits: figure_style.py
DEFAULT_SPEC or protocol.yaml `figures:` (both fixed at freeze).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade import figure_style as FS  # noqa: E402
from cascade.figure_style import f, tex  # noqa: E402

RISK_KEY = {"frame": "miss_unit", "event": "miss_event"}
RISK_NAMES = {"y_fire": "fire", "y_smoke": "smoke"}
DEV_STAMP = "DEV (val role): repeated splits, not a sealed-test result"
SRC_NAMES = {"dfire": "D-Fire", "fasdd_cv": "FASDD_CV", "pyro_sdis": "Pyro-SDIS", "flame2_det": "FLAME2"}


# ------------------------------------------------------------------ helpers
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _clip(v, vmax):
    """(plotted value, clipped?) -- a value beyond the pre-registered axis limit is drawn at the limit."""
    if np.isfinite(v) and v > vmax:
        return vmax, True
    return v, False


def _stamp(dev):
    return (r"\node[anchor=north east, font=\tiny\itshape, text=black!55] at (current bounding box.south east) "
            f"{{{DEV_STAMP}}};\n") if dev else ""


def _src(c):
    s, _, tag = str(c).partition("/")
    s = SRC_NAMES.get(s, s.replace("wildfire_ext_", "ext "))
    return tex(f"{s} {tag}" if tag and tag != "-" else s)


AXIS = (r"/pgfplots/every axis/.append style={font=\footnotesize, tick label style={font=\scriptsize}, "
        r"label style={font=\footnotesize}, legend style={font=\scriptsize, draw=none, fill=none}, legend cell align=left, "
        r"axis line style={line width=0.4pt}, tick style={line width=0.4pt}, axis lines*=left, "
        r"xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2, /pgf/number format/1000 sep={}}, "
        r"yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2, /pgf/number format/1000 sep={}}, "
        r"title style={font=\footnotesize, text height=1.7ex, text depth=0.4ex}}")


# ------------------------------------------------------------------ fig 1: architecture (pure TikZ, no data)
def fig1_architecture(out, spec, man):
    T = r"""\begin{tikzpicture}[font=\footnotesize, >={Stealth[length=4pt]}, line width=0.5pt,
  box/.style={draw, rounded corners=2pt, align=center, minimum height=0.95cm, inner sep=2pt},
  dec/.style={draw, diamond, aspect=1.5, align=center, inner sep=0.5pt, font=\scriptsize},
  res/.style={draw, rounded corners=2pt, minimum width=1.35cm, minimum height=0.55cm},
  lab/.style={font=\scriptsize, inner sep=1pt},
  tier/.style={draw, dashed, rounded corners=4pt, inner sep=5pt}]
\node[box, minimum width=1.3cm] (cam) at (0.8,2.2) {Camera\\frame};
\node[box, minimum width=1.9cm, fill=oiBlue!15] (det) at (2.95,2.2) {YOLO26\\detector\\score $s$};
\node[dec] (d1) at (5.45,2.2) {$s$ vs\\$t_{\mathrm{low}},t_{\mathrm{high}}$};
\node[res, fill=oiVerm!20] (a1) at (5.45,4.05) {Alarm};
\node[res] (si) at (5.45,0.35) {Silence};
\draw[->] (cam) -- (det);
\draw[->] (det) -- (d1);
\draw[->] (d1) -- node[lab, right] {$s\ge t_{\mathrm{high}}$} (a1);
\draw[->] (d1) -- node[lab, right] {$s<t_{\mathrm{low}}$} (si);
\node[box, minimum width=2.3cm, fill=oiOrange!22] (vlm) at (9.75,2.2) {VLM agent\\(Qwen2.5-VL)\\$g$ = Yes/No logit};
\node[dec] (d2) at (12.45,2.2) {$g$ vs\\$b,a$};
\node[res, fill=oiVerm!20] (a2) at (12.45,4.05) {Alarm};
\node[res] (rj) at (12.45,0.35) {Reject};
\draw[->] (d1) -- node[lab, above] {$t_{\mathrm{low}}\le s<t_{\mathrm{high}}$}
                  node[lab, below, text=black!65, align=center] {uplink frame\\(+ boxes)} (vlm);
\draw[->] (vlm) -- (d2);
\draw[->] (d2) -- node[lab, right] {$g\ge a$} (a2);
\draw[->] (d2) -- node[lab, right] {$g<b$} (rj);
\node[box, minimum width=2.2cm, fill=oiGreen!18] (hu) at (16.4,2.2) {Human review\\(hand-off)};
\draw[->] (d2) -- node[lab, above] {$b\le g<a$} (hu);
\begin{scope}[on background layer]
\end{scope}
\node[tier, draw=oiBlue, fit=(cam)(det)(d1)(a1)(si)] (E) {};
\node[anchor=north west, text=oiBlue, font=\footnotesize\bfseries] at (E.north west) {Edge device};
\node[tier, draw=oiOrange, fit=(vlm)(d2)(a2)(rj)] (C) {};
\node[anchor=north west, text=oiOrange, font=\footnotesize\bfseries] at (C.north west) {Cloud};
\node[tier, draw=oiGreen, fit=(hu), inner sep=8pt] (O) {};
\node[anchor=south west, text=oiGreen, font=\footnotesize\bfseries] at (O.north west) {Operator};
\node[draw=black!45, fill=black!5, rounded corners=2pt, text width=17.4cm, align=center, font=\scriptsize,
      inner sep=3pt] at (8.6,-0.8) {Offline, once: Learn-then-Test on calibration units
      (Hoeffding--Bentkus p-values, Pareto testing) selects $\theta=(t_{\mathrm{low}},t_{\mathrm{high}},b,a)$ with
      $\Pr(\mbox{certify }\theta\mbox{ whose fire or smoke event-miss risk}>\alpha)\le\delta$, then minimises
      cloud calls, hand-offs and false alarms.};
\end{tikzpicture}"""
    T = T.replace("\\begin{scope}[on background layer]\n\\end{scope}\n", "")
    rows = [{"element": e, "rule": r} for e, r in (
        ("edge", "s >= t_high -> alarm; s < t_low -> silence; else uplink"),
        ("cloud", "g >= a -> alarm; g < b -> reject; else hand-off to operator"),
        ("calibration", "LTT (HB p-values, Pareto testing), unit = split_group, joint fire/smoke event-miss risk"))]
    FS.save(out, "fig1_architecture", T, spec, rows, ["element", "rule"], man)


# ------------------------------------------------------------------ fig 2: dataset audit (label-free)
def fig2_dataset_audit(A, out, spec, man):
    rows = []
    P = []           # the four panels (pgfplots \nextgroupplot bodies)
    # (a) before / after per source
    ba = A.get("before_after_by_source") or {}
    srcs = [s for s in SRC_NAMES if s in ba]
    parts = [("kept_train", "train", "oiBlue"), ("kept_val", "val", "oiSky"), ("kept_test", "test", "oiGreen"),
             ("exact_copy_removed", "exact copy removed", "oiVerm"), ("invalid", "invalid", "oiBlack")]
    if srcs:
        tot = [sum(ba[s].get(k, 0) for k, _, _ in parts) / 1000 for s in srcs]
        xmax = max(tot) * 1.8
        body = [rf"\nextgroupplot[title={{(a) before / after cleaning, by source}}, xbar stacked, bar width=7pt,"
                rf" xmin=0, xmax={f(xmax, 1)}, ytick={{{','.join(str(i) for i in range(len(srcs)))}}},"
                rf" yticklabels={{{','.join(tex(SRC_NAMES[s]) for s in srcs)}}}, y dir=reverse,"
                rf" ymin=-0.6, ymax={len(srcs) - 0.4}, xlabel={{images (thousands)}}, area legend, clip=false,"
                r" legend columns=5, legend style={at={(0.5,-0.36)}, anchor=north, /tikz/every even column/"
                r".append style={column sep=3pt}}]"]
        for key, lab, col in parts:
            pts = " ".join(f"({f(ba[s].get(key, 0) / 1000, 3)},{i})" for i, s in enumerate(srcs))
            body.append(rf"\addplot[fill={col}, draw=none] coordinates {{{pts}}}; \addlegendentry{{{lab}}}")
            for s in srcs:
                rows.append({"panel": "a", "series": key, "subset": s, "x": ba[s].get(key, 0), "y": ""})
        for i, s in enumerate(srcs):
            rm = ba[s].get("exact_copy_removed", 0) + ba[s].get("invalid", 0)
            body.append(rf"\node[anchor=west, font=\scriptsize] at (axis cs:{f(tot[i], 3)},{i}) "
                        rf"{{{ba[s].get('raw', 0):,} raw, {rm:,} removed}};")
        P.append("\n".join(body))
    else:
        P.append(r"\nextgroupplot[title={(a) before / after cleaning}, hide axis, xmin=0, xmax=1, ymin=0, ymax=1]"
                 "\n" r"\node at (axis cs:0.5,0.5) {needs \texttt{--work} (fsclean clean\_work)};")
    # (b) original split leakage
    lk = A.get("pipeline_original_split_leakage") or {}
    keys = [k for k in ("dfire/test", "fasdd_cv/test", "fasdd_cv/val", "pyro_sdis/val") if k in lk]
    gs = A.get("groups_spanning_splits")
    cs = A.get("cross_source_near_duplicates") or {}
    body = [rf"\nextgroupplot[title={{(b) leakage of the original splits}}, xbar, bar width=5pt, xmin=0, xmax=115,"
            rf" ytick={{{','.join(str(i) for i in range(len(keys)))}}},"
            rf" yticklabels={{{','.join(_src(k).replace(chr(92) + '_', chr(92) + '_') for k in keys)}}},"
            rf" y dir=reverse, ymin=-0.6, ymax={len(keys) - 0.4},"
            r" xlabel={held-out images of the \emph{original} split (\%)}, nodes near coords,"
            r" area legend, nodes near coords style={font=\tiny, /pgf/number format/.cd, fixed, fixed zerofill, precision=1},"
            r" legend columns=2, legend style={at={(0.5,-0.36)}, anchor=north, /tikz/every even column/"
            r".append style={column sep=4pt}}]"]
    for key, lab, col, sh in (("pct", "verified near-duplicate in train", "oiVerm", "2.8pt"),
                              ("pct_group", "same camera / dup.\\ group as train", "oiOrange", "-2.8pt")):
        pts = " ".join(f"({f(lk[k][key], 2)},{i})" for i, k in enumerate(keys))
        body.append(rf"\addplot[fill={col}, draw=none, bar shift={sh}] coordinates {{{pts}}}; "
                    rf"\addlegendentry{{{lab}}}")
        for k in keys:
            rows.append({"panel": "b", "series": key, "subset": k, "x": lk[k][key], "y": ""})
    note = f"FireSmoke-Clean: {gs if gs is not None else '?'} split units span train/val/test"
    if cs:
        d2f, f2d = cs.get("dfire->fasdd_cv", 0), cs.get("fasdd_cv->dfire", 0)
        note += rf"\\cross-source duplicates: {d2f:,} D-Fire images in FASDD clusters, {f2d:,} reverse"
        rows += [{"panel": "b", "series": "cross_source", "subset": "dfire->fasdd_cv", "x": d2f, "y": ""},
                 {"panel": "b", "series": "cross_source", "subset": "fasdd_cv->dfire", "x": f2d, "y": ""}]
    body.append(rf"\node[anchor=north west, font=\scriptsize, text=oiBlue, align=left] at (rel axis cs:0,-0.56)"
                rf" {{{note}}};")
    P.append("\n".join(body))
    # (c) split unit sizes (CCDF)
    sz = A.get("split_group_sizes_by_source") or {}
    body = [r"\nextgroupplot[title={(c) split units (split\_group)}, xmode=log, ymode=log,"
            r" xlabel={split unit size (frames)}, ylabel={share of units $\ge$ size},"
            r" legend columns=2, legend style={at={(0.5,-0.36)}, anchor=north, /tikz/every even column/.append style={column sep=4pt}}, log basis x=10, log basis y=10]"]
    for s, col in (("dfire", "oiBlue"), ("fasdd_cv", "oiOrange"), ("pyro_sdis", "oiGreen"),
                   ("flame2_det", "oiPurple"), ("multi-source", "oiVerm")):
        if not sz.get(s):
            continue
        v = np.sort(np.array(sz[s]))
        uniq = np.unique(v)
        cc = np.array([(v >= u).mean() for u in uniq])
        pts = " ".join(f"({int(u)},{cc_:.6g})" for u, cc_ in zip(uniq, cc))
        body.append(rf"\addplot[const plot mark left, {col}, line width=0.7pt] coordinates {{{pts}}}; "
                    rf"\addlegendentry{{{tex(SRC_NAMES.get(s, s))} ({len(v):,} units)}}")
        rows += [{"panel": "c", "series": s, "subset": "ccdf", "x": int(u), "y": round(float(c_), 6)}
                 for u, c_ in zip(uniq, cc)]
    P.append("\n".join(body))
    # (d) pHash sensitivity
    S = A.get("phash_sensitivity") or []
    th = A.get("thresholds") or {}
    if S:
        pm, sm = th.get("phash_max", 8), th.get("scene_phash_max", 10)
        body = [r"\nextgroupplot[title={(d) sensitivity to the pHash threshold}, xmin=0, xmax=14.5, ymin=0, ymax=100,"
                r" xtick={0,2,...,14}, xlabel={pHash threshold $t$ (direct neighbours)},"
                r" ylabel={held-out with a train neighbour (\%)}, legend columns=2, legend style={at={(0.5,-0.36)},"
                r" anchor=north, /tikz/every even column/.append style={column sep=4pt}}]",
                rf"\fill[black!8] (axis cs:{sm},0) rectangle (axis cs:14.5,100);",
                rf"\draw[black!65, dashed] (axis cs:{pm},0) -- (axis cs:{pm},100);",
                rf"\node[anchor=north east, font=\tiny, text=black!65] at (axis cs:{pm},99) {{used ($t={pm}$)}};",
                rf"\node[anchor=south, font=\tiny, text=black!65, align=center] at (axis cs:{(sm + 14.5) / 2},2)"
                r" {not pixel-\\verifiable};"]
        for split, sub, col, lab in (("original", "dfire/test", "oiVerm", "D-Fire test (orig.)"),
                                     ("original", "fasdd_cv/test", "oiOrange", "FASDD test (orig.)"),
                                     ("original", "pyro_sdis/val", "oiPurple", "Pyro-SDIS val (orig.)"),
                                     ("FireSmoke-Clean", "test (all sources)", "oiBlue", "FireSmoke-Clean test")):
            for kind, ls in (("pixel_verified", "solid, line width=0.9pt"), ("hash_candidate",
                                                                             "densely dotted, line width=0.7pt")):
                pts = sorted((r["phash_t"], r["pct"]) for r in S if r["split"] == split and r["subset"] == sub
                             and r["kind"] == kind)
                if not pts:
                    continue
                c_ = " ".join(f"({a_},{f(b_, 2)})" for a_, b_ in pts)
                leg = rf"\addlegendentry{{{lab}}}" if kind == "pixel_verified" else ""
                body.append(rf"\addplot[{col}, {ls}{'' if leg else ', forget plot'}] coordinates {{{c_}}}; {leg}")
                rows += [{"panel": "d", "series": f"{split}:{sub}:{kind}", "subset": "", "x": a_, "y": b_}
                         for a_, b_ in pts]
        body.append(r"\addlegendimage{black, solid, line width=0.9pt} \addlegendentry{pixel-verified}")
        body.append(r"\addlegendimage{black, densely dotted, line width=0.7pt} \addlegendentry{hash only (look-alikes)}")
        P.append("\n".join(body))
    else:
        P.append(r"\nextgroupplot[title={(d) pHash threshold}, hide axis, xmin=0, xmax=1, ymin=0, ymax=1]"
                 "\n" r"\node at (axis cs:0.5,0.5) {needs \texttt{--work} (fsclean clean\_work)};")
    T = (r"\begin{tikzpicture}[" + AXIS + "]\n"
         r"\begin{groupplot}[group style={group name=group, group size=2 by 2, horizontal sep=2.9cm, vertical sep=3.2cm},"
         r" width=0.40\textwidth, height=4.1cm, title style={font=\footnotesize, at={(0,1)}, anchor=south west},"
         r" scaled x ticks=false]" + "\n" + "\n".join(P) + "\n\\end{groupplot}\n\\end{tikzpicture}")
    FS.save(out, "fig2_dataset_audit", T, spec, rows, ["panel", "series", "subset", "x", "y"], man)


# ------------------------------------------------------------------ fig 3: risk vs cost
def fig3_risk_cost(Pts, out, spec, man, alpha, dev):
    names = FS.ordered([p["method"] for p in Pts], spec)
    Pm = {p["method"]: p for p in Pts}
    R = spec["risk_max"]
    panels = [("calls", "cloud calls (share of frames)", spec["calls_max"]),
              ("uplink_kb", "uplink (KB / frame)", spec["uplink_kb_max"]),
              ("latency_ms", r"mean end-to-end latency (ms, server)", spec["latency_ms_max"])]
    rows, clipped, G = [], [], []
    for j, (key, xl, xmax) in enumerate(panels):
        opt = [f"xmin=0, xmax={f(xmax * 1.03)}, ymin=0, ymax={f(R * 1.03)}", f"xlabel={{{xl}}}",
               "scaled x ticks=false"]
        if j == 0:
            opt.append(r"ylabel={event miss risk (max of fire, smoke)}")
            opt.append(r"legend to name=fig3legend, legend columns=4, legend style={/tikz/every even column/"
                       r".append style={column sep=5pt}}")
        body = [rf"\nextgroupplot[{', '.join(opt)}]",
                rf"\draw[dashed, line width=0.5pt] (axis cs:0,{f(alpha)}) -- (axis cs:{f(xmax * 1.03)},{f(alpha)});"]
        if j == 0:
            body.append(rf"\node[anchor=south east, font=\scriptsize] at (axis cs:{f(xmax)},{f(alpha)}) {{$\alpha$}};")
        for m in names:
            p = Pm[m]
            x, cx = _clip(p[key], xmax)
            y, cy = _clip(p["risk"], R)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            lo, hi = p.get("lo", np.nan), p.get("hi", np.nan)
            eb = ""
            if np.isfinite(lo) and np.isfinite(hi):
                up, dn = max(min(hi, R) - y, 0), max(y - min(lo, R), 0)
                eb = f" += (0,{f(up)}) -= (0,{f(dn)})"
            leg = rf"\addlegendentry{{{tex(FS.label(m))}}}" if j == 0 else ""
            body.append(rf"\addplot[only marks, {FS.mark_style(m, p['cert'], cx or cy)}, "
                        rf"error bars/.cd, y dir=both, y explicit, error bar style={{{FS.colour(m)}, line width=0.5pt}},"
                        rf" error mark=none] coordinates {{({f(x)},{f(y)}){eb}}}; {leg}")
            if cx or cy:
                clipped.append(f"{m}:{key}")
            rows.append({"method": m, "x_metric": key, "x": p[key], "risk": p["risk"], "lo": lo, "hi": hi,
                         "certified": p["cert"], "clipped": cx or cy})
        if j == 0:
            body.append(r"\addlegendimage{only marks, mark=*, mark options={draw=black, fill=white}}"
                        rf"\addlegendentry{{hollow = not certified{' (rate $<0.5$)' if dev else ''}}}")
        G.append("\n".join(body))
    ci = "bars: trial 2.5--97.5\\% range" if dev else \
        f"bars: {int(man.get('bootstrap_reps') or 0)}-replicate unit-bootstrap 95\\% CI"
    T = (r"\begin{tikzpicture}[" + AXIS + "]\n"
         r"\begin{groupplot}[group style={group name=group, group size=3 by 1, horizontal sep=0.9cm, y descriptions at=edge left},"
         r" width=0.37\textwidth, height=4.3cm]" + "\n" + "\n".join(G) + "\n\\end{groupplot}\n"
         r"\node[anchor=south] at ($(group c2r1.north)+(0,0.15cm)$) {\pgfplotslegendfromname{fig3legend}};" + "\n"
         rf"\node[anchor=north east, font=\scriptsize] at (group c3r1.north east) {{{ci}}};" + "\n"
         + _stamp(dev) + r"\end{tikzpicture}")
    name = "fig3_risk_cost" + ("_dev" if dev else "")
    man["clipped"][name] = clipped
    FS.save(out, name, T, spec, rows, ["method", "x_metric", "x", "risk", "lo", "hi", "certified", "clipped"], man)


# ------------------------------------------------------------------ fig 4: forest plot
def fig4_forest(F, out, spec, man, alpha, risks, dev):
    names = FS.ordered({r["method"] for r in F}, spec)
    cols = list(risks or []) + ["max"]
    titles = {**{k: f"{RISK_NAMES.get(k, k)} event miss" for k in (risks or [])}, "max": "max (controlled risk)"}
    idx = {(r["method"], r["risk"]): r for r in F}
    R = spec["risk_max"]
    n = len(names)
    rows, clipped, G = [], [], []
    for j, k in enumerate(cols):
        opt = [f"title={{{titles[k]}}}", f"xmin=0, xmax={f(R * 1.03)}", f"ymin=-0.6, ymax={n - 0.4}", "y dir=reverse",
               f"ytick={{{','.join(str(i) for i in range(n))}}}", "xlabel={miss risk}", "xmajorgrids",
               "grid style={black!12, line width=0.3pt}", "ytick style={draw=none}"]
        if j == 0:
            opt.append(f"yticklabels={{{','.join('{' + tex(FS.label(m)) + '}' for m in names)}}}")
        else:
            opt.append("yticklabels={}")
        body = [rf"\nextgroupplot[{', '.join(opt)}]",
                rf"\draw[dashed, line width=0.5pt] (axis cs:{f(alpha)},-0.6) -- (axis cs:{f(alpha)},{n - 0.4});"]
        for i, m in enumerate(names):
            r = idx.get((m, k))
            if r is None or not np.isfinite(r["value"]):
                continue
            v, c = _clip(r["value"], R)
            lo, hi = r["lo"], r["hi"]
            eb = ""
            if np.isfinite(lo) and np.isfinite(hi):
                eb = f" += ({f(max(min(hi, R) - v, 0))},0) -= ({f(max(v - min(lo, R), 0))},0)"
            body.append(rf"\addplot[only marks, {FS.mark_style(m, r['cert'], c)}, error bars/.cd, x dir=both,"
                        rf" x explicit, error bar style={{{FS.colour(m)}, line width=0.6pt}}, error mark=none]"
                        rf" coordinates {{({f(v)},{i}){eb}}};")
            cl = c or (np.isfinite(hi) and hi > R)
            if cl:
                clipped.append(f"{m}:{k}")
            rows.append({"method": m, "risk": k, "value": r["value"], "lo": lo, "hi": hi, "certified": r["cert"],
                         "clipped": cl})
        G.append("\n".join(body))
    ci = "trial 2.5--97.5\\% range (dev splits)" if dev else \
        f"{int(man.get('bootstrap_reps') or 0)}-replicate unit (cluster) bootstrap 95\\% CI, sealed test"
    T = (r"\begin{tikzpicture}[" + AXIS + "]\n"
         rf"\begin{{groupplot}}[group style={{group name=group, group size={len(cols)} by 1, horizontal sep=0.45cm}},"
         rf" width={0.78 / len(cols):.3f}\textwidth, height={0.42 * n + 1.2:.2f}cm,"
         r" title style={font=\footnotesize}]" + "\n" + "\n".join(G) + "\n\\end{groupplot}\n"
         rf"\node[anchor=north, font=\scriptsize] at ($(group c{(len(cols) + 1) // 2}r1.south)+(0,-0.75cm)$)"
         rf" {{{ci}; dashed: $\alpha$; filled = certified, hollow = not certified / fixed baseline}};" + "\n"
         + _stamp(dev) + r"\end{tikzpicture}")
    name = "fig4_forest" + ("_dev" if dev else "")
    man["clipped"][name] = clipped
    FS.save(out, name, T, spec, rows, ["method", "risk", "value", "lo", "hi", "certified", "clipped"], man)


# ------------------------------------------------------------------ fig 5: edge vs cloud cost
def fig5_cost(Pts, out, spec, man, dev):
    Pm = {p["method"]: p for p in Pts}
    names = [m for m in (spec.get("cost_methods") or spec["method_order"]) if m in Pm]
    n = len(names)
    panels = [("calls", r"cloud calls (\%)", 100.0, spec["calls_max"] * 100),
              ("uplink_kb", "uplink (KB / frame)", 1.0, spec["uplink_kb_max"]),
              ("latency_ms", "latency (ms, server)", 1.0, spec["latency_ms_max"])]
    rows, clipped, G = [], [], []
    for j, (key, xl, mult, xmax) in enumerate(panels):
        opt = [f"xmin=0, xmax={f(xmax * 1.22)}", f"ymin=-0.6, ymax={n - 0.4}", "y dir=reverse",
               f"ytick={{{','.join(str(i) for i in range(n))}}}", f"xlabel={{{xl}}}", "xbar, bar width=6pt", "ytick style={draw=none}",
               "scaled x ticks=false"]
        opt.append(f"yticklabels={{{','.join('{' + tex(FS.label(m)) + '}' for m in names)}}}" if j == 0
                   else "yticklabels={}")
        body = [rf"\nextgroupplot[{', '.join(opt)}]"]
        for i, m in enumerate(names):
            v = Pm[m][key] * mult
            pv, c = _clip(v, xmax)
            if not np.isfinite(v):
                continue
            pat = ", postaction={pattern=north east lines}" if c else ""
            body.append(rf"\addplot[fill={FS.colour(m)}, draw=none, bar shift=0pt{pat}] coordinates {{({f(pv, 3)},{i})}};")
            body.append(rf"\node[anchor=west, font=\tiny] at (axis cs:{f(pv, 3)},{i}) {{{v:.1f}{'+' if c else ''}}};")
            if c:
                clipped.append(f"{m}:{key}")
            rows.append({"method": m, "metric": key, "value": v, "clipped": c})
        G.append("\n".join(body))
    T = (r"\begin{tikzpicture}[" + AXIS + "]\n"
         r"\begin{groupplot}[group style={group name=group, group size=3 by 1, horizontal sep=0.45cm},"
         rf" width=0.30\textwidth, height={0.40 * n + 1.1:.2f}cm]" + "\n" + "\n".join(G) + "\n\\end{groupplot}\n"
         + _stamp(dev) + r"\end{tikzpicture}")
    name = "fig5_edge_cloud_cost" + ("_dev" if dev else "")
    man["clipped"][name] = clipped
    FS.save(out, name, T, spec, rows, ["method", "metric", "value", "clipped"], man)


# ------------------------------------------------------------------ fig 6: by source (+ external shift), TikZ heat map
def fig6_by_source(cells, out, spec, man, alpha, dev):
    names = [m for m in (spec.get("source_methods") or spec["method_order"]) if any(c["method"] == m for c in cells)]
    cols = list(dict.fromkeys(c["subset"] for c in cells))
    if not names or not cols:
        return
    R = spec["risk_max"]
    val = {(c["method"], c["subset"]): c["value"] for c in cells}
    ext = [c_ for c_ in cols if any(x["subset"] == c_ and x["kind"] == "external" for x in cells)]
    cw, ch = 1.45, 0.52
    L = [r"\begin{tikzpicture}[font=\footnotesize]"]
    for i, m in enumerate(names):
        L.append(rf"\node[anchor=east] at (-0.1,{-i * ch:.3f}) {{{tex(FS.label(m))}}};")
        for j, c_ in enumerate(cols):
            v = val.get((m, c_))
            if v is None or not np.isfinite(v):
                continue
            hx = FS.cividis(min(v, R), R)
            dark = int(hx[:2], 16) * 0.3 + int(hx[2:4], 16) * 0.59 + int(hx[4:], 16) * 0.11 < 110
            bold = r"\bfseries" if (c_ not in ext and v > alpha) else ""
            L.append(rf"\fill[fill={{rgb,255:red,{int(hx[:2], 16)};green,{int(hx[2:4], 16)};blue,{int(hx[4:], 16)}}}]"
                     rf" ({j * cw:.3f},{-i * ch - ch / 2:.3f}) rectangle ++({cw:.3f},{ch:.3f});")
            L.append(rf"\node[font=\scriptsize{bold}, text={'white' if dark else 'black'}] at "
                     rf"({j * cw + cw / 2:.3f},{-i * ch:.3f}) {{{v:.3f}{'+' if v > R else ''}}};")
    for j, c_ in enumerate(cols):
        L.append(rf"\node[anchor=north east, rotate=35, font=\scriptsize] at ({j * cw + cw / 2:.3f},"
                 rf"{-(len(names) - 1) * ch - ch / 2 - 0.05:.3f}) {{{_src(c_)}}};")
    top = ch / 2 + 0.08
    if ext and len(ext) < len(cols):
        j0 = min(cols.index(e) for e in ext)
        L.append(rf"\draw[white, line width=2pt] ({j0 * cw:.3f},{top:.3f}) -- ({j0 * cw:.3f},"
                 rf"{-(len(names) - 1) * ch - ch / 2:.3f});")
        L.append(rf"\node[anchor=south, font=\scriptsize] at ({j0 * cw / 2:.3f},{top:.3f}) "
                 rf"{{event miss risk ({'dev' if dev else 'pooled guarantee'})}};")
        L.append(rf"\node[anchor=south, font=\scriptsize] at ({(j0 + len(cols)) * cw / 2:.3f},{top:.3f}) "
                 r"{external shift: frame miss, no guarantee};")
    else:
        L.append(rf"\node[anchor=south, font=\scriptsize] at ({len(cols) * cw / 2:.3f},{top:.3f}) "
                 rf"{{event miss risk ({'dev' if dev else 'pooled guarantee'})}};")
    # colour bar
    x0 = len(cols) * cw + 0.35
    hgt = len(names) * ch
    nb = 40
    for k in range(nb):
        hx = FS.cividis((k + 0.5) / nb * R, R)
        y0 = -(len(names) - 1) * ch - ch / 2 + k * hgt / nb
        L.append(rf"\fill[fill={{rgb,255:red,{int(hx[:2], 16)};green,{int(hx[2:4], 16)};blue,{int(hx[4:], 16)}}}]"
                 rf" ({x0:.3f},{y0:.4f}) rectangle ++(0.25,{hgt / nb + 0.002:.4f});")
    yb = -(len(names) - 1) * ch - ch / 2
    for t_ in (0, R / 2, R):
        L.append(rf"\node[anchor=west, font=\tiny] at ({x0 + 0.27:.3f},{yb + t_ / R * hgt:.3f}) {{{t_:.2f}}};")
    L.append(rf"\node[rotate=90, anchor=south, font=\scriptsize] at ({x0 + 1.25:.3f},{yb + hgt / 2:.3f})"
             r" {miss risk (bold $>\alpha$)};")
    L.append(_stamp(dev) + r"\end{tikzpicture}")
    name = "fig6_by_source" + ("_dev" if dev else "")
    rows = [dict(c, clipped=c["value"] > R) for c in cells if c["method"] in names]
    man["clipped"][name] = [f"{c['method']}:{c['subset']}" for c in rows if c["clipped"]]
    FS.save(out, name, "\n".join(L), spec, rows, ["method", "subset", "kind", "value", "n", "clipped"], man)


# ------------------------------------------------------------------ fig 7: dev head selection
def fig7_heads(Hj, out, spec, man):
    H = Hj["heads"]
    sel = Hj["selected"]
    labs = [("NMS-free (1-to-1)" if str(h["nms"]) == "false" else "NMS (1-to-many)") +
            (r"$^{\star}$" if h["name"] == sel else "") for h in H]
    xt = f"xtick={{{','.join(str(i) for i in range(len(H)))}}}, xticklabels={{{','.join('{' + s + '}' for s in labs)}}}"
    common = (f"xmin=-0.6, xmax={len(H) - 0.4}, {xt}, x tick label style={{rotate=15, anchor=north east,"
              r" font=\tiny}, ybar, bar width=9pt")
    G = []
    lo = min(min(h["fire"], h["smoke"]) for h in H)
    G.append(rf"\nextgroupplot[title={{(a) candidate recall}}, {common}, bar width=7pt, ymin={f(max(0, lo - 0.05), 3)},"
             r" ymax=1.0, ylabel={event recall at $s_{\min}$}]" + "\n" +
             r"\addplot[fill=oiVerm, draw=none] coordinates {" +
             " ".join(f"({i},{f(h['fire'], 4)})" for i, h in enumerate(H)) + "};\n" +
             r"\addplot[fill=oiSky, draw=none] coordinates {" +
             " ".join(f"({i},{f(h['smoke'], 4)})" for i, h in enumerate(H)) + "};")
    for key, yl, t in (("neg_esc", "share of negative frames", "(b) negative escalation"),
                       ("map50", "dev mAP50", "(c) dev mAP50"), ("det_ms", "ms / frame (server)",
                                                                  "(d) detector latency")):
        v = [_num(h[key]) for h in H]
        if not any(np.isfinite(x) for x in v):
            G.append(rf"\nextgroupplot[title={{{t}}}, {common}, ymin=0, ymax=1, ytick=\empty]" + "\n" +
                     r"\node[font=\scriptsize, text=black!60, align=center] at (axis cs:" +
                     f"{(len(H) - 1) / 2},0.5)" + r" {not computed\\(run with weights)};")
            continue
        fin = [x for x in v if np.isfinite(x)]
        G.append(rf"\nextgroupplot[title={{{t}}}, {common}, ymin=0, ymax={f(max(fin) * 1.2, 4)}, ylabel={{{yl}}},"
                 r" nodes near coords, nodes near coords style={font=\tiny, /pgf/number format/.cd, fixed, "
                 rf"precision={1 if key == 'det_ms' else 3}}}]" + "\n" +
                 r"\addplot[fill=oiBlue, draw=none] coordinates {" +
                 " ".join(f"({i},{f(x, 4)})" for i, x in enumerate(v) if np.isfinite(x)) + "};")
    rows = [{"head": h["name"], "nms": h["nms"], "fire_event_recall": h["fire"], "smoke_event_recall": h["smoke"],
             "neg_escalation": h["neg_esc"], "map50": h["map50"], "det_ms": h["det_ms"], "selected": h["name"] == sel}
            for h in H]
    T = (r"\begin{tikzpicture}[" + AXIS + "]\n"
         r"\begin{groupplot}[group style={group name=group, group size=4 by 1, horizontal sep=1.35cm}, width=0.25\textwidth,"
         r" height=3.9cm, title style={font=\footnotesize}, scaled y ticks=false]" + "\n" + "\n".join(G) +
         "\n\\end{groupplot}\n"
         rf"\node[anchor=north, font=\scriptsize] at ($(group c2r1.south)!0.5!(group c3r1.south)+(0,-0.9cm)$)"
         rf" {{dev (val role), $s_{{\min}}={Hj.get('s_min')}$; (a) orange = fire, blue = smoke;"
         r" $\star$ = selected by the pre-registered head rule};"
         + "\n" + r"\end{tikzpicture}")
    FS.save(out, "fig7_head_selection", T, spec, rows,
            ["head", "nms", "fire_event_recall", "smoke_event_recall", "neg_escalation", "map50", "det_ms",
             "selected"], man)


# ------------------------------------------------------------------ assembling inputs
def _manifest(kind, spec_src, extra):
    from cascade.make_protocol import analysis_code
    return {"kind": kind, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "analysis_code_sha256_now": analysis_code()[0], "spec_source": spec_src, "inputs": {}, "clipped": {},
            "figures": {}, **extra}


def dev_points(S):
    return [{"method": m, "risk": v["risk_mean"], "lo": v["risk_p2.5"], "hi": v["risk_p97.5"],
             "cert": v["certification_rate"] >= 0.5, "cert_rate": v["certification_rate"],
             "calls": v.get("calls_mean", np.nan), "uplink_kb": v.get("bytes_per_frame_mean", np.nan) / 1024,
             "latency_ms": v.get("e2e_ms_mean", np.nan)} for m, v in S.items()]


def dev_forest(S, trials_csv, risks, target):
    with open(trials_csv, newline="") as f:
        T = list(csv.DictReader(f))
    F = []
    for m, v in S.items():
        R = [r for r in T if r["method"] == m]
        cert = v["certification_rate"] >= 0.5
        F.append({"method": m, "risk": "max", "value": v["risk_mean"], "lo": v["risk_p2.5"], "hi": v["risk_p97.5"],
                  "cert": cert})
        for k in risks or []:
            col = f"{k}:{RISK_KEY[target]}"
            vals = np.array([_num(r.get(col)) for r in R])
            vals = vals[np.isfinite(vals)]
            if len(vals):
                F.append({"method": m, "risk": k, "value": float(vals.mean()), "lo": float(np.quantile(vals, 0.025)),
                          "hi": float(np.quantile(vals, 0.975)), "cert": cert})
    return F


def final_points(J):
    P, F = [], []
    target, risks = J["config"]["target"], J["config"].get("risks")
    for m, v in J["results"].items():
        r = v["sealed_test"]
        cert = bool(r.get("certified"))
        ci = r.get("risk_ci95") or [np.nan, np.nan]
        P.append({"method": m, "risk": r["risk"], "lo": ci[0], "hi": ci[1], "cert": cert, "calls": r["calls"],
                  "uplink_kb": r["bytes_per_frame"] / 1024, "latency_ms": _num(r.get("e2e_ms"))})
        F.append({"method": m, "risk": "max", "value": r["risk"], "lo": ci[0], "hi": ci[1], "cert": cert})
        for k in risks or []:
            key = f"{k}:{RISK_KEY[target]}"
            if key in r:
                c2 = r.get(f"{k}:ci95") or [np.nan, np.nan]
                F.append({"method": m, "risk": k, "value": r[key], "lo": c2[0], "hi": c2[1], "cert": cert})
    return P, F


def run_dev(a):
    from cascade.make_protocol import load_protocol
    P, _ = load_protocol(a.protocol)
    spec, src = FS.spec_from(P)
    out = Path(a.out).expanduser()
    dev_dir = Path(a.dev_dir).expanduser()
    man = _manifest("dev", src, {"role": "dev", "stamp": DEV_STAMP})
    fig1_architecture(out, spec, man)
    if a.dataset_audit and Path(a.dataset_audit).expanduser().exists():
        A = json.loads(Path(a.dataset_audit).expanduser().read_text())
        man["inputs"]["dataset_audit.json"] = FS.sha256(Path(a.dataset_audit).expanduser())
        fig2_dataset_audit(A, out, spec, man)
    if a.head_json and Path(a.head_json).expanduser().exists():
        Hj = json.loads(Path(a.head_json).expanduser().read_text())
        if Hj.get("role") != "dev":
            raise SystemExit("head json is not a dev (val role) report")
        man["inputs"]["head_report.json"] = FS.sha256(Path(a.head_json).expanduser())
        fig7_heads(Hj, out, spec, man)
    sj = dev_dir / "dev_summary.json"            # dev_* ONLY: validity_* and final/* are never opened
    if sj.exists():
        S = json.loads(sj.read_text())
        tr = dev_dir / "dev_trials.csv"
        for p_ in (sj, tr, dev_dir / "dev_by_source.json"):
            if p_.exists():
                man["inputs"][p_.name] = FS.sha256(p_)
        pts = dev_points(S)
        fig3_risk_cost(pts, out, spec, man, float(P["alpha"]), dev=True)
        fig4_forest(dev_forest(S, tr, P.get("risks"), P["target"]), out, spec, man, float(P["alpha"]),
                    P.get("risks"), dev=True)
        fig5_cost(pts, out, spec, man, dev=True)
        bs = dev_dir / "dev_by_source.json"
        if bs.exists():
            cells = [{"method": r["method"], "subset": r["source"], "kind": "pooled", "value": r["risk_mean"],
                      "n": r["trials"]} for r in json.loads(bs.read_text())]
            fig6_by_source(cells, out, spec, man, float(P["alpha"]), dev=True)
    else:
        print(f"note: {sj} not found (run STAGE=dev first); only fig1 / fig2 / fig7")
    FS.write_manifest(out, man)
    print(f"figures (dev) -> {out}: {sorted(man['figures'])}")


def run_final(a):
    from cascade.make_protocol import (ANALYSIS_CODE, FIGURE_CODE, analysis_code, lock_digest, locked_files,
                                       read_lock, read_receipt, sha256_file)
    R = read_receipt(a.protocol)
    if not R or R.get("state") != "done":
        raise SystemExit("figures final refused: the protocol receipt is not 'done' (no finished final)")
    od = Path(R["out_dir"])
    fj, fm = od / "final.json", od / "final.md"
    if not R.get("final_json_sha256"):
        raise SystemExit("figures final refused: the receipt has no final_json_sha256 (older version)")
    if sha256_file(fj) != R["final_json_sha256"] or sha256_file(fm) != R["final_md_sha256"]:
        raise SystemExit("figures final refused: final.json / final.md differ from the hashes in the receipt")
    L = read_lock(a.protocol)
    try:
        data_ok = L["sha256"] == lock_digest(Path(a.protocol).expanduser(), locked_files(L),
                                            L["analysis_code_sha256"])
    except (FileNotFoundError, KeyError):
        data_ok = False
    if not data_ok:
        raise SystemExit("figures final refused: protocol.yaml / splits.csv / locked score files changed after "
                         "freeze")
    now_sha, now_files = analysis_code()
    changed = [n for n in ANALYSIS_CODE if now_files.get(n) != (L.get("analysis_code_files") or {}).get(n)]
    fix = {}
    if changed:
        if any(n not in FIGURE_CODE for n in changed):
            raise SystemExit(f"figures final refused: analysis code changed after freeze: {changed}")
        if not a.post_freeze_figure_fix:
            raise SystemExit(f"figures final refused: figure code changed after freeze {changed}. If this is a "
                             "figure-only fix, rerun with --post-freeze-figure-fix (recorded in the manifest and "
                             "to be disclosed); final is NOT re-run.")
        fix = {n: {"locked": L["analysis_code_files"].get(n), "now": now_files[n]} for n in changed}
    P = __import__("yaml").safe_load((Path(a.protocol).expanduser() / "protocol.yaml").read_text())
    spec, src = FS.spec_from(P)
    J = json.loads(fj.read_text())
    out = Path(a.out).expanduser() if a.out else od / "figures"
    man = _manifest("final", src, {"role": "sealed_test (final.json only)", "lock_sha256": L["sha256"],
                                   "receipt_records_sha256": R["records_sha256"],
                                   "bootstrap_reps": J["config"].get("bootstrap_reps"),
                                   "post_freeze_figure_code_change": fix})
    man["inputs"] = {"final.json": R["final_json_sha256"], "final.md": R["final_md_sha256"]}
    alpha = float(J["config"]["alpha"])
    pts, F = final_points(J)
    fig3_risk_cost(pts, out, spec, man, alpha, dev=False)
    fig4_forest(F, out, spec, man, alpha, J["config"].get("risks"), dev=False)
    fig5_cost(pts, out, spec, man, dev=False)
    cells = [{"method": r["method"], "subset": r["source"], "kind": "sealed", "value": r["risk"], "n": ""}
             for r in J.get("by_source", [])]
    cells += [{"method": r["method"], "subset": r["subset"], "kind": "external", "value": r["miss_img"],
               "n": r["n_pos"]} for r in J.get("external_shift", []) if r["subset"] != "all"]
    fig6_by_source(cells, out, spec, man, alpha, dev=False)
    FS.write_manifest(out, man)
    print(f"figures (final, from final.json only) -> {out}: {sorted(man['figures'])}")
    if fix:
        print(f"NOTE: post-freeze figure-code change recorded in figures_manifest.json: {sorted(fix)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dev")
    p.add_argument("--protocol", required=True)
    p.add_argument("--dev-dir", required=True, help="run_cascade --mode dev --out-dir (reads dev_* files only)")
    p.add_argument("--head-json", default=None)
    p.add_argument("--dataset-audit", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("final")
    p.add_argument("--protocol", required=True)
    p.add_argument("--out", default=None, help="default: <final out_dir>/figures")
    p.add_argument("--post-freeze-figure-fix", action="store_true")
    a = ap.parse_args()
    {"dev": run_dev, "final": run_final}[a.cmd](a)


if __name__ == "__main__":
    main()
