"""v2.9-A/B dev probe: is there unused signal, and could any of it ever be certified? (dev labels only)

    python -m cascade.dev_probe --dataset DS --protocol P --det D --agents overlay=A1[,crop=A2,7b=A3] --out DIR

No GPU, no new inference: it only re-reads existing score CSVs. calibration / sealed labels are never read.
This is a DIAGNOSTIC: nothing here certifies anything and nothing here changes the ESVA NO-GO.

E  informativeness of the VLM score g on escalatable frames (s >= s_min), per agent x source x (S | not S):
   AUC, FPR at TPR 0.95 and 0.99, and the partial area over the high-sensitivity band TPR in [0.95, 1],
   reported as mean specificity in that band (1.0 = perfect, about 0.025 for a chance-level ranker).
   Frame-level ranking only - it does NOT replace event-level safety (see F).
F  the real policy's front: for every veto discount lambda, calibrate the cascade exactly as run_cascade does
   (LTT, Pareto testing) on one half of the dev units with g_lambda = min(1, g + lambda) inside S, and measure
   event risk / stratum risk / FA / calls on the other half. lambda = 0 is cascade_LTT; edge_LTT is reported too.
G  certifiability by SIMULATING THE ACTUAL SPLIT (not one hypergeometric draw): the dev 50/50 unit split and
   calibrate()'s own 30/70 select/certify split are replayed; the distribution of certify-fold positive units
   per constrained risk gives P(units >= n_min) and the K needed for 90%. The calibration-role counts cannot be
   computed before unsealing (labels are sealed), so only dev-side numbers are reported.
H  causal (past-only) temporal aggregation on time-stamped camera sequences: alarm when k of the last n frames
   are above an edge threshold. Event miss, false alarms and detection delay vs the single-frame rule. Pyro-SDIS
   has 2 dev units, so this is descriptive, not a generalizable claim.

Scope of the statistical statement (protocol wording): the LTT calibration model assumes i.i.d. calibration
units; exchangeability alone is not enough (perfectly dependent losses are exchangeable and carry no extra
information as n grows). The unit definition and the dependence assumption must be argued in the paper, not
swapped for a weaker word.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.dataio import pyro_camera_time  # noqa: E402
from cascade.esva_dev import eval_metrics, load_dev, split_units, take  # noqa: E402
from cascade.evidence_veto import LAMBDAS, StratumSpec, discount, in_stratum, n_min  # noqa: E402
from cascade.select import calibrate  # noqa: E402

TPR_BAND = (0.95, 1.0)          # high-sensitivity band for the partial AUC
KN = ((1, 1), (2, 3), (2, 5), (3, 5))


# ---------------------------------------------------------------- E: ranking diagnostics
def roc_stats(y, g, band=TPR_BAND):
    """AUC, FPR at TPR 0.95 / 0.99 and normalized partial AUC over the TPR band. NaN if a class is missing."""
    y = np.asarray(y).astype(bool)
    g = np.asarray(g, float)
    m = np.isfinite(g)
    y, g = y[m], g[m]
    out = {"n_pos": int(y.sum()), "n_neg": int((~y).sum())}
    if not y.any() or not (~y).any():
        return {**out, "auc": float("nan"), "fpr@tpr95": float("nan"), "fpr@tpr99": float("nan"),
                "pauc_spec_band": float("nan")}
    order = np.argsort(-g, kind="stable")
    ys, gs = y[order], g[order]
    # tie-aware ROC: one point per distinct score (ties share a threshold), AUC by the trapezoid over it
    last = np.r_[np.flatnonzero(np.diff(gs)), len(gs) - 1]
    tpr = np.r_[0.0, np.cumsum(ys)[last] / max(1, ys.sum())]
    fpr = np.r_[0.0, np.cumsum(~ys)[last] / max(1, (~ys).sum())]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    out["auc"] = float(trap(tpr, fpr))
    for q in (0.95, 0.99):
        i = int(np.searchsorted(tpr, q))
        out[f"fpr@tpr{int(q * 100)}"] = float(fpr[min(i, len(fpr) - 1)])
    lo, hi = band
    grid = np.linspace(lo, hi, 201)                       # mean specificity while sensitivity is in the band:
    out["pauc_spec_band"] = float(1.0 - np.trapezoid(np.interp(grid, tpr, fpr), grid) / (hi - lo)) \
        if hasattr(np, "trapezoid") else float(1.0 - np.trapz(np.interp(grid, tpr, fpr), grid) / (hi - lo))
    # 1.0 = perfect; a chance-level ranker gives about 1 - (lo + hi) / 2 (= 0.025 for the band 0.95-1.0)
    return out


def informativeness(D, G, inS, s_min):
    rows = []
    esc = D["s"] >= s_min
    for label in ("y", "y_smoke"):
        for src in ["ALL", *sorted(set(D["source"].tolist()))]:
            m0 = esc if src == "ALL" else esc & (D["source"] == src)
            for part, m in (("S", m0 & inS), ("notS", m0 & ~inS), ("all", m0)):
                for ag, g in G.items():
                    r = roc_stats(D[label][m] > 0, g[m])
                    rows.append({"agent": ag, "label": label, "source": src, "stratum": part, **r})
    return rows


# ---------------------------------------------------------------- I: acquisition comparison (unit bootstrap)
def auc_delta(D, G, inS, s_min, reference, reps=1000, seed=3, level=0.90):
    """AUC and FPR@TPR95 differences vs the reference agent, with a UNIT-level (cluster) bootstrap.

    Frames inside one camera / dedup group are not independent: in S most positives come from 2 Pyro units, so
    frame-level intervals would be fake precision. Units are resampled with replacement; a source whose
    interval is degenerate (1-2 units) is reported with its unit count so the reader sees why."""
    if reference not in G:
        return []
    esc = D["s"] >= s_min
    rng = np.random.default_rng(seed)
    rows = []
    for src in ["ALL", *sorted(set(D["source"].tolist()))]:
        for part, msk in (("S", inS), ("notS", ~inS)):
            m = esc & msk & (np.ones(len(esc), bool) if src == "ALL" else D["source"] == src)
            if not m.any():
                continue
            y = D["y"][m] > 0
            units = D["unit"][m]
            uu, inv = np.unique(units, return_inverse=True)
            idx_by_u = [np.flatnonzero(inv == i) for i in range(len(uu))]
            base = {ag: roc_stats(y, g[m]) for ag, g in G.items()}
            draws = {ag: {"auc": [], "fpr": []} for ag in G}
            for _ in range(reps):
                pick = rng.integers(0, len(uu), len(uu))
                sel = np.concatenate([idx_by_u[i] for i in pick]) if len(uu) else np.array([], int)
                if len(sel) == 0:
                    continue
                ys = y[sel]
                for ag, g in G.items():
                    r = roc_stats(ys, g[m][sel])
                    draws[ag]["auc"].append(r["auc"])
                    draws[ag]["fpr"].append(r["fpr@tpr95"])
            lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
            for ag in G:
                if ag == reference:
                    continue
                d_auc = np.asarray(draws[ag]["auc"], float) - np.asarray(draws[reference]["auc"], float)
                d_fpr = np.asarray(draws[ag]["fpr"], float) - np.asarray(draws[reference]["fpr"], float)
                d_auc, d_fpr = d_auc[np.isfinite(d_auc)], d_fpr[np.isfinite(d_fpr)]
                rows.append({"agent": ag, "reference": reference, "source": src, "stratum": part,
                             "n_units": int(len(uu)), "n_pos": base[ag]["n_pos"], "n_neg": base[ag]["n_neg"],
                             "auc": base[ag]["auc"], "auc_ref": base[reference]["auc"],
                             "d_auc": float(base[ag]["auc"] - base[reference]["auc"]),
                             "d_auc_lo": float(np.quantile(d_auc, lo)) if len(d_auc) else float("nan"),
                             "d_auc_hi": float(np.quantile(d_auc, hi)) if len(d_auc) else float("nan"),
                             "d_fpr95": float(base[ag]["fpr@tpr95"] - base[reference]["fpr@tpr95"]),
                             "d_fpr95_lo": float(np.quantile(d_fpr, lo)) if len(d_fpr) else float("nan"),
                             "d_fpr95_hi": float(np.quantile(d_fpr, hi)) if len(d_fpr) else float("nan")})
    return rows


# ---------------------------------------------------------------- F: the real policy's lambda front
def lambda_front(D, P, G, inS, lambdas, trials, seed, log=print):
    alpha, delta, target = float(P["alpha"]), float(P["delta"]), P["target"]
    costs = {k: float(v) for k, v in P["costs"].items() if k in ("calls", "handoff", "fa")}
    kw = dict(level="unit", opt_frac=float(P.get("select_fraction", 0.3)), s_min=float(P["s_min"]))
    rng = np.random.default_rng(seed)
    acc = defaultdict(list)
    for t in range(trials):
        cm, tm = split_units(D["unit"], 0.5, rng)
        cal, test = take(D, cm), take(D, tm)
        ev = cal["event"] if target == "event" else None
        rl = {"fire": cal["y_fire"], "smoke": cal["y_smoke"]}
        pyro_t = test["source"] == "pyro_sdis"
        def run(ag, lam_label, gc, gt, cloud):
            theta, info = calibrate(cal["s"], gc, cal["y"], cal["unit"], alpha, delta, None, costs,
                                    cloud=cloud, handoff=False, fwer="pareto", seed=seed + t,
                                    event=ev, risk_labels=rl, **kw)
            r = eval_metrics(test, gt, theta, target, inS[tm])
            key = (ag, lam_label)
            for k in ("risk", "risk_S", "fa", "calls"):
                acc[key + (k,)].append(r[k])
            acc[key + ("certified",)].append(float(bool(info.get("certified"))))
            if (test["y"][pyro_t] > 0).any():
                rp = eval_metrics(take(test, pyro_t), gt[pyro_t], theta, target, inS[tm][pyro_t])
                acc[key + ("pyro_risk",)].append(rp["risk"])
                acc[key + ("pyro_fa",)].append(rp["fa"])

        run("-", "edge_LTT", cal["g"], test["g"], False)        # no cloud stage: agent-independent
        for ag, g in G.items():
            for lam in lambdas:
                run(ag, f"{lam:.1f}", discount(g[cm], inS[cm], lam), discount(g[tm], inS[tm], lam), True)
        if (t + 1) % max(1, trials // 5) == 0:
            log(f"lambda front trial {t + 1}/{trials}")

    def m(k):
        v = np.asarray(acc.get(k, []), float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")
    keys = sorted({k[:2] for k in acc})
    return [{"agent": a, "lambda": l, **{f: m((a, l, f)) for f in ("risk", "risk_S", "fa", "calls", "certified",
                                                                   "pyro_risk", "pyro_fa")}} for a, l in keys]


# ---------------------------------------------------------------- G: certifiability under the actual split
def certify_counts(D, inS, trials, seed, alpha, delta):
    """Replay the dev 50/50 split and calibrate()'s own 30/70 select/certify split; count positive units per
    constrained risk in the certify fold."""
    nm = n_min(alpha, delta)
    risks = {"fire": D["y_fire"] > 0, "smoke": D["y_smoke"] > 0,
             "fire|S": (D["y_fire"] > 0) & inS, "smoke|S": (D["y_smoke"] > 0) & inS}
    rng = np.random.default_rng(seed)
    got = defaultdict(list)
    for t in range(trials):
        cm, _ = split_units(D["unit"], 0.5, rng)
        u = np.unique(D["unit"][cm])
        r2 = np.random.default_rng(seed + t)                       # as select.calibrate: opt_frac select units
        sel = set(r2.choice(u, max(1, int(round(0.3 * len(u)))), replace=False))
        cert_mask = cm & np.array([x not in sel for x in D["unit"]])
        for k, v in risks.items():
            got[k].append(len(np.unique(D["unit"][v & cert_mask])))
    out = {"n_min": nm, "dev_units": int(len(np.unique(D["unit"]))), "trials": trials, "risks": {}}
    for k, v in got.items():
        v = np.asarray(v)
        out["risks"][k] = {"dev_positive_units": int(len(np.unique(D["unit"][risks[k]]))),
                           "certify_units_mean": float(v.mean()), "certify_units_p05": float(np.quantile(v, .05)),
                           "p_ge_n_min": float((v >= nm).mean())}
    return out


def required_K(D, inS, alpha, delta, seed=0, trials=400, conf=0.90):
    """Smallest number of positive dev units K with P(certify-fold units >= n_min) >= conf, using the same
    two-stage split on the real unit list (not a single hypergeometric draw)."""
    nm = n_min(alpha, delta)
    units = np.unique(D["unit"])
    rng = np.random.default_rng(seed)
    lo, hi = nm, min(len(units), 20 * nm)
    while lo < hi:
        mid = (lo + hi) // 2
        ok = 0
        for t in range(trials):
            pos = set(rng.choice(units, mid, replace=False))
            m = np.array([u in pos for u in D["unit"]])
            cm, _ = split_units(D["unit"], 0.5, rng)
            u = np.unique(D["unit"][cm])
            sel = set(rng.choice(u, max(1, int(round(0.3 * len(u)))), replace=False))
            cert = cm & np.array([x not in sel for x in D["unit"]])
            ok += len(np.unique(D["unit"][m & cert])) >= nm
        if ok / trials >= conf:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


# ---------------------------------------------------------------- H: causal temporal aggregation
def temporal(D, s_edges, source="pyro_sdis"):
    """k-of-n over the last n frames of the same camera (past only). Event miss / FA / detection delay."""
    idx = np.flatnonzero(D["source"] == source)
    seq = defaultdict(list)
    for i in idx:
        ct = pyro_camera_time(str(D["stem"][i]))
        if ct is None:
            continue
        seq[(D["unit"][i], D["event"][i], ct[0])].append((ct[1], i))
    rows = []
    for k, n in KN:
        for t in s_edges:
            miss = tot = fa = neg = 0
            delays = []
            for key, lst in seq.items():
                lst.sort()
                ii = np.array([i for _, i in lst])
                ts = [x for x, _ in lst]
                hot = (D["s"][ii] >= t).astype(int)
                run = np.array([hot[max(0, j - n + 1):j + 1].sum() >= k for j in range(len(ii))])
                pos = D["y"][ii] > 0
                if pos.any():
                    tot += 1
                    fired = np.flatnonzero(run & pos)
                    if len(fired) == 0:
                        miss += 1
                    else:
                        first_pos = int(np.flatnonzero(pos)[0])
                        delays.append((ts[int(fired[0])] - ts[first_pos]).total_seconds())
                neg += int((~pos).sum())
                fa += int((run & ~pos).sum())
            rows.append({"k": k, "n": n, "s_thr": float(t), "events": tot, "event_miss": miss,
                         "event_miss_rate": miss / max(1, tot), "neg_frames": neg, "fa_frames": fa,
                         "fa_rate": fa / max(1, neg),
                         "delay_s_median": float(np.median(delays)) if delays else float("nan"),
                         "delay_s_p90": float(np.quantile(delays, .9)) if delays else float("nan")})
    return rows


# ---------------------------------------------------------------- report
def _f(v, nd=4):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def markdown(J, P, st):
    L = ["# v2.9 dev probe (diagnostic only; the ESVA NO-GO is unchanged)", "",
         f"stratum S = {st.to_dict()}; alpha={P['alpha']}, delta={P['delta']}, target={P['target']}; "
         f"agents: {', '.join(J['agents'])}", "",
         "Frame-level ranking (E) does not replace event-level safety (F). Section G replays the actual split; "
         "calibration-role counts stay unknown until unsealing.", "",
         "## E. Is the VLM score informative? (escalatable frames, s >= s_min)", "",
         "| agent | label | source | stratum | n_pos | n_neg | AUC | FPR@TPR95 | FPR@TPR99 | pAUC_spec(0.95-1) |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in J["informativeness"]:
        if r["source"] in ("ALL", "pyro_sdis", "dfire", "fasdd_cv") and r["stratum"] != "all":
            L.append(f"| {r['agent']} | {r['label']} | {r['source']} | {r['stratum']} | {r['n_pos']} | "
                     f"{r['n_neg']} | {_f(r['auc'], 3)} | {_f(r['fpr@tpr95'], 3)} | {_f(r['fpr@tpr99'], 3)} | "
                     f"{_f(r['pauc_spec_band'], 3)} |")
    if J.get("auc_delta"):
        r0 = J["auc_delta"][0]
        L += ["", f"## I. Acquisition comparison vs `{r0['reference']}` (unit-level bootstrap, 90% interval)", "",
              "| agent | source | stratum | units | AUC | dAUC | dAUC 90% | dFPR@TPR95 | dFPR 90% |",
              "|---|---|---|---:|---:|---:|---|---:|---|"]
        for r in J["auc_delta"]:
            L.append(f"| {r['agent']} | {r['source']} | {r['stratum']} | {r['n_units']} | {_f(r['auc'], 3)} | "
                     f"{_f(r['d_auc'], 3)} | [{_f(r['d_auc_lo'], 3)}, {_f(r['d_auc_hi'], 3)}] | "
                     f"{_f(r['d_fpr95'], 3)} | [{_f(r['d_fpr95_lo'], 3)}, {_f(r['d_fpr95_hi'], 3)}] |")
        L += ["", "A source with 1-2 units cannot give a meaningful interval; its row is descriptive."]
    L += ["", "## F. Real certified policy vs veto discount lambda (dev halves)", "",
          "| agent | lambda | cert rate | risk | risk_S | FA | calls | pyro risk | pyro FA |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in J["lambda_front"]:
        L.append(f"| {r['agent']} | {r['lambda']} | {_f(r['certified'], 2)} | {_f(r['risk'])} | "
                 f"{_f(r['risk_S'])} | {_f(r['fa'])} | {_f(r['calls'])} | {_f(r['pyro_risk'])} | "
                 f"{_f(r['pyro_fa'])} |")
    C = J["certifiability"]
    L += ["", f"## G. Certifiability (n_min = {C['n_min']}, dev units = {C['dev_units']}, "
              f"{C['trials']} replays of the real split)", "",
          "| risk | dev positive units | certify units (mean) | p05 | P(>= n_min) |", "|---|---:|---:|---:|---:|"]
    for k, v in C["risks"].items():
        L.append(f"| {k} | {v['dev_positive_units']} | {v['certify_units_mean']:.1f} | "
                 f"{v['certify_units_p05']:.0f} | {v['p_ge_n_min']:.3f} |")
    L += ["", f"Positive dev units needed for P(certify units >= n_min) >= 0.90 under this split: "
              f"**{J['required_K']}**", ""]
    if J["temporal"]:
        L += ["## H. Causal k-of-n on time-stamped camera sequences (descriptive; 2 dev units)", "",
              "| k | n | s_thr | events | miss rate | FA rate (frames) | delay median s | delay p90 s |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in J["temporal"]:
            L.append(f"| {r['k']} | {r['n']} | {r['s_thr']:.2f} | {r['events']} | {_f(r['event_miss_rate'], 3)} | "
                     f"{_f(r['fa_rate'], 4)} | {_f(r['delay_s_median'], 0)} | {_f(r['delay_s_p90'], 0)} |")
    L += ["", "Scope: the LTT calibration model assumes i.i.d. calibration units; exchangeability alone does not "
              "give the same guarantee, so the unit definition and the dependence assumption are argued in the "
              "paper, not relabelled."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--dataset", "--protocol", "--det", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--agents", required=True, help="name=path[,name=path...] (agent CSVs that already exist)")
    ap.add_argument("--trials", type=int, default=20, help="trials for the lambda front (F)")
    ap.add_argument("--split-trials", type=int, default=500, help="split replays for G")
    ap.add_argument("--area-max", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--reference", default="", help="agent name used as the reference in section I")
    ap.add_argument("--boot", type=int, default=1000, help="unit bootstrap replicates for section I")
    a = ap.parse_args(argv)
    specs = [x.split("=", 1) for x in a.agents.split(",") if x]
    D, P = load_dev(a.dataset, a.protocol, a.det, specs[0][1])
    G = {specs[0][0]: D["g"]}
    for name, path in specs[1:]:
        Dx, _ = load_dev(a.dataset, a.protocol, a.det, path)
        if not np.array_equal(Dx["uid"], D["uid"]):
            raise SystemExit(f"agent {name} covers different frames than {specs[0][0]}: not comparable")
        G[name] = Dx["g"]
    st = StratumSpec("small_smoke", a.area_max)
    inS = in_stratum(st, D["features"])
    alpha, delta, s_min = float(P["alpha"]), float(P["delta"]), float(P["s_min"])
    print(f"dev frames {len(D['s'])}, units {len(np.unique(D['unit']))}, S frames {int(inS.sum())}, "
          f"agents {list(G)}", flush=True)
    J = {"agents": list(G), "stratum": st.to_dict(), "area_max": a.area_max,
         "informativeness": informativeness(D, G, inS, s_min),
         "certifiability": certify_counts(D, inS, a.split_trials, a.seed, alpha, delta),
         "required_K": required_K(D, inS, alpha, delta, seed=a.seed)}
    s_edges = np.quantile(D["s"][D["s"] >= s_min], [0.5, 0.75, 0.9, 0.97]) if (D["s"] >= s_min).any() else []
    J["temporal"] = temporal(D, np.round(s_edges, 4))
    J["auc_delta"] = auc_delta(D, G, inS, s_min, a.reference or list(G)[0], reps=a.boot, seed=a.seed)
    J["lambda_front"] = lambda_front(D, P, G, inS, LAMBDAS, a.trials, a.seed,
                                     log=lambda m: print(m, flush=True))
    o = Path(a.out).expanduser()
    o.mkdir(parents=True, exist_ok=True)
    md = markdown(J, P, st)
    (o / "dev_probe.md").write_text(md)
    (o / "dev_probe.json").write_text(json.dumps(J, indent=1, default=float))
    print(md)
    print("->", o / "dev_probe.json")


if __name__ == "__main__":
    main()
