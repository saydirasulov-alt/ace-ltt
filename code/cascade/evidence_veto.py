"""ESVA-LTT: Evidence-Stratified Veto Authority for the edge -> VLM -> human cascade.

Problem (measured on dev, v2.6.2 / v2.7): the pooled event-miss guarantee holds, but the VLM *vetoes* true
smoke that the edge detector would have alarmed on, and these misses concentrate where the smoke is small
(Pyro-SDIS dev: cascade 0.08-0.11 vs edge 0.02). SmokeBench (WACV 2026) reports the same mechanism for MLLMs:
Qwen2.5-VL-7B accuracy 0.10 on very small smoke vs 0.69 on very large. A pooled guarantee averages this away.

Method. A label-free evidence stratum S is declared BEFORE calibration from the edge detector alone:
    S = {frames whose top smoke box is small (area <= area_max) and smoke evidence >= fire evidence}.
Inside S the VLM's veto is discounted by a certified amount lambda:
    g_lambda = min(1, g + lambda) on S,   g unchanged outside S          (lambda = 0: full veto, 1: no veto)
and the configuration theta = (t_low, t_high, b, a, lambda) is certified jointly for
    * the pooled fire- and smoke-event miss risks (as cascade_LTT), AND
    * the smoke-event miss risk restricted to positives inside S (event caught if a positive S-frame alarms;
      the constrained stratum risks are fixed a priori, STRATUM_RISKS = ('smoke',), never chosen from data),
by one intersection-union test (p = max of the HB p-values) inside Pareto testing (weak front, ties tested
conservative-first) with fixed-sequence certification. Statement: with probability >= 1 - delta over the calibration draw, every risk of the
certified configuration is <= alpha, pooled AND in S, simultaneously. lambda, like the thresholds, is chosen
on the select fold only; the certify fold is used only for the p-values, so FWER <= delta is kept.

Nothing uses the source, the label or the dataset id at deployment: S and g_lambda need only the detector
boxes/scores and the VLM score.

Certifiability. A risk over n exchangeable units can be certified at (alpha, delta) only if
n >= n_min = ceil(log(delta) / log(1 - alpha)) (= 45 for 0.05 / 0.10), even with zero observed misses.
If S has fewer positive units in the certify fold, ESVA cannot certify and falls back like cascade_LTT;
certifiability() reports this before any claim is made.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .policy import Grid, decide
from .select import _finish, _grid_stats, _joint_p, _objective, candidate_mask

LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class StratumSpec:
    """Label-free evidence stratum. kind: small_smoke (default), small_any, smoke_any, none."""
    kind: str = "small_smoke"
    area_max: float = 0.02

    def to_dict(self):
        return asdict(self)


def n_min(alpha=0.05, delta=0.10):
    """Smallest number of units for which zero observed misses certifies risk <= alpha at level delta."""
    return int(math.ceil(math.log(delta) / math.log(1.0 - alpha)))


def smoke_features(box_strings, s_fire, s_smoke):
    """Area of the highest-confidence SMOKE box (class 0 in dump_detector) and of the top box; 1.0 if none.
    Box string: 'cls:conf:x1:y1:x2:y2;...' in normalized coordinates, sorted by confidence."""
    n = len(box_strings)
    smoke_area = np.ones(n, float)
    top_area = np.ones(n, float)
    for i, raw in enumerate(box_strings):
        first = True
        for part in filter(None, str(raw or "").split(";")):
            try:
                c, _, x1, y1, x2, y2 = part.split(":")[:6]
                c = int(c)
                a = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
            except ValueError:
                continue
            if first:
                top_area[i] = min(a, 1.0)
                first = False
            if c == 0:
                smoke_area[i] = min(a, 1.0)
                break
    return {"smoke_area": smoke_area, "top_area": top_area,
            "s_fire": np.asarray(s_fire, float), "s_smoke": np.asarray(s_smoke, float)}


def in_stratum(spec: StratumSpec, F):
    if spec.kind == "none":
        return np.zeros(len(F["smoke_area"]), bool)
    smoke_dom = np.asarray(F["s_smoke"]) >= np.asarray(F["s_fire"])
    if spec.kind == "small_smoke":
        return (np.asarray(F["smoke_area"]) <= spec.area_max) & smoke_dom
    if spec.kind == "small_any":
        return np.asarray(F["top_area"]) <= spec.area_max
    if spec.kind == "smoke_any":
        return smoke_dom & (np.asarray(F["smoke_area"]) < 1.0)
    raise ValueError(spec.kind)


def discount(g, inS, lam):
    """VLM score with the veto discounted by lam inside S (nan = not scored stays nan)."""
    g = np.asarray(g, float)
    return np.where(inS & np.isfinite(g), np.minimum(1.0, g + lam), g)


STRATUM_RISKS = ("smoke",)      # risks constrained inside S; fixed a priori (S is a SMOKE stratum)


def stratum_risks(risk_labels, inS, which=STRATUM_RISKS):
    """{'fire': y_f, 'smoke': y_s} -> adds '<k>|S' for every k in `which`. The set is fixed in advance and never
    depends on the data: a listed stratum risk with no positive (or too few units) is NaN / not certifiable,
    so ESVA then does not certify instead of silently dropping the constraint."""
    out = dict(risk_labels)
    for k in which:
        if k not in risk_labels:
            raise ValueError(f"stratum risk {k!r} is not one of the risks {sorted(risk_labels)}")
        out[f"{k}|S"] = np.asarray(risk_labels[k]).astype(bool) & inS
    return out


def certifiability(unit, risk_labels, inS, alpha, delta, which=STRATUM_RISKS):
    """Positive units per constrained risk vs n_min (label use: only on the data passed in)."""
    unit = np.asarray(unit)
    r = {}
    for k, v in stratum_risks(risk_labels, inS, which).items():
        r[k] = int(len(np.unique(unit[np.asarray(v).astype(bool)])))
    return {"n_min": n_min(alpha, delta), "positive_units": r,
            "certifiable": all(x >= n_min(alpha, delta) for x in r.values())}


def calibrate_esva(s, g, features, y, unit, *, stratum=StratumSpec(), lambdas=LAMBDAS, alpha=0.05, delta=0.10,
                   grid=None, costs=None, nbytes=None, level="unit", opt_frac=0.30, seed=0, event=None,
                   risk_labels=None, s_min=0.02, handoff=False, stratum_risk_keys=STRATUM_RISKS):
    """Returns (theta, info); theta = (t_low, t_high, b, a) applied to discount(g, S, info['lambda'])."""
    s = np.asarray(s, float)
    g = np.asarray(g, float)
    y = np.asarray(y).astype(bool)
    unit = np.asarray(unit)
    event = None if (event is None or level != "unit") else np.asarray(event)
    base_risks = {"miss": y} if risk_labels is None else {k: np.asarray(v).astype(bool)
                                                         for k, v in risk_labels.items()}
    inS = in_stratum(stratum, features)
    if stratum.kind == "none":         # no stratum: ESVA reduces to cascade_LTT (pooled risks only)
        stratum_risk_keys = ()
    risks = stratum_risks(base_risks, inS, stratum_risk_keys)
    nb = None if nbytes is None else np.asarray(nbytes, float)
    costs = {"calls": 1.0, "fa": 10.0, **(costs or {})}
    lambdas = [float(x) for x in lambdas]
    if 0.0 not in lambdas:
        raise ValueError("lambda = 0 (full veto authority = cascade_LTT) must be in the family")

    rng = np.random.default_rng(seed)
    uu = np.unique(unit)
    sel_u = set(rng.choice(uu, max(1, int(round(opt_frac * len(uu)))), replace=False))
    ms = np.array([u in sel_u for u in unit])
    mt = ~ms
    grid = grid or Grid.default(s[ms], s_min=s_min)
    base = candidate_mask(grid, cloud=True, handoff=bool(handoff))

    def stats(mask, lam):
        return _grid_stats(s[mask], discount(g, inS, lam)[mask], y[mask], unit[mask], grid, level,
                           None if nb is None else nb[mask], None if event is None else event[mask],
                           {k: v[mask] for k, v in risks.items()})

    proposed = []                       # select fold only: per-lambda local Pareto fronts
    for li, lam in enumerate(lambdas):
        Eo, Ro, _ = stats(ms, lam)
        rmax = np.max(np.stack([np.nan_to_num(r, nan=np.inf) for r in Ro.values()]), axis=0)
        cand = base & np.isfinite(rmax) & ~np.isnan(Eo["fa"])
        rf = np.where(cand, rmax, np.inf).ravel()
        cf = np.where(cand, _objective(Eo, costs), np.inf).ravel()
        idx = np.flatnonzero(cand.ravel())
        best = np.inf
        for flat in idx[np.lexsort((rf[idx], cf[idx]))]:
            if rf[flat] <= best:        # WEAK front: keep ties (see below)
                proposed.append((li, int(flat), float(rf[flat]), float(cf[flat])))
                best = rf[flat]
    proposed.sort(key=lambda z: (z[3], z[2], z[0], z[1]))
    front, best = [], np.inf
    for z in proposed:                  # global weak Pareto front
        if z[2] <= best:
            front.append(z)
            best = z[2]
    # Testing order: safest first; among equal select-fold risk (typically 0 misses in a small stratum), the
    # MORE conservative (costlier) configuration first. A strict front would keep only the cheapest
    # zero-miss point, which in a stratum with few select units is often not safe and would stop the fixed
    # sequence at step 1. The order depends on the select fold only, so FWER <= delta is unchanged.
    front.sort(key=lambda z: (z[2], -z[3], z[0], z[1]))

    cache, pvals, valid = {}, [], []
    pall = np.ones(len(front))          # certify fold: joint p over pooled AND stratum risks (vectorized per lambda)
    for li in sorted({z[0] for z in front}):
        cache[li] = stats(mt, lambdas[li])
        Et, Rt, nt = cache[li]
        pos = [i for i, z in enumerate(front) if z[0] == li]
        pall[pos] = _joint_p(Rt, nt, alpha, np.array([front[i][1] for i in pos], int))
    for z, p in zip(front, pall):       # fixed-sequence: stop at the first non-rejection
        pvals.append(float(p))
        if p > delta:
            break
        valid.append(z)
    common = {"stratum": stratum.to_dict(), "lambdas": lambdas, "handoff": bool(handoff), "grid": grid.to_dict(),
              "n_candidates": len(front), "n_valid": len(valid), "n_select_units": len(sel_u),
              "risks": sorted(risks), "stratum_frac": float(inS.mean()), "fwer": "pareto_fixed_sequence",
              "certifiability_certify_fold": certifiability(unit[mt], {k: v[mt] for k, v in base_risks.items()},
                                                            inS[mt], alpha, delta, stratum_risk_keys)}
    if valid:
        obj = {li: _objective(cache[li][0], costs).ravel() for li in {z[0] for z in valid}}
        rmx = {li: np.max(np.stack([np.nan_to_num(r.ravel(), nan=1.0) for r in cache[li][1].values()]), axis=0)
               for li in obj}
        scored = [(float(obj[z[0]][z[1]]), float(rmx[z[0]][z[1]]), z[0], z[1], z) for z in valid]
        _, _, li, flat, chosen = min(scored)
        Et, Rt, _ = cache[li]
        idx4 = np.unravel_index(flat, base.shape)
        info = dict(common, certified=True, p_value=float(pvals[front.index(chosen)]))
    else:
        li = lambdas.index(0.0)         # fallback: lambda 0 (no discount), edge-only rule
        if li not in cache:
            cache[li] = stats(mt, lambdas[li])
        Et, Rt, _ = cache[li]
        idx4 = (0, 0, 0, 0)             # conservative edge-only fallback, as select.calibrate
        info = dict(common, certified=False, p_value=1.0)
    theta, info = _finish(grid, idx4, Et, Rt, info, alpha, delta, level)
    info["lambda"] = lambdas[li]
    return theta, info


def apply_esva(s, g, features, theta, info):
    """Deployment: decision codes of the certified ESVA configuration (policy.decide semantics)."""
    inS = in_stratum(StratumSpec(**info["stratum"]), features)
    return decide(s, discount(g, inS, info["lambda"]), theta)
