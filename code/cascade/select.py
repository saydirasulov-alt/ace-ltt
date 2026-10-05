"""Calibrate a cascade policy with Learn-then-Test and pick the cheapest certified configuration.

Several risks can be controlled at once (e.g. fire-event miss and smoke-event miss). A configuration is
certified only if EVERY risk passes: the joint p-value is the maximum of the per-risk Hoeffding-Bentkus
p-values (intersection-union test), so FWER <= delta is kept over the whole candidate family.
"""
from __future__ import annotations

import numpy as np

from .policy import Grid, evaluate_grid, event_miss_grid
from .risk import BinomCDF, hb_pvalue, unit_weights

DEFAULT_COSTS = {"calls": 1.0, "handoff": 20.0, "fa": 10.0}


def candidate_mask(grid, cloud=True, handoff=True):
    """Canonical, non-duplicated configurations: j1<=j2, k1<=k2; without cloud only (j,j,0,0)."""
    J, K = len(grid.s_edges), len(grid.g_edges)
    j1 = np.arange(J)[:, None, None, None]
    j2 = np.arange(J)[None, :, None, None]
    k1 = np.arange(K)[None, None, :, None]
    k2 = np.arange(K)[None, None, None, :]
    edge_only = (j1 == j2) & (k1 == 0) & (k2 == 0)
    cloud_cfg = (j1 < j2) & (k1 <= k2)
    if not handoff:
        cloud_cfg &= (k1 == k2)
    m = edge_only | (cloud_cfg if cloud else False)
    return np.broadcast_to(m, (J, J, K, K))


def _miss_grid(s, g, y, unit, grid, level, event):
    """Miss-risk grid for one label array and its effective sample size n."""
    if level == "unit" and event is not None:
        return event_miss_grid(s, g, y, unit, event, grid)
    if level == "unit":
        w, n = unit_weights(y, unit)
    else:
        n = int(y.sum())
        w = y / max(1, n)
    return evaluate_grid(s, g, grid, {"miss": w})["miss"], n


def _grid_stats(s, g, y, unit, grid, level, nbytes, event, risks):
    """Cost grids (fa, calls, handoff, bytes) from the any-positive label y, plus one miss grid per risk."""
    N = len(s)
    weights = {"fa": (~y) / max(1, (~y).sum()), "calls": np.full(N, 1.0 / N), "handoff": np.full(N, 1.0 / N)}
    if nbytes is not None:
        weights["bytes"] = np.asarray(nbytes, float) / N
    E = evaluate_grid(s, g, grid, weights)
    R, n = {}, {}
    for name, yk in risks.items():
        R[name], n[name] = _miss_grid(s, g, yk, unit, grid, level, event)
    return E, R, n


def _objective(E, costs):
    return sum(costs.get(k, 0.0) * np.nan_to_num(E[k]) for k in costs if k in E)


def _joint_p(R, n, alpha, idx=None):
    """max over risks of the HB p-value (nan risk, i.e. no positives of that kind -> p = 1)."""
    p = None
    for k in R:
        r = R[k] if idx is None else R[k].ravel()[idx]
        pk = hb_pvalue(np.nan_to_num(r, nan=1.0), n[k], alpha, BinomCDF(max(n[k], 1), alpha))
        p = pk if p is None else np.maximum(p, pk)
    return p


def calibrate(s, g, y, unit, alpha=0.05, delta=0.1, grid=None, costs=None, cloud=True, handoff=True,
              nbytes=None, level="unit", fwer="pareto", opt_frac=0.3, seed=0, event=None, risk_labels=None,
              s_min=0.02):
    """Returns (theta, info). theta = (t_low, t_high, b, a).
    y: any-positive label per frame (used for false alarms and costs).
    risk_labels: dict name -> label array; each is a miss-risk constrained at level alpha
                 (default {"miss": y}). E.g. {"miss_fire": y_fire, "miss_smoke": y_smoke}.
    level: 'unit' (exchangeable units, e.g. camera / near-duplicate groups) or 'image'.
    event: optional event id per frame (nested in units). Then the loss of a unit is the fraction of its
           positive events in which no positive frame raised an alarm; otherwise the fraction of positive frames.
    fwer: 'bonferroni' (all candidates at delta/m) or 'pareto' (Pareto Testing, Laufer-Goldshmidt et al. 2023):
          a 'select' part of the calibration units (opt_frac) builds the threshold grid AND orders the Pareto
          front of (max risk, cost); the disjoint 'certify' part only computes p-values of that frozen
          sequence (fixed-sequence testing). Its scores never influence the candidate set.
    grid: a pre-registered Grid (e.g. from protocol.yaml, built on dev). If None: pareto -> quantile grid of
          the select part's edge scores; bonferroni -> data-independent Grid.fixed(s_min)."""
    s = np.asarray(s, float)
    y = np.asarray(y).astype(bool)
    unit = np.asarray(unit)
    g = np.asarray(g, float)
    risks = {"miss": y} if risk_labels is None else {k: np.asarray(v).astype(bool) for k, v in risk_labels.items()}
    event = None if (event is None or level != "unit") else np.asarray(event)
    costs = dict(DEFAULT_COSTS if costs is None else costs)
    nb = None if nbytes is None else np.asarray(nbytes)

    def stats(m):
        return _grid_stats(s[m], g[m], y[m], unit[m], grid, level, None if nb is None else nb[m],
                           None if event is None else event[m], {k: v[m] for k, v in risks.items()})

    if fwer == "pareto":
        rng = np.random.default_rng(seed)
        uu = np.unique(unit)
        sel_u = set(rng.choice(uu, max(1, int(round(opt_frac * len(uu)))), replace=False))
        ms = np.array([x in sel_u for x in unit])
        grid = grid or Grid.default(s[ms], s_min=s_min)          # built from the select part only
        base = candidate_mask(grid, cloud, handoff)
        Eo, Ro, _ = stats(ms)
        Et, Rt, nt = stats(~ms)
        rmax_o = np.max(np.stack([np.nan_to_num(r, nan=np.inf) for r in Ro.values()]), axis=0)
        cand = base & np.isfinite(rmax_o) & ~np.isnan(Eo["fa"])
        ro = np.where(cand, rmax_o, np.inf).ravel()
        co = np.where(cand, _objective(Eo, costs), np.inf).ravel()
        idx = np.flatnonzero(cand.ravel())
        order = idx[np.lexsort((ro[idx], co[idx]))]                      # by cost, then risk
        front, best = [], np.inf
        for i in order:                                                   # Pareto front, cost ascending
            if ro[i] < best:
                front.append(i)
                best = ro[i]
        front = sorted(front, key=lambda i: ro[i])                        # safest first
        p = _joint_p(Rt, nt, alpha, np.array(front, int)) if front else np.array([])
        valid = []
        for i, pi in zip(front, p):                                       # fixed-sequence testing
            if pi > delta:
                break
            valid.append(i)
        info = {"n_units": nt, "n_candidates": len(front), "n_valid": len(valid), "fwer": fwer,
                "n_select_units": len(sel_u), "grid": grid.to_dict()}
        E, R, pv = Et, Rt, dict(zip(front, p))
        if valid:
            obj = _objective(Et, costs).ravel()
            rmax_t = np.max(np.stack([np.nan_to_num(r.ravel(), nan=1.0) for r in Rt.values()]), axis=0)
            k = min(valid, key=lambda i: (obj[i], rmax_t[i]))
            idx4 = np.unravel_index(k, base.shape)
            info.update(certified=True, p_value=float(pv[k]))
        else:
            idx4 = (0, 0, 0, 0)
            info.update(certified=False, p_value=1.0)
        return _finish(grid, idx4, E, R, info, alpha, delta, level)

    grid = grid or Grid.fixed(s_min)
    base = candidate_mask(grid, cloud, handoff)
    E, R, n = stats(np.ones(len(s), bool))
    rmax = np.max(np.stack([np.nan_to_num(r, nan=np.inf) for r in R.values()]), axis=0)
    cand = base & np.isfinite(rmax) & ~np.isnan(E["fa"])
    p = np.where(cand, _joint_p(R, n, alpha), 1.0)
    m = int(cand.sum())
    valid = cand & (p <= delta / max(1, m))
    info = {"n_units": n, "n_candidates": m, "n_valid": int(valid.sum()), "fwer": fwer, "grid": grid.to_dict()}
    if valid.any():
        score = np.where(valid, _objective(E, costs) + 1e-9 * rmax, np.inf)
        idx4 = np.unravel_index(np.argmin(score), score.shape)
        info["certified"] = True
    else:
        idx4 = (0, 0, 0, 0)       # nothing certifiable: most conservative edge-only rule (alarm on s >= s_min)
        info["certified"] = False
    info["p_value"] = float(p[idx4])
    return _finish(grid, idx4, E, R, info, alpha, delta, level)


def _finish(grid, idx, E, R, info, alpha, delta, level):
    j1, j2, k1, k2 = idx
    theta = (float(grid.s_edges[j1]), float(grid.s_edges[j2]), float(grid.g_edges[k1]), float(grid.g_edges[k2]))
    info.update({"cal_" + k: float(E[k][idx]) for k in E})
    info.update({"cal_" + k: float(R[k][idx]) for k in R})
    info.update(alpha=alpha, delta=delta, level=level)
    return theta, info
