"""Payload-budget risk for the cascade, and certification of a policy family under it.

Budget definition (pre-registered wording; NOT a bitrate guarantee):
    for an exchangeable unit u with N_u evaluated frames,   C_u(theta) = (1 / N_u) * sum_i B_ui(theta)
    loss   L_B(u; theta) = 1{ C_u(theta) > B0 }             (B_ui = 0 for a local decision)
    risk   R_B(theta)    = P_U( C_U(theta) > B0 ) <= beta
Every unit has equal weight; this bounds the share of units whose MEAN transmitted payload exceeds B0 bytes
per evaluated frame. It is not a per-frame cap, not a per-second rate, and it does not forbid short bursts.
A per-frame cap P(bytes of one frame > B0) <= beta would be implied by calls <= beta alone, which is why the
unit-mean form is used instead.

B_ui is the application payload measured on the SAME path the model sees (agent_vlm --jpeg-quality:
prepare -> encode -> count -> decode -> model). Network headers and retransmissions are not measured.

Certification: the budget risk joins the miss risks in one intersection-union test,
    p_joint = max(p_fire, p_smoke, p_B)
inside Pareto testing (select fold orders, certify fold tests). The candidate family contains the policy
(view, veto discount lambda) AND the budget B0, so a curve over several budgets is certified simultaneously -
the multiplicity is inside the tested sequence, not outside it.

The mechanism is standard: Pareto Testing already certifies several accuracy/cost constraints jointly
(Laufer-Goldshmidt et al., 2022) and quantile LTT controls exceedance probabilities. Nothing here claims a new
test. What is measured here is whether choosing the acquisition (which view is sent) and a selective veto
improves the risk / false-alarm / payload trade-off against (a) fixed-crop LTT and (b) the same three
constraints with a fixed view.
"""
from __future__ import annotations

import numpy as np

from .policy import Grid
from .select import _finish, _grid_stats, _joint_p, _objective, candidate_mask


def unit_prefix(s, payload, unit, grid):
    """P[u, j] = sum of payload over frames of unit u with s < s_edges[j]; N[u] = frames of u."""
    s = np.asarray(s, float)
    payload = np.nan_to_num(np.asarray(payload, float), nan=0.0)
    uu, inv = np.unique(np.asarray(unit), return_inverse=True)
    J = len(grid.s_edges)
    sb = np.searchsorted(grid.s_edges, s, side="right")        # s < s_edges[j] <=> sb <= j
    P = np.zeros((len(uu), J + 1))
    np.add.at(P, (inv, sb), payload)
    P = P.cumsum(1)[:, :J]
    N = np.bincount(inv, minlength=len(uu)).astype(float)
    return P, N, uu


def budget_risk_grid(s, payload, unit, grid, B0):
    """Share of units whose mean payload per evaluated frame exceeds B0, for every (t_low, t_high).

    Escalation depends only on (t_low, t_high), so the grid is (J, J) and is broadcast over (b, a).
    Returns (risk [J, J, K, K], n_units)."""
    P, N, uu = unit_prefix(s, payload, unit, grid)
    J, K = len(grid.s_edges), len(grid.g_edges)
    sent = P[:, None, :] - P[:, :, None]                        # [u, j1, j2] = payload of frames in [tl, th)
    mean = sent / N[:, None, None]
    risk = (mean > float(B0)).mean(axis=0)
    bad = np.arange(J)[:, None] > np.arange(J)[None, :]
    risk = np.where(bad, np.nan, risk)
    return np.broadcast_to(risk[:, :, None, None], (J, J, K, K)).copy(), int(len(uu))


def calibrate_payload(s, policies, y, unit, *, budgets, alpha=0.05, delta=0.10, beta=None, grid=None, costs=None,
                      level="unit", opt_frac=0.30, seed=0, event=None, risk_labels=None, s_min=0.02,
                      handoff=False):
    """Certify (policy, budget, thresholds) jointly. policies: list of dicts {name, g, payload}.

    beta defaults to alpha. Returns (theta, info) with info['policy'], info['budget'] and the usual fields.
    Only the select fold builds the grid and the testing order; the certify fold only computes p-values.
    """
    s = np.asarray(s, float)
    y = np.asarray(y).astype(bool)
    unit = np.asarray(unit)
    event = None if (event is None or level != "unit") else np.asarray(event)
    risks = {"miss": y} if risk_labels is None else {k: np.asarray(v).astype(bool) for k, v in risk_labels.items()}
    costs = {"calls": 1.0, "fa": 10.0, **(costs or {})}
    beta = alpha if beta is None else float(beta)
    budgets = [float(b) for b in budgets]
    if not policies:
        raise ValueError("at least one policy is required")

    rng = np.random.default_rng(seed)
    uu = np.unique(unit)
    sel_u = set(rng.choice(uu, max(1, int(round(opt_frac * len(uu)))), replace=False))
    ms = np.array([u in sel_u for u in unit])
    mt = ~ms
    grid = grid or Grid.default(s[ms], s_min=s_min)
    base = candidate_mask(grid, cloud=True, handoff=bool(handoff))

    def stats(mask, pol):
        g = np.asarray(pol["g"], float)
        pay = np.asarray(pol["payload"], float)
        E, R, n = _grid_stats(s[mask], g[mask], y[mask], unit[mask], grid, level, None,
                              None if event is None else event[mask], {k: v[mask] for k, v in risks.items()})
        RB = {}
        nB = {}
        for B0 in budgets:
            rb, nb = budget_risk_grid(s[mask], pay[mask], unit[mask], grid, B0)
            RB[B0], nB[B0] = rb, nb
        return E, R, n, RB, nB

    proposed = []                                   # select fold only
    for pi, pol in enumerate(policies):
        Eo, Ro, _, RBo, _ = stats(ms, pol)
        rmiss = np.max(np.stack([np.nan_to_num(r, nan=np.inf) for r in Ro.values()]), axis=0)
        for bi, B0 in enumerate(budgets):
            # order by the binding constraint, scaled to its own level (miss at alpha, budget at beta)
            rall = np.maximum(rmiss / max(alpha, 1e-9), np.nan_to_num(RBo[B0], nan=np.inf) / max(beta, 1e-9))
            cand = base & np.isfinite(rall) & ~np.isnan(Eo["fa"])
            rf = np.where(cand, rall, np.inf).ravel()
            cf = np.where(cand, _objective(Eo, costs), np.inf).ravel()
            idx = np.flatnonzero(cand.ravel())
            best = np.inf
            for flat in idx[np.lexsort((rf[idx], cf[idx]))]:
                if rf[flat] <= best:                 # weak front: keep ties (small strata / coarse budgets)
                    proposed.append((pi, bi, int(flat), float(rf[flat]), float(cf[flat])))
                    best = rf[flat]
    proposed.sort(key=lambda z: (z[4], z[3], z[0], z[1], z[2]))
    front, best = [], np.inf
    for z in proposed:
        if z[3] <= best:
            front.append(z)
            best = z[3]
    front.sort(key=lambda z: (z[3], -z[4], z[0], z[1], z[2]))    # safest first; ties: more conservative first

    cache = {}
    pall = np.ones(len(front))
    for pi in sorted({z[0] for z in front}):
        cache[pi] = stats(mt, policies[pi])
        Et, Rt, nt, RBt, nBt = cache[pi]
        for bi, B0 in enumerate(budgets):
            pos = [i for i, z in enumerate(front) if z[0] == pi and z[1] == bi]
            if not pos:
                continue
            flats = np.array([front[i][2] for i in pos], int)
            p_miss = _joint_p(Rt, nt, alpha, flats)
            p_b = _joint_p({"budget": RBt[B0]}, {"budget": nBt[B0]}, beta, flats)
            pall[pos] = np.maximum(p_miss, p_b)
    valid, pvals = [], []
    for z, p in zip(front, pall):
        pvals.append(float(p))
        if p > delta:
            break
        valid.append(z)

    common = {"policies": [p["name"] for p in policies], "budgets": budgets, "beta": beta,
              "n_candidates": len(front), "n_valid": len(valid), "n_select_units": len(sel_u),
              "handoff": bool(handoff), "grid": grid.to_dict(), "fwer": "pareto_fixed_sequence",
              "budget_definition": "P_unit(mean payload per evaluated frame > B0) <= beta"}
    if valid:
        obj = {pi: _objective(cache[pi][0], costs).ravel() for pi in {z[0] for z in valid}}
        scored = [(float(obj[z[0]][z[2]]), z[3], z) for z in valid]
        # per-budget answer: every valid candidate is rejected in the same fixed sequence, so the whole curve
        # (one certified configuration per budget) is valid simultaneously.
        by_budget = {}
        for o, r_, z in scored:
            k = budgets[z[1]]
            if k not in by_budget or o < by_budget[k]["objective"]:
                by_budget[k] = {"objective": o, "policy": policies[z[0]]["name"],
                                "theta_index": int(z[2]), "select_risk_ratio": r_}
        common["by_budget"] = {str(k): v for k, v in sorted(by_budget.items())}
        _, _, chosen = min(scored, key=lambda x: (x[0], x[1]))
        pi, bi, flat = chosen[0], chosen[1], chosen[2]
        Et, Rt, _, RBt, nBt = cache[pi]
        idx4 = np.unravel_index(flat, base.shape)
        info = dict(common, certified=True, p_value=float(pvals[front.index(chosen)]),
                    cal_budget_risk=float(RBt[budgets[bi]].ravel()[flat]), n_budget_units=nBt[budgets[bi]])
    else:
        common["by_budget"] = {}
        pi, bi = 0, 0
        if pi not in cache:
            cache[pi] = stats(mt, policies[pi])
        Et, Rt, _, RBt, nBt = cache[pi]
        idx4 = (0, 0, 0, 0)                          # conservative edge-only fallback (no cloud call, payload 0)
        info = dict(common, certified=False, p_value=1.0, cal_budget_risk=0.0, n_budget_units=nBt[budgets[bi]])
    theta, info = _finish(grid, idx4, Et, Rt, info, alpha, delta, level)
    info["policy"] = policies[pi]["name"]
    info["budget"] = budgets[bi]
    return theta, info


def unit_budget_stats(s, payload, unit, theta, B0):
    """Evaluation-side check of the same quantity (used on a test half / sealed test)."""
    esc = (np.asarray(s, float) >= theta[0]) & (np.asarray(s, float) < theta[1])
    pay = np.where(esc, np.nan_to_num(np.asarray(payload, float), nan=0.0), 0.0)
    uu, inv = np.unique(np.asarray(unit), return_inverse=True)
    tot = np.bincount(inv, weights=pay, minlength=len(uu))
    n = np.bincount(inv, minlength=len(uu)).astype(float)
    mean = tot / np.maximum(n, 1)
    return {"budget_risk": float((mean > float(B0)).mean()), "mean_bytes_per_frame": float(pay.mean()),
            "unit_mean_p95": float(np.quantile(mean, 0.95)), "n_units": int(len(uu))}
