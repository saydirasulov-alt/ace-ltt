"""Three-tier alarm policy and fast evaluation over a threshold grid.

Per frame: edge detector score s (max box confidence), agent score g (cloud VLM agent, only needed
when the frame is escalated).

    s >= t_high                 -> ALARM   (edge decides, no uplink)
    s <  t_low                  -> SILENT  (edge decides, no uplink)
    t_low <= s < t_high         -> ESCALATE (uplink + cloud call), then
        g >= a                  -> ALARM
        g <  b                  -> DISMISS
        b <= g < a              -> HAND-OFF to the human operator (assumed correct; costs attention)

Theta = (t_low, t_high, b, a) with t_low <= t_high and b <= a. t_low == t_high means "no cloud";
b == a means "no human tier".

All metrics are linear in per-frame weights, so they are computed for the whole grid with 2-D
prefix sums over (s-bin, g-bin): O(frames + grid) instead of O(frames * grid).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    s_edges: np.ndarray   # candidate t_low / t_high values (ascending); np.inf allowed as last
    g_edges: np.ndarray   # candidate b / a values (ascending); np.inf allowed as last

    @staticmethod
    def default(s, s_min=0.02, n_s=28, n_g=21):
        s = np.asarray(s)
        q = np.quantile(s[s >= s_min], np.linspace(0, 1, n_s - 2)[1:-1]) if (s >= s_min).sum() > 10 else []
        s_edges = np.unique(np.concatenate([[s_min], np.round(q, 4), [0.9, np.inf]]))
        g_edges = np.unique(np.concatenate([np.linspace(0.0, 1.0, n_g), [np.inf]]))
        return Grid(s_edges=s_edges, g_edges=g_edges)

    @staticmethod
    def fixed(s_min=0.02, n_s=28, n_g=21):
        """Data-independent grid (log-spaced edge thresholds), for procedures that test on all their data."""
        s_edges = np.unique(np.concatenate([[s_min], np.round(np.geomspace(s_min, 0.95, n_s - 1)[1:], 4), [np.inf]]))
        g_edges = np.unique(np.concatenate([np.linspace(0.0, 1.0, n_g), [np.inf]]))
        return Grid(s_edges=s_edges, g_edges=g_edges)

    @staticmethod
    def from_dict(d):
        return Grid(s_edges=np.array([float(x) for x in d["s_edges"]]),
                    g_edges=np.array([float(x) for x in d["g_edges"]]))

    def to_dict(self):
        return {"s_edges": [float(x) for x in self.s_edges], "g_edges": [float(x) for x in self.g_edges]}


def _cum2d(sb, gb, w, ks, kg):
    H = np.zeros((ks, kg))
    np.add.at(H, (sb, gb), w)
    return H.cumsum(0).cumsum(1)          # C[j, k] = sum w over sb <= j and gb <= k


def evaluate_grid(s, g, grid, weights):
    """weights: dict name -> per-frame weight array (e.g. miss weight on positives, fa weight on
    negatives, 1/N for rates, bytes/N). Returns dict of 4-D arrays indexed [j1, j2, k1, k2] for
    t_low=s_edges[j1], t_high=s_edges[j2], b=g_edges[k1], a=g_edges[k2] (entries with j1>j2 or
    k1>k2 are nan) with keys: miss, fa, calls, handoff, bytes (whichever weights were given)."""
    s = np.asarray(s, float)
    g = np.nan_to_num(np.asarray(g, float), nan=-1.0)
    se, ge = grid.s_edges, grid.g_edges
    Ks, Kg = len(se) + 1, len(ge) + 1
    sb = np.searchsorted(se, s, side="right")   # s < se[j]  <=>  sb <= j
    gb = np.searchsorted(ge, g, side="right")   # g < ge[k]  <=>  gb <= k
    J = np.arange(len(se))
    K = np.arange(len(ge))
    j1, j2 = J[:, None, None, None], J[None, :, None, None]
    k1, k2 = K[None, None, :, None], K[None, None, None, :]
    out = {}
    for name, w in weights.items():
        C = _cum2d(sb, gb, np.asarray(w, float), Ks, Kg)
        tot = C[-1, -1]
        below = C[:, -1]                                   # sum over s < se[j]
        esc = below[j2] - below[j1]                        # escalated mass
        esc_lt_b = C[j2, k1] - C[j1, k1]                   # escalated and g < b
        esc_lt_a = C[j2, k2] - C[j1, k2]                   # escalated and g < a
        if name == "miss":      # positives: silent, or escalated and dismissed
            v = below[j1] + esc_lt_b
        elif name == "fa":      # negatives: edge alarm, or escalated and agent alarm
            v = (tot - below[j2]) + (esc - esc_lt_a)
        elif name in ("calls", "bytes"):
            v = esc + 0 * esc_lt_a
        elif name == "handoff":
            v = esc_lt_a - esc_lt_b
        elif name == "alarm":   # any frames alarmed (edge or agent)
            v = (tot - below[j2]) + (esc - esc_lt_a)
        else:
            raise ValueError(name)
        v = np.broadcast_to(v, (len(se), len(se), len(ge), len(ge))).astype(float).copy()
        out[name] = v
    bad = (np.arange(len(se))[:, None] > np.arange(len(se))[None, :])[:, :, None, None] | \
          (np.arange(len(ge))[:, None] > np.arange(len(ge))[None, :])[None, None, :, :]
    for v in out.values():
        v[np.broadcast_to(bad, v.shape)] = np.nan
    return out


def event_unit_weights(y, unit, event):
    """Per positive event: weight so that sum(w_e * miss_e) = mean over units of the fraction of the
    unit's positive events that are missed. Returns (event ids, w_e, n_units)."""
    y = np.asarray(y).astype(bool)
    ev = np.asarray(event)[y]
    un = np.asarray(unit)[y]
    e_ids, first = np.unique(ev, return_index=True)
    e_unit = un[first]
    u_ids, inv, cnt = np.unique(e_unit, return_inverse=True, return_counts=True)
    return e_ids, 1.0 / (len(u_ids) * cnt[inv]), len(u_ids)


def event_miss_grid(s, g, y, unit, event, grid):
    """Event-level miss ("no positive frame of the event raised an alarm"), averaged per unit, for the
    whole grid. An event is caught if any of its positive frames alarms; human hand-off counts as
    caught. Returns (miss [j1, j2, k1, k2], n_units). Single-frame events use prefix sums; multi-frame
    events are evaluated exactly: miss_frame[j1, j2, k1] = silent[j1] | (below t_high[j2] & g < b[k1])."""
    s = np.asarray(s, float)
    g = np.nan_to_num(np.asarray(g, float), nan=-1.0)
    y = np.asarray(y).astype(bool)
    event = np.asarray(event)
    e_ids, w_e, n_units = event_unit_weights(y, unit, event)
    J, K = len(grid.s_edges), len(grid.g_edges)
    if n_units == 0:
        return np.full((J, J, K, K), np.nan), 0
    pos = np.flatnonzero(y)
    pe = np.searchsorted(e_ids, event[pos])
    npos = np.bincount(pe, minlength=len(e_ids))
    single = npos[pe] == 1
    w_frame = np.zeros(len(s))
    w_frame[pos[single]] = w_e[pe[single]]
    miss = evaluate_grid(s, g, grid, {"miss": w_frame})["miss"]
    sb = np.searchsorted(grid.s_edges, s, side="right")
    gb = np.searchsorted(grid.g_edges, g, side="right")
    Jr, Kr = np.arange(J), np.arange(K)
    multi = pos[~single]
    order = np.argsort(pe[~single], kind="stable")
    multi, me = multi[order], pe[~single][order]
    acc = np.zeros((J, J, K))
    if len(multi):
        cuts = np.flatnonzero(np.diff(me)) + 1
        for fr, e in zip(np.split(multi, cuts), me[np.r_[0, cuts]]):
            A = sb[fr][:, None] <= Jr[None, :]                                           # (m, J)  silent
            B = (sb[fr][:, None, None] <= Jr[None, :, None]) & (gb[fr][:, None, None] <= Kr[None, None, :])
            m = np.logical_and.reduce(A[:, :, None, None] | B[:, None, :, :], axis=0)   # (J, J, K)
            acc += w_e[e] * m
    return miss + acc[:, :, :, None], n_units


def decide(s, g, theta):
    """Per-frame decision codes: 0 silent, 1 edge alarm, 2 agent alarm, 3 dismissed, 4 hand-off."""
    tl, th, b, a = theta
    s = np.asarray(s, float)
    g = np.nan_to_num(np.asarray(g, float), nan=-1.0)
    d = np.zeros(len(s), np.int8)
    d[s >= th] = 1
    esc = (s >= tl) & (s < th)
    d[esc & (g >= a)] = 2
    d[esc & (g < b)] = 3
    d[esc & (g >= b) & (g < a)] = 4
    return d


def metrics(s, g, y, theta, unit=None, nbytes=None, event=None):
    """Direct (non-grid) evaluation; used for test sets and as a cross-check of evaluate_grid."""
    from .risk import unit_weights
    y = np.asarray(y).astype(bool)
    d = decide(s, g, theta)
    alarm = (d == 1) | (d == 2) | ((d == 4) & y)
    miss = y & ~alarm
    esc = d >= 2
    out = {
        "miss_img": float(miss[y].mean()) if y.any() else 0.0,
        "fa": float(alarm[~y].mean()) if (~y).any() else 0.0,
        "calls": float(esc.mean()),
        "handoff": float((d == 4).mean()),
        "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
    }
    if unit is not None:
        w, n_u = unit_weights(y, unit)
        out["miss_unit"] = float((w * miss).sum())
        out["n_units"] = n_u
    if unit is not None and event is not None and y.any():
        e_ids, w_e, n_u = event_unit_weights(y, unit, event)
        pe = np.searchsorted(e_ids, np.asarray(event)[y])
        caught = np.zeros(len(e_ids), bool)
        np.logical_or.at(caught, pe, alarm[y])
        out["miss_event"] = float((w_e * ~caught).sum())
        out["miss_event_raw"] = float((~caught).mean())
        out["n_events"] = len(e_ids)
    if nbytes is not None:
        out["bytes_per_frame"] = float((np.asarray(nbytes, float) * esc).mean())
    return out


def unit_losses(s, g, y, theta, unit, event=None):
    """Per-unit loss of theta for units with >= 1 positive: fraction of positive frames (event=None) or of
    positive events (no positive frame alarmed) that are missed. Returns (unit_ids, losses). The unit-level risk
    is losses.mean(); a cluster bootstrap resamples these per-unit losses."""
    y = np.asarray(y).astype(bool)
    d = decide(s, g, theta)
    alarm = (d == 1) | (d == 2) | ((d == 4) & y)
    unit = np.asarray(unit)[y]
    miss = ~alarm[y]
    if event is None:
        u, inv = np.unique(unit, return_inverse=True)
        return u, np.bincount(inv, weights=miss) / np.bincount(inv)
    ev = np.asarray(event)[y]
    e_ids, inv_e = np.unique(ev, return_inverse=True)
    caught = np.zeros(len(e_ids), bool)
    np.logical_or.at(caught, inv_e, ~miss)
    first = np.zeros(len(e_ids), int)
    first[inv_e[::-1]] = np.arange(len(ev))[::-1]
    e_unit = unit[first]
    u, inv = np.unique(e_unit, return_inverse=True)
    return u, np.bincount(inv, weights=~caught) / np.bincount(inv)

