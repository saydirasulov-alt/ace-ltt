"""Risk-certified adaptive visual acquisition for the edge--cloud cascade.

RAVC-LTT chooses exactly one cloud view for every escalated frame: the full-frame detector
overlay or a crop around the detector's top box.  The router is deliberately small and
interpretable.  Its parameters are selected on the Pareto-selection fold together with the
edge/cloud decision thresholds; the disjoint certification fold is used only for fixed-sequence
Learn-then-Test.  Consequently, adding the router does not invalidate the finite-sample guarantee.

The controlled losses are unchanged: bounded, event-averaged fire and smoke miss losses over
exchangeable units.  View routing changes the cloud score and payload, not the definition of risk.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from .policy import Grid, decide
from .select import _finish, _grid_stats, _joint_p, _objective, candidate_mask


@dataclass(frozen=True)
class RouterSpec:
    """A label-free, source-agnostic rule for choosing crop (True) or overlay (False).

    ``kind`` is one of:
      * overlay / crop: fixed-view baselines;
      * small: crop small top boxes;
      * small_isolated: additionally require at most ``max_boxes`` confident boxes;
      * small_fire: additionally require fire evidence >= smoke evidence.
    """

    name: str
    kind: str
    area_max: float = 1.0
    max_boxes: int = 10**9

    def to_dict(self):
        return asdict(self)


def top_box_features(box_strings, n_boxes_25, s_fire, s_smoke):
    """Return source-agnostic router features from detector CSV columns.

    The first serialized box is the highest-confidence box.  Its coordinates are normalized,
    so area is resolution independent.  Missing/malformed boxes receive area 1 and therefore
    default to the context-preserving overlay for all ``small*`` routers.
    """
    area = np.ones(len(box_strings), float)
    aspect = np.ones(len(box_strings), float)
    for i, raw in enumerate(box_strings):
        try:
            p = str(raw).split(";")[0].split(":")
            x1, y1, x2, y2 = map(float, p[2:6])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area[i] = np.clip(w * h, 0.0, 1.0)
            aspect[i] = w / max(h, 1e-9)
        except (ValueError, IndexError, TypeError):
            pass
    return {
        "top_area": area,
        "top_aspect": aspect,
        "n_boxes_25": np.asarray(n_boxes_25, int),
        "s_fire": np.asarray(s_fire, float),
        "s_smoke": np.asarray(s_smoke, float),
    }


def route_crop(spec: RouterSpec, features):
    """Boolean crop decision for every frame.  No label or dataset/source id is used."""
    n = len(features["top_area"])
    if spec.kind == "overlay":
        return np.zeros(n, bool)
    if spec.kind == "crop":
        return np.ones(n, bool)
    small = np.asarray(features["top_area"], float) <= spec.area_max
    if spec.kind == "small":
        return small
    isolated = np.asarray(features["n_boxes_25"], int) <= spec.max_boxes
    if spec.kind == "small_isolated":
        return small & isolated
    if spec.kind == "small_fire":
        return small & isolated & (np.asarray(features["s_fire"]) >= np.asarray(features["s_smoke"]))
    raise ValueError(f"unknown router kind: {spec.kind}")


def router_family(features, select_mask, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90), max_boxes=(1, 2, 3)):
    """Build a compact router family using feature quantiles from the selection fold only.

    Duplicate routing masks are removed.  Fixed overlay and fixed crop are always included, so
    the adaptive method can never lose a view baseline merely because of router construction.
    """
    m = np.asarray(select_mask, bool)
    a = np.asarray(features["top_area"], float)[m]
    a = a[np.isfinite(a) & (a < 1.0)]
    cuts = np.unique(np.round(np.quantile(a, quantiles), 6)) if len(a) else np.array([0.05])
    specs = [RouterSpec("overlay", "overlay"), RouterSpec("crop", "crop")]
    for q in cuts:
        specs.append(RouterSpec(f"small_a{q:.6f}", "small", float(q)))
        for b in max_boxes:
            specs.append(RouterSpec(f"isolated_a{q:.6f}_b{b}", "small_isolated", float(q), int(b)))
            specs.append(RouterSpec(f"fire_a{q:.6f}_b{b}", "small_fire", float(q), int(b)))
    out, seen = [], set()
    for spec in specs:
        key = hashlib.sha256(route_crop(spec, features)[m].tobytes()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(spec)
    return out


def routed_arrays(spec, features, g_overlay, g_crop, bytes_overlay=None, bytes_crop=None,
                  ms_overlay=None, ms_crop=None):
    """Materialize the single-view score/cost vectors induced by ``spec``."""
    use_crop = route_crop(spec, features)
    g = np.where(use_crop, np.asarray(g_crop, float), np.asarray(g_overlay, float))
    out = {"g": g, "use_crop": use_crop}
    if bytes_overlay is not None and bytes_crop is not None:
        out["nbytes"] = np.where(use_crop, np.asarray(bytes_crop, float), np.asarray(bytes_overlay, float))
    if ms_overlay is not None and ms_crop is not None:
        out["agent_ms"] = np.where(use_crop, np.asarray(ms_crop, float), np.asarray(ms_overlay, float))
    return out


def _take_features(features, mask):
    return {k: np.asarray(v)[mask] for k, v in features.items()}


def calibrate_adaptive_view(
    s,
    g_overlay,
    g_crop,
    features,
    y,
    unit,
    *,
    alpha=0.05,
    delta=0.10,
    grid=None,
    costs=None,
    nbytes_overlay=None,
    nbytes_crop=None,
    level="unit",
    opt_frac=0.30,
    seed=0,
    event=None,
    risk_labels=None,
    s_min=0.02,
    routers=None,
    handoff=False,
):
    """Jointly select a view router and cascade thresholds, then certify by fixed sequence.

    Returns ``(theta, info)``.  ``info['router']`` is the deployed RouterSpec as a dictionary.
    ``handoff=True`` adds the human tier exactly as ``select.calibrate`` does (k1 < k2 allowed); the
    operator is assumed to resolve hand-off frames correctly (protocol human_tier_assumption).
    Only the Pareto-selection fold can construct the router family, grid, and candidate order.
    The certification fold is not inspected until that sequence is frozen.
    """
    s = np.asarray(s, float)
    go, gc = np.asarray(g_overlay, float), np.asarray(g_crop, float)
    y = np.asarray(y).astype(bool)
    unit = np.asarray(unit)
    event = None if (event is None or level != "unit") else np.asarray(event)
    risks = {"miss": y} if risk_labels is None else {k: np.asarray(v).astype(bool) for k, v in risk_labels.items()}
    bo = None if nbytes_overlay is None else np.asarray(nbytes_overlay, float)
    bc = None if nbytes_crop is None else np.asarray(nbytes_crop, float)
    costs = {"calls": 1.0, "fa": 10.0, **(costs or {})}
    if not (len(s) == len(go) == len(gc) == len(y) == len(unit)):
        raise ValueError("all frame arrays must have equal length")
    required = s >= s_min
    if np.isnan(go[required]).any() or np.isnan(gc[required]).any():
        raise ValueError("both overlay and crop scores are required for every potentially escalated frame")

    rng = np.random.default_rng(seed)
    uu = np.unique(unit)
    nsel = max(1, int(round(opt_frac * len(uu))))
    sel_u = set(rng.choice(uu, nsel, replace=False))
    ms = np.array([u in sel_u for u in unit])
    mt = ~ms
    grid = grid or Grid.default(s[ms], s_min=s_min)
    routers = list(routers or router_family(features, ms))
    base = candidate_mask(grid, cloud=True, handoff=bool(handoff))

    def stats(mask, spec):
        rr = routed_arrays(spec, _take_features(features, mask), go[mask], gc[mask],
                           None if bo is None else bo[mask], None if bc is None else bc[mask])
        return _grid_stats(
            s[mask], rr["g"], y[mask], unit[mask], grid, level, rr.get("nbytes"),
            None if event is None else event[mask], {k: v[mask] for k, v in risks.items()},
        )

    # A global Pareto point must be Pareto-optimal within its own router.  Keeping only each
    # router's local front is therefore lossless and avoids retaining a large 5-D tensor.
    proposed = []
    for rid, spec in enumerate(routers):
        Eo, Ro, _ = stats(ms, spec)
        rmax = np.max(np.stack([np.nan_to_num(r, nan=np.inf) for r in Ro.values()]), axis=0)
        cand = base & np.isfinite(rmax) & ~np.isnan(Eo["fa"])
        rf = np.where(cand, rmax, np.inf).ravel()
        cf = np.where(cand, _objective(Eo, costs), np.inf).ravel()
        idx = np.flatnonzero(cand.ravel())
        order = idx[np.lexsort((rf[idx], cf[idx]))]
        best = np.inf
        for flat in order:
            if rf[flat] < best:
                proposed.append((rid, int(flat), float(rf[flat]), float(cf[flat])))
                best = rf[flat]

    # Global front: lowest cost first, keep strict improvements in risk, then test safest first.
    proposed.sort(key=lambda z: (z[3], z[2], z[0], z[1]))
    front, best = [], np.inf
    for z in proposed:
        if z[2] < best:
            front.append(z)
            best = z[2]
    front.sort(key=lambda z: (z[2], z[3], z[0], z[1]))

    cert_cache = {}
    pvals, valid = [], []
    for z in front:
        rid, flat = z[0], z[1]
        if rid not in cert_cache:
            cert_cache[rid] = stats(mt, routers[rid])
        Et, Rt, nt = cert_cache[rid]
        p = float(_joint_p(Rt, nt, alpha, np.array([flat], int))[0])
        pvals.append(p)
        if p > delta:
            break
        valid.append(z)

    common = {
        "n_units": {k: v for k, v in (next(iter(cert_cache.values()))[2] if cert_cache else {}).items()},
        "n_select_units": len(sel_u),
        "n_routers": len(routers),
        "n_candidates": len(front),
        "n_valid": len(valid),
        "fwer": "pareto_fixed_sequence",
        "handoff": bool(handoff),
        "grid": grid.to_dict(),
        "router_family": [r.to_dict() for r in routers],
    }
    if valid:
        scored = []
        for z in valid:
            rid, flat = z[0], z[1]
            Et, Rt, _ = cert_cache[rid]
            obj = float(_objective(Et, costs).ravel()[flat])
            risk = max(float(np.nan_to_num(r.ravel()[flat], nan=1.0)) for r in Rt.values())
            scored.append((obj, risk, rid, flat, z))
        _, _, rid, flat, chosen = min(scored)
        Et, Rt, _ = cert_cache[rid]
        idx4 = np.unravel_index(flat, base.shape)
        info = dict(common, certified=True, p_value=float(pvals[front.index(chosen)]))
    else:
        rid = next((i for i, r in enumerate(routers) if r.kind == "overlay"), 0)
        if rid not in cert_cache:
            cert_cache[rid] = stats(mt, routers[rid])
        Et, Rt, _ = cert_cache[rid]
        idx4 = (0, 0, 0, 0)  # conservative edge-only fallback
        info = dict(common, certified=False, p_value=1.0)

    theta, info = _finish(grid, idx4, Et, Rt, info, alpha, delta, level)
    spec = routers[rid]
    routed = routed_arrays(spec, features, go, gc, bo, bc)
    esc = decide(s, routed["g"], theta) >= 2
    info.update(
        router=spec.to_dict(),
        router_crop_fraction=float(routed["use_crop"].mean()),
        crop_call_fraction=float((esc & routed["use_crop"]).mean()),
        overlay_call_fraction=float((esc & ~routed["use_crop"]).mean()),
    )
    return theta, info

