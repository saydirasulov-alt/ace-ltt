"""Synthetic known-population check of the RAVC-LTT certification statement.

The numbers are a software/statistical validity check, never a paper result about fire detection.

Each trial draws cal_units calibration units, runs RAVC-LTT and measures the TRUE risk of the certified
configuration on the population units NOT drawn (held out). A false certification = certified and
true risk > alpha; LTT promises P(false certification) <= delta.

A validity check is informative only if it can fail, so:
  * the default is near the boundary (cal_units 3000 -> most trials certify, certified risks close to alpha);
  * --negative-control naive replaces the HB p-value by a plug-in rule (p = 0 if the empirical risk <= alpha).
    That calibrator carries no finite-sample guarantee and must show a false-certification rate above delta.
Reference (v2.7 review, 200 trials, population 30000): RAVC-LTT cal 3000 -> 87.5% certified, 0 false
(CI95 [0, 0.018]); naive cal 3000 -> 100% certified, 31/200 false (CI95 [0.108, 0.213], VIOLATED).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import adaptive_view as _av
from .adaptive_view import RouterSpec, calibrate_adaptive_view, routed_arrays
from .policy import metrics
from .run_cascade import clopper_pearson


def population(n=30000, seed=812):
    rng = np.random.default_rng(seed)
    unit = np.arange(n).astype(str)
    yf = rng.random(n) < .18
    ys = (~yf) & (rng.random(n) < .22)
    y = yf | ys
    small = rng.random(n) < .55
    area = np.where(small, rng.uniform(.002, .03, n), rng.uniform(.06, .45, n))
    nbox = np.where(small, 1, rng.integers(1, 5, n))
    # Edge confidence contains useful but imperfect evidence.
    s = np.clip(.08 + .34 * y + rng.normal(0, .16, n), 0, 1)
    sf = np.clip(s + .05 * yf - .03 * ys, 0, 1)
    ss = np.clip(s + .05 * ys - .03 * yf, 0, 1)
    # Complementary cloud evidence with deliberately non-zero residual error.
    zo = -2.3 + 4.2 * y + 1.2 * (y & ~small) - 1.0 * (y & small) + rng.normal(0, 1.2, n)
    zc = -2.3 + 4.2 * y + 1.2 * (y & small) - 1.0 * (y & ~small) + rng.normal(0, 1.2, n)
    go, gc = 1 / (1 + np.exp(-zo)), 1 / (1 + np.exp(-zc))
    feat = {"top_area": area, "top_aspect": np.ones(n), "n_boxes_25": nbox, "s_fire": sf, "s_smoke": ss}
    return dict(s=s, go=go, gc=gc, feat=feat, y=y, yf=yf, ys=ys, unit=unit,
                bf=np.full(n, 48000.), bc=np.full(n, 8000.))


def _naive_p(R, n, alpha, idx=None):
    """Negative control: plug-in 'empirical risk <= alpha' with no concentration bound."""
    p = None
    for k in R:
        r = R[k] if idx is None else R[k].ravel()[idx]
        pk = np.where(np.nan_to_num(r, nan=1.0) <= alpha, 0.0, 1.0)
        p = pk if p is None else np.maximum(p, pk)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--population", type=int, default=30000)
    ap.add_argument("--cal-units", type=int, default=3000)
    ap.add_argument("--negative-control", choices=["none", "naive"], default="none")
    ap.add_argument("--handoff", action="store_true", help="calibrate with the human tier (costs.handoff = 5)")
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--delta", type=float, default=.10)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    real_p = _av._joint_p
    if a.negative_control == "naive":
        _av._joint_p = _naive_p
    d = population(a.population, a.seed + 17)
    rng = np.random.default_rng(a.seed)
    certified = violations = 0
    true_risks, cert_risks = [], []
    for t in range(a.trials):
        ix = rng.choice(a.population, a.cal_units, replace=False)
        te = np.setdiff1d(np.arange(a.population), ix)
        feat = {k: v[ix] for k, v in d["feat"].items()}
        theta, info = calibrate_adaptive_view(
            d["s"][ix], d["go"][ix], d["gc"][ix], feat, d["y"][ix], d["unit"][ix],
            alpha=a.alpha, delta=a.delta, costs={"calls": 1., "handoff": 5., "fa": 20., "bytes": 2e-5}, handoff=a.handoff,
            nbytes_overlay=d["bf"][ix], nbytes_crop=d["bc"][ix],
            risk_labels={"fire": d["yf"][ix], "smoke": d["ys"][ix]}, seed=a.seed + t,
        )
        spec = RouterSpec(**info["router"])
        rr = routed_arrays(spec, {k: v[te] for k, v in d["feat"].items()}, d["go"][te], d["gc"][te])
        rf = metrics(d["s"][te], rr["g"], d["yf"][te], theta, d["unit"][te])["miss_unit"]
        rs = metrics(d["s"][te], rr["g"], d["ys"][te], theta, d["unit"][te])["miss_unit"]
        risk = max(rf, rs)
        true_risks.append(risk)
        certified += int(info["certified"])
        if info["certified"]:
            cert_risks.append(risk)
        violations += int(info["certified"] and risk > a.alpha)
        if (t + 1) % max(1, a.trials // 5) == 0:
            print(f"trial {t + 1}/{a.trials}", flush=True)
    _av._joint_p = real_p
    lo, hi = clopper_pearson(violations, a.trials)
    result = {
        "trials": a.trials, "population": a.population, "cal_units": a.cal_units,
        "alpha": a.alpha, "delta": a.delta, "certification_rate": certified / a.trials,
        "violations": violations, "violation_rate": violations / a.trials,
        "violation_ci95": [lo, hi], "true_risk_mean": float(np.mean(true_risks)),
        "negative_control": a.negative_control, "handoff": a.handoff, "true_risk": "held-out population units",
        "certified_true_risk_mean": float(np.mean(cert_risks)) if cert_risks else None,
        "certified_true_risk_max": float(np.max(cert_risks)) if cert_risks else None,
        "verdict": "supported" if hi <= a.delta else ("VIOLATED" if lo > a.delta else "inconclusive"),
    }
    print(json.dumps(result, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
