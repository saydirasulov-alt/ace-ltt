"""Synthetic known-population check of the ESVA-LTT certification statement (software / statistics check only).

Population: a label-free stratum S (small smoke boxes, ~30% of frames) in which the VLM vetoes true smoke
often (as SmokeBench reports for small smoke), and a complement where the VLM is informative.
Each trial draws cal_units units, runs ESVA-LTT and measures on the NOT-drawn units the true
    pooled fire / smoke miss risk and the S-restricted fire / smoke miss risk.
False certification = certified and ANY of these true risks > alpha. LTT promises P(false cert) <= delta.
--negative-control naive: plug-in p-value (p = 0 if the empirical risk <= alpha) -> must be able to fail.
--negative-control pooled_only: correct HB test but WITHOUT the S constraints; counted against the S risks,
    it shows the failure mode ESVA exists to prevent (pooled guarantee holds, S risk does not).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import evidence_veto as ev
from .evidence_veto import StratumSpec, calibrate_esva, discount, in_stratum
from .policy import metrics
from .run_cascade import clopper_pearson


def population(n=30000, seed=5, cluster=1):
    """n units; each unit = one event with `cluster` frames (label, stratum shared; scores per frame)."""
    rng = np.random.default_rng(seed)
    uid = np.repeat(np.arange(n), cluster)
    unit = uid.astype(str)
    yf = (rng.random(n) < .15)[uid]
    ys = (~yf) & (rng.random(n) < .25)[uid]
    y = yf | ys
    small = (rng.random(n) < .30)[uid]
    n = len(uid)
    smoke_area = np.where(small, rng.uniform(.001, .02, n), rng.uniform(.03, .4, n))
    s = np.clip(.10 + .35 * y + rng.normal(0, .15, n), 0, 1)
    s_smoke = np.where(ys | (~y & (rng.random(n) < .5)), s, s * .5)
    s_fire = np.where(yf, s, s * .4)
    z = -2.5 + 5.0 * y - 3.2 * (y & small & ys) + rng.normal(0, 1.1, n)       # VLM blind to small smoke
    g = 1 / (1 + np.exp(-z))
    F = {"smoke_area": np.where(ys | (~yf), smoke_area, 1.0), "top_area": smoke_area, "s_fire": s_fire,
         "s_smoke": s_smoke}
    return dict(s=s, g=g, y=y, yf=yf, ys=ys, unit=unit, event=unit.copy(), uid=uid, F=F)


def _naive_p(R, n, alpha, idx=None):
    p = None
    for k in R:
        r = R[k] if idx is None else R[k].ravel()[idx]
        pk = np.where(np.nan_to_num(r, nan=1.0) <= alpha, 0.0, 1.0)
        p = pk if p is None else np.maximum(p, pk)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--population", type=int, default=30000)
    ap.add_argument("--cal-units", type=int, default=3000)
    ap.add_argument("--negative-control", choices=["none", "naive", "pooled_only"], default="none")
    ap.add_argument("--handoff", action="store_true")
    ap.add_argument("--cluster", type=int, default=1, help="frames per unit/event (>1: event-level, clustered)")
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--delta", type=float, default=.10)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    real = ev._joint_p
    if a.negative_control == "naive":
        ev._joint_p = _naive_p
    stratum = StratumSpec("small_smoke", 0.02)
    lambdas = (0.0,) if a.negative_control == "pooled_only" else ev.LAMBDAS
    d = population(a.population, a.seed + 3, a.cluster)
    ev_arg = d["event"] if a.cluster > 1 else None
    inS_all = in_stratum(stratum, d["F"])
    rng = np.random.default_rng(a.seed)
    cert = viol = viol_pooled = 0
    cert_r, cert_rS = [], []
    try:
        for t in range(a.trials):
            iu = rng.choice(a.population, a.cal_units, replace=False)          # calibration UNITS
            cal_m = np.isin(d["uid"], iu)
            ix, te = np.flatnonzero(cal_m), np.flatnonzero(~cal_m)
            F = {k: v[ix] for k, v in d["F"].items()}
            st = StratumSpec("none") if a.negative_control == "pooled_only" else stratum
            theta, info = calibrate_esva(d["s"][ix], d["g"][ix], F, d["y"][ix], d["unit"][ix], stratum=st,
                                         lambdas=lambdas, alpha=a.alpha, delta=a.delta,
                                         costs={"calls": 1., "handoff": 5., "fa": 20.},
                                         risk_labels={"fire": d["yf"][ix], "smoke": d["ys"][ix]},
                                         seed=a.seed + t, handoff=a.handoff,
                                         event=None if ev_arg is None else ev_arg[ix])
            inS = inS_all[te]
            g_t = discount(d["g"][te], inS if st.kind != "none" else np.zeros(len(te), bool), info["lambda"])
            R = {}
            for nm, lab in (("fire", d["yf"]), ("smoke", d["ys"])):
                for suf, m in (("", np.ones(len(te), bool)), ("|S", inS)):
                    if (lab[te] & m).any():
                        mm = metrics(d["s"][te], g_t, lab[te] & m, theta, d["unit"][te], None,
                                     None if ev_arg is None else ev_arg[te])
                        R[nm + suf] = mm["miss_event"] if ev_arg is not None else mm["miss_unit"]
            pooled = max(R[k] for k in ("fire", "smoke") if k in R)
            risk_all = max(R.values())
            if info["certified"]:
                cert += 1
                cert_r.append(pooled)
                cert_rS.append(risk_all)
                viol += int(risk_all > a.alpha)
                viol_pooled += int(pooled > a.alpha)
            if (t + 1) % max(1, a.trials // 5) == 0:
                print(f"trial {t + 1}/{a.trials}", flush=True)
    finally:
        ev._joint_p = real
    lo, hi = clopper_pearson(viol, a.trials)
    res = {"trials": a.trials, "population": a.population, "cal_units": a.cal_units, "alpha": a.alpha,
           "delta": a.delta, "negative_control": a.negative_control, "handoff": a.handoff, "cluster": a.cluster,
           "certification_rate": cert / a.trials, "false_cert_any_risk": viol, "false_cert_pooled_only": viol_pooled,
           "false_cert_rate": viol / a.trials, "ci95": [lo, hi],
           "certified_true_pooled_risk_mean": float(np.mean(cert_r)) if cert_r else None,
           "certified_true_max_risk_incl_S_mean": float(np.mean(cert_rS)) if cert_rS else None,
           "certified_true_max_risk_incl_S_max": float(np.max(cert_rS)) if cert_rS else None,
           "verdict": "supported" if hi <= a.delta else ("VIOLATED" if lo > a.delta else "inconclusive")}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
