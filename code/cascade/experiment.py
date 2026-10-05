"""Calibration / evaluation experiments, baselines and distribution-shift checks.

target = "frame": risk of a unit = fraction of its positive frames that are missed.
target = "event": cluster-averaged event miss risk: risk of a unit = fraction of its positive events with no
                  alarm on any positive frame. Only Pyro-SDIS has multi-frame events (camera sequences); for
                  single images (D-Fire, FASDD, FLAME2 frames) an event is the image itself, so there the event
                  risk equals the image miss risk ("singleton events").

What the Learn-then-Test guarantee says: over the draw of the calibration data,
    P( some configuration with true risk > alpha is certified ) <= delta.
It says nothing about the empirical risk on a finite test set. Hence we report, per method:
    certification_rate            share of trials in which a configuration was certified
    cert_and_test_exceeds_alpha   diagnostic only: certified AND empirical test risk > alpha
                                  (finite-test noise alone can push this above 0 even when every true risk <= alpha)
    risk_mean / risk_p2.5 / risk_p97.5   empirical test risk across trials
The theoretical delta is checked only in run_cascade --synthetic-validity, where the true risk is known.
"""
from __future__ import annotations

import numpy as np

from .policy import metrics
from .select import calibrate

RISK_KEY = {"frame": "miss_unit", "event": "miss_event"}


def split_units(unit, frac, rng):
    u = np.unique(unit)
    cal = set(rng.choice(u, int(round(frac * len(u))), replace=False))
    m = np.array([x in cal for x in unit])
    return m, ~m


PLAIN_G = ("pure_cloud_plain",)       # methods evaluated with the plain-view agent score g_plain


def data_for(name, d):
    """The score set a method is evaluated on (pure cloud baseline uses the detector-independent g_plain)."""
    return dict(d, g=d["g_plain"]) if name in PLAIN_G else d


def methods(alpha, delta, costs, s_min, target="frame", payload="nbytes", fixed=None,
            risks=None, fwer="pareto", select_fraction=0.3, grid=None, seed=0, all_cloud=False, pure_cloud=False):
    """name -> function(cal_dict) -> (theta, info).
    fixed: dict edge_threshold / cascade_t_low / cascade_t_high / agent_threshold for the uncalibrated
           baselines (from protocol.yaml; defaults 0.25 / 0.10 / 0.60 / 0.50).
    risks: label keys constrained jointly (e.g. ["y_fire", "y_smoke"]); default: the any-positive label y.
    grid: optional pre-registered Grid; otherwise built inside calibrate() from the select part only.
    all_cloud: 'all_frames_cloud_overlay' (overlay agent on every frame; agent run with s_min = 0).
    pure_cloud: 'pure_cloud_plain' (plain-view agent on every frame, detector-independent; needs g_plain)."""
    f = {"edge_threshold": 0.25, "cascade_t_low": 0.10, "cascade_t_high": 0.60, "agent_threshold": 0.50,
         **(fixed or {})}
    t_edge, tl, th, ga = (float(f[k]) for k in ("edge_threshold", "cascade_t_low", "cascade_t_high",
                                                 "agent_threshold"))

    def ltt(cloud, handoff, level="unit", fw=None):
        def f(d):
            ev = d.get("event") if target == "event" else None
            rl = {k: d[k] for k in risks} if risks else None
            return calibrate(d["s"], d["g"], d["y"], d["unit"], alpha, delta, grid, costs, cloud, handoff,
                             d.get(payload), level, fw or fwer, select_fraction, seed, event=ev, risk_labels=rl,
                             s_min=s_min)
        return f

    M = {
        "edge_fixed": lambda d: ((t_edge, t_edge, 0.0, 0.0), {"certified": False}),
        "edge_LTT": ltt(cloud=False, handoff=False),
        "edge_LTT_image": ltt(cloud=False, handoff=False, level="image"),      # image-exchangeable calibration
        "detector_gated_agent": lambda d: ((s_min, np.inf, ga, ga), {"certified": False}),
        "cascade_fixed": lambda d: ((tl, th, ga, ga), {"certified": False}),
        "cascade_LTT_image": ltt(cloud=True, handoff=False, level="image"),
        "cascade_LTT_bonf": ltt(cloud=True, handoff=False, fw="bonferroni"),
        "cascade_LTT": ltt(cloud=True, handoff=False),
        "cascade_LTT+human": ltt(cloud=True, handoff=True),
    }
    if all_cloud:
        M["all_frames_cloud_overlay"] = lambda d: ((-np.inf, np.inf, ga, ga), {"certified": False})
    if pure_cloud:
        M["pure_cloud_plain"] = lambda d: ((-np.inf, np.inf, ga, ga), {"certified": False})
    return M


def _take(d, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in d.items()}


def evaluate(d, theta, payload="nbytes", risks=None):
    """Metrics of theta on d. With risks, also per-risk miss rates (<key>:miss_unit / <key>:miss_event)."""
    r = metrics(d["s"], d["g"], d["y"], theta, d["unit"], d.get(payload), d.get("event"))
    for k in risks or []:
        yk = np.asarray(d[k]) > 0
        if yk.any():
            m = metrics(d["s"], d["g"], yk, theta, d["unit"], None, d.get("event"))
            for key in ("miss_unit", "miss_event", "miss_img"):
                if key in m:
                    r[f"{k}:{key}"] = m[key]
    return r


def risk_value(r, target, risks=None):
    """The controlled quantity: max over the constrained risks (or the any-positive risk)."""
    key = RISK_KEY[target]
    if not risks:
        return r[key]
    vals = [r[f"{k}:{key}"] for k in risks if f"{k}:{key}" in r]
    return max(vals) if vals else r[key]


def run_trials(data, n_trials=100, cal_frac=0.5, alpha=0.05, delta=0.1, costs=None, s_min=0.02, seed=0,
               log=print, target="frame", payload="nbytes", only=None, by=None, risks=None, **mkw):
    """Repeated random unit splits. by: optional key (e.g. 'source') for per-group test metrics."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trials):
        M = methods(alpha, delta, costs, s_min, target, payload, risks=risks, seed=seed + t, **mkw)
        if only:
            M = {k: v for k, v in M.items() if k in only}
        cm, tm = split_units(data["unit"], cal_frac, rng)
        cal, test = _take(data, cm), _take(data, tm)
        for name, f in M.items():
            theta, info = f(cal)
            tst = data_for(name, test)
            r = evaluate(tst, theta, payload, risks)
            r.update(method=name, trial=t, theta=theta, certified=info.get("certified"),
                     risk=risk_value(r, target, risks))
            if by:
                r["by"] = {str(v): evaluate(_take(tst, tst[by] == v), theta, payload, risks)
                           for v in np.unique(tst[by]) if tst["y"][tst[by] == v].any()}
            rows.append(r)
        if log and (t + 1) % max(1, n_trials // 5) == 0:
            log(f"  trial {t + 1}/{n_trials}")
    return rows


STAT_KEYS = ("miss_img", "miss_unit", "miss_event", "fa", "calls", "handoff", "bytes_per_frame")


def summarize(rows, alpha, target="frame"):
    out = {}
    for name in dict.fromkeys(r["method"] for r in rows):
        R = [r for r in rows if r["method"] == name]
        mu = np.array([r["risk"] for r in R])
        cert = np.array([bool(r["certified"]) for r in R])
        agg = {"n_trials": len(R), "certification_rate": float(cert.mean()),
               "risk_mean": float(mu.mean()), "risk_p2.5": float(np.quantile(mu, 0.025)),
               "risk_p97.5": float(np.quantile(mu, 0.975)),
               "cert_and_test_exceeds_alpha": float((cert & (mu > alpha)).mean()),
               "n_cert_and_exceeds": int((cert & (mu > alpha)).sum())}
        for k in STAT_KEYS:
            if k in R[0]:
                v = np.array([r[k] for r in R])
                agg[k + "_mean"] = float(v.mean())
        for k in R[0]:
            if ":" in k:
                agg[k + "_mean"] = float(np.mean([r[k] for r in R if k in r]))
        out[name] = agg
    return out


def shift_loso(data, alpha=0.05, delta=0.1, costs=None, s_min=0.02, key="source", target="frame",
               payload="nbytes", keep=("edge_LTT", "cascade_LTT", "cascade_LTT+human"), risks=None, **mkw):
    """Calibrate on all groups but one, test on the held-out group (exchangeability deliberately broken)."""
    res = {}
    M = methods(alpha, delta, costs, s_min, target, payload, risks=risks, **mkw)
    for gname in np.unique(data[key]):
        m = data[key] == gname
        cal, test = _take(data, ~m), _take(data, m)
        if test["y"].sum() == 0:
            continue
        for name in keep:
            theta, info = M[name](cal)
            r = evaluate(data_for(name, test), theta, payload, risks)
            r.update(theta=theta, certified=info.get("certified"), held_out=str(gname), method=name,
                     risk=risk_value(r, target, risks))
            res[f"{gname}/{name}"] = r
    return res


def format_table(summary, alpha, target="frame", extra_cols=()):
    cols = ["certification_rate", "risk_mean", "risk_p2.5", "risk_p97.5", "cert_and_test_exceeds_alpha",
            "fa_mean", "calls_mean", "handoff_mean", "bytes_per_frame_mean", *extra_cols]
    head = "| method | " + " | ".join(cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for name, a in summary.items():
        vals = []
        for c in cols:
            v = a.get(c, float("nan"))
            vals.append(f"{v:.0f}" if (c.startswith("bytes") or c.endswith("_ms_mean")) else f"{v:.4f}")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines.append(f"\nalpha = {alpha}; risk = empirical test {RISK_KEY[target]} (max over constrained risks). "
                 "cert_and_test_exceeds_alpha is a DIAGNOSTIC, not the LTT guarantee (see experiment.py).")
    return "\n".join(lines)
