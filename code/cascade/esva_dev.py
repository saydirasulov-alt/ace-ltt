"""Dev-only, pre-registered evaluation of ESVA-LTT against edge_LTT and cascade_LTT (same splits, same data).

    python -m cascade.esva_dev register --protocol P --det D --agent A --out REG.json
    python -m cascade.esva_dev run --dataset DS --protocol P --det D --agent A --registration REG.json --out DIR

protocol.yaml must contain (written before registration; never changed afterwards):
    esva: {stratum: {kind: small_smoke, area_max: 0.02}, lambdas: [0.0, 0.1, ..., 1.0]}
register writes once: UTC time, sha256 of protocol.yaml / splits.csv / inputs, analysis_code_sha256, the esva
block and the acceptance rule. run refuses on any mismatch and refuses a second run (ravc_dev semantics).

Methods (repeated 50/50 unit splits of the dev units; calibration on one half, metrics on the other):
    edge_LTT, cascade_LTT, cascade_LTT+human      (select.calibrate, as run_cascade)
    ESVA_LTT, ESVA_LTT+human                      (evidence_veto.calibrate_esva)
Only rows whose frozen role is `dev` are read.

Acceptance (decision method ESVA_LTT+human; all must hold):
    1 pooled risk_p97.5 <= alpha
    2 certification_rate >= 0.90
    3 stratum risk (smoke|S: smoke-event miss inside S, the constrained stratum risk) p97.5 <= alpha
    4 Pyro-SDIS mean risk <= 0.05            (the view_rule condition that rejected every VLM cascade so far;
                                               Pyro has ~2 dev units, so this is a noisy but pre-registered check)
    5 objective_mean <= 0.90 * edge_LTT objective_mean   (must beat the current primary method by >= 10%)
ACCEPT -> ESVA_LTT+human becomes the proposed method (edge_LTT stays the main baseline);
REJECT -> edge_LTT stays primary and ESVA is reported as a pre-registered negative result.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.dataio import frame_label, read_csv_by_uid, read_manifest  # noqa: E402
from cascade.evidence_veto import (StratumSpec, calibrate_esva, discount, in_stratum,  # noqa: E402
                                   n_min, smoke_features)
from cascade.policy import metrics  # noqa: E402
from cascade.select import calibrate  # noqa: E402

METHODS = ("edge_LTT", "cascade_LTT", "cascade_LTT+human", "ESVA_LTT", "ESVA_LTT+human")
DECISION = "ESVA_LTT+human"
ACCEPTANCE = {
    "decision_method": DECISION,
    "pooled_risk_p97.5_le_alpha": "risk_p97.5 <= alpha",
    "certification_rate_ge_0.90": "certification_rate >= 0.90",
    "stratum_risk_p97.5_le_alpha": "smoke|S p97.5 <= alpha (trials whose test half has no S positive are skipped)",
    "pyro_risk_mean_le_0.05": "by-source pyro_sdis risk_mean <= 0.05",
    "objective_le_0.90_edge_LTT": "objective_mean <= 0.90 * edge_LTT objective_mean",
    "accept": "all five true",
}


def _sha(p):
    from cascade.make_protocol import sha256_file
    return sha256_file(p)


def _meta(p):
    return json.loads(Path(p).expanduser().with_suffix(".meta.json").read_text())


def read_protocol(protocol, need_esva=True):
    import yaml
    P = yaml.safe_load((Path(protocol) / "protocol.yaml").read_text())
    E = P.get("esva")
    if need_esva and (not isinstance(E, dict) or "stratum" not in E):
        raise SystemExit("protocol.yaml has no `esva: {stratum: {...}, lambdas: [...]}` block: register it first")
    return P


def esva_cfg(P):
    E = P.get("esva") or {}
    st = StratumSpec(**E.get("stratum", {}))
    lam = [float(x) for x in E.get("lambdas", [round(0.1 * i, 1) for i in range(11)])]
    return st, lam


def load_dev(dataset, protocol, det_path, agent_path):
    """Dev rows only (frozen role 'dev'); labels read only for those rows. Features from the detector boxes."""
    P = read_protocol(protocol, need_esva=False)
    s_min = float(P["s_min"])
    am = _meta(agent_path)
    problems = []
    if am.get("dry"):
        problems.append("dry agent scores are forbidden")
    if float(am.get("s_min", -1)) != s_min:
        problems.append(f"agent s_min {am.get('s_min')} != protocol s_min {s_min}")
    if am.get("det_csv_sha256") != _sha(det_path):
        problems.append("agent meta det_csv_sha256 != sha256(--det)")
    if problems:
        raise SystemExit("score contract failed:\n  " + "\n  ".join(problems))
    with open(Path(protocol) / "splits.csv", newline="") as f:
        split = {r["uid"]: r for r in csv.DictReader(f)}
    det, ag = read_csv_by_uid(det_path), read_csv_by_uid(agent_path)
    rows = [r for r in read_manifest(dataset, ("val",)) if split.get(r["uid"], {}).get("role") == "dev"]
    cols = ("uid", "source", "unit", "event", "y", "y_fire", "y_smoke", "s", "s_fire", "s_smoke", "g", "boxes",
            "stem")
    out = {k: [] for k in cols}
    missing = []
    for r in rows:
        u, d = r["uid"], det.get(r["uid"])
        if d is None or (float(d["s"]) >= s_min and u not in ag):
            missing.append(u)
            continue
        y, ys, yf = frame_label(r)
        sp = split[u]
        vals = dict(uid=u, source=r["source"], unit=sp["unit"], event=sp["event"], y=y, y_fire=yf, y_smoke=ys,
                    s=float(d["s"]), s_fire=float(d["s_fire"]), s_smoke=float(d["s_smoke"]),
                    g=float(ag[u]["g"]) if u in ag else np.nan, boxes=d.get("boxes", ""),
                    stem=Path(r["image"]).stem.split("__", 1)[-1])
        for k, v in vals.items():
            out[k].append(v)
    if missing:
        raise SystemExit(f"fail-closed: {len(missing)} dev frames without detector/agent score, e.g. {missing[:3]}")
    D = {k: np.asarray(v) for k, v in out.items()}
    for k in ("y", "y_fire", "y_smoke"):
        D[k] = (D[k].astype(float) > 0).astype(int)
    D["features"] = smoke_features(D.pop("boxes"), D["s_fire"], D["s_smoke"])
    return D, P


def take(d, m):
    out = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in d.items()}
    if "features" in d:
        out["features"] = {k: v[m] for k, v in d["features"].items()}
    return out


def split_units(unit, frac, rng):
    u = np.unique(unit)
    a = set(rng.choice(u, max(1, int(round(frac * len(u)))), replace=False))
    m = np.array([x in a for x in unit])
    return m, ~m


def eval_metrics(d, g_eff, theta, target, inS):
    """Pooled + per-risk + stratum risks; NaN where a label has no positive."""
    key = "miss_event" if target == "event" else "miss_unit"
    ev = d["event"] if target == "event" else None
    r = metrics(d["s"], g_eff, d["y"], theta, d["unit"], None, ev)
    for name, lab in (("fire", d["y_fire"]), ("smoke", d["y_smoke"])):
        for suf, m in (("", np.ones(len(lab), bool)), ("|S", inS)):
            yk = np.asarray(lab).astype(bool) & m
            r[name + suf] = float(metrics(d["s"], g_eff, yk, theta, d["unit"], None, ev)[key]) if yk.any() \
                else float("nan")
    fin = [r[k] for k in ("fire", "smoke") if np.isfinite(r[k])]
    finS = [r[k] for k in ("smoke|S",) if np.isfinite(r[k])]          # = evidence_veto.STRATUM_RISKS
    r["risk"] = max(fin) if fin else float("nan")
    r["risk_S"] = max(finS) if finS else float("nan")
    return r


def objective(r, costs):
    return float(sum(float(costs.get(k, 0.0)) * r.get(k, 0.0) for k in ("calls", "handoff", "fa")))


def run_trials(D, P, trials, seed, log=print):
    st, lam = esva_cfg(P)
    alpha, delta, target = float(P["alpha"]), float(P["delta"]), P["target"]
    costs = {k: float(v) for k, v in P["costs"].items() if k in ("calls", "handoff", "fa")}
    kw = dict(level="unit", opt_frac=float(P.get("select_fraction", 0.3)), s_min=float(P["s_min"]))
    inS_all = in_stratum(st, D["features"])
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(trials):
        cm, tm = split_units(D["unit"], 0.5, rng)
        cal, test = take(D, cm), take(D, tm)
        inS_c, inS_t = inS_all[cm], inS_all[tm]
        ev = cal["event"] if target == "event" else None
        rl = {"fire": cal["y_fire"], "smoke": cal["y_smoke"]}
        for name in METHODS:
            lam_used = 0.0
            if name.startswith("ESVA"):
                theta, info = calibrate_esva(cal["s"], cal["g"], cal["features"], cal["y"], cal["unit"],
                                             stratum=st, lambdas=lam, alpha=alpha, delta=delta, costs=costs,
                                             seed=seed + t, event=ev, risk_labels=rl,
                                             handoff=name.endswith("+human"), **kw)
                lam_used = info["lambda"]
            else:
                theta, info = calibrate(cal["s"], cal["g"], cal["y"], cal["unit"], alpha, delta, None, costs,
                                        cloud=name != "edge_LTT", handoff=name.endswith("+human"),
                                        fwer="pareto", seed=seed + t, event=ev, risk_labels=rl, **kw)
            g_t = discount(test["g"], inS_t, lam_used)
            r = eval_metrics(test, g_t, theta, target, inS_t)
            r.update(method=name, trial=t, certified=bool(info.get("certified")), theta=[float(x) for x in theta],
                     lam=lam_used, objective=objective(r, costs))
            r["by_source"] = {}
            for src in np.unique(test["source"]):
                m = test["source"] == src
                if test["y"][m].any():
                    rs = eval_metrics(take(test, m), g_t[m], theta, target, inS_t[m])
                    r["by_source"][str(src)] = {k: rs[k] for k in ("risk", "risk_S", "fa", "calls", "handoff")}
            rows.append(r)
        if (t + 1) % max(1, trials // 10) == 0:
            log(f"trial {t + 1}/{trials}")
    return rows


def _nm(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _q(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float("nan")


def summarize(rows):
    S = {}
    for name in METHODS:
        rr = [r for r in rows if r["method"] == name]
        if not rr:
            continue
        a = {"n_trials": len(rr), "certification_rate": float(np.mean([r["certified"] for r in rr]))}
        for k in ("risk", "risk_S"):
            v = [r[k] for r in rr]
            a[k + "_mean"], a[k + "_p97.5"] = _nm(v), _q(v, .975)
        for k in ("fire", "smoke", "fire|S", "smoke|S", "fa", "calls", "handoff", "objective", "lam"):
            a[k + "_mean"] = _nm([r.get(k, np.nan) for r in rr])
        srcs = sorted(set().union(*(r["by_source"] for r in rr)))
        a["by_source"] = {s: {k: _nm([r["by_source"][s][k] for r in rr if s in r["by_source"]])
                              for k in ("risk", "risk_S", "fa", "calls", "handoff")} | {
                              "trials": sum(s in r["by_source"] for r in rr)} for s in srcs}
        S[name] = a
    return S


def acceptance(S, P):
    a, e = S[DECISION], S["edge_LTT"]
    pyro = a["by_source"].get("pyro_sdis", {}).get("risk", float("nan"))
    checks = {
        "pooled_risk_p97.5_le_alpha": bool(a["risk_p97.5"] <= float(P["alpha"])),
        "certification_rate_ge_0.90": bool(a["certification_rate"] >= 0.90),
        "stratum_risk_p97.5_le_alpha": bool(a["risk_S_p97.5"] <= float(P["alpha"])),
        "pyro_risk_mean_le_0.05": bool(np.isfinite(pyro) and pyro <= 0.05),
        "objective_le_0.90_edge_LTT": bool(a["objective_mean"] <= 0.90 * e["objective_mean"]),
    }
    return {"method": DECISION, "accepted": all(checks.values()), "checks": checks}


def _f(v, nd=4):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def markdown(S, P, dec, cert):
    st, lam = esva_cfg(P)
    cols = ("certification_rate", "risk_mean", "risk_p97.5", "risk_S_mean", "risk_S_p97.5", "fa_mean",
            "calls_mean", "handoff_mean", "objective_mean", "lam_mean")
    L = ["# ESVA-LTT dev-only report (pre-registered)", "",
         f"stratum S = {st.to_dict()}, lambdas = {lam}; alpha={P['alpha']}, delta={P['delta']}, "
         f"target={P['target']}, costs={P['costs']}", "",
         f"Dev certifiability of S (all dev units; a certify fold has ~35% of them): {cert}", "",
         "| method | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for n, a in S.items():
        L.append(f"| {n} | " + " | ".join(_f(a[c]) for c in cols) + " |")
    L += ["", f"Pre-registered decision ({DECISION}): **{'ACCEPT' if dec['accepted'] else 'REJECT'}**", ""]
    L += [f"- [{'x' if v else ' '}] {k}" for k, v in dec["checks"].items()]
    srcs = sorted(set().union(*(a["by_source"] for a in S.values())))
    L += ["", "### Mean risk / FA by source (diagnostic; no per-source guarantee)", "",
          "| source | " + " | ".join(f"{n} risk" for n in S) + " | " + " | ".join(f"{n} fa" for n in S) + " |",
          "|---|" + "---:|" * (2 * len(S))]
    for s in srcs:
        L.append(f"| {s} | " + " | ".join(_f(S[n]["by_source"].get(s, {}).get("risk", np.nan)) for n in S) + " | "
                 + " | ".join(_f(S[n]["by_source"].get(s, {}).get("fa", np.nan)) for n in S) + " |")
    L += ["", "Repeated-split quantiles are dev diagnostics, not the final LTT guarantee."]
    return "\n".join(L) + "\n"


def _state(a):
    from cascade.make_protocol import analysis_code
    P = read_protocol(a.protocol)
    inp = {}
    for k in ("det", "agent"):
        p = Path(getattr(a, k)).expanduser()
        inp[k] = {"path": str(p), "sha256": _sha(p), "meta_sha256": _sha(p.with_suffix(".meta.json"))}
    return {"protocol_sha256": _sha(Path(a.protocol) / "protocol.yaml"),
            "splits_sha256": _sha(Path(a.protocol) / "splits.csv"), "analysis_code_sha256": analysis_code()[0],
            "esva": P["esva"], "costs": P["costs"], "acceptance": ACCEPTANCE, "methods": list(METHODS),
            "inputs": inp}


def cmd_register(a):
    out = Path(a.out).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists: a registration is written once")
    st = {"registered_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **_state(a)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(out)
    print(json.dumps({k: st[k] for k in ("registered_utc", "protocol_sha256", "analysis_code_sha256", "esva")},
                     indent=1))
    print("->", out)


def cmd_run(a):
    reg = json.loads(Path(a.registration).expanduser().read_text())
    st = _state(a)
    bad = [k for k in st if reg.get(k) != st[k]]
    if bad:
        raise SystemExit(f"registration mismatch at {bad}: code, protocol or inputs changed after registration")
    out = Path(a.out).expanduser()
    if (out / "esva_dev.json").exists():
        raise SystemExit(f"{out / 'esva_dev.json'} exists: the pre-registered run was already made")
    D, P = load_dev(a.dataset, a.protocol, a.det, a.agent)
    stc, _ = esva_cfg(P)
    inS = in_stratum(stc, D["features"])
    cert = {"n_min": n_min(float(P["alpha"]), float(P["delta"])),
            "S_frames": int(inS.sum()),
            "S_positive_units": {k: int(len(np.unique(D["unit"][(D[k] > 0) & inS]))) for k in ("y_fire", "y_smoke")}}
    print(f"dev frames {len(D['s'])}, units {len(np.unique(D['unit']))}; {cert}", flush=True)
    rows = run_trials(D, P, a.trials, a.seed, log=lambda m: print(m, flush=True))
    S = summarize(rows)
    dec = acceptance(S, P)
    md = markdown(S, P, dec, cert)
    out.mkdir(parents=True, exist_ok=True)
    (out / "esva_dev.md").write_text(md)
    tmp = out / "esva_dev.json.tmp"
    tmp.write_text(json.dumps({"registration": reg, "trials": a.trials, "seed": a.seed, "certifiability": cert,
                               "summary": S, "acceptance": dec, "rows": rows}, indent=1, default=float))
    tmp.replace(out / "esva_dev.json")
    print(md)
    print("->", out / "esva_dev.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("register", "run"):
        p = sub.add_parser(name)
        for k in ("--protocol", "--det", "--agent", "--out"):
            p.add_argument(k, required=True)
        if name == "run":
            p.add_argument("--dataset", required=True)
            p.add_argument("--registration", required=True)
            p.add_argument("--trials", type=int, default=100)
            p.add_argument("--seed", type=int, default=20260922)
    a = ap.parse_args(argv)
    (cmd_register if a.cmd == "register" else cmd_run)(a)


if __name__ == "__main__":
    main()
