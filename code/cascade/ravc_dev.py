"""Dev-only evaluation of RAVC-LTT (overlay, crop and adaptive one-call routing), pre-registered.

    python -m cascade.ravc_dev register --protocol P --det D --overlay AO --crop AC --crop-bytes CB --out REG.json
    python -m cascade.ravc_dev run --dataset DS --protocol P --det D --overlay AO --crop AC --crop-bytes CB \
                                   --registration REG.json --out OUTDIR [--trials 100]

register : writes, once, the UTC time, sha256 of protocol.yaml, analysis_code_sha256, the protocol `ravc:` block,
           the acceptance rule below and sha256 of every input file. Refuses if REG.json exists.
run      : refuses unless every one of those hashes still matches, and refuses if OUTDIR/ravc_dev.json exists
           (one pre-registered run; a crashed run leaves no ravc_dev.json and may be repeated).

Only rows whose frozen role is `dev` are read; calibration / sealed labels are never read.
Byte price: protocol.yaml `ravc: {byte_cost: ...}` (required; no default in code). The main cascade objective
(`costs:`) is not changed by it.

Crop payload: nbytes of the exact crop the VLM saw (cascade.view_bytes), not dump_detector's 448-px nbytes_crop.

Two families are evaluated on the same repeated splits:
  no_human : overlay_LTT, crop_LTT, RAVC_LTT                   (cloud only, handoff=False)
  human    : overlay_LTT+human, crop_LTT+human, RAVC_LTT+human (operator tier, as cascade_LTT+human)
The DECISION family is `human`, because under the pre-registered costs 1/5/20 the operator tier is kept
(dev handoff > 0). The no_human family is reported as a diagnostic with the same checks.

The primary method of the paper is already fixed by view_rule_outcome.txt (edge_LTT). This report decides only
whether RAVC-LTT enters the paper as the method for the VLM cascade; it cannot change the primary method.
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

from cascade.adaptive_view import (RouterSpec, calibrate_adaptive_view, routed_arrays,  # noqa: E402
                                   top_box_features)
from cascade.dataio import frame_label, read_csv_by_uid, read_manifest  # noqa: E402
from cascade.policy import decide, metrics  # noqa: E402

VIEWS = {"overlay": [RouterSpec("overlay", "overlay")], "crop": [RouterSpec("crop", "crop")], "RAVC": None}
FAMILIES = {"no_human": False, "human": True}
DECISION_FAMILY = "human"


def mname(view, fam):
    return f"{view}_LTT" + ("+human" if FAMILIES[fam] else "")


METHODS = [(mname(v, f), v, f) for f in FAMILIES for v in VIEWS]

ACCEPTANCE = {
    "decision_family": DECISION_FAMILY,
    "pooled_risk_p97.5_le_alpha": "RAVC risk_p97.5 <= alpha",
    "certification_within_5pp": "RAVC certification_rate >= max(fixed views) - 0.05",
    "objective_improves_at_least_10pct": "RAVC objective_mean <= 0.90 * min(fixed views objective_mean)",
    "nondegenerate_router_at_least_90pct": "router kind not in {overlay, crop} in >= 90% of trials",
    "no_source_worse_by_more_than_2pp": "for every source with a finite mean risk in all three views: "
                                        "RAVC <= min(fixed views) + 0.02",
    "accept": "all five checks true in the decision family",
}


def _take(d, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in d.items()}


def _split_units(unit, frac, rng):
    u = np.unique(unit)
    a = set(rng.choice(u, max(1, int(round(frac * len(u)))), replace=False))
    m = np.array([x in a for x in unit])
    return m, ~m


def _meta(path):
    return json.loads(Path(path).expanduser().with_suffix(".meta.json").read_text())


def _sha(p):
    from cascade.make_protocol import sha256_file
    return sha256_file(p)


def _check_score_contract(det, overlay, crop, crop_bytes, s_min):
    om, cm = _meta(overlay), _meta(crop)
    problems = []
    if om.get("view") != "overlay" or cm.get("view") != "crop":
        problems.append(f"views must be overlay/crop, got {om.get('view')}/{cm.get('view')}")
    for k in ("model", "revision", "quant", "max_side", "s_min", "det_csv_sha256", "det_meta_sha256", "dry"):
        if om.get(k) != cm.get(k):
            problems.append(f"overlay/crop meta differs at {k}: {om.get(k)!r} vs {cm.get(k)!r}")
    if om.get("det_csv_sha256") != _sha(det):
        problems.append("agent meta det_csv_sha256 != sha256(--det)")
    if float(om.get("s_min", -1)) != float(s_min):
        problems.append(f"agent s_min={om.get('s_min')} but protocol s_min={s_min}")
    if om.get("dry") or cm.get("dry"):
        problems.append("dry agent scores are forbidden")
    bm = _meta(crop_bytes)
    if bm.get("view") != "crop" or int(bm.get("max_side", -1)) != int(cm.get("max_side", -2)):
        problems.append(f"crop-bytes view/max_side {bm.get('view')}/{bm.get('max_side')} != crop agent "
                        f"crop/{cm.get('max_side')}")
    if bm.get("agent_csv_sha256") != _sha(crop) or bm.get("det_csv_sha256") != _sha(det):
        problems.append("crop-bytes were computed from other agent / detector files")
    if bm.get("csv_sha256") != _sha(crop_bytes):
        problems.append("crop-bytes CSV changed after it was written")
    if problems:
        raise SystemExit("RAVC score contract failed:\n  " + "\n  ".join(problems))


def _protocol(protocol):
    import yaml
    P = yaml.safe_load((Path(protocol) / "protocol.yaml").read_text())
    rv = P.get("ravc")
    if not isinstance(rv, dict) or "byte_cost" not in rv:
        raise SystemExit("protocol.yaml has no `ravc: {byte_cost: ...}` block: pre-register the byte price first")
    return P


def load_dev(dataset, protocol, det_path, overlay_path, crop_path, crop_bytes_path):
    P = _protocol(protocol)
    s_min = float(P["s_min"])
    _check_score_contract(det_path, overlay_path, crop_path, crop_bytes_path, s_min)
    with open(Path(protocol) / "splits.csv", newline="") as f:
        split = {r["uid"]: r for r in csv.DictReader(f)}
    det, ao, ac = read_csv_by_uid(det_path), read_csv_by_uid(overlay_path), read_csv_by_uid(crop_path)
    cb = read_csv_by_uid(crop_bytes_path)
    rows = [r for r in read_manifest(dataset, ("val",)) if split.get(r["uid"], {}).get("role") == "dev"]
    out = {k: [] for k in ("uid", "source", "unit", "event", "y", "y_fire", "y_smoke", "s", "s_fire",
                           "s_smoke", "go", "gc", "bf", "bc", "boxes", "n_boxes_25")}
    missing = []
    for r in rows:
        uid, d = r["uid"], det.get(r["uid"])
        if d is None:
            missing.append(f"det:{uid}")
            continue
        need = float(d["s"]) >= s_min
        if need and (uid not in ao or uid not in ac or uid not in cb):
            missing.append(f"agent/bytes:{uid}")
            continue
        y, ys, yf = frame_label(r)
        sp = split[uid]
        vals = dict(uid=uid, source=r["source"], unit=sp["unit"], event=sp["event"], y=y, y_fire=yf,
                    y_smoke=ys, s=float(d["s"]), s_fire=float(d["s_fire"]), s_smoke=float(d["s_smoke"]),
                    go=float(ao[uid]["g"]) if need else np.nan, gc=float(ac[uid]["g"]) if need else np.nan,
                    bf=float(d["nbytes_frame"]), bc=float(cb[uid]["nbytes_view"]) if need else 0.0,
                    boxes=d.get("boxes", ""), n_boxes_25=int(d.get("n_boxes_25", 0)))
        for k, v in vals.items():
            out[k].append(v)
    if missing:
        raise SystemExit(f"fail-closed: {len(missing)} missing scores, e.g. {missing[:3]}")
    for k in out:
        out[k] = np.asarray(out[k])
    out["features"] = top_box_features(out.pop("boxes"), out.pop("n_boxes_25"), out["s_fire"], out["s_smoke"])
    return out, P


def _risk_metrics(d, theta, spec, target):
    """Pooled or per-source metrics. A risk whose label has no positive frame here is NaN (not an error)."""
    rr = routed_arrays(spec, d["features"], d["go"], d["gc"], d["bf"], d["bc"])
    key = "miss_event" if target == "event" else "miss_unit"
    base = metrics(d["s"], rr["g"], d["y"], theta, d["unit"], rr["nbytes"], d.get("event"))
    risks = {}
    for k in ("y_fire", "y_smoke"):
        yk = np.asarray(d[k]).astype(bool)
        risks[k] = float(metrics(d["s"], rr["g"], yk, theta, d["unit"], None, d.get("event"))[key]) \
            if yk.any() else float("nan")
    fin = [v for v in risks.values() if np.isfinite(v)]
    base.update(risk=max(fin) if fin else float("nan"), **risks)
    esc = decide(d["s"], rr["g"], theta) >= 2
    base["crop_calls"] = float((esc & rr["use_crop"]).mean())
    base["overlay_calls"] = float((esc & ~rr["use_crop"]).mean())
    return base


def _objective(r, costs, byte_cost):
    o = (costs.get("calls", 0.0) * r.get("calls", 0.0) + costs.get("handoff", 0.0) * r.get("handoff", 0.0)
         + costs.get("fa", 0.0) * r.get("fa", 0.0))
    return float(o + byte_cost * r.get("bytes_per_frame", 0.0)), float(o)


def run_trials(data, P, trials, seed, log=print):
    rng = np.random.default_rng(seed)
    rows = []
    costs = {k: float(v) for k, v in P["costs"].items() if k in ("calls", "handoff", "fa")}
    byte_cost = float(P["ravc"]["byte_cost"])
    cal_costs = {**costs, "bytes": byte_cost}
    for t in range(trials):
        cm, tm = _split_units(data["unit"], 0.5, rng)
        cal, test = _take(data, cm), _take(data, tm)
        cal["features"] = _take(data["features"], cm)
        test["features"] = _take(data["features"], tm)
        for name, view, fam in METHODS:
            theta, info = calibrate_adaptive_view(
                cal["s"], cal["go"], cal["gc"], cal["features"], cal["y"], cal["unit"],
                alpha=float(P["alpha"]), delta=float(P["delta"]), costs=cal_costs,
                nbytes_overlay=cal["bf"], nbytes_crop=cal["bc"], level="unit",
                opt_frac=float(P["select_fraction"]), seed=seed + t,
                event=cal["event"] if P["target"] == "event" else None,
                risk_labels={"fire": cal["y_fire"], "smoke": cal["y_smoke"]}, s_min=float(P["s_min"]),
                routers=VIEWS[view], handoff=FAMILIES[fam],
            )
            spec = RouterSpec(**info["router"])
            r = _risk_metrics(test, theta, spec, P["target"])
            r["objective"], r["objective_nobytes"] = _objective(r, costs, byte_cost)
            r.update(method=name, view=view, family=fam, trial=t, certified=bool(info["certified"]),
                     theta=[float(x) for x in theta], router=info["router"], p_value=info["p_value"])
            r["by_source"] = {}
            for src in np.unique(test["source"]):
                m = test["source"] == src
                if test["y"][m].any():
                    ds = _take(test, m)
                    ds["features"] = _take(test["features"], m)
                    r["by_source"][str(src)] = _risk_metrics(ds, theta, spec, P["target"])
            rows.append(r)
        if (t + 1) % max(1, trials // 10) == 0:
            log(f"trial {t + 1}/{trials}")
    return rows


def _nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def summarize(rows):
    out = {}
    for name, view, fam in METHODS:
        rr = [r for r in rows if r["method"] == name]
        if not rr:
            continue
        risk = np.array([r["risk"] for r in rr], float)
        rf = risk[np.isfinite(risk)]
        a = {"view": view, "family": fam, "n_trials": len(rr),
             "certification_rate": float(np.mean([r["certified"] for r in rr])),
             "risk_mean": float(rf.mean()) if len(rf) else float("nan"),
             "risk_p2.5": float(np.quantile(rf, .025)) if len(rf) else float("nan"),
             "risk_p97.5": float(np.quantile(rf, .975)) if len(rf) else float("nan")}
        for k in ("y_fire", "y_smoke", "fa", "calls", "handoff", "bytes_per_frame", "crop_calls", "overlay_calls",
                  "objective", "objective_nobytes"):
            a[k + "_mean"] = _nanmean([r.get(k, np.nan) for r in rr])
        sources = sorted(set().union(*(r["by_source"] for r in rr)))
        a["by_source_risk_mean"] = {src: _nanmean([r["by_source"][src]["risk"] for r in rr if src in r["by_source"]])
                                    for src in sources}
        a["by_source_fa_mean"] = {src: _nanmean([r["by_source"][src]["fa"] for r in rr if src in r["by_source"]])
                                  for src in sources}
        routers = {}
        for r in rr:
            routers[r["router"]["name"]] = routers.get(r["router"]["name"], 0) + 1
        a["router_frequency"] = dict(sorted(routers.items(), key=lambda kv: (-kv[1], kv[0])))
        a["nondegenerate_router_rate"] = float(np.mean([r["router"]["kind"] not in ("overlay", "crop") for r in rr]))
        out[name] = a
    return out


def acceptance(summary, P, fam):
    a = summary[mname("RAVC", fam)]
    fixed = [summary[mname("overlay", fam)], summary[mname("crop", fam)]]
    checks = {
        "pooled_risk_p97.5_le_alpha": bool(a["risk_p97.5"] <= float(P["alpha"])),
        "certification_within_5pp": bool(a["certification_rate"] >= max(x["certification_rate"] for x in fixed) - .05),
        "objective_improves_at_least_10pct": bool(a["objective_mean"] <= .90 * min(x["objective_mean"] for x in fixed)),
        "nondegenerate_router_at_least_90pct": bool(a["nondegenerate_router_rate"] >= .90),
    }
    sources = [s for s in sorted(a["by_source_risk_mean"])
               if np.isfinite(a["by_source_risk_mean"][s])
               and all(np.isfinite(x["by_source_risk_mean"].get(s, np.nan)) for x in fixed)]
    checks["no_source_worse_by_more_than_2pp"] = bool(all(
        a["by_source_risk_mean"][s] <= min(x["by_source_risk_mean"][s] for x in fixed) + .02 for s in sources))
    return {"family": fam, "accepted": all(checks.values()), "checks": checks, "sources_compared": sources}


def _f(v, nd=4):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def markdown(summary, P, decisions):
    cols = ("certification_rate", "risk_mean", "risk_p2.5", "risk_p97.5", "y_fire_mean", "y_smoke_mean",
            "fa_mean", "calls_mean", "handoff_mean", "crop_calls_mean", "bytes_per_frame_mean", "objective_mean",
            "objective_nobytes_mean", "nondegenerate_router_rate")
    dec = decisions[DECISION_FAMILY]
    L = ["# RAVC-LTT dev-only report", "",
         "Primary method of the paper: **edge_LTT** (fixed earlier by view_rule_outcome.txt). This report only "
         "decides whether RAVC-LTT enters as the method for the VLM cascade.", "",
         "No calibration or sealed-test label is read. Source is used only for diagnostics, never by the router.", "",
         f"alpha={P['alpha']}, delta={P['delta']}, target={P['target']}, costs={P['costs']}, "
         f"byte_cost={P['ravc']['byte_cost']} per byte", "",
         "| method | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, a in summary.items():
        L.append(f"| {name} | " + " | ".join(_f(a[k], 0 if k.startswith("bytes") else 4) for k in cols) + " |")
    L += ["", f"Pre-registered decision (family `{DECISION_FAMILY}`): "
              f"**{'ACCEPT RAVC' if dec['accepted'] else 'REJECT RAVC'}**", ""]
    for k, v in dec["checks"].items():
        L.append(f"- [{'x' if v else ' '}] {k}")
    d2 = decisions["no_human"]
    L += ["", f"Diagnostic family `no_human`: {'would accept' if d2['accepted'] else 'would reject'} "
              f"({', '.join(k for k, v in d2['checks'].items() if not v) or 'all checks pass'})", ""]
    for fam in FAMILIES:
        names = [mname(v, fam) for v in VIEWS]
        L += [f"### Mean risk / FA by source, family `{fam}` (diagnostic; no per-source guarantee)", "",
              "| source | " + " | ".join(f"{n} risk" for n in names) + " | " + " | ".join(f"{n} fa" for n in names)
              + " |", "|---|" + "---:|" * (2 * len(names))]
        sources = sorted(set().union(*(summary[n]["by_source_risk_mean"] for n in names)))
        for s in sources:
            L.append(f"| {s} | " + " | ".join(_f(summary[n]["by_source_risk_mean"].get(s, np.nan)) for n in names)
                     + " | " + " | ".join(_f(summary[n]["by_source_fa_mean"].get(s, np.nan)) for n in names) + " |")
        L.append("")
    L += ["Router frequencies are in ravc_dev.json. Repeated-split quantiles are dev robustness diagnostics, "
          "not the final LTT guarantee. objective_nobytes = the main-protocol objective (calls/handoff/fa only)."]
    return "\n".join(L) + "\n"


def _inputs(a):
    d = {}
    for k in ("det", "overlay", "crop", "crop_bytes"):
        p = Path(getattr(a, k)).expanduser()
        d[k] = {"path": str(p), "sha256": _sha(p), "meta_sha256": _sha(p.with_suffix(".meta.json"))}
    return d


def _state(a):
    from cascade.make_protocol import analysis_code
    P = _protocol(a.protocol)
    return {"protocol_sha256": _sha(Path(a.protocol) / "protocol.yaml"),
            "splits_sha256": _sha(Path(a.protocol) / "splits.csv"),
            "analysis_code_sha256": analysis_code()[0], "ravc": P["ravc"], "costs": P["costs"],
            "acceptance": ACCEPTANCE, "methods": [m[0] for m in METHODS], "inputs": _inputs(a)}


def cmd_register(a):
    out = Path(a.out).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists: a registration is written once")
    st = _state(a)
    st = {"registered_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **st}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(out)
    print(json.dumps({k: st[k] for k in ("registered_utc", "protocol_sha256", "analysis_code_sha256", "ravc")},
                     indent=1))
    print("->", out)


def cmd_run(a):
    reg = json.loads(Path(a.registration).expanduser().read_text())
    st = _state(a)
    bad = [k for k in st if reg.get(k) != st[k]]
    if bad:
        raise SystemExit(f"registration mismatch at {bad}: code, protocol or inputs changed after registration")
    out = Path(a.out).expanduser()
    if (out / "ravc_dev.json").exists():
        raise SystemExit(f"{out / 'ravc_dev.json'} exists: the pre-registered run was already made")
    data, P = load_dev(a.dataset, a.protocol, a.det, a.overlay, a.crop, a.crop_bytes)
    print(f"dev frames: {len(data['s'])}, units: {len(np.unique(data['unit']))}", flush=True)
    rows = run_trials(data, P, a.trials, a.seed, log=lambda m: print(m, flush=True))
    summary = summarize(rows)
    decisions = {fam: acceptance(summary, P, fam) for fam in FAMILIES}
    out.mkdir(parents=True, exist_ok=True)
    md = markdown(summary, P, decisions)
    res = {"registration": reg, "trials": a.trials, "seed": a.seed, "summary": summary,
           "acceptance": decisions[DECISION_FAMILY], "acceptance_by_family": decisions, "rows": rows}
    (out / "ravc_dev.md").write_text(md)
    tmp = out / "ravc_dev.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(out / "ravc_dev.json")
    print(md)
    print("->", out / "ravc_dev.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("register", "run"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", required=True)
        p.add_argument("--det", required=True)
        p.add_argument("--overlay", required=True)
        p.add_argument("--crop", required=True)
        p.add_argument("--crop-bytes", required=True)
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--dataset", required=True)
            p.add_argument("--registration", required=True)
            p.add_argument("--trials", type=int, default=100)
            p.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args(argv)
    (cmd_register if a.cmd == "register" else cmd_run)(a)


if __name__ == "__main__":
    main()
