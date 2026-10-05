"""Dev-only CEILING diagnostic: where do the cascade's misses come from, and can any veto fix pay off?

Run this BEFORE building / registering a method (lesson of RAVC: measure the ceiling first).

    python -m cascade.dev_diagnose --dataset DS --protocol P --det D --agent A --out DIR [--trials 50]

Reads only frozen-role `dev` labels. Reports, over repeated 50/50 unit splits (cascade_LTT calibrated on one
half exactly as run_cascade; metrics on the other half):

A  label-free unit / frame counts per source x role (splits.csv) -> what can ever be certified per source;
   n_min = units needed to certify any risk at (alpha, delta) even with zero misses.
B  miss attribution per source: missed positive events that were edge-silent vs vetoed by the VLM.
C  VLM veto behaviour on escalated frames, per source x inside/outside S: share of positives dismissed
   (harmful veto) and of negatives dismissed (useful veto).
D  what-if ceilings for S = small_smoke(area_max) at several area_max (0.02 is the value fixed a priori from
   SmokeBench / PyroNear2025, not from these data), with the calibrated theta kept fixed:
     no_veto_in_S : lambda = 1 inside S (the strongest ESVA setting)
     oracle       : vetoes removed only on positive frames (upper bound for ANY veto-side method)
   metrics: pooled risk, S risk, pyro risk, FA, calls, handoff.

GO / NO-GO rule for ESVA (fixed here, before the diagnostic is run):
   GO if, for area_max = 0.02, no_veto_in_S has mean pyro risk <= 0.05 AND mean pooled FA <= 0.90 * edge_LTT FA
   AND dev S has >= ceil(n_min / 0.35) positive smoke units (in an esva_dev trial the certify fold holds ~35% of
   the dev units: 50% calibration half x 70% certify part).
   Otherwise ESVA is not registered or run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.esva_dev import eval_metrics, load_dev, split_units, take  # noqa: E402
from cascade.evidence_veto import StratumSpec, discount, in_stratum, n_min  # noqa: E402
from cascade.policy import decide  # noqa: E402
from cascade.select import calibrate  # noqa: E402

AREAS = (0.005, 0.01, 0.02, 0.05, 0.10)
PRIMARY_AREA = 0.02


def role_counts(protocol):
    with open(Path(protocol) / "splits.csv", newline="") as f:
        R = list(csv.DictReader(f))
    units, frames = defaultdict(set), Counter()
    for r in R:
        k = (r.get("role", ""), r.get("source", ""))
        units[k].add(r["unit"])
        frames[k] += 1
    ev_units = defaultdict(set)
    for r in R:
        ev_units[(r.get("source", ""), r["event"])].add(r["unit"])
    span = Counter(k[0] for k, v in ev_units.items() if len(v) > 1)
    out = {f"{k[0]}/{k[1]}": {"units": len(units[k]), "frames": frames[k]} for k in sorted(units)}
    # label-free check of the nesting the event risk assumes: an event should lie inside one unit
    out["_events_spanning_units"] = {s: int(n) for s, n in span.items()}
    return out


def attribution(d, g, theta):
    """Missed positive events (unit, event) -> 'edge_silent' or 'vlm_veto'."""
    dec = decide(d["s"], g, theta)
    alarm = (dec == 1) | (dec == 2) | ((dec == 4) & (d["y"] > 0))
    out = defaultdict(Counter)
    pos = np.flatnonzero(d["y"] > 0)
    ev = defaultdict(list)
    for i in pos:
        ev[(d["unit"][i], d["event"][i])].append(i)
    for idx in ev.values():
        idx = np.array(idx)
        src = d["source"][idx[0]]
        out[src]["events"] += 1
        if alarm[idx].any():
            continue
        out[src]["vlm_veto" if (dec[idx] == 3).any() else "edge_silent"] += 1
    return out


def veto_rates(d, g, theta, inS):
    dec = decide(d["s"], g, theta)
    esc = dec >= 2
    res = defaultdict(Counter)
    for i in np.flatnonzero(esc):
        k = f"{d['source'][i]}|{'S' if inS[i] else 'notS'}"
        lab = "pos" if d["y"][i] > 0 else "neg"
        res[k][lab + "_esc"] += 1
        res[k][lab + "_vetoed"] += int(dec[i] == 3)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--dataset", "--protocol", "--det", "--agent", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    D, P = load_dev(a.dataset, a.protocol, a.det, a.agent)
    alpha, delta, target = float(P["alpha"]), float(P["delta"]), P["target"]
    costs = {k: float(v) for k, v in P["costs"].items() if k in ("calls", "handoff", "fa")}
    kw = dict(level="unit", opt_frac=float(P.get("select_fraction", 0.3)), s_min=float(P["s_min"]))
    nm = n_min(alpha, delta)
    S_masks = {A: in_stratum(StratumSpec("small_smoke", A), D["features"]) for A in AREAS}
    out = {"n_min": nm, "role_counts": role_counts(a.protocol),
           "dev_S": {str(A): {"frames": int(m.sum()),
                              "pos_units_smoke": int(len(np.unique(D["unit"][(D["y_smoke"] > 0) & m]))),
                              "pos_units_fire": int(len(np.unique(D["unit"][(D["y_fire"] > 0) & m]))),
                              "frames_by_source": dict(Counter(D["source"][m].tolist()))}
                     for A, m in S_masks.items()}}
    rng = np.random.default_rng(a.seed)
    acc = defaultdict(list)
    attr_tot, veto_tot = defaultdict(Counter), defaultdict(Counter)
    for t in range(a.trials):
        cm, tm = split_units(D["unit"], 0.5, rng)
        cal, test = take(D, cm), take(D, tm)
        ev = cal["event"] if target == "event" else None
        rl = {"fire": cal["y_fire"], "smoke": cal["y_smoke"]}
        th_c, ic = calibrate(cal["s"], cal["g"], cal["y"], cal["unit"], alpha, delta, None, costs, cloud=True,
                             handoff=False, fwer="pareto", seed=a.seed + t, event=ev, risk_labels=rl, **kw)
        th_e, _ = calibrate(cal["s"], cal["g"], cal["y"], cal["unit"], alpha, delta, None, costs, cloud=False,
                            handoff=False, fwer="pareto", seed=a.seed + t, event=ev, risk_labels=rl, **kw)
        pyro = test["source"] == "pyro_sdis"
        has_pyro = bool((test["y"][pyro] > 0).any())

        def rec(tag, g_eff, theta, inS):
            r = eval_metrics(test, g_eff, theta, target, inS)
            acc[tag + ":risk"].append(r["risk"])
            acc[tag + ":risk_S"].append(r["risk_S"])
            for k in ("fa", "calls", "handoff"):
                acc[f"{tag}:{k}"].append(r[k])
            if has_pyro:
                rp = eval_metrics(take(test, pyro), g_eff[pyro], theta, target, inS[pyro])
                acc[tag + ":pyro_risk"].append(rp["risk"])
                acc[tag + ":pyro_fa"].append(rp["fa"])

        inS0 = S_masks[PRIMARY_AREA][tm]
        rec("edge_LTT", test["g"], th_e, inS0)
        rec("cascade_LTT", test["g"], th_c, inS0)
        for A in AREAS:
            inS = S_masks[A][tm]
            rec(f"no_veto_in_S@{A}", discount(test["g"], inS, 1.0), th_c, inS)
        g_or = np.where((test["y"] > 0) & np.isfinite(test["g"]), np.maximum(test["g"], 1.0), test["g"])
        rec("oracle_no_false_veto", g_or, th_c, inS0)
        for src, c in attribution(test, test["g"], th_c).items():
            attr_tot[src].update(c)
        for k, c in veto_rates(test, test["g"], th_c, inS0).items():
            veto_tot[k].update(c)
        if (t + 1) % max(1, a.trials // 5) == 0:
            print(f"trial {t + 1}/{a.trials}", flush=True)

    def m(k):
        v = np.asarray(acc.get(k, []), float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    tags = ["edge_LTT", "cascade_LTT"] + [f"no_veto_in_S@{A}" for A in AREAS] + ["oracle_no_false_veto"]
    table = {t_: {k: m(f"{t_}:{k}") for k in ("risk", "risk_S", "pyro_risk", "fa", "pyro_fa", "calls", "handoff")}
             for t_ in tags}
    out["whatif"] = table
    out["miss_attribution"] = {s: dict(c) for s, c in attr_tot.items()}
    out["veto_rates"] = {k: dict(c) | {
        "pos_veto_rate": c["pos_vetoed"] / c["pos_esc"] if c["pos_esc"] else None,
        "neg_veto_rate": c["neg_vetoed"] / c["neg_esc"] if c["neg_esc"] else None} for k, c in sorted(veto_tot.items())}
    prim = table[f"no_veto_in_S@{PRIMARY_AREA}"]
    go = {"pyro_risk_le_0.05": bool(np.isfinite(prim["pyro_risk"]) and prim["pyro_risk"] <= 0.05),
          "fa_le_0.90_edge": bool(prim["fa"] <= 0.90 * table["edge_LTT"]["fa"]),
          "S_smoke_units_ge_nmin_over_0.35": bool(out["dev_S"][str(PRIMARY_AREA)]["pos_units_smoke"]
                                                  >= int(np.ceil(nm / 0.35)))}
    out["go_rule"] = {"checks": go, "GO": all(go.values())}

    L = ["# Dev ceiling diagnostic (before any new method)", "",
         f"n_min (units to certify at alpha={alpha}, delta={delta}) = {nm}", "",
         "## A. Units per role/source (label-free)", "", "| role/source | units | frames |", "|---|---:|---:|"]
    L += [f"| {k} | {v['units']} | {v['frames']} |" for k, v in out["role_counts"].items() if not k.startswith("_")]
    L += ["", f"Events spanning more than one unit (should be 0): {out['role_counts']['_events_spanning_units'] or 0}"]
    L += ["", "## Dev stratum S = small smoke box (area <= area_max, smoke >= fire)", "",
          "| area_max | frames | pos units smoke | pos units fire | frames by source |", "|---:|---:|---:|---:|---|"]
    L += [f"| {A} | {v['frames']} | {v['pos_units_smoke']} | {v['pos_units_fire']} | {v['frames_by_source']} |"
          for A, v in out["dev_S"].items()]
    L += ["", "## B. Missed positive events (test halves, summed over trials)", "",
          "| source | events | edge_silent | vlm_veto |", "|---|---:|---:|---:|"]
    L += [f"| {s} | {c.get('events', 0)} | {c.get('edge_silent', 0)} | {c.get('vlm_veto', 0)} |"
          for s, c in sorted(out["miss_attribution"].items())]
    L += ["", f"## C. VLM veto on escalated frames (S = area <= {PRIMARY_AREA})", "",
          "| source/stratum | pos esc | pos veto rate (harmful) | neg esc | neg veto rate (useful) |",
          "|---|---:|---:|---:|---:|"]
    for k, c in out["veto_rates"].items():
        pr, nr = c["pos_veto_rate"], c["neg_veto_rate"]
        L.append(f"| {k} | {c.get('pos_esc', 0)} | {'-' if pr is None else f'{pr:.3f}'} | {c.get('neg_esc', 0)} | "
                 f"{'-' if nr is None else f'{nr:.3f}'} |")
    L += ["", "## D. What-if ceilings (theta of cascade_LTT kept fixed; NOT certified numbers)", "",
          "| setting | risk | risk_S | pyro risk | FA | pyro FA | calls | handoff |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t_, r in table.items():
        L.append(f"| {t_} | " + " | ".join("nan" if not np.isfinite(r[k]) else f"{r[k]:.4f}"
                                          for k in ("risk", "risk_S", "pyro_risk", "fa", "pyro_fa", "calls",
                                                    "handoff")) + " |")
    L += ["", f"## GO / NO-GO for ESVA (rule fixed in the code header): **{'GO' if out['go_rule']['GO'] else 'NO-GO'}**",
          ""] + [f"- [{'x' if v else ' '}] {k}" for k, v in go.items()]
    md = "\n".join(L) + "\n"
    o = Path(a.out).expanduser()
    o.mkdir(parents=True, exist_ok=True)
    (o / "dev_diagnose.md").write_text(md)
    (o / "dev_diagnose.json").write_text(json.dumps(out, indent=1, default=float))
    print(md)
    print("->", o / "dev_diagnose.json")


if __name__ == "__main__":
    main()
