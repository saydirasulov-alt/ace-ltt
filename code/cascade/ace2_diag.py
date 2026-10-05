"""Paired POST-REGISTRATION DIAGNOSTIC for ACE-LTT Study 2: edge-only vs the registered selected cascade.

    python -m cascade.ace2_diag --dataset DS --protocol P --det DET.csv \
        --agent crop=... --agent overlay=... --registration REG2.json --run RUNDIR [--policy NAME]

This module is DESCRIPTIVE. It is deliberately NOT part of ANALYSIS_CODE, so it cannot change any code hash
a registration pinned, and it is structurally incapable of certifying anything:

  * no p-value is computed, no boundary is compared against, no GO / NO-GO rule is applied;
  * no candidate is selected, re-ordered, added or removed;
  * the policy it reports on is read from the completed run, not chosen here.

What it does: put the edge-only policy and the registered selected cascade on EXACTLY the same footing -
the same certify units, the same (t_low, t_high), the same detector scores - and report both, with raw
numerators and denominators, plus the PAIRED per-unit differences.

Edge-only is the same object with the uncertain band collapsed (t_low = t_high): every frame is either
silent or an edge alarm, nothing escalates, so cloud calls and hand-offs are zero by construction. That is
what makes the comparison same-threshold rather than same-name.

Any interval printed here is a descriptive cluster (unit) bootstrap on a policy that was chosen after
seeing the certify fold. It is POST-SELECTION and is not a confidence statement about the selected policy's
risk. The certified guarantee of the run is unaffected by anything in this file, in either direction.

Context, not the comparison: the registered dev probe measured `edge_LTT` at FA 0.0505 with zero calls.
That was a different fold and different thresholds, so it is reported as background only; the same-threshold
edge-only column below is the one to read.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .ace2_register import DECLARATION, load_all
from .ace_register import certify_mask
from .evidence_veto import StratumSpec, discount, in_stratum
from .policy import decide, unit_losses

RISKS = ("fire", "smoke")


def _counts(s, gv, y, theta, unit, event):
    """Unit-averaged event-miss risk plus the raw event counts behind it."""
    u, L = unit_losses(s, gv, y.astype(bool), theta, unit, event)
    d = decide(s, gv, theta)
    caught_frame = np.isin(d, (1, 2, 4)) & y.astype(bool)
    ev = np.asarray(event)[y.astype(bool)]
    e_ids, inv = np.unique(ev, return_inverse=True)
    caught = np.zeros(len(e_ids), bool)
    np.logical_or.at(caught, inv, caught_frame[y.astype(bool)])
    return {"risk_unit_averaged": float(L.mean()) if len(L) else 0.0, "n_units": int(len(L)),
            "events_missed": int((~caught).sum()), "events_total": int(len(e_ids)),
            "event_miss_unweighted": float((~caught).mean()) if len(e_ids) else 0.0}, u, L


def _frame_stats(s, gv, y, theta):
    d = decide(s, gv, theta)
    yb = y.astype(bool)
    alarm = np.isin(d, (1, 2))
    return {"fa_numerator": int(alarm[~yb].sum()), "fa_denominator": int((~yb).sum()),
            "fa": float(alarm[~yb].mean()) if (~yb).any() else 0.0,
            "calls_numerator": int((d >= 2).sum()), "calls_denominator": int(len(d)),
            "calls": float((d >= 2).mean()),
            "handoff_numerator": int((d == 4).sum()), "handoff": float((d == 4).mean())}


def _boot_diff(dc, de, seed=20260924, reps=5000, lo=5, hi=95):
    """Descriptive cluster bootstrap of the paired per-unit difference (cascade - edge)."""
    d = np.asarray(dc, float) - np.asarray(de, float)
    if not len(d):
        return {"mean": 0.0, "interval": [0.0, 0.0], "n_units": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    m = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "interval": [float(np.percentile(m, lo)), float(np.percentile(m, hi))],
            "n_units": int(len(d)), "n_units_worse_under_cascade": int((d > 0).sum()),
            "n_units_better_under_cascade": int((d < 0).sum()), "reps": reps,
            "kind": "descriptive cluster bootstrap, 90% percentile interval, POST-SELECTION"}


def run(a, log=print):
    from .make_protocol import sha256_file
    rec = json.loads(Path(a.registration).expanduser().read_text())
    rep_run = json.loads((Path(a.run).expanduser() / "ace2_dev.json").read_text())
    op = (rep_run.get("operating_point") or {}).get("policy") or {}
    name = a.policy or op.get("name")
    if not name:
        raise SystemExit("the run certified nothing and no --policy was given: there is nothing to compare")
    chosen = next((p for p in rec["preconditions"]["frozen_family"]["policies"] if p["name"] == name), None)
    if chosen is None:
        raise SystemExit(f"policy {name!r} is not in the registered family")
    log(f"comparing edge-only vs the registered selected cascade {name!r} "
        f"(run {rep_run['run_utc']}, registration {rec['registered_utc']})")

    agents = dict(x.split("=", 1) for x in a.agent)
    D, P = load_all(a.dataset, a.protocol, a.det, agents)
    spec = StratumSpec(DECLARATION["veto"]["stratum"]["kind"], DECLARATION["veto"]["stratum"]["area_max"])
    inS = in_stratum(spec, D["features"])
    cert = certify_mask(D["unit"], float(P.get("select_fraction", 0.3)), DECLARATION["split"]["seed"])

    tl, th, b, A = chosen["t_low"], chosen["t_high"], chosen["b"], chosen["A"]
    lam = chosen["veto"]["S"]
    s = D["s"][cert]
    gv = discount(D[f"g_{chosen['acq']}"][cert], inS[cert], lam)
    unit, event = D["unit"][cert], D["event"][cert]
    # edge-only: the SAME object with the uncertain band collapsed; nothing escalates
    theta_casc, theta_edge = (tl, th, b, A), (th, th, b, A)

    out = {"kind": "paired post-registration diagnostic; descriptive only, no p-value, no decision rule",
           "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "registration": {"utc": rec["registered_utc"], "sha256": sha256_file(a.registration),
                            "study": rec["study"], "spec_version": rec["spec_version"]},
           "run_utc": rep_run["run_utc"], "policy": chosen,
           "thresholds": {"t_low": tl, "t_high": th, "b": b, "A": A, "lambda_S": lam,
                          "edge_only": "same (t_high); t_low collapsed to t_high so nothing escalates"},
           "certify_units": int(len(set(np.asarray(unit).tolist()))), "risks": {}, "frames": {},
           "paired_unit_differences": {},
           "context_not_the_comparison": ("the registered dev probe measured edge_LTT at FA 0.0505 with "
                                          "zero calls, on a different fold and different thresholds; the "
                                          "same-threshold edge-only column here is the comparison")}
    for k in RISKS:
        y = D[f"y_{k}"][cert].astype(int)
        cc, uc, Lc = _counts(s, gv, y, theta_casc, unit, event)
        ce, ue, Le = _counts(s, gv, y, theta_edge, unit, event)
        if not np.array_equal(uc, ue):
            raise SystemExit(f"{k}: the two policies were scored over different units; not paired")
        out["risks"][k] = {"cascade": cc, "edge_only": ce,
                           "difference_unit_averaged": cc["risk_unit_averaged"] - ce["risk_unit_averaged"]}
        out["paired_unit_differences"][k] = _boot_diff(Lc, Le)
    y_all = D["y"][cert].astype(int)
    out["frames"] = {"cascade": _frame_stats(s, gv, y_all, theta_casc),
                     "edge_only": _frame_stats(s, gv, y_all, theta_edge)}
    for f in ("fa", "calls", "handoff"):
        out["frames"][f"difference_{f}"] = out["frames"]["cascade"][f] - out["frames"]["edge_only"][f]

    d = Path(a.run).expanduser()
    (d / "ace2_diag.json").write_text(json.dumps(out, indent=1))
    md = report_md(out)
    (d / "ace2_diag.md").write_text(md)
    log("\n" + md)
    log(f"-> {d / 'ace2_diag.json'}")
    return out


def report_md(o):
    L = [f"# Study 2 — paired post-registration diagnostic ({o['utc']})", "",
         f"**{o['kind']}**", "",
         f"policy `{o['policy']['name']}` from registration {o['registration']['utc']} "
         f"({o['registration']['spec_version']}), run {o['run_utc']}",
         f"same certify units ({o['certify_units']}), same t_high = {o['thresholds']['t_high']:.4f}, "
         f"same detector. Edge-only = {o['thresholds']['edge_only']}.", "",
         "## Event-miss risk (unit averaged) with raw event counts", "",
         "| risk | cascade | edge-only | difference | cascade missed / total | edge missed / total |",
         "|---|---|---|---|---|---|"]
    for k in RISKS:
        r = o["risks"][k]
        c, e = r["cascade"], r["edge_only"]
        L.append(f"| {k} | {c['risk_unit_averaged']:.4f} | {e['risk_unit_averaged']:.4f} | "
                 f"{r['difference_unit_averaged']:+.4f} | {c['events_missed']} / {c['events_total']} | "
                 f"{e['events_missed']} / {e['events_total']} |")
    L += ["", "## Frame-level rates with raw numerators", "",
          "| quantity | cascade | edge-only | difference |", "|---|---|---|---|"]
    c, e = o["frames"]["cascade"], o["frames"]["edge_only"]
    L.append(f"| false alarm | {c['fa']:.4f} ({c['fa_numerator']}/{c['fa_denominator']}) | "
             f"{e['fa']:.4f} ({e['fa_numerator']}/{e['fa_denominator']}) | "
             f"{o['frames']['difference_fa']:+.4f} |")
    L.append(f"| cloud calls | {c['calls']:.4f} ({c['calls_numerator']}/{c['calls_denominator']}) | "
             f"{e['calls']:.4f} ({e['calls_numerator']}/{e['calls_denominator']}) | "
             f"{o['frames']['difference_calls']:+.4f} |")
    L.append(f"| hand-off | {c['handoff']:.4f} ({c['handoff_numerator']}) | "
             f"{e['handoff']:.4f} ({e['handoff_numerator']}) | {o['frames']['difference_handoff']:+.4f} |")
    L += ["", "## Paired per-unit differences (cascade − edge-only)", "",
          "| risk | mean | 90% interval | units worse | units better | units |",
          "|---|---|---|---|---|---|"]
    for k in RISKS:
        b = o["paired_unit_differences"][k]
        L.append(f"| {k} | {b['mean']:+.5f} | [{b['interval'][0]:+.5f}, {b['interval'][1]:+.5f}] | "
                 f"{b['n_units_worse_under_cascade']} | {b['n_units_better_under_cascade']} | "
                 f"{b['n_units']} |")
    L += ["", "*Intervals are a descriptive cluster (unit) bootstrap on a policy chosen after the certify "
              "fold was seen. They are POST-SELECTION: not a confidence statement about its risk, and not a "
              "test. No p-value and no decision rule appear in this diagnostic.*", "",
          f"*{o['context_not_the_comparison']}*", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--agent", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--registration", required=True)
    ap.add_argument("--run", required=True, help="the completed run directory (holds ace2_dev.json)")
    ap.add_argument("--policy", default=None, help="override the policy to compare (default: the run's)")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
