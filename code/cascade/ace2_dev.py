"""ACE-LTT Study 2 run (b-threshold ladder, spec v2.3): execute the registered procedure once.

    python -m cascade.ace2_dev --dataset DS --protocol P --det DET.csv \
        --agent crop=AG_CROP.csv --agent overlay=AG_OVERLAY.csv \
        --registration REG2.json --out DIR

This module chooses nothing. The registration froze the thresholds, the b ladder, the family, its chain
decomposition, the cover id and the design-time loss capacity, all from select-fold data. The run
re-derives every one of them and refuses on any mismatch, then - and only then - touches the certify fold.

Reported, never used to decide anything: the REALIZED loss capacity on the certify fold. Under v2.3 a
realized capacity is diagnostic only; admitting it into family design would let the family adapt to the
labels it is about to be tested on.

Status: exploratory / method-development. See the registration's `status` and `predecessor` fields.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .ace2_register import (DECLARATION, declared_losses, design_capacity, family, load_all,
                            select_stage)
from .ace_ltt import (NOT_CERTIFIED, Capacity, Contract, InferenceRegistration, ObservedLoss, Policy,
                      build_chains,
                      certification_boundary, certification_margin, certify_family, chain_capacity,
                      chain_cover_id, choose_operating_point, membership_digest, unreachable_from)
from .ace_register import certify_mask, membership_digests
from .evidence_veto import StratumSpec, discount, in_stratum, n_min
from .policy import decide, metrics, unit_losses

RISKS = ("fire", "smoke")


def check_registration(reg_path, protocol, det, agents):
    from .make_protocol import analysis_code, sha256_file, verify_tree
    rec = json.loads(Path(reg_path).expanduser().read_text())
    if rec.get("study") != DECLARATION["study"]:
        raise SystemExit(f"registration is for {rec.get('study')!r}, not {DECLARATION['study']!r}")
    now = {"protocol_sha256": sha256_file(Path(protocol) / "protocol.yaml"),
           "splits_sha256": sha256_file(Path(protocol) / "splits.csv"),
           "analysis_code_sha256": analysis_code()[0], "det_sha256": sha256_file(det),
           "agent_sha256": {k: sha256_file(v) for k, v in sorted(agents.items())}}
    bad = [k for k, v in now.items() if rec.get(k) != v]
    if bad:
        raise SystemExit("the registered state no longer holds; differing: " + ", ".join(bad))
    if rec["declaration"] != DECLARATION:
        raise SystemExit("the installed DECLARATION differs from the registered one")
    if not verify_tree()["ok"]:
        raise SystemExit("the installed package does not match its MANIFEST.sha256")
    return rec


def registered_inference(rec, losses):
    """The InferenceRegistration this study registered, refused if the record does not carry one.

    The registration pins the p-value machinery by CODE identity (v2.4.2). A record written before that
    field existed cannot be executed by this release - which is the same rule `analysis_code_sha256`
    already enforces, stated for contract I.
    """
    got = rec.get("preconditions", {}).get("P7_inference_registration")
    if not got:
        raise SystemExit("the registration carries no P7_inference_registration: this release pins the "
                         "p-value construction by code identity, so the study must be re-registered "
                         "(or executed from the package it was registered against)")
    ireg = InferenceRegistration(got["registration"], got["registry_id"], tuple(got["inferences"]))
    bad = ireg.problems(losses)
    if bad:
        raise SystemExit("the registered inference contract no longer describes this code: " + "; ".join(bad))
    return ireg


def _theta(p: Policy):
    return (p.t_low, p.t_high, p.b, p.A)


def _gv(D, mask, p: Policy, inS):
    return discount(D[f"g_{p.acq}"][mask], inS[mask], p.veto_at("S"))


def make_certify_eval(D, cert, inS, digests, record):
    def ev(p: Policy):
        gv = _gv(D, cert, p, inS)
        out = {}
        for k in RISKS:
            u, L = unit_losses(D["s"][cert], gv, D[f"y_{k}"][cert].astype(bool), _theta(p),
                              D["unit"][cert], D["event"][cert])
            # v2.4.1: the observation carries the evaluated unit SET, which the engine re-verifies
            out[k] = ObservedLoss(float(L.mean()) if len(L) else 0.0, int(len(L)), membership_digest(u))
            if int(len(L)) != digests[k]["n_units"]:
                raise SystemExit(f"{p.name}: {k} on {len(L)} units, audited {digests[k]['n_units']}")
        record[p.name] = {k: v.to_dict() for k, v in out.items()}
        return out
    return ev


def realized_capacity(D, cert, stage):
    """The same Gamma, recomputed on the certify fold. DIAGNOSTIC ONLY (v2.3 Sec. 4.5)."""
    caps = design_capacity(D, cert, stage)
    return {k: Capacity(k, v.value, "realized", "certify fold of the dev split; diagnostic only")
            for k, v in caps.items()}


def op_stats(D, cert, inS, p: Policy):
    gv = _gv(D, cert, p, inS)
    m = metrics(D["s"][cert], gv, D["y"][cert].astype(bool), _theta(p), D["unit"][cert], None,
                D["event"][cert])
    return {k: m[k] for k in ("fa", "calls", "handoff", "miss_event", "miss_unit") if k in m}


def stratum_diag(D, cert, inS, delta_m, alpha):
    need = n_min(alpha, delta_m)
    out = {"n_min_at_delta_m": need, "delta_m": delta_m}
    for k in RISKS:
        pos = cert & (D[f"y_{k}"].astype(int) > 0) & inS
        out[f"{k}|S"] = {"certify_units": len(set(np.asarray(D["unit"])[pos].tolist()))}
        out[f"{k}|S"]["certifiable"] = out[f"{k}|S"]["certify_units"] >= need
    return out


def run(a, log=print):
    from .make_protocol import analysis_code, sha256_file
    out = Path(a.out).expanduser()
    rep_path = out / "ace2_dev.json"
    if rep_path.exists() and not a.resume:
        raise SystemExit(f"{rep_path} exists: the registered run is executed once")
    agents = dict(x.split("=", 1) for x in a.agent)
    rec = check_registration(a.registration, a.protocol, a.det, agents)
    log(f"registration {rec['registered_utc']} ({rec['spec_version']}) verified")

    D, P = load_all(a.dataset, a.protocol, a.det, agents)
    spec = StratumSpec(DECLARATION["veto"]["stratum"]["kind"], DECLARATION["veto"]["stratum"]["area_max"])
    inS = in_stratum(spec, D["features"])
    sf = float(P.get("select_fraction", 0.3))
    cert = certify_mask(D["unit"], sf, DECLARATION["split"]["seed"])
    digests, split = membership_digests(D, sf, DECLARATION["split"]["seed"])
    if digests != rec["preconditions"]["P5_audit_membership"]:
        raise SystemExit("the audit membership differs from the registered one")
    counts = {k: v["n_units"] for k, v in digests.items()}

    # re-derive the frozen family; nothing here may differ from the registration
    stage = select_stage(D, ~cert, P, counts, log=lambda *_: None)
    rstar = stage.pop("_r_star")
    if json.dumps(stage, sort_keys=True) != json.dumps(rec["preconditions"]["frozen_family"]["thresholds"],
                                                       sort_keys=True):
        raise SystemExit("the re-derived thresholds / b ladder differ from the registered ones")
    losses = declared_losses()
    pol = family(stage)
    chains = build_chains(pol, losses)
    cover = chain_cover_id(chains)
    if cover != rec["preconditions"]["P2_chains"]["cover_id"]:
        raise SystemExit("the chain cover differs from the registered one")
    log(f"frozen family re-derived: M={len(chains)}, cover {cover[:12]}, "
        f"{[len(c) for c in chains]} members per chain")

    out.mkdir(parents=True, exist_ok=True)
    (out / "ace2_candidates.json").write_text(json.dumps(
        {"written_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "cover_id": cover, "thresholds": stage, "chains": [[p.name for p in c] for c in chains],
         "policies": [p.to_dict() for p in pol],
         "note": "re-derived from the registration BEFORE any certify-fold quantity was computed"}, indent=1))

    observed = {}
    res = certify_family(pol, losses, delta=float(DECLARATION["delta"]),
                         contracts=lambda _p: Contract(True, True, counts, split["n_certify_units"],
                                                       {k: v["sha256"] for k, v in digests.items()},
                                                       0.0, float("inf")),
                         certify_eval=make_certify_eval(D, cert, inS, digests, observed), plan=None,
                         inference_registration=registered_inference(rec, losses))

    u = DECLARATION["utility"]
    stats = {p.name: op_stats(D, cert, inS, p) for p in res.certified}

    def utility(p):
        s = stats[p.name]
        return (float(u["calls"]) * s["calls"] + float(u["fa"]) * s["fa"]
                + float(u["handoff"]) * s.get("handoff", 0.0))

    best, info = choose_operating_point(res, utility) if res.ok else (None, {})
    dm = float(DECLARATION["chains"]["delta_m"])
    caps_real = realized_capacity(D, cert, stage)
    anchors = {c[0].name: observed.get(c[0].name) for c in res.chains}
    rep = {"study": DECLARATION["study"], "spec_version": rec["spec_version"],
           "status": DECLARATION["status"],
           "run_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "registration": {"utc": rec["registered_utc"], "sha256": sha256_file(a.registration)},
           "analysis_code_sha256": analysis_code()[0], "cover_id": cover, "split": split,
           "thresholds": stage, "select_bar_r_star": rstar,
           "certification": res.to_dict(), "certify_risks": observed,
           "operating_point": ({"policy": best.to_dict(), **info, "stats": stats[best.name],
                                "caveat": u["caveat"]} if best is not None else
                               {"policy": None, "reason": "NO_REJECTION"}),
           "capacity": {
               "design": rec["preconditions"]["P6_design_capacity"],
               "realized_DIAGNOSTIC_ONLY": {k: v.to_dict() for k, v in caps_real.items()},
               "status_realized": chain_capacity(caps_real, counts, losses, dm, rstar),
               "margin": {k: {"r_star": certification_boundary(counts[k], 0.05, dm),
                              "g_star": certification_margin(counts[k], 0.05, dm)} for k in RISKS},
               "anchor_unreachable": {
                   k: {c: (None if anchors[c] is None else
                           unreachable_from(anchors[c][k]["risk"], caps_real[k].value, counts[k], 0.05, dm))
                       for c in anchors} for k in RISKS},
               "note": DECLARATION["capacity"]["realized_capacity_note"]},
           "diagnostics": {"stratum": stratum_diag(D, cert, inS, dm, float(DECLARATION["alpha"]))},
           "selection_side_stats": stats}
    rep_path.write_text(json.dumps(rep, indent=1))
    md = report_md(rep, res)
    (out / "ace2_dev.md").write_text(md)
    log("\n" + md)
    log(f"-> {rep_path}")
    return rep


def report_md(rep, res):
    L = [f"# ACE-LTT Study 2 (b ladder) — {rep['run_utc']}", "",
         f"**{rep['status']}**", "",
         f"registration {rep['registration']['utc']} · spec {rep['spec_version']} · "
         f"code `{rep['analysis_code_sha256'][:12]}` · cover `{rep['cover_id'][:12]}`",
         f"split: {rep['split']['n_select_units']} select / {rep['split']['n_certify_units']} certify units",
         "", "## Frozen family (registered from the select fold)", "",
         "| acquisition | t_low | t_high | A | b ladder (safest first) | select-fold joint slack |",
         "|---|---|---|---|---|---|"]
    for acq, st in rep["thresholds"].items():
        L.append(f"| {acq} | {st['t_low']:.4f} | {st['t_high']:.4f} | {st['A']:.4f} | "
                 f"{len(st['b_ladder'])} members, b {st['b_ladder'][0]:.3f}..{st['b_ladder'][-1]:.3f} | "
                 f"{st['spanning']['status']} ({st['joint_slack'][0]:+.4f} .. "
                 f"{st['joint_slack'][-1]:+.4f}) |")
    L += ["", "## Certification", "",
          f"delta = {res.delta:g}, M = {len(res.chains)}, delta_m = "
          f"{', '.join(f'{d:g}' for d in res.deltas)}", "",
          "A member is tested only if every safer member of its chain was rejected (fail-safe path).", "",
          "| policy | fire risk | smoke risk | p_joint | delta_m | certified |", "|---|---|---|---|---|---|"]
    certified = {p.name for p in res.certified}
    for c in res.chains:
        for p in c:
            o, pv = rep["certify_risks"].get(p.name), res.pvalues.get(p.name)
            if o is None or pv is None:
                L.append(f"| {p.name} | — | — | — | — | not tested (sequence stopped earlier) |")
                continue
            L.append(f"| {p.name} | {o['fire']['risk']:.4f} | {o['smoke']['risk']:.4f} | "
                     f"{pv['p_joint']:.5f} | {pv['delta_m']:.4f} | "
                     f"{'YES' if p.name in certified else 'no'} |")
    op = rep["operating_point"]
    L += ["", "## Operating point", ""]
    if op.get("policy"):
        s = op["stats"]
        L += [f"chosen: **{op['policy']['name']}** (utility {op['utility']:.4f} over "
              f"{op['n_certified']} certified candidates)",
              f"selection-side on the certify fold: FA {s['fa']:.4f}, calls {s['calls']:.4f}, "
              f"hand-off {s.get('handoff', 0.0):.4f}", "", f"*{op['caveat']}*"]
    else:
        L += ["**NOT CERTIFIED** — " + ", ".join(f"{r}: {NOT_CERTIFIED.get(r, '')}" for r in res.reasons)]
    cap = rep["capacity"]
    L += ["", "## Sec. 4.5 capacity", "",
          "| loss | g* | design Γ (registered) | realized Γ (diagnostic) | design status |",
          "|---|---|---|---|---|"]
    for k in RISKS:
        L.append(f"| {k} | {cap['margin'][k]['g_star']:.4f} | {cap['design'][k]['value']:.4f} | "
                 f"{cap['realized_DIAGNOSTIC_ONLY'][k]['value']:.4f} | "
                 f"{'MARGIN_CAPABLE' if cap['design'][k]['value'] >= cap['margin'][k]['g_star'] else 'MARGIN_INCAPABLE'} |")
    L += ["", f"*{cap['note']}*", "", "## Diagnostics (never constrained)", ""]
    d = rep["diagnostics"]["stratum"]
    for k in RISKS:
        v = d[f"{k}|S"]
        L.append(f"- `{k}|S`: {v['certify_units']} certify units, n_min = {d['n_min_at_delta_m']} → "
                 f"{'certifiable' if v['certifiable'] else 'NOT certifiable (clause C3 would refuse)'}")
    if res.reasons:
        L += ["", "clauses that fired: " + ", ".join(res.reasons)]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--agent", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--registration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
