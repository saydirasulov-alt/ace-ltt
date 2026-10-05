"""ACE-LTT dev run: execute the registered procedure once and report what it returns.

    python -m cascade.ace_dev --dataset DS --protocol P --det DET.csv \
        --agent crop=AG_CROP.csv --agent overlay=AG_OVERLAY.csv \
        --registration REG.json --out DIR

This module runs the study declared in `ace_register.DECLARATION`. It adds no choices of its own: every
quantity that could steer the result - the acquisitions, the lambda grid, the chain structure, the per-chain
budget, the constrained losses and their monotonicity contracts, the select/certify split, the audit
membership and the operating-point utility - is read from the registration and CHECKED, not re-decided.

Order of operations (clause C5 made visible on disk, not only in the call order):

    1. registration + input hashes verified against the installed code and files;
    2. select fold -> one threshold vector per acquisition, at lambda = 0, by the pre-registered objective;
    3. the candidate list, its chain decomposition and the chain cover id are WRITTEN OUT;
    4. only then is the certify fold touched, by certify_family, which re-reads that file and refuses on
       any mismatch.

The run is idempotent-by-refusal: an existing report is not overwritten.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .ace_ltt import (NOT_CERTIFIED, Contract, InferenceRegistration, ObservedLoss, Policy, build_chains,
                      certify_family, chain_cover_id, choose_operating_point, membership_digest, miss_loss)
from .ace_register import DECLARATION, certify_mask, membership_digests
from .evidence_veto import StratumSpec, discount, in_stratum, n_min
from .policy import Grid, decide, metrics, unit_losses
from .select import _grid_stats, _objective, candidate_mask

RISKS = ("fire", "smoke")


# ------------------------------------------------------------------ registration

def check_registration(reg_path, protocol, det, agents):
    """Every hash the registration pinned must still hold, or the run is not the registered one."""
    from .make_protocol import analysis_code, sha256_file, verify_tree
    rec = json.loads(Path(reg_path).expanduser().read_text())
    if rec.get("study") != DECLARATION["study"]:
        raise SystemExit(f"registration is for {rec.get('study')!r}, not {DECLARATION['study']!r}")
    now = {"protocol_sha256": sha256_file(Path(protocol) / "protocol.yaml"),
           "splits_sha256": sha256_file(Path(protocol) / "splits.csv"),
           "analysis_code_sha256": analysis_code()[0],
           "det_sha256": sha256_file(det),
           "agent_sha256": {k: sha256_file(v) for k, v in sorted(agents.items())}}
    bad = [k for k, v in now.items() if rec.get(k) != v]
    if bad:
        raise SystemExit("the registered state no longer holds; differing: " + ", ".join(bad) + "\n" +
                         json.dumps({k: {"registered": rec.get(k), "now": now[k]} for k in bad}, indent=1))
    if rec["declaration"] != DECLARATION:
        raise SystemExit("the installed DECLARATION differs from the registered one")
    if not verify_tree()["ok"]:
        raise SystemExit("the installed package does not match its MANIFEST.sha256")
    return rec


def declared_losses():
    return tuple(miss_loss(L["name"], L["alpha"], L["lemma"], catch_set=L["catch_set"],
                           aggregation=L["aggregation"], weighting=L["weighting"],
                           empty_denominator=L["empty_denominator"]) for L in DECLARATION["losses"])


# ------------------------------------------------------------------ 2. select fold

def select_thresholds(D, sel, P, log=print):
    """One (t_low, t_high, b, A) per acquisition, chosen on the SELECT fold at lambda = 0.

    lambda is the chain variable, so the thresholds must not depend on it: they are chosen once, with the
    veto authority off. Rule: among candidates whose select-fold risk estimate is <= alpha for EVERY
    constrained risk, minimise the pre-registered objective (protocol `costs`). handoff = False forces
    b = A, the machine-only family.
    """
    alpha = float(P["alpha"])
    costs = {k: float(v) for k, v in P["costs"].items() if k in ("calls", "handoff", "fa")}
    out = {}
    for acq in DECLARATION["acquisitions"]:
        s, g = D["s"][sel], D[f"g_{acq}"][sel]
        grid = Grid.default(s, s_min=float(P["s_min"]))
        risks = {k: D[f"y_{k}"][sel].astype(bool) for k in RISKS}
        E, R, _ = _grid_stats(s, g, D["y"][sel].astype(bool), D["unit"][sel], grid, "unit", None,
                              D["event"][sel], risks)
        rmax = np.max(np.stack([np.nan_to_num(r, nan=np.inf) for r in R.values()]), axis=0)
        cand = candidate_mask(grid, cloud=True, handoff=False) & (rmax <= alpha) & ~np.isnan(E["fa"])
        if not cand.any():
            raise SystemExit(f"{acq}: no select-fold candidate meets alpha={alpha} for every risk")
        obj = np.where(cand, _objective(E, costs), np.inf)
        j1, j2, k1, k2 = np.unravel_index(int(np.argmin(obj)), obj.shape)
        th = (float(grid.s_edges[j1]), float(grid.s_edges[j2]), float(grid.g_edges[k1]),
              float(grid.g_edges[k2]))
        out[acq] = {"theta": th, "select_objective": float(obj[j1, j2, k1, k2]),
                    "select_risk_max": float(rmax[j1, j2, k1, k2]), "grid": grid.to_dict()}
        log(f"  {acq:8s} theta=({th[0]:.4f}, {th[1]:.4f}, b=A={th[2]:.4f})  "
            f"select risk<= {out[acq]['select_risk_max']:.4f}  obj={out[acq]['select_objective']:.4f}")
    return out


def family(thetas):
    """The declared family: one lambda ladder per acquisition, safest member first."""
    pol = []
    for acq in DECLARATION["acquisitions"]:
        tl, th, b, A = thetas[acq]["theta"]
        for lam in sorted(DECLARATION["veto"]["lambda_grid"], reverse=True):
            pol.append(Policy(acq, tl, th, b, A, {"S": float(lam), "rest": 0.0}, float("inf"),
                              f"{acq}|lambda={lam:g}"))
    return tuple(pol)


# ------------------------------------------------------------------ 4. certify fold

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


def make_certify_eval(D, cert, inS, digests, record):
    """(risk, n units) per constrained loss on the certify fold, with the declared catch set.

    `unit_losses` counts d in {1, 2, 4} as caught on positive frames - the response set - and aggregates as
    (missed positive events of the unit) / (positive events of the unit), averaged over units with at least
    one. That is exactly what the registration declares.
    """
    def ev(p: Policy):
        gv = discount(D[f"g_{p.acq}"][cert], inS[cert], p.veto_at("S"))
        out = {}
        for k in RISKS:
            u, L = unit_losses(D["s"][cert], gv, D[f"y_{k}"][cert].astype(bool), _theta(p),
                               D["unit"][cert], D["event"][cert])
            # v2.4.1: the observation carries the evaluated unit SET, which the engine re-verifies
            out[k] = ObservedLoss(float(L.mean()) if len(L) else 0.0, int(len(L)), membership_digest(u))
            if int(len(L)) != digests[k]["n_units"]:
                raise SystemExit(f"{p.name}: {k} evaluated on {len(L)} units, audited {digests[k]['n_units']}")
        record[p.name] = {k: v.to_dict() for k, v in out.items()}
        return out
    return ev


def contracts_from(digests, n_cert_units):
    def contracts(_p):
        return Contract(measurement_coherent=True, score_contract_ok=True,
                        positive_units={k: v["n_units"] for k, v in digests.items()},
                        budget_units=int(n_cert_units),
                        audit_membership={k: v["sha256"] for k, v in digests.items()},
                        max_payload_bytes=0.0, b_max=float("inf"),
                        note="no payload claim is made in this study (payload is diagnostic)")
    return contracts


def operating_point_stats(D, cert, inS, p: Policy):
    """Selection-side FA / calls of one policy on the certify fold (NOT an unbiased performance estimate)."""
    gv = discount(D[f"g_{p.acq}"][cert], inS[cert], p.veto_at("S"))
    m = metrics(D["s"][cert], gv, D["y"][cert].astype(bool), _theta(p), D["unit"][cert], None,
                D["event"][cert])
    return {k: m[k] for k in ("fa", "calls", "handoff", "miss_event", "miss_unit") if k in m}


# ------------------------------------------------------------------ diagnostics (never constrained)

def stratum_diagnostics(D, cert, inS, delta_m, alpha):
    """Why the stratum-restricted risks are not in the constrained family - with the numbers."""
    need = n_min(alpha, delta_m)
    out = {"n_min_at_delta_m": need, "delta_m": delta_m}
    for k in RISKS:
        pos = cert & (D[f"y_{k}"].astype(int) > 0) & inS
        units = sorted(set(np.asarray(D["unit"])[pos].tolist()))
        out[f"{k}|S"] = {"certify_units": len(units), "certifiable": len(units) >= need}
    return out


def handoff_diagnostic(D, cert, inS, p: Policy):
    """Descriptive only: the same policy scored with the operator tier NOT counted as a catch."""
    gv = discount(D[f"g_{p.acq}"][cert], inS[cert], p.veto_at("S"))
    d = decide(D["s"][cert], gv, _theta(p))
    y = D["y"][cert].astype(bool)
    alarm_only = np.isin(d, (1, 2))
    return {"handoff_rate": float((d == 4).mean()),
            "positive_frames_caught_by_handoff_only": float(((d == 4) & y).mean()),
            "note": ("b = A in this family, so the operator tier is empty and the response set equals the "
                     "alarm set; the number is reported to show that, not to compare two families."),
            "alarm_only_positive_rate": float(alarm_only[y].mean()) if y.any() else 0.0}


# ------------------------------------------------------------------ run

def run(a, log=print):
    from .esva_dev import load_dev
    from .make_protocol import analysis_code, sha256_file
    out = Path(a.out).expanduser()
    rep_path = out / "ace_dev.json"
    if rep_path.exists() and not a.resume:
        raise SystemExit(f"{rep_path} exists: the registered run is executed once")
    agents = dict(x.split("=", 1) for x in a.agent)
    if sorted(agents) != sorted(DECLARATION["acquisitions"]):
        raise SystemExit(f"--agent names {sorted(agents)} != declared {sorted(DECLARATION['acquisitions'])}")
    rec = check_registration(a.registration, a.protocol, a.det, agents)
    log(f"registration {rec['registered_utc']} verified (protocol, splits, code, detector, agents)")

    D, P = None, None
    for name, path in sorted(agents.items()):
        Di, P = load_dev(a.dataset, a.protocol, a.det, path)
        if D is None:
            D = {k: v for k, v in Di.items() if k != "g"}
            D["unit"], D["event"] = np.asarray(Di["unit"]), np.asarray(Di["event"])
        elif not np.array_equal(np.asarray(D["uid"]), np.asarray(Di["uid"])):
            raise SystemExit(f"acquisition {name!r} covers different dev rows than the first one")
        D[f"g_{name}"] = np.asarray(Di["g"], float)
    spec = StratumSpec(DECLARATION["veto"]["stratum"]["kind"], DECLARATION["veto"]["stratum"]["area_max"])
    inS = in_stratum(spec, D["features"])

    sf = float(P.get("select_fraction", 0.3))
    cert = certify_mask(D["unit"], sf, DECLARATION["split"]["seed"])
    digests, split = membership_digests(D, sf, DECLARATION["split"]["seed"])
    if digests != rec["preconditions"]["P5_audit_membership"]:
        raise SystemExit("the audit membership differs from the registered one")
    log(f"split: {split['n_select_units']} select / {split['n_certify_units']} certify units "
        f"(seed {split['seed']}); audit fire {digests['fire']['n_units']}, smoke {digests['smoke']['n_units']}")

    log("select fold -> thresholds (lambda = 0, pre-registered objective):")
    thetas = select_thresholds(D, ~cert, P, log=log)
    losses = declared_losses()
    pol = family(thetas)
    chains = build_chains(pol, losses)
    cover = chain_cover_id(chains)
    if cover != rec["preconditions"]["P2_chains"]["cover_id"]:
        raise SystemExit(f"chain cover {cover[:12]} != registered {rec['preconditions']['P2_chains']['cover_id'][:12]}")

    out.mkdir(parents=True, exist_ok=True)
    cand_path = out / "ace_candidates.json"
    cand = {"written_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cover_id": cover, "thresholds": thetas,
            "chains": [[p.name for p in c] for c in chains],
            "policies": [p.to_dict() for p in pol],
            "note": "written BEFORE any certify-fold quantity is computed (clause C5)"}
    cand_path.write_text(json.dumps(cand, indent=1))
    log(f"candidates + cover {cover[:12]} written to {cand_path.name}; the certify fold is untouched so far")

    observed = {}
    cert_res = certify_family(pol, losses, delta=float(DECLARATION["delta"]),
                              contracts=contracts_from(digests, split["n_certify_units"]),
                              certify_eval=make_certify_eval(D, cert, inS, digests, observed), plan=None,
                              inference_registration=registered_inference(rec, losses))
    if [[p.name for p in c] for c in cert_res.chains] != cand["chains"]:
        raise SystemExit("the tested chains differ from the ones written before the certify fold")

    util = DECLARATION["utility"]
    stats = {p.name: operating_point_stats(D, cert, inS, p) for p in cert_res.certified}

    def utility(p):
        s = stats[p.name]
        return float(util["calls"]) * s["calls"] + float(util["fa"]) * s["fa"]

    best, info = choose_operating_point(cert_res, utility) if cert_res.ok else (None, {})

    rep = {"study": DECLARATION["study"], "run_utc": _dt.datetime.now(_dt.timezone.utc)
           .strftime("%Y-%m-%dT%H:%M:%SZ"),
           "registration": {"utc": rec["registered_utc"], "sha256": sha256_file(a.registration)},
           "analysis_code_sha256": analysis_code()[0], "cover_id": cover,
           "split": split, "thresholds": thetas,
           "certification": cert_res.to_dict(), "certify_risks": observed,
           "operating_point": ({"policy": best.to_dict(), **info, "stats": stats[best.name],
                                "caveat": util["caveat"]} if best is not None else
                               {"policy": None, "reason": "NO_REJECTION"}),
           "diagnostics": {
               "stratum": stratum_diagnostics(D, cert, inS, float(DECLARATION["chains"]["delta_m"]),
                                              float(DECLARATION["alpha"])),
               "handoff": (handoff_diagnostic(D, cert, inS, best) if best is not None else {}),
               "payload": ("not measured on these agent CSVs; the measured payload lives in the "
                           "acquisition-ladder outputs. No payload claim is made in this study."),
           },
           "selection_side_stats": stats}
    rep_path.write_text(json.dumps(rep, indent=1))
    (out / "ace_dev.md").write_text(report_md(rep, cert_res))
    log("\n" + report_md(rep, cert_res))
    log(f"-> {rep_path}")
    return rep


def report_md(rep, cert_res):
    L = [f"# ACE-LTT dev run — {rep['run_utc']}", "",
         f"registration {rep['registration']['utc']} · code `{rep['analysis_code_sha256'][:12]}` · "
         f"cover `{rep['cover_id'][:12]}`",
         f"split: {rep['split']['n_select_units']} select / {rep['split']['n_certify_units']} certify units",
         "", "## Thresholds chosen on the select fold (lambda = 0)", "",
         "| acquisition | t_low | t_high | b = A | select risk | objective |", "|---|---|---|---|---|---|"]
    for acq, t in rep["thresholds"].items():
        th = t["theta"]
        L.append(f"| {acq} | {th[0]:.4f} | {th[1]:.4f} | {th[2]:.4f} | {t['select_risk_max']:.4f} | "
                 f"{t['select_objective']:.4f} |")
    L += ["", "## Certification", "",
          f"delta = {cert_res.delta:g}, M = {len(cert_res.chains)}, delta_m = "
          f"{', '.join(f'{d:g}' for d in cert_res.deltas)}", "",
          "A member is tested only if every safer member of its chain was rejected (fail-safe path).", "",
          "| policy | fire risk | smoke risk | p_joint | delta_m | certified |", "|---|---|---|---|---|---|"]
    certified = {p.name for p in cert_res.certified}
    for c in cert_res.chains:
        for p in c:
            o, pv = rep["certify_risks"].get(p.name), cert_res.pvalues.get(p.name)
            if o is None or pv is None:      # the fixed sequence stopped before reaching this member
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
              f"selection-side on the certify fold: FA {s['fa']:.4f}, calls {s['calls']:.4f}",
              "", f"*{op['caveat']}*"]
    else:
        L += ["**NOT CERTIFIED** — " + ", ".join(f"{r}: {NOT_CERTIFIED.get(r, '')}" for r in cert_res.reasons)]
    d = rep["diagnostics"]["stratum"]
    L += ["", "## Diagnostics (never constrained)", "",
          f"stratum-restricted risks at delta_m = {d['delta_m']:g} need n_min = {d['n_min_at_delta_m']} units:"]
    for k in RISKS:
        v = d[f"{k}|S"]
        L.append(f"- `{k}|S`: {v['certify_units']} certify units → "
                 f"{'certifiable' if v['certifiable'] else 'NOT certifiable (clause C3 would refuse)'}")
    L += ["", rep["diagnostics"]["payload"], ""]
    if cert_res.reasons:
        L += ["clauses that fired: " + ", ".join(cert_res.reasons), ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--agent", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--registration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="allow overwriting a partial report after a crash")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
