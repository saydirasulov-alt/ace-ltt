"""One-shot pre-registration of ACE-LTT Study 2 (b-threshold ladder), under spec v2.3.

    python -m cascade.ace2_register --dataset DS --protocol P --det DET.csv \
        --agent crop=AG_CROP.csv --agent overlay=AG_OVERLAY.csv --out REG2.json [--dry-run]

Relation to Study 1. `ace_ltt_dev` was registered and executed under spec **v2.2** and stays valid as that
experiment; its artifact is cascade_v2_19.zip. Its observed failure motivated Sec. 4.5 of **v2.3**, whose
`chain_margin_capacity` guard did not exist when Study 1 was registered. Under v2.3 Study 1's lambda family
would be refused for a NEW registration - that is a statement about v2.3 admissibility, never a claim that
the v2.2 registration was invalid.

Status: **exploratory / method-development.** All 5989 dev units were consumed by Study 1 (1797 select +
4192 certify) and its results have been seen, so this procedure is born from them. Its HB / fixed-sequence
output is method-development evidence and is NOT a 90% confirmatory guarantee. The confirmatory test is a
later study on fresh sealed units under the frozen protocol.

Everything the run could otherwise choose is frozen HERE, from select-fold data only: the thresholds, the
b-ladder, the family, its chain decomposition and cover id, and the design-time loss capacity. The runner
recomputes all of it and refuses on any mismatch, so the certify fold meets a family that was fixed before
it was touched.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .ace_ltt import (CHECKLIST, SPEC_VERSION, Capacity, Contract, Policy, allocate_delta,
                      boundary_spanning, build_chains, certification_boundary, certification_slack,
                      chain_capacity, chain_cover_id, declare_inference, miss_loss, preflight_checklist)
# study-agnostic helpers, first written for Study 1 and unchanged here
from .ace_register import certify_mask, events_spanning_units, membership_digests
from .evidence_veto import StratumSpec, discount, in_stratum
from .policy import Grid
from .select import _grid_stats, _objective, candidate_mask

STUDY = "ace_ltt_study2_bladder"
LEMMA = "ACE-LTT spec v2.3 Sec. 4.3 Lemma, event-miss instance (2026-09-24)"
RISKS = ("fire", "smoke")

DECLARATION = {
    "study": STUDY,
    "spec_version": SPEC_VERSION,
    "status": "exploratory / method-development; NOT a confirmatory guarantee on fresh data",
    "question": (
        "With the chain built on the coordinate the constrained loss actually responds to - b, which governs "
        "the response set - does one registered ACE-LTT procedure produce a certified set whose fixed "
        "sequence performs real selection rather than stopping at its first member?"),

    "predecessor": {
        "study": "ace_ltt_dev", "spec_version": "v2.2", "registered_utc": "2026-09-23T10:31:20Z",
        "artifact": "cascade_v2_19.zip",
        "relation": ("Study 1 remains a valid experiment under the v2.2 protocol frozen at its "
                     "registration. Sec. 4.5 and the chain_margin_capacity guard are v2.3 additions made "
                     "after its result; under v2.3 its lambda family would be refused for a new "
                     "registration. That is an admissibility statement about v2.3, not a retroactive "
                     "invalidation of Study 1."),
        "exploratory_reason": ("Study 1 consumed all 5989 dev units and its results have been seen, so no "
                               "untouched dev units remain and this study cannot be confirmatory."),
    },

    # ---- family ------------------------------------------------------------------------------------
    "acquisitions": ["crop", "overlay"],
    "veto": {"stratum": {"kind": "small_smoke", "area_max": 0.02},
             "partition": ["S", "rest"],
             "lambda_fixed": 0.4,
             "why": ("lambda is FROZEN at the pre-Study-1 Table F working point, not selected from Study 1's "
                     "result, and is not the chain coordinate: Sec. 4.5 showed it cannot span the required "
                     "margin because it acts only inside S.")},
    "chain_coordinate": {
        "name": "b",
        "why": ("the D8 code audit established that the constrained miss loss uses the RESPONSE set, and the "
                "response set is governed by b; A only separates hand-off from agent alarm and does not "
                "enter the miss loss at all (the event-miss grid is exactly constant along A). b is "
                "therefore the loss-aligned coordinate, and it acts on the whole escalated population "
                "rather than on a stratum."),
        "order": "ascending b: smallest (largest response set, safest) first",
        "ladder_rule": ("every g-grid edge b <= A that forms a valid candidate at the chosen (t_low, "
                        "t_high, A), ascending. The g grid is fixed in cascade.policy.Grid, so the ladder "
                        "is a deterministic function of the chosen thresholds and of nothing observed. "
                        "Study 1's certify-fold numbers are not used anywhere in its construction."),
        "min_members": 3,
        "not_all_members_feasible": ("members beyond the certifiable side are DELIBERATE: a ladder whose "
                                     "every member is select-feasible cannot be boundary spanning, and the "
                                     "fixed sequence would have nothing to reject."),
    },
    "thresholds": {
        "rule": ("per acquisition, at the frozen lambda, choose (t_low, t_high, A) on the SELECT fold "
                 "among configurations whose b ladder satisfies BOTH: (i) the safe endpoint (smallest b) "
                 "is joint-select-feasible, s_joint = min_k (r*_k - R_sel_k) >= 0; and (ii) the ladder is "
                 "BOUNDARY_SPANNING (Sec. 4.6): max_pi s_joint >= 0 and min_pi s_joint < 0. Among those, "
                 "minimise the pre-registered utility over the ladder members that are on the certifiable "
                 "side. r*_k = certification_boundary(n_k, alpha_k, delta_m) is a function of the "
                 "registered (n, alpha, delta_m) alone."),
        "why_not_the_old_rule": ("the first proposed rule (minimise utility subject to select risk <= r*, "
                                 "capacity checked afterwards) was refused at design audit 2A: it chose a "
                                 "narrow, low t_high band in which almost every positive event is "
                                 "edge-alarmed, leaving the b coordinate almost inert. Putting the "
                                 "boundary-spanning requirement INSIDE the selection makes the thresholds "
                                 "and the chain coordinate be chosen together."),
        "frozen_at": "registration (from select-fold data only); the runner recomputes and must match",
    },
    "chains": {"M": 2, "rule": "one chain per acquisition; the chain is the b ladder",
               "delta": 0.10, "delta_m": 0.05, "allocation": "uniform (delta/M); no DeltaPlan"},

    # ---- losses ------------------------------------------------------------------------------------
    "losses": [
        {"name": "fire", "alpha": 0.05, "catch_set": "response", "aggregation": "unit_mean_over_events",
         "weighting": "uniform", "empty_denominator": "exclude_unit", "lemma": LEMMA},
        {"name": "smoke", "alpha": 0.05, "catch_set": "response", "aggregation": "unit_mean_over_events",
         "weighting": "uniform", "empty_denominator": "exclude_unit", "lemma": LEMMA},
    ],
    "capacity": {
        "domain_S": "positive events that are escalated and not edge-alarmed, i.e. the events whose catch "
                    "status b can change at the frozen (t_low, t_high)",
        "select_bar": ("r*_k. The rule guarantees only that the SAFE ENDPOINT is select-feasible, so the "
                       "margin Sec. 4.5 asks the coordinate to span is h*_k = max(0, r*_k - r*_k) = 0 and "
                       "the capacity condition is satisfied trivially. That is correct, not a loophole: "
                       "there is no anchor left to pull across. Gamma is still reported as the upper bound "
                       "on the ladder's select-fold risk span, and the work of showing the chain is "
                       "informative is done by Sec. 4.6 boundary spanning, not by Gamma."),
        "source": "design",
        "basis": "select fold of the dev split, at the frozen thresholds",
        "requirement": "MARGIN_CAPABLE for fire and smoke (Sec. 4.5), checked at registration",
        "realized_capacity_note": ("a capacity recomputed on the certify fold is DIAGNOSTIC ONLY and never "
                                   "accepts a family, changes a chain or changes an order"),
    },
    "diagnostic_only": {
        "stratum_risks": "fire|S and smoke|S: too few units to constrain (clause C3); reported with counts",
        "payload": "C_U reported if measured; no payload loss in the constrained family, no B0 fitted",
    },

    "alpha": 0.05, "delta": 0.10, "unit": "split_group (global dedup component)", "target": "event",
    "split": {"rule": "select/certify split of the dev units, select_fraction from protocol.yaml",
              "seed": 20260924,
              "reproduction": "cascade.ace_register.certify_mask (identical to cascade.select.calibrate)"},
    "utility": {"calls": 1.0, "fa": 10.0, "handoff": 20.0,
                "caveat": ("the operating point is chosen inside the certified set, which does not weaken "
                           "the guarantee; but FA, calls and hand-off measured on the certify fold are "
                           "selection-side quantities, not an unbiased performance estimate. With b < A the "
                           "operator tier is non-empty, so hand-off load is priced by the utility.")},
    "outcome_rule": ("the study reports what the procedure returns: certified set, operating point, chain "
                     "certificates, every NOT CERTIFIED clause that fired, and the realized capacity as a "
                     "diagnostic. Nothing is added, removed or re-ordered after the certify fold is "
                     "touched. An empty certified set is a reportable outcome."),
}


# ------------------------------------------------------------------ frozen family construction

def select_stage(D, sel, P, counts, log=print):
    """Choose (t_low, t_high, A) AND the b ladder together, on the SELECT fold, at the frozen lambda.

    The grid already holds the select-fold risk of every (t_low, t_high, b, A), so the ladder at a given
    (t_low, t_high, A) is just the grid's b edges up to A. For each configuration:

        s_k(pi)      = r*_k - R_sel_k(pi)             certification slack, select fold
        s_joint(pi)  = min_k s_k(pi)

    admissible iff the safe endpoint (smallest b) has s_joint >= 0 and the ladder is BOUNDARY_SPANNING
    (max s_joint >= 0 and min s_joint < 0). Among admissible configurations, minimise the pre-registered
    utility over the ladder members on the certifiable side.

    Note the miss risk is non-increasing in the response set, hence non-decreasing in b, so s_joint is
    non-increasing along the ladder: the spanning test is decided by its two endpoints. The implementation
    computes every member anyway and records the whole slack profile.
    """
    alpha, dm = float(P["alpha"]), float(DECLARATION["chains"]["delta_m"])
    costs = {k: float(v) for k, v in P["costs"].items() if k in ("calls", "handoff", "fa")}
    rstar = {k: certification_boundary(int(counts[k]), alpha, dm) for k in RISKS}
    spec = StratumSpec(DECLARATION["veto"]["stratum"]["kind"], DECLARATION["veto"]["stratum"]["area_max"])
    inS = in_stratum(spec, D["features"])
    lam = float(DECLARATION["veto"]["lambda_fixed"])
    minmem = int(DECLARATION["chain_coordinate"]["min_members"])
    out = {"_r_star": rstar}
    for acq in DECLARATION["acquisitions"]:
        s = D["s"][sel]
        gv = discount(D[f"g_{acq}"][sel], inS[sel], lam)
        grid = Grid.default(s, s_min=float(P["s_min"]))
        risks = {k: D[f"y_{k}"][sel].astype(bool) for k in RISKS}
        E, R, _ = _grid_stats(s, gv, D["y"][sel].astype(bool), D["unit"][sel], grid, "unit", None,
                              D["event"][sel], risks)
        valid = candidate_mask(grid, cloud=True, handoff=True) & ~np.isnan(E["fa"])
        sj = np.min(np.stack([rstar[k] - np.nan_to_num(R[k], nan=np.inf) for k in RISKS]), axis=0)
        obj = _objective(E, costs)
        J, K = valid.shape[0], valid.shape[2]
        best = None
        for k2 in range(K):                                   # A = g_edges[k2]
            V = valid[:, :, :k2 + 1, k2]
            if not V.any():
                continue
            S = np.where(V, sj[:, :, :k2 + 1, k2], np.nan)
            U = np.where(V & (sj[:, :, :k2 + 1, k2] >= 0), obj[:, :, :k2 + 1, k2], np.nan)
            allnan_s, allnan_u = np.all(np.isnan(S), axis=2), np.all(np.isnan(U), axis=2)
            S = np.where(np.isnan(S), -np.inf, S)
            U = np.where(np.isnan(U), np.inf, U)
            hi = np.where(allnan_s, -np.inf, S.max(axis=2))
            lo = np.where(allnan_s, np.inf, np.where(np.isneginf(S), np.inf, S).min(axis=2))
            util = np.where(allnan_u, np.inf, U.min(axis=2))
            ok = (V.sum(2) >= minmem) & (hi >= 0.0) & (lo < 0.0) & np.isfinite(util)
            if not ok.any():
                continue
            u = np.where(ok, util, np.inf)
            j1, j2 = np.unravel_index(int(np.argmin(u)), u.shape)
            if best is None or u[j1, j2] < best[0]:
                best = (float(u[j1, j2]), int(j1), int(j2), int(k2))
        if best is None:
            raise SystemExit(f"{acq}: no threshold configuration gives a boundary-spanning b ladder whose "
                             f"safe endpoint is select-feasible (r* = "
                             f"{ {k: round(v, 4) for k, v in rstar.items()} })")
        u, j1, j2, k2 = best
        ks = [int(k) for k in range(k2 + 1) if valid[j1, j2, k, k2]]
        lad = [float(grid.g_edges[k]) for k in ks]
        prof = [float(sj[j1, j2, k, k2]) for k in ks]
        per = [{k: float(rstar[k] - R[k][j1, j2, kk, k2]) for k in RISKS} for kk in ks]
        out[acq] = {"t_low": float(grid.s_edges[j1]), "t_high": float(grid.s_edges[j2]),
                    "A": float(grid.g_edges[k2]), "b_ladder": lad, "joint_slack": prof,
                    "slack_per_loss": per, "select_utility": u,
                    "select_risk_safe_endpoint": {k: float(R[k][j1, j2, ks[0], k2]) for k in RISKS},
                    "grid": grid.to_dict()}
        st, info = boundary_spanning(prof)
        out[acq]["spanning"] = {"status": st, **info}
        log(f"  {acq:8s} t=({out[acq]['t_low']:.4f}, {out[acq]['t_high']:.4f})  A={out[acq]['A']:.4f}  "
            f"{len(lad)} members  slack {prof[0]:+.4f} .. {prof[-1]:+.4f}  {st}  util={u:.4f}")
    return out


def family(stage):
    """One chain per acquisition: same (t_low, t_high, A, lambda), b ascending (safest first)."""
    lam = float(DECLARATION["veto"]["lambda_fixed"])
    pol = []
    for acq in DECLARATION["acquisitions"]:
        st = stage[acq]
        for b in st["b_ladder"]:
            pol.append(Policy(acq, st["t_low"], st["t_high"], float(b), st["A"], {"S": lam, "rest": 0.0},
                              float("inf"), f"{acq}|b={b:g}"))
    return tuple(pol)


def design_capacity(D, sel, stage):
    """Gamma_k(S) on the SELECT fold: the share of a unit's positive k-events whose catch status b can change.

    S = positive events that are escalated at the frozen (t_low, t_high) and have no edge-alarmed frame.
    An event outside S is caught (or missed) regardless of b, so it contributes no reach to the chain.
    The acquisitions share (t_low, t_high) only by coincidence, so the minimum over acquisitions is taken:
    the family is admissible only if EVERY chain can span the margin.
    """
    out = {}
    for k in RISKS:
        per_acq = []
        for acq in DECLARATION["acquisitions"]:
            st = stage[acq]
            s, y = D["s"][sel], D[f"y_{k}"][sel].astype(bool)
            ev, un = np.asarray(D["event"][sel]), np.asarray(D["unit"][sel])
            esc = (s >= st["t_low"]) & (s < st["t_high"])
            edge = s >= st["t_high"]
            e_ids, inv = np.unique(ev[y], return_inverse=True)
            has_edge = np.zeros(len(e_ids), bool)
            has_esc = np.zeros(len(e_ids), bool)
            np.logical_or.at(has_edge, inv, edge[y])
            np.logical_or.at(has_esc, inv, esc[y])
            sensitive = has_esc & ~has_edge
            first = np.zeros(len(e_ids), int)
            first[inv[::-1]] = np.arange(int(y.sum()))[::-1]
            e_unit = un[y][first]
            uu, uinv = np.unique(e_unit, return_inverse=True)
            frac = np.bincount(uinv, weights=sensitive) / np.bincount(uinv)
            per_acq.append(float(frac.mean()) if len(frac) else 0.0)
        out[k] = Capacity(k, min(per_acq), "design", DECLARATION["capacity"]["basis"])
    return out


# ------------------------------------------------------------------ preconditions

def verify(protocol, D, P, log=print):
    from .make_protocol import verify_tree
    rep, ok = {}, True
    sf = float(P.get("select_fraction", 0.3))

    span = events_spanning_units(protocol)
    rep["P1_events_spanning_units"] = span
    ok &= not span

    cert = certify_mask(D["unit"], sf, DECLARATION["split"]["seed"])
    dig, split = membership_digests(D, sf, DECLARATION["split"]["seed"])
    rep["P5_audit_membership"], rep["P5_split"] = dig, split
    counts = {k: v["n_units"] for k, v in dig.items()}
    ok &= all(v >= 59 for v in counts.values())

    log("select fold -> thresholds and b ladder chosen TOGETHER (frozen here, not at run time):")
    stage = select_stage(D, ~cert, P, counts, log=log)
    rstar = stage.pop("_r_star")
    losses = declared_losses()
    pol = family(stage)
    chains = build_chains(pol, losses)
    deltas = allocate_delta(chains, DECLARATION["delta"], None)
    rep["P2_chains"] = {"M": len(chains), "declared_M": DECLARATION["chains"]["M"], "delta_m": list(deltas),
                        "cover_id": chain_cover_id(chains), "select_bar_r_star": rstar,
                        "chains": [[p.name for p in c] for c in chains]}
    rep["frozen_family"] = {"thresholds": stage, "policies": [p.to_dict() for p in pol]}
    # contract I (v2.4.2): the construction each loss names, and the code identity of the implementation
    rep["P7_inference_registration"] = declare_inference(DECLARATION["study"], losses).to_dict()
    ok &= (len(chains) == DECLARATION["chains"]["M"]
           and all(abs(d - DECLARATION["chains"]["delta_m"]) < 1e-12 for d in deltas)
           and all(len(c) >= DECLARATION["chain_coordinate"]["min_members"] for c in chains))

    caps = design_capacity(D, ~cert, stage)
    rep["P6_design_capacity"] = {k: v.to_dict() for k, v in caps.items()}
    dm = float(DECLARATION["chains"]["delta_m"])
    rep["P6_margin"] = {k: {"n_units": counts[k], "r_star": rstar[k], "select_bar": rstar[k],
                            "required_margin_h_star": max(0.0, rstar[k] - rstar[k]),
                            "note": "the rule's select bar is r*, so Sec. 4.5 asks the coordinate to span "
                                    "h* = 0; the informativeness work is done by Sec. 4.6"} for k in RISKS}
    cap_status = chain_capacity(caps, counts, losses, float(DECLARATION["chains"]["delta_m"]), rstar)
    rep["P7_boundary_spanning"] = {a: stage[a]["spanning"] for a in DECLARATION["acquisitions"]}
    rep["P6_status"] = {k: {kk: v[kk] for kk in ("capacity", "source", "status", "usable_for_registration")}
                        for k, v in cap_status.items()}

    def contracts(_p):
        return Contract(True, True, counts, split["n_certify_units"],
                        {k: v["sha256"] for k, v in dig.items()}, 0.0, float("inf"))

    import yaml
    Y = yaml.safe_load((Path(protocol) / "protocol.yaml").read_text())
    rows = preflight_checklist(pol, losses, contracts=contracts, plan=None, payload_semantics=None,
                               weights_sha256=str((Y.get("detector") or {}).get("weights_sha256") or ""),
                               capacities=caps, delta=float(DECLARATION["delta"]), select_bar=rstar,
                               joint_slacks={a: stage[a]["joint_slack"] for a in DECLARATION["acquisitions"]},
                               inference_registration=declare_inference(DECLARATION["study"], losses))
    rep["P3_checklist"] = rows
    ok &= all(r["ok"] for r in rows)

    tree = verify_tree()
    rep["P4_runtime_tree"] = {k: tree[k] for k in ("ok", "extra", "missing", "changed")}
    ok &= tree["ok"]
    return rep, bool(ok)


def declared_losses():
    return tuple(miss_loss(L["name"], L["alpha"], L["lemma"], catch_set=L["catch_set"],
                           aggregation=L["aggregation"], weighting=L["weighting"],
                           empty_denominator=L["empty_denominator"]) for L in DECLARATION["losses"])


def load_all(dataset, protocol, det, agents):
    """Dev rows for every acquisition, with one g column each and a shared unit/event index."""
    from .esva_dev import load_dev
    D, P = None, None
    for name, path in sorted(agents.items()):
        Di, P = load_dev(dataset, protocol, det, path)
        if D is None:
            D = {k: v for k, v in Di.items() if k != "g"}
            D["unit"], D["event"] = np.asarray(Di["unit"]), np.asarray(Di["event"])
        elif not np.array_equal(np.asarray(D["uid"]), np.asarray(Di["uid"])):
            raise SystemExit(f"acquisition {name!r} covers different dev rows than the first one")
        D[f"g_{name}"] = np.asarray(Di["g"], float)
    return D, P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--agent", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from .make_protocol import analysis_code, sha256_file

    out = Path(a.out).expanduser()
    if out.exists() and not a.dry_run:
        raise SystemExit(f"{out} exists: a registration is written once")
    agents = dict(x.split("=", 1) for x in a.agent)
    if sorted(agents) != sorted(DECLARATION["acquisitions"]):
        raise SystemExit(f"--agent names {sorted(agents)} != declared {sorted(DECLARATION['acquisitions'])}")

    D, P = load_all(a.dataset, a.protocol, a.det, agents)
    rep, ok = verify(a.protocol, D, P)
    print(json.dumps(rep, indent=1))
    print("\npreconditions:", "ALL HOLD" if ok else "NOT SATISFIED - nothing is registered")
    if not ok:
        # provenance: a refused family is evidence that the design guards worked, so it is kept - clearly
        # named as a design audit and never as a registration.
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        au = out.with_name(out.stem + f".design_audit_{stamp}.json")
        au.parent.mkdir(parents=True, exist_ok=True)
        au.write_text(json.dumps({"study": STUDY, "spec_version": SPEC_VERSION, "outcome": "REFUSED",
                                  "utc": stamp, "declaration": DECLARATION, "preconditions": rep,
                                  "note": "design audit only; nothing was registered"}, indent=1))
        print("design audit ->", au)
        raise SystemExit(1)
    if a.dry_run:
        print("--dry-run: nothing written")
        return
    rec = {"study": STUDY, "spec_version": SPEC_VERSION,
           "registered_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "declaration": DECLARATION, "preconditions": rep, "checklist_items": list(CHECKLIST),
           "protocol_sha256": sha256_file(Path(a.protocol) / "protocol.yaml"),
           "splits_sha256": sha256_file(Path(a.protocol) / "splits.csv"),
           "analysis_code_sha256": analysis_code()[0], "det_sha256": sha256_file(a.det),
           "agent_sha256": {k: sha256_file(v) for k, v in sorted(agents.items())}, "note": a.note}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    print(json.dumps({k: rec[k] for k in ("study", "spec_version", "registered_utc", "protocol_sha256",
                                          "analysis_code_sha256")}, indent=1))
    print("cover_id:", rep["P2_chains"]["cover_id"])
    print("->", out)


if __name__ == "__main__":
    main()
