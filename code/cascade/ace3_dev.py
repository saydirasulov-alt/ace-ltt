"""ACE-LTT Study 3 run: A -> formal C3 -> ONE execution of the registered procedure.

    python -m cascade.ace3_dev --dataset DS --protocol P --det DET.csv \
        --agent crop=AG_CROP.csv --agent overlay=AG_OVERLAY.csv \
        --registration REG3.json --out DIR

This module chooses nothing. `cascade.ace3_register` froze the family, the arm-wise cover, the intra-chain
order, the delta shares and every R1 status (CAPABLE, KNIFE_EDGE, NO_REACH, R2) from calibration-SELECT data
and the design reference counts alone. Here, in this order and no other:

    1. verify the registration receipt and re-derive the frozen design; refuse on any difference;
    2. open the audit metadata A - Z_k, the realized n_cert_k, the membership digests. A is computed here,
       before any verdict on it exists;
    3. FORMAL C3 against A: n_cert_k >= n_min(alpha_k, delta_m) for the branch ALREADY CHOSEN, plus a
       policy-independent, declared audit membership, plus the nine pre-freeze items re-run against the
       realized contract. PASS or REFUSE, nothing else. There is no full -> fallback switch, no re-planning
       and no re-seeding: a shortfall refuses. The AUDIT RECEIPT is written at the end of this step, carrying
       A and that verdict together - it is the first artefact in this study containing a certify-fold
       quantity, and it is written whether C3 passes or refuses;
    4. ONE execution: one certify_family call per arm at delta_arm = 0.05. There is no second path, no
       ad-hoc recovery and no retry.

Two boundaries, two namespaces, never interchanged:

    r_star_ref  = certification_boundary(n_ref_k,  alpha_k, delta_m)   registered; design, h*_ref, CAPABLE,
                                                                       T, S, N, KNIFE_EDGE, R2
    r_star_cert = certification_boundary(n_cert_k, alpha_k, delta_m)   opened HERE; the certificate and X

Every T/S/N verdict below reads the REGISTERED h_star_ref and the REGISTERED CAPABLE bits. This module never
recomputes them from n_cert: that is what makes 'A may only refuse' true of the code and not only of the
prose. X is the one primitive that touches r_star_cert, and it is secondary and confounded.

CHAIN_TERMINAL is fail-closed at study level: any post-registration CHAIN_TERMINAL affecting a registered
primary pair sets P-4.5 = REFUSED. The denominator is never shrunk after execution and R2 is never recomputed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .ace3_register import (ARMS, DECLARATION, DELTA, DELTA_ARM, RISKS, all_rows, arm_covers,
                            chain_envelope_capacity, choose_branch, cover_is_arm_pure, declared_losses,
                            arm_min_capacity, design_reference_counts, load_certify, load_select,
                            positive_unit_ids, resolve_role_split, select_counts, select_stage,
                            split_params)
from .ace_ltt import (CHECKLIST, Contract, InferenceRegistration, NOT_CERTIFIED, ObservedLoss, Policy,
                      allocate_delta, certification_boundary, certify_family, choose_operating_point,
                      membership_digest, n_min, preflight_checklist)
from .evidence_veto import StratumSpec, discount, in_stratum
from .policy import metrics, unit_losses

TAU = float(DECLARATION["predictions"]["tolerance"]["tau"])


# ------------------------------------------------------------------ (1) the registration receipt

def check_registration(reg_path, protocol, det, agents):
    from .make_protocol import analysis_code, sha256_file, verify_tree
    rec = json.loads(Path(reg_path).expanduser().read_text())
    if rec.get("study") != DECLARATION["study"]:
        raise SystemExit(f"registration is for {rec.get('study')!r}, not {DECLARATION['study']!r}")
    if rec.get("receipt") != "R1":
        raise SystemExit("this is not an R1 registration receipt")
    if rec.get("audit_metadata_present", None) is not False:
        raise SystemExit("the registration receipt claims to carry audit metadata; under this study's stage "
                         "order (R1 -> A) it must not, and a receipt that does cannot be executed here")
    from .ace3_register import p0_protocol_state
    state = p0_protocol_state(protocol)
    if not state["ok"]:
        raise SystemExit("the protocol state no longer satisfies P0: " + "; ".join(state["problems"]))
    locked = (rec.get("preconditions", {}).get("P0_protocol_state") or {}).get("lock_sha256", "")
    if locked and locked != state["lock_sha256"]:
        raise SystemExit("protocol.lock differs from the one R1 was registered against")
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
    got = rec.get("preconditions", {}).get("R1_inference_registration")
    if not got:
        raise SystemExit("the registration carries no R1_inference_registration: this release pins the "
                         "p-value construction by code identity, so the study must be re-registered")
    ireg = InferenceRegistration(got["registration"], got["registry_id"], tuple(got["inferences"]))
    bad = ireg.problems(losses)
    if bad:
        raise SystemExit("the registered inference contract no longer describes this code: " + "; ".join(bad))
    return ireg


def rederive_design(rec, D, P, rs, log=print):
    """Re-derive everything R1 froze and refuse on any difference. SELECT fold only, by construction.

    `D` came from `load_select`, so this function has no certify row in reach. The certify fold is opened
    later, by `run()`, after the audit receipt exists.
    """
    if D.get("_side") != "select":
        raise SystemExit(f"rederive_design was handed a {D.get('_side')!r} fold; it takes the select fold")
    sel = all_rows(D)
    n_sel = select_counts(D, sel)
    n_ref = design_reference_counts(n_sel, rs.select_fraction)
    reg = rec["preconditions"]["R0_design_reference"]
    if n_sel != reg["n_sel"] or n_ref != reg["n_ref"]:
        raise SystemExit(f"the re-derived design reference counts differ: n_sel {n_sel} vs {reg['n_sel']}, "
                         f"n_ref {n_ref} vs {reg['n_ref']}")
    branch_name, branch = choose_branch(n_ref)
    if branch_name != reg["branch"]:
        raise SystemExit(f"the re-derived branch {branch_name!r} differs from the registered "
                         f"{reg['branch']!r}")
    stage = select_stage(D, sel, P, n_ref, branch["modes"], branch["delta_m"], log=lambda *_: None)
    r_star_ref = stage.pop("_r_star_ref")
    if json.dumps(stage, sort_keys=True) != json.dumps(rec["preconditions"]["R1_family"]["thresholds"],
                                                       sort_keys=True):
        raise SystemExit("the re-derived thresholds / ladders differ from the registered ones")
    losses = declared_losses()
    covers = arm_covers(stage, branch["modes"], losses)
    mixed = cover_is_arm_pure(covers)
    if mixed:
        raise SystemExit("the re-derived cover mixes arms: " + "; ".join(mixed))
    for arm in ARMS:
        if covers[arm]["cover_id"] != rec["preconditions"]["R1_cover"][arm]["cover_id"]:
            raise SystemExit(f"arm {arm}: the chain cover differs from the registered one")
        got = [[p.name for p in c] for c in covers[arm]["chains"]]
        if got != rec["preconditions"]["R1_cover"][arm]["chains"]:
            raise SystemExit(f"arm {arm}: the chain membership or intra-chain order differs from the "
                             f"registered one")
    names = [p.name for arm in ARMS for p in covers[arm]["policies"]]
    if names != [p["name"] for p in rec["preconditions"]["R1_family"]["policies"]]:
        raise SystemExit("the re-derived family differs from the registered one")
    if {k: float(v) for k, v in r_star_ref.items()} != {
            k: float(v) for k, v in rec["preconditions"]["R1_family"]["r_star_ref"].items()}:
        raise SystemExit("the re-derived r_star_ref differs from the registered one")
    log(f"frozen design re-derived: branch {branch_name!r}, "
        f"M={sum(len(v['chains']) for v in covers.values())}, "
        f"{ {a: covers[a]['cover_id'][:12] for a in ARMS} }")
    return {"sel": sel, "n_sel": n_sel, "n_ref": n_ref, "branch_name": branch_name,
            "branch": branch, "stage": stage, "r_star_ref": r_star_ref, "losses": losses,
            "covers": covers}


# ------------------------------------------------------------------ (2) the audit receipt

def open_audit_metadata(Dc, rs):
    """A: Z_k (as the audited unit sets), the realized n_cert_k, and the membership digests.

    `Dc` is the CERTIFY fold, loaded by `run()` only after R1 was verified against the select fold. This is
    the first certify label this study reads, and the read is candidate-independent by construction: it is a
    function of the split and the labels, and of no policy coordinate.
    """
    if Dc.get("_side") != "certify":
        raise SystemExit(f"open_audit_metadata was handed a {Dc.get('_side')!r} fold")
    cert = all_rows(Dc)
    out = {}
    for k in RISKS:
        ids = positive_unit_ids(Dc, cert, k)
        out[k] = {"n_units": len(ids), "sha256": membership_digest(ids)}
    unit = np.asarray(Dc["unit"])
    split = {"n_certify_frames": int(len(unit)), "n_certify_units": int(len(set(unit.tolist()))),
             "n_select_units": int(rs.n_select_units),
             "n_calibration_units": int(len(rs.units))}
    return out, split


def formal_c3(audit, split, branch, covers, losses, stage, caps_sel, r_star_ref, ireg, weights_sha256,
              log=print):
    """Formal C3 against A, plus the nine pre-freeze items on the REALIZED contract. PASS or REFUSE.

    No branch switch is reachable from here: `branch` is the one R1 chose, and a realized count below its
    n_min refuses even where the other branch's n_min would have been met. That is the registered rule.
    """
    problems = []
    need = {L.name: n_min(L.alpha, float(branch["delta_m"])) for L in losses}
    for L in losses:
        got = int(audit[L.name]["n_units"])
        if got < need[L.name]:
            problems.append(f"{L.name}: {got} realized certify units < n_min({L.alpha:g}, "
                            f"{branch['delta_m']:g}) = {need[L.name]} for the registered branch "
                            f"(no full -> fallback switch exists)")
        if not audit[L.name]["sha256"]:
            problems.append(f"{L.name}: the audit unit membership is not declared")
    counts = {k: int(v["n_units"]) for k, v in audit.items()}
    digs = {k: v["sha256"] for k, v in audit.items()}

    def realized_contracts(_p):
        return Contract(True, True, counts, int(split["n_certify_units"]), digs, 0.0, float("inf"))

    # The same two-bar discipline as R1, with the counts now REALIZED. `select_bar` stays the selection
    # guarantee r*_ref, while `counts` is n_cert, so the checklist's own boundary is r*_cert and the margin
    # it asks for is h*_C3 = max(0, r*_ref - r*_cert). If the realized audit support is worse than the design
    # reference count led the design to assume, r*_cert < r*_ref, the margin becomes positive, and A can
    # REFUSE here - which is exactly "audit metadata may refuse, never choose". It never recomputes CAPABLE,
    # the R1 statuses or R2: those are registered quantities of the design fold at bar = alpha.
    rows = {}
    for arm in ARMS:
        rows[arm] = preflight_checklist(
            covers[arm]["policies"], losses, contracts=realized_contracts, plan=None,
            payload_semantics=None, weights_sha256=weights_sha256,
            capacities=arm_min_capacity(caps_sel, arm), delta=DELTA_ARM,
            select_bar={k: float(r_star_ref[k]) for k in RISKS},
            joint_slacks={a: stage[a]["joint_slack"] for a in branch["modes"]},
            inference_registration=ireg)
        problems += [f"{arm}/{r['item']}: {r['detail']}" for r in rows[arm] if not r["ok"]]
    verdict = "PASS" if not problems else "REFUSE"
    log(f"formal C3 against A: {verdict}  n_cert={counts}  n_min={need}")
    r_cert = {L.name: certification_boundary(int(counts[L.name]), float(L.alpha),
                                             float(branch["delta_m"])) for L in losses}
    return {"verdict": verdict, "n_cert": counts, "n_min_required": need,
            "branch": {k: (list(v) if isinstance(v, tuple) else v) for k, v in branch.items()},
            "bars": {"checklist_select_bar_r_star_ref": {k: float(v) for k, v in r_star_ref.items()},
                     "checklist_counts_n_cert": dict(counts),
                     "r_star_cert": {k: float(v) for k, v in r_cert.items()},
                     "h_star_C3": {k: max(0.0, float(r_star_ref[k]) - float(r_cert[k])) for k in RISKS},
                     "note": ("h*_C3 = max(0, r*_ref - r*_cert): positive only when the realized support is "
                              "worse than the design reference count assumed, and then A may REFUSE. "
                              "CAPABLE, the R1 statuses and R2 are never recomputed here."),
                     "capacity_minimum_scope": "within one arm; never over the union of the arms"},
            "checklist_items": list(CHECKLIST), "checklist_realized": rows, "problems": problems,
            "powers": "verify or refuse only; A may not alter the family, cover, order, shares or branch"}


# ------------------------------------------------------------------ (4) the single execution

def _theta(p: Policy):
    return (p.t_low, p.t_high, p.b, p.A)


def _gv(D, mask, p: Policy, inS):
    return discount(D[f"g_{p.acq}"][mask], inS[mask], p.veto_at("S"))


def certify_risks_all_members(D, cert, inS, covers, audit):
    """R_cert_k and the evaluated unit set for EVERY registered member, computed once.

    Every member is needed because sp_cert_k(C) is a property of the chain's two endpoints, and the fixed
    sequence may stop before reaching the last one. Computing them here decides nothing: the family, the
    cover and every R1 status are already frozen, and this function cannot change any of them.
    """
    obs = {}
    for arm in ARMS:
        for p in covers[arm]["policies"]:
            gv = _gv(D, cert, p, inS)
            per = {}
            for k in RISKS:
                u, L = unit_losses(D["s"][cert], gv, D[f"y_{k}"][cert].astype(bool), _theta(p),
                                   D["unit"][cert], D["event"][cert])
                per[k] = ObservedLoss(float(L.mean()) if len(L) else 0.0, int(len(L)), membership_digest(u))
                if int(len(L)) != int(audit[k]["n_units"]):
                    raise SystemExit(f"{p.name}: {k} evaluated on {len(L)} units, audited "
                                     f"{audit[k]['n_units']} - C3 count identity broken")
            obs[p.name] = per
    return obs


def op_stats(D, cert, inS, p: Policy):
    gv = _gv(D, cert, p, inS)
    m = metrics(D["s"][cert], gv, D["y"][cert].astype(bool), _theta(p), D["unit"][cert], None,
                D["event"][cert])
    return {k: m[k] for k in ("fa", "calls", "handoff", "miss_event", "miss_unit") if k in m}


# ------------------------------------------------------------------ the P-4.5 primitives

def evaluate_predictions(rec, covers, obs, r_star_cert, caps_cert):
    """T, S, N, X per registered (C,k), from the REGISTERED h_star_ref and CAPABLE bits.

    Nothing here is recomputed from n_cert except X, which is what X is for.
    """
    reg = rec["preconditions"]["R1_statuses"]
    h = {k: float(v) for k, v in reg["h_star_ref"].items()}
    pairs = {}
    for arm in ARMS:
        for c in covers[arm]["chains"]:
            key = f"{arm}|{c[0].acq}"
            for k in RISKS:
                pk = f"{key}|{k}"
                r = reg["pairs"].get(pk)
                if r is None:
                    continue
                anchor, last = obs[c[0].name][k].risk, obs[c[-1].name][k].risk
                sp = float(last - anchor)
                gam = float(r["gamma_sel"])
                row = {"chain": key, "loss": k, "arm": arm, "mode": c[0].acq, "J": len(c),
                       "gamma_sel_registered": gam, "h_star_ref_registered": h[k],
                       "capable_registered": bool(r["capable"]),
                       "knife_edge_registered": bool(r["knife_edge"]),
                       "no_reach_registered": bool(r["no_reach"]),
                       "r_cert_anchor": float(anchor), "r_cert_last": float(last), "sp_cert": sp,
                       "gamma_cert_DIAGNOSTIC": float(caps_cert[key][k].value),
                       "r_star_cert": float(r_star_cert[k]),
                       "T": bool(sp <= gam + TAU),
                       "S": (bool(sp >= h[k] - TAU) if r["capable"] else None),
                       "N": (bool(sp < h[k] - TAU) if not r["capable"] else None),
                       "X": bool(anchor <= float(r_star_cert[k])),
                       "X_antecedent_registered": bool(
                           r["capable"] and float(rec["preconditions"]["R1_family"]["r_star_ref"][k])
                           < float(anchor))}
                pairs[pk] = row
    return {"h_star_ref_registered": h, "r_star_cert": {k: float(v) for k, v in r_star_cert.items()},
            "pairs": pairs,
            "namespace_note": ("T, S, N read the registered h_star_ref and CAPABLE bits; only X reads "
                               "r_star_cert. The two boundaries are never interchanged."),
            "N_is_derived_never_evidence": True}


def chain_terminal_of(arm, res, covers):
    """Which REGISTERED primary pairs a CHAIN_TERMINAL touched, and whether one was observed at all.

    Returns (pairs, observed). It iterates the REGISTERED chains, never `res.chains`, because
    `certify_family` DROPS a chain that goes terminal in its stages (4) and (5) - the authoritative
    re-check at the share, and the chain certificate - so the engine's surviving chains do not contain the
    very chains this rule is about. Up to v2.4.8 the extraction looped `res.chains`, and those two shapes
    therefore produced an empty mapping while `reasons` carried CHAIN_TERMINAL: the study reported its
    registered R2 verdict instead of REFUSED.

    A registered chain is terminal under either of two independent criteria:

      (a) it is absent from `res.chains`. `certify_family` builds its cover over the survivors and
          `res.chains` is a sub-tuple of it; the caller has already refused unless `res.cover_id` equals the
          registered cover id, so the engine's cover IS the registered cover and absence can only mean the
          chain was dropped after the cover was fixed. This criterion reads no reason string;
      (b) one of its members is in `res.inadmissible`. This is the shape where the chain survives the cover
          but a member fails later - the C3_CERTIFIABILITY_AUDIT mismatch between the evaluated and the
          audited unit set.

    `observed` is additionally true whenever the engine reported CHAIN_TERMINAL, so a future change to either
    criterion cannot silence the refusal.
    """
    survived = {tuple(p.name for p in c) for c in res.chains}
    pairs, seen = set(), "CHAIN_TERMINAL" in res.reasons
    for c in covers[arm]["chains"]:
        dropped = tuple(p.name for p in c) not in survived
        member_bad = any(p.name in res.inadmissible for p in c)
        if dropped or member_bad:
            seen = True
            for k in RISKS:
                pairs.add(f"{arm}|{c[0].acq}|{k}")
    return pairs, seen


def p45_verdict(rec, preds, terminal_pairs, terminal_observed=False):
    """P-4.5 in {INFORMATIVE, UNINFORMATIVE, REFUSED}. Fail-closed on CHAIN_TERMINAL.

    Two arguments, not one, and the distinction is the whole point (v2.4.9). `terminal_observed` is the
    REFUSAL AUTHORITY: the engine reported CHAIN_TERMINAL, or a registered chain did not survive the cover.
    `terminal_pairs` is the DIAGNOSTIC: which registered primary pairs it touched. Up to v2.4.8 the refusal
    was conditioned on the diagnostic being non-empty, so any failure of the mapping from a terminal chain to
    its pairs silently restored the registered R2 verdict and quietly kept the full denominator. A mapping is
    a place a bug can live; the observation is not. So the observation refuses, and the mapping only says
    where.
    """
    r2 = rec["preconditions"]["R1_rule_R2"]
    if terminal_observed or terminal_pairs:
        return {"verdict": "REFUSED", "reason": "CHAIN_TERMINAL on registered primary pair(s)",
                "terminal_pairs": sorted(terminal_pairs),
                "chain_terminal_observed": True,
                "refusal_authority": ("the observation of CHAIN_TERMINAL, not the pair mapping: an empty "
                                      "terminal_pairs list with a terminal observation still REFUSES"),
                "rule": DECLARATION["predictions"]["CHAIN_TERMINAL"],
                "R2_registered": r2["verdict"], "R2_recomputed": False,
                "denominator_shrunk": False,
                "T_verdict": None, "S_verdict": None}
    elig = [p for k, p in preds["pairs"].items()
            if not p["knife_edge_registered"] and not p["no_reach_registered"]]
    T_pairs = [p for p in preds["pairs"].values() if not p["no_reach_registered"]]
    S_pairs = [p for p in elig if p["capable_registered"]]
    return {"verdict": r2["verdict"], "reason": "as registered at R1; never recomputed",
            "terminal_pairs": [], "chain_terminal_observed": False,
            "R2_registered": r2["verdict"], "R2_recomputed": False,
            "denominator_shrunk": False,
            "T_verdict": {"n": len(T_pairs), "held": sum(bool(p["T"]) for p in T_pairs),
                          "all_held": all(p["T"] for p in T_pairs) if T_pairs else None},
            "S_verdict": {"n": len(S_pairs), "held": sum(bool(p["S"]) for p in S_pairs),
                          "all_held": all(p["S"] for p in S_pairs) if S_pairs else None},
            "interpretation_gate": (DECLARATION["predictions"]["uninformative_rule"]
                                    if r2["verdict"] == "UNINFORMATIVE" else
                                    "confirmatory reading of T and S is licensed, as ONE instance")}


# ------------------------------------------------------------------ run

def run(a, log=print):
    from .make_protocol import analysis_code, sha256_file
    out = Path(a.out).expanduser()
    rep_path = out / "ace3_dev.json"
    audit_path = out / "ace3_audit_receipt.json"
    if rep_path.exists():
        raise SystemExit(f"{rep_path} exists: the registered run is executed ONCE. There is no resume path "
                         f"and no ad-hoc recovery path in this study.")
    agents = dict(x.split("=", 1) for x in a.agent)
    rec = check_registration(a.registration, a.protocol, a.det, agents)
    log(f"R1 receipt {rec['registered_utc']} ({rec['spec_version']}) verified")

    # ---- (1) stage 1 metadata, then the SELECT fold only. No certify row is opened in this block. ---
    from .esva_dev import read_protocol
    P0 = read_protocol(a.protocol, need_esva=False)
    sf, seed = split_params(P0)
    rs = resolve_role_split(a.protocol, sf, seed)
    log(f"role split (metadata only): {rs.n_select_units} select / {rs.n_certify_units} certify units")
    Ds, P = load_select(a.dataset, a.protocol, a.det, agents, rs)
    st = rederive_design(rec, Ds, P, rs, log=log)
    sel, covers, losses = st["sel"], st["covers"], st["losses"]
    branch, stage = st["branch"], st["stage"]
    spec = StratumSpec(DECLARATION["veto"]["stratum"]["kind"], DECLARATION["veto"]["stratum"]["area_max"])

    caps_sel = chain_envelope_capacity(Ds, sel, stage, covers, "design",
                                       "calibration-select fold, at the frozen thresholds")
    ireg = registered_inference(rec, losses)

    # ---- (2) R1 is verified. ONLY NOW is the certify fold opened and A computed. ---------------------
    log("R1 verified against the select fold; opening the certify fold")
    Dc, _Pc = load_certify(a.dataset, a.protocol, a.det, agents, rs)
    inS = in_stratum(spec, Dc["features"])
    cert = all_rows(Dc)
    audit, split = open_audit_metadata(Dc, rs)
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    Y = yaml.safe_load((Path(a.protocol) / "protocol.yaml").read_text())
    w_sha = str((Y.get("detector") or {}).get("weights_sha256") or "")

    # ---- (3) FORMAL C3: verify or refuse; nothing else, then the receipt carrying A AND that verdict -
    c3 = formal_c3(audit, split, branch, covers, losses, stage, caps_sel, st["r_star_ref"], ireg, w_sha,
                   log=log)
    receipt = {"study": DECLARATION["study"], "receipt": "AUDIT", "spec_version": rec["spec_version"],
               "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "verifies_registration": {"utc": rec["registered_utc"],
                                         "sha256": sha256_file(a.registration)},
               "audit_metadata": {"Z_k": "exclude_unit: the audited unit set per loss, label-derived",
                                  "membership": audit, "split": split,
                                  "candidate_independent": True},
               "formal_C3": c3,
               "powers": ("these values may only verify the already frozen procedure or refuse it; they "
                          "cannot alter the family, the cover, the order, the shares or the branch"),
               "carries_no_design_quantity": True}
    audit_path.write_text(json.dumps(receipt, indent=1))
    log(f"audit receipt -> {audit_path}")
    if c3["verdict"] != "PASS":
        (out / "ace3_REFUSED.json").write_text(json.dumps(
            {"study": DECLARATION["study"], "outcome": "REFUSED", "stage": "formal C3 against A",
             "P-4.5": "REFUSED", "problems": c3["problems"],
             "note": "nothing was executed; no certify-fold risk was evaluated"}, indent=1))
        raise SystemExit("formal C3 REFUSED:\n  " + "\n  ".join(c3["problems"]))

    # ---- (4) ONE execution -------------------------------------------------------------------------
    counts = c3["n_cert"]
    digs = {k: v["sha256"] for k, v in audit.items()}
    obs = certify_risks_all_members(Dc, cert, inS, covers, audit)

    def contracts(_p):
        return Contract(True, True, counts, split["n_certify_units"], digs, 0.0, float("inf"))

    results, terminal_pairs, terminal_observed = {}, set(), False
    for arm in ARMS:
        res = certify_family(covers[arm]["policies"], losses, delta=DELTA_ARM, contracts=contracts,
                             certify_eval=lambda p: obs[p.name], plan=None, inference_registration=ireg)
        if res.cover_id != covers[arm]["cover_id"]:
            raise SystemExit(f"arm {arm}: the engine's cover id differs from the registered one")
        results[arm] = res
        tp, seen = chain_terminal_of(arm, res, covers)
        terminal_pairs |= tp
        terminal_observed = terminal_observed or seen
        log(f"arm {arm}: {len(res.certified)}/{len(covers[arm]['policies'])} certified, "
            f"reasons {list(res.reasons) or ['-']}")

    r_star_cert = {L.name: certification_boundary(int(counts[L.name]), float(L.alpha),
                                                  float(branch["delta_m"])) for L in losses}
    caps_cert = chain_envelope_capacity(Dc, cert, stage, covers, "realized",
                                        "certify fold; DIAGNOSTIC ONLY")
    preds = evaluate_predictions(rec, covers, obs, r_star_cert, caps_cert)
    verdict = p45_verdict(rec, preds, terminal_pairs, terminal_observed)

    u = DECLARATION["utility"]
    certified = [p for arm in ARMS for p in results[arm].certified]
    stats = {p.name: op_stats(Dc, cert, inS, p) for p in certified}

    def utility(p):
        s = stats[p.name]
        return (float(u["calls"]) * s["calls"] + float(u["fa"]) * s["fa"]
                + float(u["handoff"]) * s.get("handoff", 0.0))

    per_arm = {}
    for arm in ARMS:                      # the operating point is reported per arm, never pooled
        if results[arm].ok:
            b, i = choose_operating_point(results[arm], utility)
            per_arm[arm] = {"policy": b.to_dict(), **i, "stats": stats[b.name]}

    rep = {"study": DECLARATION["study"], "spec_version": rec["spec_version"],
           "status": DECLARATION["status"],
           "run_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "stages_executed": ["R1 verified", "A opened", "formal C3 PASS", "one execution"],
           "registration": {"utc": rec["registered_utc"], "sha256": sha256_file(a.registration)},
           "audit_receipt": {"path": audit_path.name,
                             "created_utc": receipt["created_utc"],
                             "n_cert": counts, "membership": digs},
           "analysis_code_sha256": analysis_code()[0],
           "branch": st["branch_name"], "n_sel": st["n_sel"], "n_ref": st["n_ref"], "n_cert": counts,
           "role_split_metadata_only": rs.to_dict(),
           "access_boundary": ("the select fold and the certify fold are loaded by two separate calls; the "
                               "certify call happens after R1 is verified, so no certify label, score or "
                               "feature is read before R1"),
           "boundaries": {"r_star_ref_registered": {k: float(v) for k, v in st["r_star_ref"].items()},
                          "r_star_cert": {k: float(v) for k, v in r_star_cert.items()},
                          "note": preds["namespace_note"]},
           "thresholds": stage, "split": split,
           "cover": {arm: {"cover_id": covers[arm]["cover_id"],
                           "chains": [[p.name for p in c] for c in covers[arm]["chains"]],
                           "delta_arm": DELTA_ARM,
                           "delta_m": [float(d) for d in allocate_delta(covers[arm]["chains"],
                                                                        DELTA_ARM, None)]}
                     for arm in ARMS},
           "delta_total": DELTA,
           "certification": {arm: results[arm].to_dict() for arm in ARMS},
           "certify_risks": {n: {k: v.to_dict() for k, v in per.items()} for n, per in obs.items()},
           "capacity": {"design_registered": rec["preconditions"]["R1_capacity"],
                        "realized_DIAGNOSTIC_ONLY": {key: {k: v.to_dict() for k, v in per.items()}
                                                     for key, per in caps_cert.items()}},
           "P45_predictions": preds, "P45": verdict,
           "operating_point": {"per_arm": per_arm, "pooled": None,
                               "why_not_pooled": ("the arms are two registered families with their own "
                                                  "delta; a pooled operating point would compare across "
                                                  "two separate guarantees")},
           "utility_caveat": u["caveat"],
           "selection_side_stats": stats,
           "one_fold_two_uses": DECLARATION["one_fold_two_uses"],
           "source_concentration": DECLARATION["source_concentration"]}
    rep_path.write_text(json.dumps(rep, indent=1))
    md = report_md(rep, results)
    (out / "ace3_dev.md").write_text(md)
    log("\n" + md)
    log(f"-> {rep_path}")
    return rep


def report_md(rep, results):
    L = [f"# ACE-LTT Study 3 — prospective stress test of the Sec. 4.5 extension — {rep['run_utc']}", "",
         f"**{rep['status']}**", "",
         f"R1 {rep['registration']['utc']} · spec {rep['spec_version']} · "
         f"code `{rep['analysis_code_sha256'][:12]}` · branch `{rep['branch']}`",
         f"n_sel {rep['n_sel']} → n_ref {rep['n_ref']} → n_cert {rep['n_cert']}",
         f"r*_ref {({k: round(v, 5) for k, v in rep['boundaries']['r_star_ref_registered'].items()})} · "
         f"r*_cert {({k: round(v, 5) for k, v in rep['boundaries']['r_star_cert'].items()})}", "",
         "## Registered cover (per arm; arms are never mixed)", "",
         "| arm | M | δ_arm | δ_m | chains |", "|---|---|---|---|---|"]
    for arm, c in rep["cover"].items():
        L.append(f"| {arm} | {len(c['chains'])} | {c['delta_arm']:g} | "
                 f"{', '.join(f'{d:g}' for d in c['delta_m'])} | "
                 f"{'; '.join(str(len(x)) + ' members' for x in c['chains'])} |")
    L += ["", "## Certification", ""]
    for arm in rep["cover"]:
        res = results[arm]
        cert = {p.name for p in res.certified}
        L += [f"### arm {arm} (δ = {res.delta:g})", "",
              "| policy | fire | smoke | p_joint | δ_m | certified |", "|---|---|---|---|---|---|"]
        for c in res.chains:
            for p in c:
                o, pv = rep["certify_risks"].get(p.name), res.pvalues.get(p.name)
                if o is None or pv is None:
                    L.append(f"| {p.name} | — | — | — | — | not tested (sequence stopped earlier) |")
                    continue
                L.append(f"| {p.name} | {o['fire']['risk']:.4f} | {o['smoke']['risk']:.4f} | "
                         f"{pv['p_joint']:.5f} | {pv['delta_m']:.4f} | "
                         f"{'YES' if p.name in cert else 'no'} |")
        if res.reasons:
            L.append("")
            L.append("clauses that fired: " + ", ".join(f"`{r}`" for r in res.reasons))
        L.append("")
    v = rep["P45"]
    L += ["## P-4.5", "", f"**{v['verdict']}** — {v['reason']}", ""]
    if v["verdict"] != "REFUSED":
        L += [f"T: {v['T_verdict']['held']}/{v['T_verdict']['n']} pairs · "
              f"S: {v['S_verdict']['held']}/{v['S_verdict']['n']} capable pairs", "",
              "| (C,k) | J | Γ_sel | h*_ref | CAPABLE | sp_cert | T | S | N | X | flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, p in rep["P45_predictions"]["pairs"].items():
            flags = ",".join(x for x, y in (("KNIFE_EDGE", p["knife_edge_registered"]),
                                            ("NO_REACH", p["no_reach_registered"])) if y) or "—"
            L.append(f"| {key} | {p['J']} | {p['gamma_sel_registered']:.4f} | "
                     f"{p['h_star_ref_registered']:.4f} | {'yes' if p['capable_registered'] else 'no'} | "
                     f"{p['sp_cert']:.4f} | {'✓' if p['T'] else '✗'} | "
                     f"{'—' if p['S'] is None else ('✓' if p['S'] else '✗')} | "
                     f"{'—' if p['N'] is None else ('✓' if p['N'] else '✗')} | "
                     f"{'✓' if p['X'] else '✗'} | {flags} |")
        L += ["", "*N is derived and never counted as evidence; X is secondary and confounded. T, S and N "
                  "read the registered h\\*_ref and CAPABLE bits; only X reads r\\*_cert.*"]
    else:
        L += ["terminal pairs: " + (", ".join(f"`{x}`" for x in v["terminal_pairs"])
                                    or "none identified — the refusal rests on the OBSERVATION of "
                                       "CHAIN_TERMINAL, not on this mapping"), "",
              f"*{v['rule']}*"]
    op = rep["operating_point"]["per_arm"]
    L += ["", "## Operating point (reported per arm; never pooled)", ""]
    if op:
        for arm, o in op.items():
            s = o["stats"]
            L.append(f"- arm {arm}: **{o['policy']['name']}** (utility {o['utility']:.4f} over "
                     f"{o['n_certified']} certified) — FA {s['fa']:.4f}, calls {s['calls']:.4f}, "
                     f"hand-off {s.get('handoff', 0.0):.4f}")
        L += ["", f"*{rep['utility_caveat']}*"]
    else:
        L += ["**NOT CERTIFIED** — " + ", ".join(
            f"`{r}`: {NOT_CERTIFIED.get(r, '')}" for arm in rep["cover"] for r in results[arm].reasons)]
    L += ["", "## Scope", "",
          f"- {rep['source_concentration']}",
          f"- allowed: {'; '.join(rep['one_fold_two_uses']['allowed'])}",
          f"- never written: {'; '.join(rep['one_fold_two_uses']['not_allowed'])} — "
          f"{rep['one_fold_two_uses']['why']}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--agent", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--registration", required=True)
    ap.add_argument("--out", required=True)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
