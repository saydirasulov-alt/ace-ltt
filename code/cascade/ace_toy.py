"""ACE-LTT on a GENERIC synthetic instance: the whole method, end to end, with no application data.

    python -m cascade.ace_toy [--out DIR] [--seed 20260925]

Purpose. The two registered studies are fire/smoke studies. A reader cannot tell from them which parts of
ACE-LTT are method and which are wildfire. This module answers that: it instantiates the full path -
policy object, admissibility contract, chain cover, delta plan, fixed-sequence certification, operating
point, and the two family-design diagnostics - on an abstract two-tier stream of 8 candidates that has
nothing to do with fire, smoke, cameras or YOLO.

It is an ILLUSTRATION, not evidence. The generator below is a declared, seeded, three-line model, and its
constants were chosen so that the instance exhibits the behaviours the method is built around (a chain that
truncates, a chain that does not, a refusal). Nothing here is a measurement of the real system, and no
number from this module may be reported as an empirical result. What it does establish is that the method
runs, and what each of its objects does, on data no application supplies.

It is DELIBERATELY NOT part of ANALYSIS_CODE. Adding it there would change analysis_code_sha256 and both
registrations would stop verifying against the installed tree. It reads no protocol, no dataset and no
detector output, so it cannot influence any registered quantity in either direction.

Scenes:
  1  nominal   - family design on the SELECT fold (capacity, slack, spanning, 8/8 checklist), then
                 certification on the CERTIFY fold and the operating point.
  2  ordering  - the strict-Pareto-front counterexample, measured: the strict front discards a certifiable
                 candidate that the ACE inclusion order keeps.
  3  refusals  - C3 (audit support below n_min), C6 (a loss with no monotonicity lemma), C7 (payload
                 semantics not frozen) and contract I (the p-value machinery is not the registered
                 machinery): four fail-closed paths, each returning a reason code, not a number.
  4  diagnostics - the same chain re-scored against a smaller acting domain S, so that the SAME family is
                 MARGIN_CAPABLE for one declaration of S and MARGIN_INCAPABLE for another; and a family
                 that is NOT_SPANNING.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

from .ace_ltt import (BOUNDARY_SPANNING, Capacity, Contract, DeltaPlan, Policy, PayloadSemantics,
                      SPEC_VERSION, InferenceRegistration, declare_inference,
                      boundary_spanning, build_chains, certification_boundary, certification_margin,
                      certification_slack, chain_capacity, chain_cover_id, chain_signature, certify_family,
                      choose_operating_point, miss_loss, n_min, payload_loss, preflight_checklist,
                      select_front)

# ------------------------------------------------------------------ the declared generator (illustration)

GEN = {
    "n_units": 900, "seed": 20260925,
    "frames_per_unit": [4, 5, 6, 7, 8],          # uniform over this set
    "events_per_unit": [0, 0, 1, 1, 2, 3],       # uniform: ~2/3 of units carry at least one event
    "frames_per_event": [1, 2, 3],
    "s_positive": (0.46, 0.15), "s_negative": (0.15, 0.10),     # edge score ~ N(mu, sd), clipped to [0,1]
    "g_gain": {"a1": 0.30, "a2": 0.14},          # verifier lift on a positive frame, per acquisition mode
    "g_base": (0.40, 0.20),                      # verifier score ~ N(mu, sd) + gain*y, clipped to [0,1]
    "bytes_per_escalated_frame": {"a1": 22.0, "a2": 41.0},      # KiB, per acquisition mode
    # The acquisition contract's DETERMINISTIC transport cap: a protocol parameter, not a policy coordinate.
    # In this instance every escalated frame transmits exactly bytes_per_escalated_frame[a] KiB by
    # construction, so |r_a(x)| <= that constant for every input x - which is what clause C4 needs. It is a
    # static bound, never an observed maximum.
    "b_max_kib": 48.0,
    "band": (0.20, 0.36), "A": 0.55,
    "b_ladder": (0.00, 0.15, 0.30, 0.45),
    "lambda_S": 0.0, "select_fraction": 0.3,
}
ALPHA, BETA, DELTA = 0.05, 0.05, 0.10
B0 = 30.0                                        # KiB per evaluated frame, on the registered grid
REGISTRATION = "ace_ltt_toy_generic/v1"
# Contract I, registration side (v2.4.2): the construction each loss names, and the code identity of the
# implementation, are frozen HERE and re-verified by certify_family before any p-value is computed.


def synth(gen=GEN):
    """Abstract two-tier stream. Units -> frames; some frames belong to a positive event.

    Nothing is domain-specific: a unit is a group that must not be split across folds, an event is a thing
    that must be caught at least once, a frame is one observation carrying a cheap score s and, if it is
    escalated, a verifier score g_a that depends on the acquisition mode a.
    """
    rng = np.random.default_rng(gen["seed"])
    unit, event, y = [], [], []
    for u in range(gen["n_units"]):
        n_f = int(rng.choice(gen["frames_per_unit"]))
        n_e = int(rng.choice(gen["events_per_unit"]))
        ev = [f"u{u}e{j}" for j in range(n_e)]
        pos = []
        for e in ev:
            pos += [e] * int(rng.choice(gen["frames_per_event"]))
        pos = pos[:max(0, n_f - 1)]                                  # at least one negative frame per unit
        for e in pos:
            unit.append(f"u{u}"); event.append(e); y.append(1)
        for _ in range(n_f - len(pos)):
            unit.append(f"u{u}"); event.append(""); y.append(0)
    y = np.array(y, bool)
    n = len(y)
    mu, sd = np.where(y, gen["s_positive"][0], gen["s_negative"][0]), np.where(y, gen["s_positive"][1],
                                                                              gen["s_negative"][1])
    s = np.clip(rng.normal(mu, sd), 0.0, 1.0)
    g = {}
    for a, lift in gen["g_gain"].items():
        g[a] = np.clip(rng.normal(gen["g_base"][0], gen["g_base"][1], n) + lift * y, 0.0, 1.0)
    return {"unit": np.array(unit), "event": np.array(event), "y": y, "s": s, **{f"g_{a}": v for a, v in g.items()},
            "n_frames": n}


def split(unit, fraction, seed):
    """Group-level select/certify split: a unit is wholly on one side. Deterministic in (unit id, seed)."""
    ids = np.unique(unit)
    key = np.array([int(hashlib.sha256(f"{seed}|{u}".encode()).hexdigest()[:8], 16) for u in ids])
    sel = set(ids[np.argsort(key)[:int(round(fraction * len(ids)))]].tolist())
    return np.array([u in sel for u in unit]), np.array([u not in sel for u in unit])


# ------------------------------------------------------------------ the policy's measured quantities

def decide(s, gv, p: Policy):
    """0 silent, 1 edge alarm, 2 verifier alarm, 3 dismissed, 4 hand-off. response = {1, 2, 4}."""
    d = np.zeros(len(s), int)
    edge = s >= p.t_high
    esc = (s >= p.t_low) & ~edge
    d[edge] = 1
    d[esc & (gv >= p.A)] = 2
    d[esc & (gv >= p.b) & (gv < p.A)] = 4
    d[esc & (gv < p.b)] = 3
    return d


def gv_of(D, p: Policy, mask):
    """The verifier score under the policy's declared veto authority on the single stratum 'S'."""
    return np.minimum(1.0, D[f"g_{p.acq}"][mask] + p.veto_at("S"))


def miss_risk(D, p: Policy, mask):
    """unit_mean_over_events / exclude_unit, catch_set = response. Returns (risk, n_audit_units)."""
    y, unit, event = D["y"][mask], D["unit"][mask], D["event"][mask]
    d = decide(D["s"][mask], gv_of(D, p, mask), p)
    caught_frame = np.isin(d, (1, 2, 4)) & y
    ev, inv = np.unique(event[y], return_inverse=True)
    caught = np.zeros(len(ev), bool)
    np.logical_or.at(caught, inv, caught_frame[y])
    ev_unit = np.array([unit[y][i] for i in np.unique(inv, return_index=True)[1][np.argsort(
        np.unique(inv, return_index=True)[0])]]) if len(ev) else np.array([])
    per_unit = {}
    for e_i, u in enumerate(ev_unit):
        per_unit.setdefault(u, []).append(0.0 if caught[e_i] else 1.0)
    L = np.array([float(np.mean(v)) for v in per_unit.values()]) if per_unit else np.array([])
    return (float(L.mean()) if len(L) else 0.0), int(len(L)), per_unit


def audit_units(D, mask):
    """WHICH units enter the miss risk: a function of the split and the labels only, never of a policy."""
    y, unit = D["y"][mask], D["unit"][mask]
    return sorted(set(unit[y].tolist()))


def toy_digest(ids):
    """This instance's audit-membership digest, used on BOTH sides of the certify-time identity check.

    From v2.4.1 every observation must say which units it was evaluated on, and the engine compares that
    digest with the audited one; the two sides must therefore be produced by one function.
    """
    return hashlib.sha256("|".join(sorted({str(u) for u in ids})).encode()).hexdigest()


def payload_risk(D, p: Policy, mask, b0=B0):
    """Z_{B,U} = 1{C_U > B0}, C_U = mean KiB per evaluated frame. Constant along a b ladder by construction."""
    unit = D["unit"][mask]
    d = decide(D["s"][mask], gv_of(D, p, mask), p)
    # every escalated frame pays: d in {2, 3, 4} all mean the verifier was called on a transmitted payload
    kib = np.where(d >= 2, GEN["bytes_per_escalated_frame"][p.acq], 0.0)
    ids = np.unique(unit)
    c = np.array([kib[unit == u].mean() for u in ids])
    return float((c > b0).mean()), int(len(ids)), float(c.max())


def frame_costs(D, p: Policy, mask):
    """Deployment costs: false alarm, verifier calls, hand-off. Never constrained losses (P2)."""
    y = D["y"][mask]
    d = decide(D["s"][mask], gv_of(D, p, mask), p)
    alarm = np.isin(d, (1, 2))
    return {"fa": float(alarm[~y].mean()), "calls": float((d >= 2).mean()), "handoff": float((d == 4).mean())}


def acting_domain(D, p: Policy, mask):
    """S: the events whose catch status the chain's coordinate b can change at all.

    An event is in S iff none of its positive frames is edge-alarmed (s >= t_high) and at least one is
    escalated. Computed from thresholds and labels, not from b, so it is the same for every chain member.
    """
    y, event = D["y"][mask], D["event"][mask]
    s = D["s"][mask]
    edge, esc = s >= p.t_high, (s >= p.t_low) & (s < p.t_high)
    out = {}
    for e, ed, es in zip(event[y], edge[y], esc[y]):
        r = out.setdefault(e, [False, False])
        r[0] |= bool(ed); r[1] |= bool(es)
    return {e for e, (ed, es) in out.items() if (not ed) and es}


def capacity_gamma(D, p: Policy, mask, source, basis):
    """Gamma(S) = (1/|U|) sum_u |E_u n S| / |E_u|, with the loss's own weights and denominators."""
    _, _, per_unit = miss_risk(D, p, mask)
    S = acting_domain(D, p, mask)
    y, unit, event = D["y"][mask], D["unit"][mask], D["event"][mask]
    ev_of_unit = {}
    for u, e in zip(unit[y], event[y]):
        ev_of_unit.setdefault(u, set()).add(e)
    vals = [len(es & S) / len(es) for u, es in ev_of_unit.items() if es and u in per_unit]
    return Capacity("miss", float(np.mean(vals)) if vals else 0.0, source, basis)


# ------------------------------------------------------------------ the declared family

MISS = miss_loss("miss", ALPHA, "L-toy-1: unit_mean_over_events is non-increasing in the response set",
                 catch_set="response", aggregation="unit_mean_over_events", empty_denominator="exclude_unit")
PAY = payload_loss(BETA, "budget")
LOSSES = (MISS, PAY)
# Contract I, registration side (v2.4.2): the construction each loss names AND the code identity of the
# implementation are frozen here; certify_family recomputes both and refuses on any difference.
IREG = declare_inference(REGISTRATION, LOSSES)
SEM = PayloadSemantics(REGISTRATION, "split_group (a unit is never split across folds)", (B0,), BETA,
                       b_max=float("inf"))
UTILITY_WEIGHTS = {"calls": 1.0, "fa": 10.0, "handoff": 4.0}


def family(gen=GEN):
    tl, th = gen["band"]
    return tuple(Policy(a, tl, th, b, gen["A"], (("S", gen["lambda_S"]),), B0, f"{a}|b={b:g}")
                 for a in ("a1", "a2") for b in gen["b_ladder"])


def contracts_for(D, cert, sel, pol_list):
    """One Contract per candidate. Counts AND membership are identical across the family by construction.

    The audit counts and membership are functions of the SPLIT and the LABELS only (clause C3 / assumption
    A4), never of a candidate's behaviour, which is why one digest serves the whole family.

    Clause C4 is discharged STATICALLY, not from data. The quantity compared against the declared transport
    cap is the construction bound |r_a(x)| <= bytes_per_escalated_frame[a], which holds for every input by
    the definition of the acquisition mode. A maximum observed on the design fold would be a sample
    statement and would not establish a forall-x bound, so it is reported separately as a diagnostic and is
    never the cap. No fold-dependent quantity enters C4 at all.
    """
    units = audit_units(D, cert)
    dig = toy_digest(units)
    _, n_units, _ = miss_risk(D, pol_list[0], cert)
    n_budget = len(np.unique(D["unit"][cert]))
    dig_b = toy_digest(np.unique(D["unit"][cert]))
    static = GEN["bytes_per_escalated_frame"]
    cap = float(GEN["b_max_kib"])
    design_max = {a: max(payload_risk(D, p, sel)[2] for p in pol_list if p.acq == a)
                  for a in sorted({p.acq for p in pol_list})}
    return (lambda p: Contract(True, True, {"miss": n_units}, n_budget,
                               {"miss": dig, "budget": dig_b},
                               float(static[p.acq]), cap, True),
            {"n_miss_units": n_units, "n_budget_units": n_budget, "membership_sha256": dig,
             "C4_static_bound_kib": {a: float(v) for a, v in static.items()},
             "C4_declared_transport_cap_kib": cap,
             "design_fold_empirical_max_C_u": design_max,
             "C4_note": "the cap is the construction bound; the design-fold maximum is a diagnostic"})


# ------------------------------------------------------------------ scenes

def scene_nominal(D, sel, cert, log):
    pol = family()
    chains = build_chains(pol, LOSSES)
    cover = chain_cover_id(chains)
    plan = DeltaPlan(REGISTRATION, {chain_signature(c): 1.0 for c in chains}, cover)
    mk, audit = contracts_for(D, cert, sel, pol)

    # --- select fold: family design only
    sel_rows, slacks = [], []
    for p in pol:
        r, n_u, _ = miss_risk(D, p, sel)
        pz, n_b, _ = payload_risk(D, p, sel)
        per, joint = certification_slack({"miss": r}, {"miss": n_u}, LOSSES, DELTA / len(chains))
        sel_rows.append({"policy": p.name, "risk_select": r, "n_units": n_u, "payload_select": pz,
                         "joint_slack": joint, **frame_costs(D, p, sel)})
        slacks.append(joint)
    caps = {c[0].acq: capacity_gamma(D, c[0], sel, "design", "select fold, toy generator") for c in chains}
    cap_rows, span_rows = {}, {}
    for c in chains:
        a = c[0].acq
        r0, n_u, _ = miss_risk(D, c[0], sel)
        bar = certification_boundary(n_u, ALPHA, DELTA / len(chains))
        cap_rows[a] = chain_capacity({"miss": caps[a]}, {"miss": n_u}, LOSSES, DELTA / len(chains),
                                     select_bar={"miss": bar})
        s_c = [row["joint_slack"] for row in sel_rows if row["policy"].startswith(a)]
        st, det = boundary_spanning(s_c)
        span_rows[a] = {"status": st, **det}

    # The checklist runs at FAMILY level: capacity is audited against the weakest chain (minimum Gamma over
    # chains) at the select bar the rule actually used, and the "model identity" of this instance is the
    # digest of the declared generator - the toy's analogue of the detector weights hash.
    worst = min(caps.values(), key=lambda c: c.value)
    bar0 = certification_boundary(miss_risk(D, chains[0][0], sel)[1], ALPHA, DELTA / len(chains))
    gen_digest = hashlib.sha256(json.dumps(GEN, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    chk = preflight_checklist(pol, LOSSES, contracts=mk, plan=plan, payload_semantics=SEM,
                              weights_sha256=gen_digest, capacities={"miss": worst},
                              delta=DELTA, select_bar={"miss": bar0},
                              joint_slacks={c[0].acq: [row["joint_slack"] for row in sel_rows
                                                       if row["policy"].startswith(c[0].acq)]
                                            for c in chains},
                              inference_registration=IREG)

    # --- certify fold
    def ev(p):
        r, n_u, per_unit = miss_risk(D, p, cert)
        pz, n_b, _ = payload_risk(D, p, cert)
        # v2.4.1: report WHICH units were evaluated, not only how many; the engine re-verifies the set
        return {"miss": (r, n_u, toy_digest(per_unit.keys())),
                "budget": (pz, n_b, toy_digest(np.unique(D["unit"][cert])))}

    C = certify_family(pol, LOSSES, delta=DELTA, contracts=mk, certify_eval=ev, plan=plan,
                       payload_semantics=SEM, inference_registration=IREG)
    util = lambda p: sum(UTILITY_WEIGHTS[k] * v for k, v in frame_costs(D, p, cert).items())
    op, info = choose_operating_point(C, util)
    cert_rows = [{"policy": p.name, **{k: v for k, v in zip(("risk_certify", "n_units"), ev(p)["miss"])},
                  "payload_certify": ev(p)["budget"][0],
                  "p_joint": C.pvalues.get(p.name, {}).get("p_joint"),
                  "certified": any(q.name == p.name for q in C.certified),
                  "utility": util(p), **frame_costs(D, p, cert)} for p in pol]
    log(f"  cover {cover[:12]}...  M={len(chains)}  delta_m={DELTA/len(chains):g}  "
        f"n_min={n_min(ALPHA, DELTA/len(chains))}  certified {len(C.certified)}/{len(pol)}")
    return {"cover_id": cover, "M": len(chains), "delta_m": DELTA / len(chains), "audit": audit,
            "n_min": n_min(ALPHA, DELTA / len(chains)),
            "r_star": {c[0].acq: certification_boundary(miss_risk(D, c[0], cert)[1], ALPHA, DELTA / len(chains))
                       for c in chains},
            "g_star": {c[0].acq: certification_margin(miss_risk(D, c[0], cert)[1], ALPHA, DELTA / len(chains))
                       for c in chains},
            "select": sel_rows, "capacity": cap_rows, "spanning": span_rows,
            "checklist": chk, "certify": cert_rows, "reasons": list(C.reasons),
            "operating_point": {"policy": op.name if op else None, **info}}


def scene_ordering(D, sel, cert, log):
    """The strict-front counterexample, with measured select-fold ties and measured certify-fold outcomes."""
    tl, th = GEN["band"]
    # A tie on the select fold is not a contrivance: on a fine b grid, adjacent members differ only where an
    # event's verifier score falls between them, so equal select risk with unequal deployment cost is the
    # generic case. The grid is declared; the tie groups are FOUND, not assumed.
    grid = tuple(round(0.01 * i, 2) for i in range(0, int(round(100 * GEN["A"])) + 1))
    pol = [Policy("a2", tl, th, b, GEN["A"], (("S", 0.0),), B0, f"a2|b={b:.2f}") for b in grid]
    mk, _ = contracts_for(D, cert, sel, pol)

    def ev(p):
        r, n_u, per_unit = miss_risk(D, p, cert)
        pz, n_b, _ = payload_risk(D, p, cert)
        # v2.4.1: report WHICH units were evaluated, not only how many; the engine re-verifies the set
        return {"miss": (r, n_u, toy_digest(per_unit.keys())),
                "budget": (pz, n_b, toy_digest(np.unique(D["unit"][cert])))}

    def run_family(fam):
        ch = build_chains(fam, LOSSES)
        C = certify_family(fam, LOSSES, delta=DELTA, contracts=mk, certify_eval=ev,
                           plan=DeltaPlan(REGISTRATION, {chain_signature(c): 1.0 for c in ch},
                                          chain_cover_id(ch)), payload_semantics=SEM, inference_registration=IREG)
        return {"family": [p.name for p in fam], "M": len(ch), "delta_m": DELTA / max(1, len(ch)),
                "certified": [p.name for p in C.certified], "reasons": list(C.reasons),
                "p_joint": {p.name: C.pvalues.get(p.name, {}).get("p_joint") for p in fam}}

    items, rows, by_risk = [], [], {}
    for p in pol:
        r, n_u, _ = miss_risk(D, p, sel)
        cost = frame_costs(D, p, sel)["handoff"]        # operator load: a cost, never a constrained loss
        items.append((p, r, cost))
        rc, nc, _ = miss_risk(D, p, cert)
        rows.append({"policy": p.name, "risk_select": r, "cost_select": cost, "risk_certify": rc})
        by_risk.setdefault(round(r, 9), []).append(p)
    tie_groups = {f"{k:.6f}": [p.name for p in v] for k, v in by_risk.items() if len(v) > 1}

    out = {"grid": list(grid), "ladder": rows, "tie_groups": tie_groups,
           "cost": "hand-off rate on the select fold (operator load)"}
    out["full_grid_ACE"] = run_family(select_front(items, strict=False))
    out["full_grid_strict"] = run_family(select_front(items, strict=True))

    # The witness: the tie group whose cheapest member fails on the certify fold while a tied, more
    # conservative member passes. A strict front reduces that group to its cheapest member alone.
    r_star = certification_boundary(ev(pol[0])["miss"][1], ALPHA, DELTA)
    witness = None
    for k, group in sorted(by_risk.items()):
        if len(group) < 2:
            continue
        sub = [(p, miss_risk(D, p, sel)[0], frame_costs(D, p, sel)["handoff"]) for p in group]
        ace, strict = run_family(select_front(sub, strict=False)), run_family(select_front(sub, strict=True))
        if ace["certified"] and not strict["certified"]:
            witness = {"select_risk": k, "r_star_singleton_chain": r_star,
                       "members": [{"policy": p.name, "cost_select": c,
                                    "risk_certify": miss_risk(D, p, cert)[0],
                                    "p_joint": ace["p_joint"].get(p.name) or strict["p_joint"].get(p.name)}
                                   for p, _, c in sub],
                       "ACE": ace, "strict": strict}
            break
    out["witness"] = witness
    log(f"  {len(tie_groups)} tie group(s) on the select fold; full grid: ACE certifies "
        f"{len(out['full_grid_ACE']['certified'])}, strict front {len(out['full_grid_strict']['certified'])}; "
        f"witness: {'yes' if witness else 'none at this seed'}")
    return out


def scene_refusals(D, sel, cert, log):
    pol = family()
    mk, audit = contracts_for(D, cert, sel, pol)

    def ev(p):
        r, n_u, per_unit = miss_risk(D, p, cert)
        pz, n_b, _ = payload_risk(D, p, cert)
        # v2.4.1: report WHICH units were evaluated, not only how many; the engine re-verifies the set
        return {"miss": (r, n_u, toy_digest(per_unit.keys())),
                "budget": (pz, n_b, toy_digest(np.unique(D["unit"][cert])))}

    chains = build_chains(pol, LOSSES)
    plan = DeltaPlan(REGISTRATION, {chain_signature(c): 1.0 for c in chains}, chain_cover_id(chains))
    out = {}

    # C3: the audit population is below n_min for the share this candidate is tested at
    starved = lambda p: Contract(True, True, {"miss": 12}, audit["n_budget_units"],
                                 {"miss": audit["membership_sha256"], "budget": "d" * 64}, 0.0, float("inf"), True)
    C = certify_family(pol, LOSSES, delta=DELTA, contracts=starved, certify_eval=ev, plan=plan,
                       payload_semantics=SEM, inference_registration=IREG)
    out["C3_audit_support"] = {"reasons": list(C.reasons), "certified": len(C.certified),
                               "n_min": n_min(ALPHA, DELTA / len(chains)), "declared_units": 12}

    # C6: a miss loss with no monitonicity lemma cannot be carried along a chain
    from .ace_ltt import LossSpec
    unlemmad = LossSpec("miss", ALPHA, False, kind="miss", catch_set="response",
                        aggregation="unit_mean_over_events", empty_denominator="exclude_unit")
    ch2 = build_chains(pol, (unlemmad, PAY))
    out["C6_no_lemma"] = {"chains_with_lemma": [len(c) for c in chains],
                          "chains_without_lemma": [len(c) for c in ch2],
                          "note": "without a written lemma every candidate is its own chain: M rises from "
                                  f"{len(chains)} to {len(ch2)}, delta_m falls from {DELTA/len(chains):g} to "
                                  f"{DELTA/len(ch2):g}, and no fail-safe path exists"}

    # C7: a payload loss with no frozen semantics
    C = certify_family(pol, LOSSES, delta=DELTA, contracts=mk, certify_eval=ev, plan=plan,
                       payload_semantics=None, inference_registration=IREG)
    out["C7_payload_semantics"] = {"reasons": list(C.reasons), "certified": len(C.certified)}

    # contract I: the p-value machinery is not the registered machinery (here: a stale registry id, which
    # is what a substituted or re-pointed implementation produces)
    stale = InferenceRegistration(REGISTRATION, "0" * 64, IREG.inferences)
    C = certify_family(pol, LOSSES, delta=DELTA, contracts=mk, certify_eval=ev, plan=plan,
                       payload_semantics=SEM, inference_registration=stale)
    out["I_inference_registration"] = {"reasons": list(C.reasons), "certified": len(C.certified),
                                       "registered_registry_id": IREG.registry_id[:16] + "..."}
    log("  refusals: " + "; ".join(f"{k} -> {v.get('reasons', v.get('note'))}" for k, v in out.items())[:180])
    return out


def scene_diagnostics(D, sel, cert, log):
    """The SAME family, two declarations of the acting domain S, and a NOT_SPANNING family."""
    pol = family()
    chains = build_chains(pol, LOSSES)
    dm = DELTA / len(chains)
    out = {"delta_m": dm, "per_acq": {}}
    for c in chains:
        a = c[0].acq
        r0, n_u, _ = miss_risk(D, c[0], sel)
        bar_rstar = certification_boundary(n_u, ALPHA, dm)
        full = capacity_gamma(D, c[0], sel, "design", "select fold, S = escalated-only events")
        narrow = Capacity("miss", full.value / 5.0, "design",
                          "select fold, S restricted to a fifth of the acting events (declared variant)")
        rows = {}
        for tag, cap, bar in (("bar=r*, S full", full, bar_rstar),
                              ("bar=alpha, S full", full, ALPHA),
                              ("bar=alpha, S narrow", narrow, ALPHA)):
            rows[tag] = chain_capacity({"miss": cap}, {"miss": n_u}, LOSSES, dm, select_bar={"miss": bar})
        out["per_acq"][a] = {"n_select_units": n_u, "r_star_select": bar_rstar,
                             "g_star_select": certification_margin(n_u, ALPHA, dm),
                             "gamma_full": full.value, "gamma_narrow": narrow.value, "capacity": rows}
    # a NOT_SPANNING family: only the two safest members of each ladder
    safe = tuple(p for p in pol if p.b <= 0.10)
    sl = []
    for p in safe:
        r, n_u, _ = miss_risk(D, p, sel)
        _, joint = certification_slack({"miss": r}, {"miss": n_u}, LOSSES, dm)
        sl.append(joint)
    st, det = boundary_spanning(sl)
    out["not_spanning_subfamily"] = {"members": [p.name for p in safe], "status": st, **det}
    log(f"  spanning of the safe-only subfamily: {st}")
    return out


# ------------------------------------------------------------------ report

def report_md(o):
    g = o["generator"]
    n1 = o["nominal"]
    L = [f"# ACE-LTT on a generic synthetic instance ({o['utc']})", "",
         "**Illustration, not evidence.** The generator is declared and seeded; its constants were chosen so "
         "that the instance exhibits a truncating chain, a non-truncating chain and the refusal paths. No "
         "number here is a measurement of any real system.", "",
         f"generator seed {g['seed']}, {g['n_units']} units, {o['n_frames']} frames; "
         f"band ({g['band'][0]:g}, {g['band'][1]:g}), A = {g['A']:g}, b ladder {list(g['b_ladder'])}, "
         f"lambda_S = {g['lambda_S']:g}", "",
         f"alpha = {ALPHA:g}, beta = {BETA:g}, delta = {DELTA:g}; "
         f"M = {n1['M']}, delta_m = {n1['delta_m']:g}, n_min = {n1['n_min']}; "
         f"cover `{n1['cover_id']}`", "",
         f"audit population: {n1['audit']['n_miss_units']} miss units, {n1['audit']['n_budget_units']} budget "
         f"units, membership `{n1['audit']['membership_sha256'][:16]}...` (identical for all 8 candidates)", "",
         "C4, discharged statically: every escalated frame of mode `a` transmits exactly "
         + ", ".join(f"{v:g} KiB (`{a}`)" for a, v in n1["audit"]["C4_static_bound_kib"].items())
         + f" by construction, against the declared transport cap B_max = "
           f"{n1['audit']['C4_declared_transport_cap_kib']:g} KiB. As a non-certifying diagnostic, the design-fold "
           f"empirical maximum of C_u is "
         + ", ".join(f"{v:.1f} (`{a}`)" for a, v in n1["audit"]["design_fold_empirical_max_C_u"].items())
         + " KiB - a sample statement, never the cap. Note that it lies strictly BELOW the construction "
           "bound: the design fold never observed the worst case, which is exactly why a sample maximum "
           "cannot discharge a forall-x cap.", "",
         "## 1 Family design, select fold only", "",
         "| candidate | R_sel(miss) | Z_sel(budget) | joint slack | FA | calls | hand-off |",
         "|---|---|---|---|---|---|---|"]
    for r in n1["select"]:
        L.append(f"| `{r['policy']}` | {r['risk_select']:.4f} | {r['payload_select']:.4f} | "
                 f"{r['joint_slack']:+.5f} | {r['fa']:.4f} | {r['calls']:.4f} | {r['handoff']:.4f} |")
    L += ["", "| chain | Gamma(S) design | select bar | required h* | status | spanning | max/min joint slack |",
          "|---|---|---|---|---|---|---|"]
    for a, cap in n1["capacity"].items():
        m = cap["miss"]
        sp = n1["spanning"][a]
        L.append(f"| `{a}` | {m['capacity']:.5f} | {m['select_bar']:.5f} | {m['required_margin']:.5f} | "
                 f"{m['status']} | {sp['status']} | {sp['max_joint_slack']:+.5f} / {sp['min_joint_slack']:+.5f} |")
    ok = sum(1 for r in n1["checklist"] if r["ok"])
    L += ["", f"pre-freeze checklist: **{ok}/{len(n1['checklist'])}**", "",
          "| item | ok |", "|---|---|"]
    for r in n1["checklist"]:
        L.append(f"| {r['item']} | {'yes' if r['ok'] else 'NO'} |")
    L += ["", "## 2 Certification, certify fold", "",
          f"r* = " + ", ".join(f"{a} {v:.5f}" for a, v in n1["r_star"].items()) +
          "; g* = " + ", ".join(f"{a} {v:.5f}" for a, v in n1["g_star"].items()), "",
          "| candidate | R_cert(miss) | Z_cert(budget) | p_joint | certified | utility |",
          "|---|---|---|---|---|---|"]
    for r in n1["certify"]:
        pj = "-" if r["p_joint"] is None else f"{r['p_joint']:.3g}"
        L.append(f"| `{r['policy']}` | {r['risk_certify']:.4f} | {r['payload_certify']:.4f} | {pj} | "
                 f"{'yes' if r['certified'] else 'no'} | {r['utility']:.4f} |")
    opn = n1["operating_point"]["policy"]
    L += ["", f"reasons: {n1['reasons'] or 'none'}; operating point **`{opn}`** "
              f"(utility {n1['operating_point'].get('utility', float('nan')):.4f} over "
              f"{n1['operating_point'].get('n_certified', 0)} certified candidates)", "",
          "## 3 Ordering: the strict-front counterexample", ""]
    s2 = o["ordering"]
    L += [f"b grid {s2['grid'][0]:g} .. {s2['grid'][-1]:g} step 0.01 ({len(s2['grid'])} candidates), "
          f"cost = {s2['cost']}. Select-fold tie groups (equal R_sel, unequal cost):", ""]
    for k, v in s2["tie_groups"].items():
        L.append(f"- R_sel = {k}: `{'`, `'.join(v)}`")
    fa, fs = s2["full_grid_ACE"], s2["full_grid_strict"]
    L += ["", f"Whole grid: ACE (weak front + inclusion order) certifies **{len(fa['certified'])}** of "
              f"{len(fa['family'])} kept candidates; the strict front keeps {len(fs['family'])} and certifies "
              f"**{len(fs['certified'])}**.", ""]
    w = s2["witness"]
    if w:
        L += [f"**The witness.** Tie group at R_sel = {w['select_risk']:.6f}, tested as its own family "
              f"(M = 1, delta_m = {DELTA:g}, r* = {w['r_star_singleton_chain']:.5f}):", "",
              "| member | cost_sel | R_cert | p_joint | kept by strict front |", "|---|---|---|---|---|"]
        cheapest = min(w["members"], key=lambda m: m["cost_select"])["policy"]
        for m in w["members"]:
            pj = "-" if m["p_joint"] is None else f"{m['p_joint']:.3g}"
            L.append(f"| `{m['policy']}` | {m['cost_select']:.4f} | {m['risk_certify']:.4f} | {pj} | "
                     f"{'yes (cheapest)' if m['policy'] == cheapest else 'no'} |")
        L += ["", f"- ACE family {w['ACE']['family']} -> certified **{w['ACE']['certified']}**",
              f"- strict front family {w['strict']['family']} -> certified **{w['strict']['certified']}**, "
              f"reasons {w['strict']['reasons']}", "",
              "The strict front is not merely less efficient here: it returns the empty certified set on an "
              "instance where the ACE order certifies. The candidates it discarded are risk-equivalent to "
              "the one it kept on the fold that chose them, and strictly safer on the fold that tests them.", ""]
    else:
        L += ["", "No witness at this seed: every tie group's cheapest member also passed.", ""]
    L += ["## 4 Fail-closed refusals", ""]
    s3 = o["refusals"]
    L += ["| clause | outcome |", "|---|---|",
          f"| C3 audit support | declared {s3['C3_audit_support']['declared_units']} units < n_min "
          f"{s3['C3_audit_support']['n_min']} -> {s3['C3_audit_support']['reasons']}, "
          f"{s3['C3_audit_support']['certified']} certified |",
          f"| C6 no monotonicity lemma | chain sizes {s3['C6_no_lemma']['chains_with_lemma']} -> "
          f"{s3['C6_no_lemma']['chains_without_lemma']} |",
          f"| C7 payload semantics | {s3['C7_payload_semantics']['reasons']}, "
          f"{s3['C7_payload_semantics']['certified']} certified |",
          f"| contract I registration | stale registry id (registered "
          f"{s3['I_inference_registration']['registered_registry_id']}) -> "
          f"{s3['I_inference_registration']['reasons']}, "
          f"{s3['I_inference_registration']['certified']} certified |", "",
          s3["C6_no_lemma"]["note"], "",
          "## 5 The two diagnostics are declarations, not properties of the data", ""]
    s4 = o["diagnostics"]
    L += ["| chain | select bar / S | Gamma | required h* | status |", "|---|---|---|---|---|"]
    for a, v in s4["per_acq"].items():
        for tag, cap in v["capacity"].items():
            m = cap["miss"]
            L.append(f"| `{a}` | {tag} | {m['capacity']:.5f} | {m['required_margin']:.5f} | {m['status']} |")
    ns = s4["not_spanning_subfamily"]
    L += ["", f"safe-only subfamily {ns['members']}: **{ns['status']}** "
              f"(max {ns['max_joint_slack']:+.5f}, min {ns['min_joint_slack']:+.5f}) - "
              f"{ns['reason'] or 'spans the boundary'}", ""]
    return "\n".join(L)


def run(out_dir=None, seed=None, log=print):
    if seed:
        GEN["seed"] = int(seed)
    D = synth()
    sel, cert = split(D["unit"], GEN["select_fraction"], GEN["seed"])
    log(f"toy instance: {GEN['n_units']} units, {D['n_frames']} frames; "
        f"{len(np.unique(D['unit'][sel]))} select / {len(np.unique(D['unit'][cert]))} certify units")
    o = {"utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "kind": "generic synthetic illustration of ACE-LTT; not an empirical result",
         "spec_version": SPEC_VERSION,
         "registration": REGISTRATION, "generator": GEN, "n_frames": int(D["n_frames"]),
         "alpha": ALPHA, "beta": BETA, "delta": DELTA,
         "n_select_units": int(len(np.unique(D["unit"][sel]))),
         "n_certify_units": int(len(np.unique(D["unit"][cert])))}
    log("scene 1/4 nominal"); o["nominal"] = scene_nominal(D, sel, cert, log)
    log("scene 2/4 ordering"); o["ordering"] = scene_ordering(D, sel, cert, log)
    log("scene 3/4 refusals"); o["refusals"] = scene_refusals(D, sel, cert, log)
    log("scene 4/4 diagnostics"); o["diagnostics"] = scene_diagnostics(D, sel, cert, log)
    md = report_md(o)
    if out_dir:
        d = Path(out_dir).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        (d / "ace_toy.json").write_text(json.dumps(o, indent=1, default=str))
        (d / "ace_toy.md").write_text(md)
        log(f"-> {d / 'ace_toy.md'}")
    log("\n" + md)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", default=None)
    a = ap.parse_args(argv)
    run(a.out, a.seed)


if __name__ == "__main__":
    main()
