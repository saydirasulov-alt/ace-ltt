"""One-shot pre-registration of the ACE-LTT dev study.

    python -m cascade.ace_register --dataset DS --protocol P --det DET.csv \
        --agent crop=AG_CROP.csv --agent overlay=AG_OVERLAY.csv --out REG.json [--note "..."]

The DECLARATION below is the immutable content of the study: it is part of the analysis code, so it is
covered by analysis_code_sha256 and cannot change after freeze without invalidating the lock. This module
writes WHEN the declaration was fixed, against WHICH inputs, and - unlike a prose registration - it refuses
to register unless the preconditions the declaration depends on actually hold right now:

  P1  every event lies inside exactly one unit (label-free, from splits.csv). The constrained loss maps
      events to units by the unit of the event's first positive frame; if an event spanned two units the
      aggregation would not be the declared one and the Lemma would be proved for a different object.
  P2  the declared family decomposes into exactly the declared number of chains, with the declared per-chain
      budget, and the chain cover id is recorded here so the run can be checked against it.
  P3  the ACE pre-freeze checklist passes 6/6 against THIS declaration (not the shipped example).
  P4  the installed package matches its MANIFEST.sha256.
  P5  the audit unit membership per constrained risk, computed from the registered select/certify split, is
      identical for every acquisition in the family - and its digests are frozen here, so the run cannot
      silently audit a different set of units.

A second registration is refused; a changed declaration is a new study with a new file.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .ace_ltt import (CHECKLIST, Capacity, Contract, DeltaPlan, Policy, allocate_delta, build_chains,
                      chain_cover_id, declare_inference, miss_loss, preflight_checklist)

STUDY = "ace_ltt_dev"
LEMMA = "ACE-LTT spec v2.2 Sec. 4.3 Lemma, event-miss instance (2026-09-23)"

DECLARATION = {
    "study": STUDY,
    "question": (
        "Does ONE registered ACE-LTT procedure, run without human intervention on one candidate family, "
        "certify the crop cascade at a veto authority that controls the source-specific risk, and decline to "
        "certify the overlay cascade, on the dev split?"),

    # ---- the candidate family (Sec. 5 step 1) -------------------------------------------------------
    "acquisitions": ["crop", "overlay"],
    "veto": {
        "stratum": {"kind": "small_smoke", "area_max": 0.02,
                    "definition": "top smoke box area <= area_max AND s_smoke >= s_fire; label-free, from the "
                                  "detector boxes only"},
        "partition": ["S", "rest"],
        "lambda_grid": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "applies": "v = (S: lambda, rest: 0.0); g_v = min(1, g + v(x)). Larger lambda = more inclusive.",
    },
    "thresholds": (
        "(t_low, t_high, b, A) are chosen ON THE SELECT FOLD by the pre-registered objective, once per "
        "acquisition, and are NOT enumerated in the family. The concrete candidates and their order are "
        "written to disk with the chain cover id BEFORE any certify-fold quantity is touched; certification "
        "verifies the cover it tests against that file and refuses on mismatch."),
    "chains": {
        "M": 2, "rule": "one chain per acquisition; the chain is the lambda ladder ordered 1.0 -> 0.0",
        "delta": 0.10, "delta_m": 0.05, "allocation": "uniform (delta/M); no DeltaPlan",
        "note": ("M = 2 is a property of THIS design (equal allocation, delta = 0.10, worst required "
                 "p ~ 0.0150 on the registered lambda front): 0.10/6 = 0.0167 > 0.0150 while "
                 "0.10/7 = 0.0143 < 0.0150, so M_max = 6 for a registered equal-allocation design. ACE-LTT "
                 "itself does not forbid M > 6; a non-uniform registered DeltaPlan is permitted by the "
                 "specification and is deliberately NOT used here. The multiplicity price is paid over 2 "
                 "chains, not over 12 candidates."),
    },

    # ---- constrained losses (Sec. 3) ----------------------------------------------------------------
    "losses": [
        {"name": "fire", "alpha": 0.05, "catch_set": "response", "aggregation": "unit_mean_over_events",
         "weighting": "uniform", "empty_denominator": "exclude_unit", "lemma": LEMMA},
        {"name": "smoke", "alpha": 0.05, "catch_set": "response", "aggregation": "unit_mean_over_events",
         "weighting": "uniform", "empty_denominator": "exclude_unit", "lemma": LEMMA},
    ],
    "catch_set_audit": (
        "catch_set = response was established from the code, not chosen: on positive frames policy.decide "
        "counts d in {1, 2, 4} as caught (hand-off included), i.e. {s >= t_high} u {escalated and g_v >= b}, "
        "governed by b. Numerically, the event-miss grid is exactly constant along the A axis "
        "(max |miss(A) - miss(A_0)| = 0.0) and varies along b. For the machine-only family handoff=False "
        "forces b = A, where response and alarm coincide; response is the general statement."),
    "diagnostic_only": {
        "stratum_risks": ("fire|S and smoke|S are NOT constrained: the certifiability replay gives 4 and 17 "
                          "positive dev units against n_min = 149 required for P(certify >= n_min) >= 0.9. "
                          "Constraining them would make every candidate NOT CERTIFIED under clause C3. They "
                          "are reported with their audit counts, and that refusal is the fail-closed "
                          "demonstration, not a failure of the run."),
        "payload": ("C_U (KiB per evaluated frame) and its distribution are REPORTED; no payload loss enters "
                    "the constrained family and no B0 is fitted. There is no B0 from an independent "
                    "deployment source yet; choosing one from the observed payload would contradict ACE's "
                    "own admissibility principle. Before the sealed test, (B0, beta, unit, normalization) "
                    "are preregistered separately once an uplink requirement exists, and payload then "
                    "becomes a confirmatory constraint."),
        "handoff_variant": ("the +human family (b < A, operator counted as a catch) is reported as a "
                            "diagnostic, not as a second constrained family, because it would double M for "
                            "a descriptive comparison."),
    },

    # ---- inference and reporting --------------------------------------------------------------------
    "alpha": 0.05, "delta": 0.10, "unit": "split_group (global dedup component)", "target": "event",
    "split": {"rule": "select/certify split of the dev units, select_fraction from protocol.yaml",
              "seed": 20260923,
              "reproduction": "numpy.random.default_rng(seed).choice(sorted unique units, "
                              "max(1, round(select_fraction * n_units)), replace=False) = select; rest = certify "
                              "(identical to cascade.select.calibrate); the run MUST use "
                              "cascade.ace_register.certify_mask so there is one implementation",
              "no_dev_holdout": (
                  "ALL dev units are used as the calibration set of this dry run; there is no held-out dev "
                  "half. Consequence, consistent with the utility caveat: the FA and calls of the chosen "
                  "operating point are selection-side quantities, and final performance is measured once on "
                  "the sealed test."),
              "relation_to_the_certifiability_replay": (
                  "The registered dev probe reported ~573 fire and ~526 smoke certify units, this "
                  "registration ~1138 and ~1079. These are different objects, not a contradiction: the probe "
                  "REPLAYS the deployment pipeline, where dev is first halved into calibration/test and the "
                  "select/certify split happens inside the calibration half (0.5 x 0.7 = 0.35 of the "
                  "positive units), while this dry run calibrates on all dev units (0.7). The ratio is "
                  "1138/573 = 1.99 = 1/0.5. The n_min = 149 figure belongs to the replayed pipeline and is "
                  "what governs the sealed-test path; the audit here is against n_min(0.05, delta_m = 0.05) "
                  "= 59, which both risks clear by more than an order of magnitude."),
              },
    "audit_membership": ("per constrained risk, sha256 of the newline-joined sorted certify unit ids that "
                         "carry a positive of that risk; one value per risk, identical for every acquisition "
                         "by construction, frozen in this registration and re-checked by the run"),
    "utility": {"calls": 1.0, "fa": 10.0,
                "caveat": ("the operating point is chosen by minimising this utility INSIDE the certified "
                           "set, which does not weaken the risk guarantee - the policy is already in the "
                           "simultaneously certified set. But the FA and calls values used for that choice "
                           "are measured on the certify fold and are therefore NOT an unbiased final "
                           "performance estimate; they are reported as selection-side quantities. Final "
                           "FA/calls performance is measured once on the sealed test.")},
    "outcome_rule": (
        "The study reports what the single procedure returns: the certified set, the chosen operating point, "
        "the chain certificates, and every NOT CERTIFIED clause that fired. No candidate is added, removed "
        "or re-ordered after the certify fold is touched. An empty certified set is a reportable outcome, "
        "not a reason to change the family."),
    "framing": ("The veto regime selected by ACE lowers the risk associated with the earlier fixed-view "
                "failure mode; the previously registered view_rule outcome stands unchanged. ACE is a new "
                "registered study, not a retroactive correction."),
    "predecessors": ["acquisition_ladder (2026-09-23T03:27:11Z)", "view_rule amendment"],
}


# ------------------------------------------------------------------ preconditions

def events_spanning_units(protocol):
    """P1, label-free: events whose frames fall in more than one unit, per source (must be empty)."""
    with open(Path(protocol) / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    ev = defaultdict(set)
    for r in rows:
        ev[(r.get("source", ""), r["event"])].add(r["unit"])
    bad = defaultdict(int)
    for (src, _), units in ev.items():
        if len(units) > 1:
            bad[src] += 1
    return dict(bad)


def certify_mask(unit, select_fraction, seed):
    """The registered select/certify split, reproducing cascade.select.calibrate exactly."""
    unit = np.asarray(unit)
    uu = np.unique(unit)
    rng = np.random.default_rng(int(seed))
    sel = set(rng.choice(uu, max(1, int(round(float(select_fraction) * len(uu)))), replace=False))
    return ~np.array([u in sel for u in unit])


def membership_digests(D, select_fraction, seed):
    """P5: per constrained risk, the digest of the certify units carrying a positive of that risk.

    Returns (digests, split) where `split` counts FRAMES and UNITS separately - the mask is per frame, and
    reporting its sum as a unit count would put a wrong number into the registration.
    """
    unit = np.asarray(D["unit"])
    cert = certify_mask(unit, select_fraction, seed)
    out = {}
    for k in ("fire", "smoke"):
        pos = cert & (np.asarray(D[f"y_{k}"]).astype(int) > 0)
        ids = sorted(set(unit[pos].tolist()))
        out[k] = {"n_units": len(ids), "sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    split = {"n_dev_frames": int(len(unit)), "n_dev_units": int(len(set(unit.tolist()))),
             "n_certify_frames": int(cert.sum()), "n_certify_units": int(len(set(unit[cert].tolist()))),
             "n_select_units": int(len(set(unit[~cert].tolist()))),
             "select_fraction": float(select_fraction), "seed": int(seed)}
    return out, split


def declared_family():
    """The family's STRUCTURE, with symbolic thresholds: the chain decomposition depends only on the group
    key (acquisition, band, B0) and on the lambda order, never on the threshold values."""
    lam = DECLARATION["veto"]["lambda_grid"]
    pol = []
    for acq in DECLARATION["acquisitions"]:
        for x in sorted(lam, reverse=True):                    # safest first
            pol.append(Policy(acq, 0.02, 0.90, 0.10, 0.10, {"S": x, "rest": 0.0}, float("inf"),
                              f"{acq}|lambda={x:g}"))
    losses = tuple(miss_loss(L["name"], L["alpha"], L["lemma"], catch_set=L["catch_set"],
                             aggregation=L["aggregation"], weighting=L["weighting"],
                             empty_denominator=L["empty_denominator"]) for L in DECLARATION["losses"])
    return tuple(pol), losses


def lambda_capacity(D, sel):
    """Gamma_k(S) of the lambda coordinate: the S mass per risk (spec v2.3 Sec. 4.5).

    ADDED UNDER v2.3, AFTER Study 1 was registered and executed. It does not alter that experiment, which
    stands under the v2.2 protocol frozen at its registration; it only means this family would be REFUSED
    for a new v2.3 registration, because lambda acts solely inside S and S is too small to span g*.
    """
    from .evidence_veto import StratumSpec, in_stratum
    spec = StratumSpec(DECLARATION["veto"]["stratum"]["kind"], DECLARATION["veto"]["stratum"]["area_max"])
    inS = in_stratum(spec, D["features"])[sel]
    out = {}
    for k in ("fire", "smoke"):
        y = np.asarray(D[f"y_{k}"][sel]).astype(bool)
        ev, un = np.asarray(D["event"][sel]), np.asarray(D["unit"][sel])
        e_ids, inv = np.unique(ev[y], return_inverse=True)
        inS_ev = np.zeros(len(e_ids), bool)
        np.logical_or.at(inS_ev, inv, inS[y])
        first = np.zeros(len(e_ids), int)
        first[inv[::-1]] = np.arange(int(y.sum()))[::-1]
        uu, uinv = np.unique(un[y][first], return_inverse=True)
        frac = np.bincount(uinv, weights=inS_ev) / np.bincount(uinv)
        out[k] = Capacity(k, float(frac.mean()) if len(frac) else 0.0, "design",
                          "select fold of the dev split; S mass of the lambda coordinate")
    return out


def verify(protocol, D, select_fraction):
    """Run every precondition. Returns (report, ok)."""
    from .make_protocol import verify_tree
    rep, ok = {}, True

    span = events_spanning_units(protocol)
    rep["P1_events_spanning_units"] = span
    ok &= not span

    policies, losses = declared_family()
    chains = build_chains(policies, losses)
    deltas = allocate_delta(chains, DECLARATION["delta"], None)
    rep["P2_chains"] = {"M": len(chains), "declared_M": DECLARATION["chains"]["M"],
                        "delta_m": list(deltas), "declared_delta_m": DECLARATION["chains"]["delta_m"],
                        "cover_id": chain_cover_id(chains),
                        "chains": [[p.name for p in c] for c in chains]}
    ok &= (len(chains) == DECLARATION["chains"]["M"]
           and all(abs(d - DECLARATION["chains"]["delta_m"]) < 1e-12 for d in deltas)
           and all(len(c) == len(DECLARATION["veto"]["lambda_grid"]) for c in chains))

    dig, split = membership_digests(D, select_fraction, DECLARATION["split"]["seed"])
    rep["P5_audit_membership"] = dig
    rep["P5_split"] = split
    ok &= all(v["n_units"] >= 59 for v in dig.values())        # n_min(alpha=0.05, delta_m=0.05)

    def contracts(_p):
        return Contract(True, True, {k: v["n_units"] for k, v in dig.items()}, split["n_certify_units"],
                        {k: v["sha256"] for k, v in dig.items()}, 0.0, float("inf"))

    import yaml
    P = yaml.safe_load((Path(protocol) / "protocol.yaml").read_text())
    caps = lambda_capacity(D, ~certify_mask(D["unit"], select_fraction, DECLARATION["split"]["seed"]))
    rep["P6_design_capacity"] = {k: v.to_dict() for k, v in caps.items()}
    # contract I (v2.4.2): the construction each loss names, and the code identity of the implementation
    ireg = declare_inference(DECLARATION["study"], losses)
    rep["P7_inference_registration"] = ireg.to_dict()
    rows = preflight_checklist(policies, losses, contracts=contracts, plan=None, payload_semantics=None,
                               weights_sha256=str((P.get("detector") or {}).get("weights_sha256") or ""),
                               capacities=caps, delta=float(DECLARATION["delta"]),
                               inference_registration=ireg)
    rep["P3_checklist"] = rows
    ok &= all(r["ok"] for r in rows)

    tree = verify_tree()
    rep["P4_runtime_tree"] = {"ok": tree["ok"], "extra": tree["extra"], "missing": tree["missing"],
                              "changed": tree["changed"]}
    ok &= tree["ok"]
    return rep, bool(ok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--agent", action="append", required=True, metavar="NAME=PATH",
                    help="one per acquisition, e.g. crop=...csv overlay=...csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true", help="verify the preconditions, write nothing")
    a = ap.parse_args(argv)

    from .esva_dev import load_dev
    from .make_protocol import analysis_code, sha256_file

    out = Path(a.out).expanduser()
    if out.exists() and not a.dry_run:
        raise SystemExit(f"{out} exists: a registration is written once")
    agents = dict(x.split("=", 1) for x in a.agent)
    if sorted(agents) != sorted(DECLARATION["acquisitions"]):
        raise SystemExit(f"--agent names {sorted(agents)} != declared acquisitions "
                         f"{sorted(DECLARATION['acquisitions'])}")

    D = None
    for name, path in sorted(agents.items()):
        Di, P = load_dev(a.dataset, a.protocol, a.det, path)      # also enforces the score contract
        if D is None:
            D = Di
        elif not np.array_equal(np.asarray(D["unit"]), np.asarray(Di["unit"])):
            raise SystemExit(f"acquisition {name!r} covers different dev rows than the first one")
    rep, ok = verify(a.protocol, D, float(P.get("select_fraction", 0.3)))

    print(json.dumps(rep, indent=1))
    print("\npreconditions:", "ALL HOLD" if ok else "NOT SATISFIED - nothing is registered")
    if not ok:
        raise SystemExit(1)
    if a.dry_run:
        print("--dry-run: nothing written")
        return

    rec = {"study": STUDY,
           "registered_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "declaration": DECLARATION,
           "preconditions": rep,
           "checklist_items": list(CHECKLIST),
           "protocol_sha256": sha256_file(Path(a.protocol) / "protocol.yaml"),
           "splits_sha256": sha256_file(Path(a.protocol) / "splits.csv"),
           "analysis_code_sha256": analysis_code()[0],
           "det_sha256": sha256_file(a.det),
           "agent_sha256": {k: sha256_file(v) for k, v in sorted(agents.items())},
           "note": a.note}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    print(json.dumps({k: rec[k] for k in ("study", "registered_utc", "protocol_sha256",
                                          "analysis_code_sha256")}, indent=1))
    print("cover_id:", rep["P2_chains"]["cover_id"])
    print("->", out)


if __name__ == "__main__":
    main()
