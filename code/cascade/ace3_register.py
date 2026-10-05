"""One-shot pre-registration of ACE-LTT Study 3 (prospective stress test of the Sec. 4.5 extension).

    python -m cascade.ace3_register --dataset DS --protocol P --det DET.csv \
        --agent crop=AG_CROP.csv --agent overlay=AG_OVERLAY.csv --out REG3.json [--dry-run]

This is the literal implementation of the frozen R_0 text (`ACE_study3_preregistration_v5_R0.md`). It
chooses nothing that R_0 did not already fix, and it closes with R_1: the family, the arm-wise cover, the
intra-chain order and the delta shares, frozen BEFORE any certify-fold quantity is read.

THE GOVERNING INVARIANT

    design chooses; the audit metadata A may only refuse.

Two consequences are structural in this module, not commentary:

  * The design stage never sees a certify-fold count. It sees n_sel (positive units on calibration-SELECT)
    and the DESIGN REFERENCE COUNT n_ref_k = floor(n_sel_k * (1 - sf) / sf) = floor(7 n_sel_k / 3) at
    sf = 0.3. n_ref is a design reference only: never an estimator, never a validity quantity.
  * Hence two boundaries with two namespaces, and they are never interchanged:

        r_star_ref[k]  = certification_boundary(n_ref_k,  alpha_k, delta_m)   design, h*_ref, CAPABLE,
                                                                              T/S/N, KNIFE_EDGE, R2
        r_star_cert[k] = certification_boundary(n_cert_k, alpha_k, delta_m)   the certificate itself and X

    This module computes ONLY r_star_ref. `r_star_cert` does not exist here and cannot: n_cert is opened by
    cascade.ace3_dev, after this registration is written.

The design stage does NOT test clause C3. It applies a DESIGN-TIME SUPPORT PROXY - the C3 threshold
n_min(alpha_k, delta_m) evaluated at n_ref - which may shape the family and the branch precisely because
n_ref is a select-fold quantity. Being one, it is not the audit population, it discharges no premise of
Theorem 1, and it is nowhere called C3. Formal C3 is tested once, by cascade.ace3_dev, against the realized
n_cert and the audit membership, and there it may only PASS or REFUSE.

Status: **prospective and pre-registered.** Calibration units were never read by Studies 1 or 2. The fresh
fold is not a new-domain evaluation: 88.6% of calibration units come from the dominant source.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ace_ltt import (CHECKLIST, SPEC_VERSION, Capacity, Contract, Policy, allocate_delta,
                      boundary_spanning, build_chains, certification_boundary, chain_cover_id,
                      declare_inference, membership_digest, miss_loss, preflight_checklist)
from .ace_register import certify_mask, events_spanning_units
from .dataio import EVAL_SPLITS, frame_label, read_csv_by_uid, read_manifest
from .evidence_veto import StratumSpec, discount, in_stratum, smoke_features
from .policy import Grid
from .select import _grid_stats, _objective, candidate_mask

STUDY = "ace_ltt_study3_p45_stress"
LEMMA = "ACE-LTT spec v2.4 Sec. 4.3 Lemma, event-miss instance (2026-09-24)"
RISKS = ("fire", "smoke")
ARMS = ("B", "L")
ROLE = "calibration"

#: The two branches, declared LITERALLY (R_0 §2). The branch is chosen from n_ref alone; the realized cover
#: must then reproduce (M, delta_m) exactly, and no full -> fallback switch exists at any later stage.
BRANCHES = {
    "full":     {"modes": ("crop", "overlay"), "M": 4, "delta_m": 0.025, "n_min": 72},
    "fallback": {"modes": ("overlay",),        "M": 2, "delta_m": 0.05,  "n_min": 59},
}
DELTA = 0.10
DELTA_ARM = 0.05                     # delta_B = delta_L; 2 arms x 0.05 = delta

DECLARATION = {
    "study": STUDY,
    "spec_version": SPEC_VERSION,
    "status": ("prospective / pre-registered on units never read by Studies 1-2; the P-4.5 predictions are "
               "finite-sample predictions evaluated once, carrying no p-value and no population claim"),
    "question": ("Does the Sec. 4.5 design-time capacity statement survive a prospective test: does the "
                 "design-fold influence-envelope capacity of a chain's varying coordinate bound the reach "
                 "that coordinate realizes on a fresh certify fold (T), and is capability sufficient for a "
                 "reach at least as large as the finite-sample margin (S)?"),

    "governing_invariant": "design chooses; the audit metadata A may only refuse",
    "stages": ["P0", "R0", "design on calibration-select", "R1", "A", "C3 verify/refuse", "one execution"],

    # ---- the design reference count and the branch (R_0 §2) -----------------------------------------
    "design_reference_count": {
        "rule": "n_ref_k = floor(n_sel_k * (1 - select_fraction) / select_fraction)",
        "at_sf_0_3": "floor(7 * n_sel_k / 3)",
        "floor_is_deliberate": ("certification_boundary takes an integer n, and a fractional future unit "
                               "earns the design no credit"),
        "status": ("DESIGN REFERENCE ONLY - never an estimator, never a validity quantity. It is the only "
                   "count the design stage may see."),
    },
    "branches": {k: dict(v, modes=list(v["modes"])) for k, v in BRANCHES.items()},
    "branch_rule": ("from n_ref ALONE: full if n_ref_k >= 72 for both k; else fallback if n_ref_k >= 59 for "
                    "both k; else design-stage REFUSE (C3_CERTIFIABILITY_AUDIT). 72 = n_min(0.05, 0.025), "
                    "59 = n_min(0.05, 0.05). The realized cover must then reproduce the branch's declared "
                    "(M, delta_m) to 1e-12 or the registration REFUSES; no re-planning, no re-seeding."),
    "no_switch": ("there is no full -> fallback switch at the audit stage or at execution. A realized "
                  "n_cert below the chosen branch's n_min REFUSES."),
    "support_proxy": ("the design stage applies the C3 THRESHOLD n_min(alpha_k, delta_m) at n_ref. This is "
                     "a DESIGN-TIME SUPPORT PROXY, not clause C3: n_ref is not the audit population, so it "
                     "discharges no premise. Formal C3 is tested once, after A is opened."),

    # ---- family: two arms, arm-wise cover (R_0 §3) --------------------------------------------------
    "acquisitions": ["crop", "overlay"],
    "veto": {"stratum": {"kind": "small_smoke", "area_max": 0.02},
             "partition": ["S", "rest"],
             "lambda_fixed": 0.4,
             "why": "the pre-Study-1 working point Study 2 registered as lambda_fixed; not selected from "
                    "any study's result"},
    "arms": {
        "B": {"coordinate": "b", "fixed": {"lambda": 0.4},
              "ladder_rule": ("every g-grid edge b <= A forming a valid candidate at the chosen (t_low, "
                              "t_high, A), ASCENDING. The g grid is fixed in cascade.policy.Grid, so the "
                              "ladder is a deterministic function of the chosen thresholds and of nothing "
                              "observed."),
              "min_members": 3, "envelope": "S_B", "order": "ascending b: safest (largest response set) first"},
        "L": {"coordinate": "lambda", "fixed": {"b": 0.25},
              "ladder_rule": "lambda in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, safest first (descending lambda)",
              "min_members": 3, "envelope": "S_L", "order": "descending lambda: safest first",
              "b_provenance": ("b = 0.25 is carried forward from the operating point Study 2's "
                               "pre-registered utility selected in its historical development run. It is "
                               "used ONLY as pre-existing design information; no validity claim from Study "
                               "2 is imported into Study 3. Arm B's safest endpoint was considered and "
                               "rejected: as b -> 0 the response set absorbs the whole escalated "
                               "population, so lambda becomes inert with respect to the response-miss risk "
                               "it is supposed to move."),
              "refuse_if": "b_L = 0.25 is not a valid candidate at the mode's chosen (t_low, t_high, A)"},
    },
    "lambda_grid": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "b_fixed_L": 0.25,
    "cover": {
        "rule": ("the cover is computed WITHIN AN ARM, never over the union of the arms; the two arms are "
                 "two registered families, each with its own delta"),
        "why": ("build_chains groups by (acq, t_low, t_high, B0, strata) and the group key holds the "
                "stratification KEYS, not the lambda values, so on one mode at the same thresholds arms B "
                "and L share a group and cross-arm pairs are >=_cert-comparable: the minimum cover "
                "interleaves them (measured: ['B|b=0','B|b=0.1','B|b=0.25','L|lam=0.4','B|b=0.4']). Such a "
                "chain has two varying coordinates, so neither S_B nor S_L is its envelope and the (C,k) "
                "primitives have nothing to attach to."),
        "delta_split": {"delta": DELTA, "delta_B": DELTA_ARM, "delta_L": DELTA_ARM,
                        "arithmetic": "4 x 0.025 = 0.10 on the full branch; 2 x 0.05 = 0.10 on the fallback"},
        "validity": ("any partition into >=_cert-chains fixed before the test fold gives the guarantee, and "
                     "minimality is a POWER property, not a validity property (Sec. 3.4). The price of "
                     "covering per arm is a little power; the return is chains whose varying coordinate, "
                     "and therefore whose influence envelope, is well defined."),
        "invariant": "no chain of the registered cover may contain members of both arms",
    },
    "thresholds": {
        "rule": ("per mode, at the frozen lambda, choose (t_low, t_high, A) on calibration-SELECT among "
                 "configurations whose b ladder satisfies BOTH: (i) the safe endpoint is joint-select-"
                 "feasible, s_joint = min_k (r_star_ref_k - R_sel_k) >= 0; and (ii) the ladder is "
                 "BOUNDARY_SPANNING, max_pi s_joint >= 0 and min_pi s_joint < 0. Among those, minimise the "
                 "declared utility over the ladder members on the certifiable side."),
        "boundary": "r_star_ref ONLY; the design stage never sees r_star_cert",
        "re_derived": ("(t_low, t_high, A) are re-derived on calibration-select, not carried over from dev: "
                       "calibration-select is Study 3's design fold, so this keeps the family a function of "
                       "the fresh fold rather than of the fold Studies 1 and 2 used"),
        "frozen_at": "registration (from select-fold data only); the runner recomputes and must match",
    },

    # ---- losses and envelopes (R_0 §4) --------------------------------------------------------------
    "losses": [
        {"name": "fire", "alpha": 0.05, "catch_set": "response", "aggregation": "unit_mean_over_events",
         "weighting": "uniform", "empty_denominator": "exclude_unit", "lemma": LEMMA},
        {"name": "smoke", "alpha": 0.05, "catch_set": "response", "aggregation": "unit_mean_over_events",
         "weighting": "uniform", "empty_denominator": "exclude_unit", "lemma": LEMMA},
    ],
    "envelopes": {
        "definition": ("an INFLUENCE ENVELOPE S is any event set CONTAINING every event whose catch status "
                       "the coordinate can change. Minimality gives the tightest bound; a superset is "
                       "admissible and gives a looser one. Gamma_k,D(S) = |D|^-1 sum_u |E_ku n S|/|E_ku| "
                       "over units carrying at least one positive k-event, computed PER CHAIN AT ITS OWN "
                       "MODE (Study 2's minimum over modes is kept only for the pre-freeze checklist)."),
        "S_B": ("for mode a at its frozen (t_low, t_high): a positive k-event e is in S_B iff some positive "
                "frame of e has t_low <= s < t_high AND no positive frame of e has s >= t_high. This is the "
                "MINIMAL envelope for b: an event outside it is caught or missed regardless of b."),
        "S_L": ("a positive k-event e is in S_L iff at least one positive frame of e satisfies "
                "StratumSpec(kind='small_smoke', area_max=0.02) - top smoke box area <= area_max and "
                "s_smoke >= s_fire, label-free, from the detector boxes only - the predicate the released "
                "lambda_capacity implements, carried over unchanged."),
        "S_L_scope": ("S_L is a CONSERVATIVE influence envelope, not the minimal acting domain: it includes "
                      "events already caught by the edge stage, whose catch status lambda cannot change. So "
                      "Gamma_L is a valid reach upper bound but may be loose, and the direction of that "
                      "looseness is stated rather than glossed: a larger Gamma_L makes T EASIER to satisfy "
                      "and makes CAPABLE_L easier to trigger, which makes S HARDER. No tightened envelope "
                      "is substituted after the fact."),
    },

    # ---- the predictions (R_0 §5); evaluated by cascade.ace3_dev, never by this module --------------
    "predictions": {
        "margin": "h_star_ref_k = max(0, bar_k - r_star_ref_k) with bar_k = alpha_k (pre-registered "
                  "reference bar); CAPABLE_k(C) = 1{Gamma_sel_k(S) >= h_star_ref_k}",
        "bar_why": ("Study 2 registered the tight bound R_sel_k <= r_star_k, which makes the margin 0 and "
                    "the capacity condition vacuous. Since r_star_ref_k <= alpha_k the selection rule also "
                    "guarantees R_sel_k <= alpha_k, so Study 3 pre-registers the weaker but still "
                    "selection-guaranteed reference bar bar_k = alpha_k in order to evaluate a non-vacuous "
                    "reach margin. The effect is not one-directional: a larger h_star_ref_k makes CAPABLE "
                    "harder to reach."),
        "T": "sp_cert_k(C) <= Gamma_sel_k(S)                              primary; cross-fold reach bound",
        "S": "CAPABLE_k(C) = 1 => sp_cert_k(C) >= h_star_ref_k            primary",
        "N": "CAPABLE_k(C) = 0 and T => sp_cert_k(C) < h_star_ref_k       DERIVED; never evidence",
        "X": ("CAPABLE_k(C) = 1 and R_sel_k(pi_1) > r_star_ref_k => X_k(C) = 1, where "
              "X_k(C) = 1{R_cert_k(pi_1) <= r_star_cert_k}. Secondary and confounded. X is the ONLY "
              "primitive that touches r_star_cert; its antecedent is an R1 quantity."),
        "chain_label": "AND_k CAPABLE_k(C) is DERIVED, reported for continuity with Sec. 3.6, used in no "
                       "prediction",
        "what_they_are": ("pre-registered FINITE-SAMPLE predictions, evaluated once on the fresh certify "
                          "fold. No p-value, confidence statement, sampling distribution or population-"
                          "level generalisation attaches to any of them. The only statements carrying a "
                          "probability are the certificates of Theorem 1, which do not depend on T/S/N/X."),
        "tolerance": {"tau": 1e-12, "knife_edge": 1e-06},
        "comparison_semantics": ("float64, absolute tolerance tau = 1e-12: status uses plain >=; T holds iff "
                                 "sp_cert_k <= Gamma_sel_k + tau; S holds iff sp_cert_k >= h_star_ref_k - "
                                 "tau; X uses the plain test and must match the engine bit for bit"),
        "KNIFE_EDGE": ("|Gamma_sel_k - h_star_ref_k| <= 1e-06: excluded from the S verdict and from R2's "
                       "count, kept in T, reported with its numbers. Both quantities are known at R1, so "
                       "the exclusion is decided blind to the outcome."),
        "NO_REACH": ("a pair whose chain has |C| = 1 has sp_D_k == 0, so T holds trivially and S fails "
                     "trivially for any h_star_ref_k > 0: no reach exists to measure. Decided at R1, "
                     "excluded from the T, S and N verdicts and from R2's count, reported."),
        "CHAIN_TERMINAL": ("ANY post-registration CHAIN_TERMINAL affecting a registered primary pair => "
                           "P-4.5 = REFUSED. The terminal chain's pair-level numbers are reported; other "
                           "chains may be reported descriptively; no study-level T or S confirmatory "
                           "verdict is given; R2 is NOT recomputed; the denominator is NOT shrunk after "
                           "execution."),
        "R2": ("INFORMATIVE iff, among the pairs of the registered cover that are neither KNIFE_EDGE nor "
               "NO_REACH, there exists (C,k) with CAPABLE = 1 AND (C',k') with CAPABLE = 0. Decided at R1, "
               "before any certify outcome, never recomputed."),
        "verdict_space": ["INFORMATIVE", "UNINFORMATIVE", "REFUSED"],
        "uninformative_rule": ("no confirmatory interpretation of T, S or N; the run still executes and is "
                              "reported as an independent registered ACE-LTT certification on fresh units; "
                              "no certify-fold outcome is used to redesign the P-4.5 family, in this paper "
                              "or a later one; no recovery of the form 'after execution we happened to see "
                              "both types' is permitted. A bigger n_ref_k raises r_star_ref_k and shrinks "
                              "h_star_ref_k, so a coordinate that was incapable on a smaller design "
                              "reference count can come back capable - accepted, not designed around. The "
                              "realized n_cert_k cannot move CAPABLE, R2 or the KNIFE_EDGE set at all."),
    },

    "alpha": 0.05, "delta": DELTA, "unit": "split_group (global dedup component)", "target": "event",
    "role": ROLE,
    "split": {"rule": "select/certify split of the CALIBRATION units; select_fraction and calibrate_seed "
                      "from protocol.yaml, quoted by value into the receipt",
              "reproduction": "cascade.ace_register.certify_mask (identical to cascade.select.calibrate)"},
    "utility": {"calls": 1.0, "fa": 10.0, "handoff": 20.0,
                "scope": "the reported operating point ONLY; never T, S, N or X",
                "caveat": ("FA, calls and hand-off measured on the certify fold are selection-side "
                           "quantities, not an unbiased performance estimate")},
    "one_fold_two_uses": {
        "fact": "the fresh certify fold both issues the ACE-LTT certificate and evaluates T, S and X",
        "allowed": ["an independent fresh-fold test of the Sec. 4.5 extension",
                    "evaluated on units not used to design the family"],
        "not_allowed": ["independent of the ACE-LTT certification result"],
        "why": "they are the same units",
    },
    "source_concentration": ("the prospective fold is fresh with respect to the registered unit split, but "
                             "it is not a new-domain evaluation: 88.6% of calibration units come from the "
                             "dominant source, and the smallest source contributes only two units"),
    "outcome_rule": ("the study reports what the procedure returns: certified set per arm, operating point, "
                     "chain certificates, every NOT CERTIFIED clause that fired, and the (C,k) statuses as "
                     "measured. Nothing is added, removed or re-ordered after the certify fold is touched. "
                     "An empty certified set is a reportable outcome."),
}


# ------------------------------------------------------------------ the calibration fold, in two stages

# The access boundary is a property of the CODE, not of a comment. A single-stage loader that materialises
# the whole calibration role - labels, agent scores, detector features - and then applies a select mask has
# already READ the certify fold before R1, whatever it does with it afterwards. Functional independence
# (perturb certify, get the same design) is a weaker statement than the one A1 and C5 make. So the loader is
# split in two, and the split is enforced by construction:
#
#   stage 1  resolve_role_split()   splits.csv ONLY - uid, role, unit, event. No label, no score, no feature.
#                                   Produces the select / certify UID sets from the frozen seed.
#   stage 2  load_uids(select)      opens detector / agent CSVs and the manifest for the SELECT uids ONLY.
#
# cascade.ace3_dev calls load_uids again, for the certify uids, AFTER R1 is verified and the audit receipt
# is written. Nothing in this module can reach a certify row: `load_select` passes only select uids, and
# every row outside that set is skipped before any numeric cell is parsed.


@dataclass(frozen=True)
class RoleSplit:
    """The select / certify partition of one frozen role, derived from METADATA ONLY."""
    role: str
    select_fraction: float
    seed: int
    units: tuple                       # every unit of the role, sorted
    select_units: frozenset
    certify_units: frozenset
    select_uids: frozenset
    certify_uids: frozenset
    roles_present: tuple
    units_by_role: tuple               # (role, n_units) pairs, for P0

    @property
    def n_select_units(self):
        return len(self.select_units)

    @property
    def n_certify_units(self):
        return len(self.certify_units)

    def to_dict(self):
        return {"role": self.role, "select_fraction": self.select_fraction, "seed": self.seed,
                "n_units": len(self.units), "n_select_units": self.n_select_units,
                "n_certify_units": self.n_certify_units,
                "n_select_uids": len(self.select_uids), "n_certify_uids": len(self.certify_uids),
                "roles_present": list(self.roles_present),
                "derived_from": "splits.csv only (uid, role, unit, event); no label, score or feature read",
                "select_units_digest": membership_digest(sorted(self.select_units)),
                "certify_units_digest": membership_digest(sorted(self.certify_units))}


def read_splits(protocol):
    """splits.csv as a list of rows. Label-free by construction: it has no label or score column."""
    with open(Path(protocol) / "splits.csv", newline="") as f:
        return list(csv.DictReader(f))


def resolve_role_split(protocol, sf, seed, role=ROLE):
    """STAGE 1. The select / certify partition, from splits.csv alone.

    The partition is produced by the frozen `certify_mask` applied to the role's sorted unique unit ids.
    That is the same RNG draw the per-frame call makes - `certify_mask` starts with `np.unique(unit)`, so
    feeding it an already-unique sorted array gives an identical `uu`, an identical draw and an identical
    select set - which is why this function reuses it instead of reimplementing the draw.
    """
    rows = read_splits(protocol)
    roles = sorted({r["role"] for r in rows})
    by_role = {}
    for r in rows:
        by_role.setdefault(r["role"], set()).add(r["unit"])
    mine = [r for r in rows if r["role"] == role]
    if not mine:
        raise SystemExit(f"no rows with split role {role!r} in splits.csv: nothing to register "
                         f"(roles present: {roles})")
    units = tuple(sorted({r["unit"] for r in mine}))
    cert_u = certify_mask(np.asarray(units), sf, seed)
    select_units = frozenset(u for u, c in zip(units, cert_u) if not c)
    certify_units = frozenset(u for u, c in zip(units, cert_u) if c)
    return RoleSplit(role=role, select_fraction=float(sf), seed=int(seed), units=units,
                     select_units=select_units, certify_units=certify_units,
                     select_uids=frozenset(r["uid"] for r in mine if r["unit"] in select_units),
                     certify_uids=frozenset(r["uid"] for r in mine if r["unit"] in certify_units),
                     roles_present=tuple(roles),
                     units_by_role=tuple(sorted((k, len(v)) for k, v in by_role.items())))


def load_uids(dataset, protocol, det_path, agents, uids, *, side, role=ROLE):
    """STAGE 2. Labels, scores and features for the GIVEN uids only.

    `uids` is the gate: a row whose uid is not in it is skipped before any numeric cell is parsed, so a
    certify row cannot be read by a caller that did not ask for certify uids. `side` is recorded on the
    returned dict so a downstream function cannot mistake one fold for the other.
    """
    from .esva_dev import _meta, _sha, read_protocol
    uids = frozenset(uids)
    P = read_protocol(protocol, need_esva=False)
    s_min = float(P["s_min"])
    split = {r["uid"]: r for r in read_splits(protocol) if r["uid"] in uids}
    det = read_csv_by_uid(det_path)
    # EVAL_SPLITS, not ("val",): the frozen ROLE selects the rows, and the roles live in different dataset
    # splits - make_protocol maps dataset split "val" -> role "dev", while "calibration" and "sealed_test"
    # are the two halves of dataset split "test". Filtering on ("val",) here, as the dev-only loaders do,
    # selects nothing for this study. Same pattern as cascade.build_records.
    rows = [r for r in read_manifest(dataset, EVAL_SPLITS)
            if r["uid"] in uids and split.get(r["uid"], {}).get("role") == role]
    if not rows:
        raise SystemExit(f"no {side} frames for role {role!r}: nothing to load")
    D = None
    for name, path in sorted(agents.items()):
        am = _meta(path)
        problems = []
        if am.get("dry"):
            problems.append(f"{name}: dry agent scores are forbidden")
        if float(am.get("s_min", -1)) != s_min:
            problems.append(f"{name}: agent s_min {am.get('s_min')} != protocol s_min {s_min}")
        if am.get("det_csv_sha256") != _sha(det_path):
            problems.append(f"{name}: agent meta det_csv_sha256 != sha256(--det)")
        if problems:
            raise SystemExit("score contract failed:\n  " + "\n  ".join(problems))
        ag = read_csv_by_uid(path)
        cols = ("uid", "source", "unit", "event", "y", "y_fire", "y_smoke", "s", "s_fire", "s_smoke", "g",
                "boxes", "stem")
        out = {k: [] for k in cols}
        missing = []
        for r in rows:
            u, d = r["uid"], det.get(r["uid"])
            if d is None or (float(d["s"]) >= s_min and u not in ag):
                missing.append(u)
                continue
            y, ys, yf = frame_label(r)
            sp = split[u]
            vals = dict(uid=u, source=r["source"], unit=sp["unit"], event=sp["event"], y=y, y_fire=yf,
                        y_smoke=ys, s=float(d["s"]), s_fire=float(d["s_fire"]),
                        s_smoke=float(d["s_smoke"]),
                        g=float(ag[u]["g"]) if u in ag else np.nan, boxes=d.get("boxes", ""),
                        stem=Path(r["image"]).stem.split("__", 1)[-1])
            for k, v in vals.items():
                out[k].append(v)
        if missing:
            raise SystemExit(f"fail-closed: {len(missing)} {side} frames without detector/agent score for "
                             f"{name}, e.g. {missing[:3]}")
        Di = {k: np.asarray(v) for k, v in out.items()}
        for k in ("y", "y_fire", "y_smoke"):
            Di[k] = (Di[k].astype(float) > 0).astype(int)
        if D is None:
            D = {k: v for k, v in Di.items() if k != "g"}
            D["features"] = smoke_features(D.pop("boxes"), D["s_fire"], D["s_smoke"])
        elif not np.array_equal(np.asarray(D["uid"]), np.asarray(Di["uid"])):
            raise SystemExit(f"acquisition {name!r} covers different {side} rows than the first one")
        D[f"g_{name}"] = np.asarray(Di["g"], float)
    D["_side"] = side
    return D, P


def load_select(dataset, protocol, det_path, agents, rs: RoleSplit):
    """The design fold, and nothing else. The only loader this module calls."""
    D, P = load_uids(dataset, protocol, det_path, agents, rs.select_uids, side="select", role=rs.role)
    got = set(np.asarray(D["unit"]).tolist())
    if got != set(rs.select_units):
        raise SystemExit(f"the loaded select rows cover {len(got)} units, expected "
                         f"{rs.n_select_units} from the metadata-only role resolution")
    return D, P


def load_certify(dataset, protocol, det_path, agents, rs: RoleSplit):
    """The certify fold. Called ONLY by cascade.ace3_dev, after R1 is verified."""
    D, P = load_uids(dataset, protocol, det_path, agents, rs.certify_uids, side="certify", role=rs.role)
    got = set(np.asarray(D["unit"]).tolist())
    if got != set(rs.certify_units):
        raise SystemExit(f"the loaded certify rows cover {len(got)} units, expected {rs.n_certify_units}")
    return D, P


def all_rows(D):
    """The mask that selects every loaded row. A fold-sized D needs no sub-mask (see the two-stage note)."""
    return np.ones(len(np.asarray(D["uid"])), bool)


def split_params(P):
    """(select_fraction, seed) for the calibration split, quoted from the protocol by value."""
    return float(P.get("select_fraction", 0.3)), int(P.get("calibrate_seed", 1))


def positive_unit_ids(D, mask, k):
    unit = np.asarray(D["unit"])
    pos = mask & (np.asarray(D[f"y_{k}"]).astype(int) > 0)
    return sorted(set(unit[pos].tolist()))


def select_counts(D, sel):
    """n_sel_k: positive-unit counts on calibration-SELECT. A design-fold quantity."""
    return {k: len(positive_unit_ids(D, sel, k)) for k in RISKS}


def design_reference_counts(n_sel, sf):
    """n_ref_k = floor(n_sel_k * (1 - sf) / sf); at sf = 0.3 this is floor(7 n_sel_k / 3)."""
    return {k: int(np.floor(int(v) * (1.0 - float(sf)) / float(sf))) for k, v in n_sel.items()}


def choose_branch(n_ref):
    """Rule R1, from n_ref ALONE. Returns (name, branch) or raises the design-stage refusal."""
    for name in ("full", "fallback"):
        if all(int(n_ref[k]) >= BRANCHES[name]["n_min"] for k in RISKS):
            return name, BRANCHES[name]
    raise SystemExit("C3_CERTIFIABILITY_AUDIT: design-stage refusal - the design reference counts "
                     f"{n_ref} do not reach n_min for either branch "
                     f"({ {k: v['n_min'] for k, v in BRANCHES.items()} })")


# ------------------------------------------------------------------ frozen family construction

def select_stage(D, sel, P, counts_ref, modes, delta_m, log=print):
    """Choose (t_low, t_high, A) AND the b ladder together on calibration-SELECT, at the frozen lambda.

    The literal contract of R_0 §6. `counts_ref` is n_ref and nothing else: this function must never be
    handed a certify-fold count, which is why the caller passes it explicitly rather than reading it here.

        r_star_ref_k = certification_boundary(n_ref_k, alpha, delta_m)
        s_k(pi)      = r_star_ref_k - R_sel_k(pi)
        s_joint(pi)  = min_k s_k(pi)

    A configuration is admissible iff it has >= min_members valid members, its safe endpoint is
    joint-select-feasible (max_pi s_joint >= 0) and the ladder is BOUNDARY_SPANNING (min_pi s_joint < 0).
    Among admissible configurations, minimise the declared utility over the members on the certifiable side.
    """
    alpha, dm = float(P["alpha"]), float(delta_m)
    costs = {k: float(v) for k, v in P["costs"].items() if k in ("calls", "handoff", "fa")}
    rstar_ref = {k: certification_boundary(int(counts_ref[k]), alpha, dm) for k in RISKS}
    spec = StratumSpec(DECLARATION["veto"]["stratum"]["kind"], DECLARATION["veto"]["stratum"]["area_max"])
    inS = in_stratum(spec, D["features"])
    lam = float(DECLARATION["veto"]["lambda_fixed"])
    minmem = int(DECLARATION["arms"]["B"]["min_members"])
    out = {"_r_star_ref": rstar_ref}
    for acq in modes:
        s = D["s"][sel]
        gv = discount(D[f"g_{acq}"][sel], inS[sel], lam)
        grid = Grid.default(s, s_min=float(P["s_min"]))
        risks = {k: D[f"y_{k}"][sel].astype(bool) for k in RISKS}
        E, R, _ = _grid_stats(s, gv, D["y"][sel].astype(bool), D["unit"][sel], grid, "unit", None,
                              D["event"][sel], risks)
        valid = candidate_mask(grid, cloud=True, handoff=True) & ~np.isnan(E["fa"])
        sj = np.min(np.stack([rstar_ref[k] - np.nan_to_num(R[k], nan=np.inf) for k in RISKS]), axis=0)
        obj = _objective(E, costs)
        K = valid.shape[2]
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
            raise SystemExit(f"{acq}: no threshold configuration gives a boundary-spanning ladder whose "
                             f"safe endpoint is select-feasible (r_star_ref = "
                             f"{ {k: round(v, 4) for k, v in rstar_ref.items()} })")
        u, j1, j2, k2 = best
        ks = [int(k) for k in range(k2 + 1) if valid[j1, j2, k, k2]]
        lad = [float(grid.g_edges[k]) for k in ks]
        prof = [float(sj[j1, j2, k, k2]) for k in ks]
        per = [{k: float(rstar_ref[k] - R[k][j1, j2, kk, k2]) for k in RISKS} for kk in ks]
        out[acq] = {"t_low": float(grid.s_edges[j1]), "t_high": float(grid.s_edges[j2]),
                    "A": float(grid.g_edges[k2]), "b_ladder": lad, "joint_slack": prof,
                    "slack_per_loss": per, "select_utility": u,
                    "select_risk_safe_endpoint": {k: float(R[k][j1, j2, ks[0], k2]) for k in RISKS},
                    "grid": grid.to_dict()}
        st, info = boundary_spanning(prof)
        out[acq]["spanning"] = {"status": st, **info}
        log(f"  {acq:8s} t=({out[acq]['t_low']:.4f}, {out[acq]['t_high']:.4f})  A={out[acq]['A']:.4f}  "
            f"{len(lad)} b members  slack {prof[0]:+.4f} .. {prof[-1]:+.4f}  {st}  util={u:.4f}")
    return out


def family_arm(stage, modes, arm):
    """The policies of ONE arm, safest first within each mode. Arms are never mixed."""
    lam_fixed = float(DECLARATION["veto"]["lambda_fixed"])
    b_L = float(DECLARATION["b_fixed_L"])
    pol = []
    for acq in modes:
        st = stage[acq]
        if arm == "B":
            for b in st["b_ladder"]:                                   # ascending b = safest first
                pol.append(Policy(acq, st["t_low"], st["t_high"], float(b), st["A"],
                                  {"S": lam_fixed, "rest": 0.0}, float("inf"), f"B|{acq}|b={b:g}"))
        elif arm == "L":
            if b_L > float(st["A"]) + 1e-12:
                raise SystemExit(f"{acq}: arm L's declared b = {b_L:g} is not a valid candidate at the "
                                 f"chosen A = {st['A']:g} (b <= A is required); design-stage REFUSE")
            for x in sorted(DECLARATION["lambda_grid"], reverse=True):  # descending lambda = safest first
                pol.append(Policy(acq, st["t_low"], st["t_high"], b_L, st["A"], {"S": float(x), "rest": 0.0},
                                  float("inf"), f"L|{acq}|lam={x:g}"))
        else:
            raise ValueError(arm)
    return tuple(pol)


def arm_covers(stage, modes, losses):
    """{arm: (chains, cover_id)}. The cover is built WITHIN an arm, never over the union (R_0 §3)."""
    out = {}
    for arm in ARMS:
        pol = family_arm(stage, modes, arm)
        chains = build_chains(pol, losses)
        out[arm] = {"policies": pol, "chains": chains, "cover_id": chain_cover_id(chains)}
    return out


def declared_vs_realized(covers, branch):
    """Verify the REALIZED cover against the branch's literally declared (M, delta_m). REFUSE on mismatch.

    `select_stage` consumes delta_m before it can choose a threshold, so delta_m is DECLARED and then
    VERIFIED rather than derived - the only acyclic arrangement that does not require rewriting the released
    selection rule. Nothing here re-plans, re-seeds or switches branch on a mismatch.
    """
    shares = [float(d) for v in covers.values() for d in allocate_delta(v["chains"], DELTA_ARM, None)]
    M = sum(len(v["chains"]) for v in covers.values())
    m_ok = M == int(branch["M"])
    dm_ok = bool(shares) and all(abs(d - float(branch["delta_m"])) < 1e-12 for d in shares)
    sum_ok = abs(sum(shares) - DELTA) < 1e-12
    mem_ok = all(len(c) >= int(DECLARATION["arms"][arm]["min_members"])
                 for arm, v in covers.items() for c in v["chains"])
    return {"M_declared": int(branch["M"]), "M_realized": M, "M_ok": bool(m_ok),
            "delta_m_declared": float(branch["delta_m"]), "delta_m_realized": shares,
            "delta_m_ok": bool(dm_ok), "delta_sum_ok": bool(sum_ok), "min_members_ok": bool(mem_ok),
            "ok": bool(m_ok and dm_ok and sum_ok and mem_ok),
            "on_mismatch": "REFUSE; no re-planning, no re-seeding, no branch switch"}


def cover_is_arm_pure(covers):
    """The registered invariant: no chain may contain members of both arms."""
    bad = []
    for arm, v in covers.items():
        for c in v["chains"]:
            arms = {p.name.split("|", 1)[0] for p in c}
            if arms != {arm}:
                bad.append(f"{arm}: chain {[p.name for p in c]} mixes arms {sorted(arms)}")
    return bad


# ------------------------------------------------------------------ the influence envelopes

def _event_frac_per_unit(D, mask, k, event_pred):
    """|E_ku n S| / |E_ku| averaged over units with at least one positive k-event.

    `event_pred(frame_index_array) -> per-event bool` is applied by reducing the positive frames of each
    event with logical_or, which is what both envelopes need.
    """
    y = np.asarray(D[f"y_{k}"][mask]).astype(bool)
    if not y.any():
        return 0.0
    ev, un = np.asarray(D["event"][mask]), np.asarray(D["unit"][mask])
    e_ids, inv = np.unique(ev[y], return_inverse=True)
    inS_ev = event_pred(e_ids, inv, y)
    first = np.zeros(len(e_ids), int)
    first[inv[::-1]] = np.arange(int(y.sum()))[::-1]
    _, uinv = np.unique(un[y][first], return_inverse=True)
    frac = np.bincount(uinv, weights=inS_ev) / np.bincount(uinv)
    return float(frac.mean()) if len(frac) else 0.0


def envelope_B(D, mask, st, k):
    """S_B: MINIMAL envelope of b at the frozen (t_low, t_high) - escalated and not edge-alarmed."""
    s = np.asarray(D["s"][mask])
    y = np.asarray(D[f"y_{k}"][mask]).astype(bool)
    esc = (s >= float(st["t_low"])) & (s < float(st["t_high"]))
    edge = s >= float(st["t_high"])

    def pred(e_ids, inv, yy):
        has_edge = np.zeros(len(e_ids), bool)
        has_esc = np.zeros(len(e_ids), bool)
        np.logical_or.at(has_edge, inv, edge[yy])
        np.logical_or.at(has_esc, inv, esc[yy])
        return has_esc & ~has_edge

    del y
    return _event_frac_per_unit(D, mask, k, pred)


def envelope_L(D, mask, k):
    """S_L: the released `lambda_capacity` predicate, carried over unchanged. CONSERVATIVE, not minimal."""
    spec = StratumSpec(DECLARATION["veto"]["stratum"]["kind"], DECLARATION["veto"]["stratum"]["area_max"])
    inS = in_stratum(spec, D["features"])[mask]

    def pred(e_ids, inv, yy):
        out = np.zeros(len(e_ids), bool)
        np.logical_or.at(out, inv, inS[yy])
        return out

    return _event_frac_per_unit(D, mask, k, pred)


def chain_envelope_capacity(D, mask, stage, covers, source, basis):
    """Gamma per (arm, chain, k), each chain at ITS OWN mode and with ITS OWN envelope."""
    out = {}
    for arm, v in covers.items():
        for c in v["chains"]:
            acq = c[0].acq
            st = stage[acq]
            key = f"{arm}|{acq}"
            out[key] = {}
            for k in RISKS:
                g = envelope_B(D, mask, st, k) if arm == "B" else envelope_L(D, mask, k)
                out[key][k] = Capacity(k, g, source, f"{basis}; envelope S_{arm}, mode {acq}")
    return out


def arm_min_capacity(caps, arm):
    """Study 2's minimum over modes, taken WITHIN ONE ARM. Pre-freeze checklist only, never a (C,k) status.

    Over the union of the arms it would be wrong twice: the arms are two registered families with their own
    delta, and arm L's envelope S_L is a different object from arm B's S_B, so a minimum across them mixes
    two coordinates. Taking it across arms also destroys the point of the arm-wise cover: a small Gamma_L
    would fail arm B's checklist, for a chain arm B does not contain.
    """
    mine = {key: per for key, per in caps.items() if key.split("|", 1)[0] == arm}
    if not mine:
        raise SystemExit(f"no registered chain in arm {arm!r}: the capacity minimum is undefined")
    return {k: Capacity(k, min(v[k].value for v in mine.values()), next(iter(mine.values()))[k].source,
                        f"minimum over arm {arm}'s registered chains; pre-freeze checklist only")
            for k in RISKS}


# ------------------------------------------------------------------ the R1 statuses (all known at R1)

def r1_statuses(caps_sel, r_star_ref):
    """CAPABLE, NO_REACH, KNIFE_EDGE and R2 - functions of the design fold, n_ref and declared constants.

    Nothing here may be recomputed from a certify-fold count: that is the whole point of the namespace.
    """
    alpha = {L["name"]: float(L["alpha"]) for L in DECLARATION["losses"]}
    tol = float(DECLARATION["predictions"]["tolerance"]["knife_edge"])
    h = {k: max(0.0, alpha[k] - float(r_star_ref[k])) for k in RISKS}
    out = {"h_star_ref": h, "r_star_ref": {k: float(v) for k, v in r_star_ref.items()}, "pairs": {}}
    for key, per_k in caps_sel.items():
        for k in RISKS:
            g = float(per_k[k].value)
            out["pairs"][f"{key}|{k}"] = {
                "chain": key, "loss": k, "arm": key.split("|", 1)[0], "mode": key.split("|", 1)[1],
                "gamma_sel": g, "h_star_ref": h[k], "capable": bool(g >= h[k]),
                "knife_edge": bool(abs(g - h[k]) <= tol), "no_reach": False}
    return out


def mark_no_reach(statuses, covers):
    for arm, v in covers.items():
        for c in v["chains"]:
            if len(c) == 1:
                for k in RISKS:
                    key = f"{arm}|{c[0].acq}|{k}"
                    if key in statuses["pairs"]:
                        statuses["pairs"][key]["no_reach"] = True
    return statuses


def rule_r2(statuses):
    """INFORMATIVE iff both a CAPABLE and an INCAPABLE eligible pair exist. Decided at R1, never redone."""
    elig = [p for p in statuses["pairs"].values() if not p["knife_edge"] and not p["no_reach"]]
    yes = any(p["capable"] for p in elig)
    no = any(not p["capable"] for p in elig)
    return {"verdict": "INFORMATIVE" if (yes and no) else "UNINFORMATIVE",
            "n_eligible": len(elig), "n_capable": sum(p["capable"] for p in elig),
            "n_incapable": sum(not p["capable"] for p in elig),
            "excluded_knife_edge": [k for k, p in statuses["pairs"].items() if p["knife_edge"]],
            "excluded_no_reach": [k for k, p in statuses["pairs"].items() if p["no_reach"]],
            "decided_at": "R1", "never_recomputed": True}


# ------------------------------------------------------------------ P0 and R1

def declared_losses():
    return tuple(miss_loss(L["name"], L["alpha"], L["lemma"], catch_set=L["catch_set"],
                           aggregation=L["aggregation"], weighting=L["weighting"],
                           empty_denominator=L["empty_denominator"]) for L in DECLARATION["losses"])


def p0_set_operations(protocol, rs: RoleSplit):
    """P0: ACTUAL id-set operations on splits.csv, then a digest as attestation of what was asserted.

    METADATA ONLY. It takes the stage-1 RoleSplit rather than a loaded fold, so P0 cannot be the step that
    opens a label: splits.csv carries uid, split, role, source, unit, event and nothing else.
    """
    rows = read_splits(protocol)
    by_role = {}
    for r in rows:
        by_role.setdefault(r["role"], set()).add(r["unit"])
    U_sel, U_cert = set(rs.select_units), set(rs.certify_units)
    U_sealed = by_role.get("sealed_test", set())
    pairs = {"select&certify": U_sel & U_cert, "select&sealed": U_sel & U_sealed,
             "certify&sealed": U_cert & U_sealed}
    every_unit_one_role = all(len(v) == len({r["unit"] for r in rows if r["role"] == k})
                              for k, v in by_role.items())
    roles_of = {}
    for r in rows:
        roles_of.setdefault(r["unit"], set()).add(r["role"])
    multi = sorted(u for u, rs in roles_of.items() if len(rs) > 1)
    calib = by_role.get(ROLE, set())
    partition_ok = (U_sel | U_cert) == calib
    rep = {"disjointness": {k: sorted(v)[:5] for k, v in pairs.items()},
           "disjoint": all(not v for v in pairs.values()),
           "units_in_more_than_one_role": multi[:5], "every_unit_exactly_one_role": not multi,
           "select_union_certify_equals_calibration": bool(partition_ok),
           "role_sizes": {k: len(v) for k, v in sorted(by_role.items())},
           "n_select_units": len(U_sel), "n_certify_units": len(U_cert),
           "select_fraction": float(rs.select_fraction), "calibrate_seed": int(rs.seed),
           "reproduces_from_seed_alone": True,
           "derived_from": "splits.csv only; no label, score or feature was read to compute this",
           "digests": {"select": membership_digest(sorted(U_sel)),
                       "certify": membership_digest(sorted(U_cert))},
           "digest_note": ("the digests are ATTESTATION of the asserted sets; the set operations above are "
                           "what is checked, and they are checked first"),
           "_ignore": bool(every_unit_one_role)}
    rep.pop("_ignore")
    ok = (rep["disjoint"] and rep["every_unit_exactly_one_role"]
          and rep["select_union_certify_equals_calibration"])
    return rep, bool(ok)


def p0_protocol_state(protocol):
    """P0, the part that is about the protocol rather than the split: frozen, unchanged, and still sealed.

    Any failure here refuses BEFORE the design stage, so no label of any fold is read.
    """
    from .make_protocol import lock_status, read_lock, read_receipt
    problems = list(lock_status(protocol))
    L = read_lock(protocol) or {}
    R = read_receipt(protocol)
    if R is not None:
        problems.append(f"an unseal/final receipt already exists (state {R.get('state')!r}): the sealed "
                        f"fold has been touched, so this study cannot be registered against this protocol")
    return {"protocol_lock_present": bool(L),
            "locked_analysis_code_sha256": L.get("analysis_code_sha256", ""),
            "lock_sha256": L.get("sha256", ""),
            "sealed_test_unseal_receipt": (None if R is None else R.get("state")),
            "problems": problems, "ok": not problems,
            "gate": "freeze must precede the design stage; a stale lock or an opened seal refuses here"}


def verify(protocol, D, P, rs: RoleSplit, log=print):
    """P0 -> design on calibration-select -> R1.

    `D` is the SELECT fold and only the select fold: it was loaded through `load_select`, which was handed
    the select uid set that stage 1 derived from splits.csv. There is no certify row in this function's
    reach - not behind a mask, not in memory - so the boundary A1 and C5 assert is a property of the call
    graph rather than of a mask applied after the fact.
    """
    from .make_protocol import verify_tree
    rep, ok = {}, True
    sf = rs.select_fraction
    rep["stages"] = list(DECLARATION["stages"])
    rep["P0_role_split"] = rs.to_dict()
    if D.get("_side") != "select":
        raise SystemExit(f"verify() was handed a {D.get('_side')!r} fold; it accepts the select fold only")

    state = p0_protocol_state(protocol)
    rep["P0_protocol_state"] = state
    ok &= state["ok"]
    if not state["ok"]:
        log("P0 REFUSES before the design stage: " + "; ".join(state["problems"]))
        return rep, False

    span = events_spanning_units(protocol)
    rep["P0_events_spanning_units"] = span
    ok &= not span

    p0, p0_ok = p0_set_operations(protocol, rs)
    rep["P0_set_operations"] = p0
    ok &= p0_ok

    sel = all_rows(D)                 # D IS the select fold; there is no sub-mask to apply

    # ---- design reference counts and the branch: n_ref ONLY ----------------------------------------
    n_sel = select_counts(D, sel)
    n_ref = design_reference_counts(n_sel, sf)
    branch_name, branch = choose_branch(n_ref)
    rep["R0_design_reference"] = {
        "n_sel": n_sel, "n_ref": n_ref, "rule": DECLARATION["design_reference_count"]["rule"],
        "select_fraction": sf, "branch": branch_name, "M_declared": branch["M"],
        "delta_m_declared": branch["delta_m"], "n_min_of_branch": branch["n_min"],
        "modes": list(branch["modes"]),
        "support_proxy_note": DECLARATION["support_proxy"],
        "certify_counts_read": False}
    log(f"n_sel {n_sel} -> n_ref {n_ref} -> branch {branch_name!r} "
        f"(M={branch['M']}, delta_m={branch['delta_m']}, modes={list(branch['modes'])})")

    # ---- the family, on calibration-select only ----------------------------------------------------
    log("calibration-select -> thresholds and b ladder chosen TOGETHER (frozen here):")
    stage = select_stage(D, sel, P, n_ref, branch["modes"], branch["delta_m"], log=log)
    r_star_ref = stage.pop("_r_star_ref")
    losses = declared_losses()
    covers = arm_covers(stage, branch["modes"], losses)

    mixed = cover_is_arm_pure(covers)
    rep["R1_cover"] = {arm: {"M": len(v["chains"]), "cover_id": v["cover_id"],
                             "chains": [[p.name for p in c] for c in v["chains"]],
                             "delta_arm": DELTA_ARM,
                             "delta_m": [float(d) for d in allocate_delta(v["chains"], DELTA_ARM, None)]}
                       for arm, v in covers.items()}
    rep["R1_cover"]["arm_purity_violations"] = mixed
    rep["R1_cover"]["M_total"] = sum(len(v["chains"]) for v in covers.values())
    rep["R1_cover"]["combined_cover_id"] = hashlib.sha256(
        "\n".join(covers[a]["cover_id"] for a in ARMS).encode()).hexdigest()
    rep["R1_cover"]["delta_split"] = DECLARATION["cover"]["delta_split"]
    ok &= not mixed

    # declared (M, delta_m) vs the REALIZED cover; refuse on any mismatch, no re-planning
    dvr = declared_vs_realized(covers, branch)
    rep["R1_declared_vs_realized"] = dvr
    ok &= dvr["ok"]

    pol_all = tuple(p for arm in ARMS for p in covers[arm]["policies"])
    rep["R1_family"] = {"thresholds": stage, "r_star_ref": {k: float(v) for k, v in r_star_ref.items()},
                        "policies": [p.to_dict() for p in pol_all],
                        "arms": {arm: [p.name for p in covers[arm]["policies"]] for arm in ARMS},
                        "boundary_namespace": ("r_star_ref only; r_star_cert is opened by ace3_dev and is "
                                               "used by the certificate and by X, never by a design or R1 "
                                               "quantity")}
    rep["R1_boundary_spanning"] = {a: stage[a]["spanning"] for a in branch["modes"]}
    rep["R1_inference_registration"] = declare_inference(DECLARATION["study"], losses).to_dict()

    # ---- the design-fold capacities and every R1 status --------------------------------------------
    caps_sel = chain_envelope_capacity(D, sel, stage, covers, "design",
                                       "calibration-select fold, at the frozen thresholds")
    rep["R1_capacity"] = {key: {k: v.to_dict() for k, v in per.items()} for key, per in caps_sel.items()}
    statuses = mark_no_reach(r1_statuses(caps_sel, r_star_ref), covers)
    rep["R1_statuses"] = statuses
    rep["R1_rule_R2"] = rule_r2(statuses)
    log(f"R1: {rep['R1_rule_R2']['verdict']} "
        f"({rep['R1_rule_R2']['n_capable']} capable / {rep['R1_rule_R2']['n_incapable']} incapable of "
        f"{rep['R1_rule_R2']['n_eligible']} eligible pairs)")

    # ---- the 9-item pre-freeze checklist, per arm --------------------------------------------------
    # The audit-membership item is verified HERE against the design fold, as a proxy: the audit population
    # does not exist yet, and manufacturing one would be exactly the dependence A1 forbids. cascade.ace3_dev
    # re-runs the same nine items against the realized audit metadata before any risk is evaluated.
    # TWO DIFFERENT BARS, and they are not interchangeable. `preflight_checklist`'s `select_bar` is the
    # bound the SELECTION STAGE guarantees for the candidates it hands over; Study 3's select_stage
    # guarantees the safe endpoint at r*_ref, so select_bar = r*_ref and the capacity item is vacuous
    # (h* = max(0, r*_ref - r*_ref) = 0) - the same situation Study 2 registered, and correct rather than a
    # loophole: there is no anchor left to pull across. Study 3's bar_k = alpha_k is a PREDICTION-REFERENCE
    # bar, declared in R_0 §5 to make the reach margin non-vacuous, and it lives in R1_statuses. Feeding
    # alpha to the checklist conflates the two and makes a family-DESIGN guard forbid the very incapable
    # coordinate R2 needs in order to be INFORMATIVE - a gate cannot also be the object under test.
    # `counts` is n_ref for the same reason: r*_ref is the only design-side boundary R_0 defines.
    sel_digests = {k: membership_digest(positive_unit_ids(D, sel, k)) for k in RISKS}

    def design_contracts(_p):
        return Contract(True, True, n_ref, rs.n_select_units, sel_digests, 0.0, float("inf"))

    import yaml
    Y = yaml.safe_load((Path(protocol) / "protocol.yaml").read_text())
    rep["R1_checklist_design_proxy"] = {}
    for arm in ARMS:
        rows = preflight_checklist(
            covers[arm]["policies"], losses, contracts=design_contracts, plan=None,
            payload_semantics=None,
            weights_sha256=str((Y.get("detector") or {}).get("weights_sha256") or ""),
            capacities=arm_min_capacity(caps_sel, arm), delta=DELTA_ARM,
            select_bar={k: float(r_star_ref[k]) for k in RISKS},
            joint_slacks={a: stage[a]["joint_slack"] for a in branch["modes"]},
            inference_registration=declare_inference(DECLARATION["study"], losses))
        rep["R1_checklist_design_proxy"][arm] = rows
        ok &= all(r["ok"] for r in rows)
    rep["R1_checklist_bars"] = {
        "checklist_select_bar": {k: float(r_star_ref[k]) for k in RISKS},
        "checklist_counts": dict(n_ref),
        "checklist_h_star": {k: 0.0 for k in RISKS},
        "why_vacuous": ("the selection rule guarantees the safe endpoint at r*_ref, so the margin the "
                        "coordinate is asked to span is max(0, r*_ref - r*_ref) = 0. The item verifies the "
                        "SELECTION GUARANTEE; it is not the study's capacity statement."),
        "prediction_reference_bar": {k: float(DECLARATION["alpha"]) for k in RISKS},
        "where_the_non_vacuous_statuses_live": "R1_statuses (h_star_ref = alpha - r*_ref, per (C,k))",
        "capacity_minimum_scope": "within one arm; never over the union of the arms"}
    rep["R1_checklist_scope"] = (
        "nine items per arm, closed on the design fold. `policy_independent_audit_membership` is verified "
        "here as a DESIGN PROXY on calibration-select, because the audit population does not exist before A "
        "is opened; cascade.ace3_dev re-runs all nine against the realized audit metadata, before any risk "
        "is evaluated, and refuses on any item that does not close there.")

    tree = verify_tree()
    rep["R1_runtime_tree"] = {k: tree[k] for k in ("ok", "extra", "missing", "changed")}
    ok &= tree["ok"]
    return rep, bool(ok)


# ------------------------------------------------------------------ CLI

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

    # stage 1: metadata only. stage 2: the select fold only. The certify fold is never opened here.
    from .esva_dev import read_protocol
    P0 = read_protocol(a.protocol, need_esva=False)
    sf, seed = split_params(P0)
    rs = resolve_role_split(a.protocol, sf, seed)
    print(f"role split (metadata only): {rs.n_select_units} select / {rs.n_certify_units} certify units "
          f"of {len(rs.units)}; certify uids are NOT loaded by this module")
    D, P = load_select(a.dataset, a.protocol, a.det, agents, rs)
    rep, ok = verify(a.protocol, D, P, rs)
    print(json.dumps(rep, indent=1))
    print("\npreconditions:", "ALL HOLD" if ok else "NOT SATISFIED - nothing is registered")
    if not ok:
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
    rec = {"study": STUDY, "spec_version": SPEC_VERSION, "receipt": "R1",
           "registered_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "declaration": DECLARATION, "preconditions": rep, "checklist_items": list(CHECKLIST),
           "protocol_sha256": sha256_file(Path(a.protocol) / "protocol.yaml"),
           "splits_sha256": sha256_file(Path(a.protocol) / "splits.csv"),
           "analysis_code_sha256": analysis_code()[0], "det_sha256": sha256_file(a.det),
           "agent_sha256": {k: sha256_file(v) for k, v in sorted(agents.items())},
           "audit_metadata_present": False,
           "audit_metadata_note": ("this is the REGISTRATION receipt. It carries no audit metadata by "
                                   "construction: Z_k, the realized n_k and the membership digests are "
                                   "recorded in the AUDIT receipt, created afterwards by cascade.ace3_dev, "
                                   "and they may only verify this frozen procedure or refuse it."),
           "note": a.note}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    print(json.dumps({k: rec[k] for k in ("study", "spec_version", "receipt", "registered_utc",
                                          "protocol_sha256", "analysis_code_sha256")}, indent=1))
    print("cover ids:", {a_: rep["R1_cover"][a_]["cover_id"][:12] for a_ in ARMS})
    print("R2:", rep["R1_rule_R2"]["verdict"])
    print("->", out)


if __name__ == "__main__":
    main()
