"""Tests for Study 3's registration and run modules — the refuse-only invariant, made executable.

Seven properties, each one a thing that would silently break the study if it stopped holding:

1  the design stage never sees the audit: change the certify-fold labels as violently as you like and
   n_sel, n_ref, the branch, the thresholds, the ladders, the arm covers, the intra-chain order, the delta
   shares, r*_ref and every R1 status come back BIT-FOR-BIT identical;
2  formal C3 REFUSES when the realized support is short of the registered branch's n_min;
3  there is no full -> fallback switch: a realized count that would have satisfied the fallback still
   refuses, and formal C3 returns the branch it was given;
4  the arms never share a chain - and the merged cover really would interleave them, so the arm-wise rule is
   what prevents it rather than luck;
5  a realized cover that does not reproduce the declared (M, delta_m) REFUSES, with no re-planning;
6  a post-registration CHAIN_TERMINAL on a registered primary pair sets P-4.5 = REFUSED, does not recompute
   R2 and does not shrink the denominator;
7  the two boundaries do not mix: T, S and N are functions of the registered r*_ref only, and changing
   r*_cert can move X and nothing else.

No dataset and no protocol directory: the fold is a synthetic two-tier stream built here.
"""
from __future__ import annotations

import contextlib
import csv
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from .ace3_dev import chain_terminal_of, evaluate_predictions, formal_c3, p45_verdict
from .ace3_register import (ARMS, DECLARATION, DELTA_ARM, RISKS, ROLE, all_rows, arm_covers,
                            arm_min_capacity, chain_envelope_capacity, choose_branch, cover_is_arm_pure,
                            declared_losses, declared_vs_realized, design_reference_counts, family_arm,
                            load_certify, load_select, mark_no_reach, r1_statuses, resolve_role_split,
                            rule_r2, select_counts, select_stage)
from .ace_ltt import (BUILTIN_INFERENCE, AdmissibilityResult, Contract, InferenceRegistration,
                      ObservedLoss, allocate_delta, build_chains, certification_boundary, certify_family,
                      chain_cover_id, inference_registry_id, membership_digest)
from .ace_register import certify_mask

SF, SEED = 0.3, 1
P = {"alpha": 0.05, "s_min": 0.02, "select_fraction": SF, "calibrate_seed": SEED,
     "costs": {"calls": 1.0, "fa": 10.0, "handoff": 20.0}}
IREG = InferenceRegistration("ace3-tests-2026-09-27", inference_registry_id([BUILTIN_INFERENCE]),
                             (BUILTIN_INFERENCE,))


# ------------------------------------------------------------------ a synthetic calibration fold

def synth(n_units=1400, seed=7):
    """Units -> frames; some frames belong to a positive fire or smoke event. Nothing domain-specific."""
    rng = np.random.default_rng(seed)
    unit, event, yf, ys = [], [], [], []
    for u in range(n_units):
        n_e = int(rng.integers(1, 3))
        n_f = int(rng.integers(3, 7))
        pos = []
        for j in range(n_e):
            kind = "f" if rng.random() < 0.5 else "s"
            pos += [(f"u{u}e{j}", kind)] * int(rng.integers(1, 3))
        pos = pos[:max(0, n_f - 1)]
        for e, kind in pos:
            unit.append(f"u{u}"); event.append(e)
            yf.append(int(kind == "f")); ys.append(int(kind == "s"))
        for _ in range(n_f - len(pos)):
            unit.append(f"u{u}"); event.append(""); yf.append(0); ys.append(0)
    yf, ys = np.array(yf, int), np.array(ys, int)
    y = ((yf + ys) > 0).astype(int)
    n = len(y)
    s = np.clip(rng.normal(np.where(y > 0, 0.42, 0.11), np.where(y > 0, 0.20, 0.09)), 0.0, 1.0)
    D = {"unit": np.array(unit), "event": np.array(event), "y": y, "y_fire": yf, "y_smoke": ys, "s": s}
    for acq, lift in (("crop", 0.34), ("overlay", 0.22)):
        D[f"g_{acq}"] = np.clip(rng.normal(0.30, 0.16, n) + lift * (y > 0), 0.0, 1.0)
    s_f = np.clip(s + rng.normal(0, 0.03, n), 0, 1)
    s_s = np.clip(s + rng.normal(0, 0.03, n), 0, 1)
    D["features"] = {"smoke_area": rng.random(n) * 0.06, "top_area": rng.random(n) * 0.2,
                     "s_fire": s_f, "s_smoke": s_s}
    return D


def perturb_certify_side(D, cert, seed=99):
    """Rewrite the CERTIFY-side labels and scores. A's counts and membership move; the design must not."""
    rng = np.random.default_rng(seed)
    E = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in D.items()}
    E["features"] = {k: v.copy() for k, v in D["features"].items()}
    m = cert
    E["y_fire"][m] = rng.integers(0, 2, int(m.sum()))
    E["y_smoke"][m] = rng.integers(0, 2, int(m.sum()))
    E["y"] = ((E["y_fire"] + E["y_smoke"]) > 0).astype(int)
    E["s"][m] = rng.random(int(m.sum()))
    for acq in ("crop", "overlay"):
        E[f"g_{acq}"][m] = rng.random(int(m.sum()))
    return E


def design(D, modes=("crop", "overlay")):
    """The whole design stage on an in-memory fold, returned as a comparable dict.

    This helper keeps the mask form because it works on one synthetic array that carries both sides; the
    REAL loader never does (see test_certify_rows_are_not_read_before_R1).
    """
    cert = certify_mask(np.asarray(D["unit"]), SF, SEED)
    sel = ~cert
    n_sel = select_counts(D, sel)
    n_ref = design_reference_counts(n_sel, SF)
    branch_name, branch = choose_branch(n_ref)
    use = tuple(m for m in branch["modes"] if m in modes)
    stage = select_stage(D, sel, P, n_ref, use, branch["delta_m"], log=lambda *_: None)
    r_star_ref = stage.pop("_r_star_ref")
    losses = declared_losses()
    covers = arm_covers(stage, use, losses)
    caps = chain_envelope_capacity(D, sel, stage, covers, "design", "select")
    st = mark_no_reach(r1_statuses(caps, r_star_ref), covers)
    return {"cert": cert, "sel": sel, "stage": stage, "covers": covers, "losses": losses,
            "branch_name": branch_name, "branch": branch, "r_star_ref": r_star_ref, "caps": caps,
            "frozen": {
                "n_sel": n_sel, "n_ref": n_ref, "branch": branch_name,
                "thresholds": json.loads(json.dumps(stage, sort_keys=True)),
                "r_star_ref": {k: float(v) for k, v in r_star_ref.items()},
                "cover": {a: {"id": covers[a]["cover_id"],
                              "chains": [[p.name for p in c] for c in covers[a]["chains"]],
                              "delta_m": [float(d) for d in allocate_delta(covers[a]["chains"],
                                                                           DELTA_ARM, None)]}
                          for a in ARMS},
                "statuses": json.loads(json.dumps(st, sort_keys=True)),
                "R2": rule_r2(st)}}


def fake_rec(d):
    """The R1 receipt the run module reads, built from a design result."""
    return {"study": DECLARATION["study"], "receipt": "R1", "registered_utc": "2026-09-27T00:00:00Z",
            "audit_metadata_present": False, "declaration": DECLARATION,
            "preconditions": {
                "R0_design_reference": {"n_sel": d["frozen"]["n_sel"], "n_ref": d["frozen"]["n_ref"],
                                        "branch": d["branch_name"]},
                "R1_family": {"thresholds": d["stage"], "r_star_ref": d["frozen"]["r_star_ref"],
                              "policies": [p.to_dict() for a in ARMS for p in d["covers"][a]["policies"]]},
                "R1_cover": {a: d["frozen"]["cover"][a] | {"cover_id": d["covers"][a]["cover_id"]}
                             for a in ARMS},
                "R1_statuses": d["frozen"]["statuses"],
                "R1_rule_R2": d["frozen"]["R2"]}}


def audit_of(D, cert):
    out = {}
    unit = np.asarray(D["unit"])
    for k in RISKS:
        pos = cert & (np.asarray(D[f"y_{k}"]).astype(int) > 0)
        ids = sorted(set(unit[pos].tolist()))
        out[k] = {"n_units": len(ids), "sha256": membership_digest(ids)}
    return out, {"n_certify_units": int(len(set(unit[cert].tolist())))}


# ------------------------------------------------------------------ 1

def test_design_stage_never_sees_the_audit():
    D = synth()
    a = design(D)
    b = design(perturb_certify_side(D, a["cert"]))
    aud_a, _ = audit_of(D, a["cert"])
    aud_b, _ = audit_of(perturb_certify_side(D, a["cert"]), a["cert"])
    assert aud_a != aud_b, "the perturbation did not move A, so the test proves nothing"
    assert a["frozen"] == b["frozen"], "a certify-fold change moved a design quantity"
    # and specifically, item by item, so a failure names itself
    for key in ("n_sel", "n_ref", "branch", "thresholds", "r_star_ref", "cover", "statuses", "R2"):
        assert a["frozen"][key] == b["frozen"][key], key


# ------------------------------------------------------------------ 2 and 3

def _c3(d, D, n_cert):
    """formal C3 with the realized counts forced to `n_cert`; everything else as registered."""
    aud, split = audit_of(D, d["cert"])
    aud = {k: {"n_units": int(n_cert[k]), "sha256": aud[k]["sha256"]} for k in RISKS}
    return formal_c3(aud, split, d["branch"], d["covers"], d["losses"], d["stage"], d["caps"],
                     d["r_star_ref"], IREG, "ab" * 32, log=lambda *_: None)


def test_formal_c3_refuses_insufficient_actual_support():
    D = synth()
    d = design(D)
    need = d["branch"]["n_min"]
    ok = _c3(d, D, {k: need + 500 for k in RISKS})
    assert ok["verdict"] == "PASS", ok["problems"]
    for short in RISKS:
        bad = _c3(d, D, {k: (need - 1 if k == short else need + 500) for k in RISKS})
        assert bad["verdict"] == "REFUSE"
        assert any(p.startswith(short) and "n_min" in p for p in bad["problems"]), bad["problems"]
    # an undeclared membership is equally terminal
    aud, split = audit_of(D, d["cert"])
    aud["fire"]["sha256"] = ""
    r = formal_c3(aud, split, d["branch"], d["covers"], d["losses"], d["stage"], d["caps"],
                  d["r_star_ref"], IREG, "ab" * 32, log=lambda *_: None)
    assert r["verdict"] == "REFUSE" and any("not declared" in p for p in r["problems"])


def test_no_full_to_fallback_switch():
    D = synth()
    d = design(D)
    assert d["branch_name"] == "full", "this fold should reach the full branch; adjust the synth size"
    between = {k: 65 for k in RISKS}                  # >= fallback's 59, < full's 72
    r = _c3(d, D, between)
    assert r["verdict"] == "REFUSE", "a count short of the REGISTERED branch must refuse"
    assert any("no full -> fallback switch" in p for p in r["problems"]), r["problems"]
    assert r["branch"]["M"] == d["branch"]["M"] and r["branch"]["delta_m"] == d["branch"]["delta_m"], \
        "formal C3 returned a different branch than the one R1 chose"
    # and the branch rule itself is a function of n_ref alone: it never receives a certify count
    assert choose_branch(d["frozen"]["n_ref"])[0] == d["branch_name"]
    assert choose_branch({k: 65 for k in RISKS})[0] == "fallback", \
        "sanity: 65 WOULD have chosen the fallback at design time - which is exactly why the run may not"


# ------------------------------------------------------------------ 4

def test_arms_never_share_a_chain():
    D = synth()
    d = design(D)
    assert not cover_is_arm_pure(d["covers"]), "an arm-wise cover contained a mixed chain"
    for arm in ARMS:
        for c in d["covers"][arm]["chains"]:
            assert {p.name.split("|", 1)[0] for p in c} == {arm}
    # the merged cover really would interleave: the arm-wise rule is doing the work, not luck
    modes = tuple(m for m in d["branch"]["modes"])
    merged = build_chains(family_arm(d["stage"], modes, "B") + family_arm(d["stage"], modes, "L"),
                          d["losses"])
    mixed = [c for c in merged if len({p.name.split("|", 1)[0] for p in c}) > 1]
    assert mixed, ("the merged cover did not interleave on this fold, so this instance cannot witness the "
                   "blocker; the arm-wise rule is still registered")
    assert chain_cover_id(merged) != d["covers"]["B"]["cover_id"]


# ------------------------------------------------------------------ 5

def test_declared_realized_mismatch_refuses():
    D = synth()
    d = design(D)
    good = declared_vs_realized(d["covers"], d["branch"])
    assert good["ok"] and good["M_ok"] and good["delta_m_ok"] and good["delta_sum_ok"], good

    wrong_M = dict(d["branch"], M=int(d["branch"]["M"]) + 1)
    r = declared_vs_realized(d["covers"], wrong_M)
    assert not r["ok"] and not r["M_ok"] and r["M_realized"] == good["M_realized"]

    wrong_dm = dict(d["branch"], delta_m=0.0125)
    r = declared_vs_realized(d["covers"], wrong_dm)
    assert not r["ok"] and not r["delta_m_ok"]
    assert r["delta_m_realized"] == good["delta_m_realized"], "the check re-planned instead of refusing"
    assert "no re-planning" in r["on_mismatch"] and "no branch switch" in r["on_mismatch"]


# ------------------------------------------------------------------ 6 and 7

def _obs_for(d, shift=0.0):
    """A certify-fold observation per registered member: risk rising along each chain."""
    obs = {}
    for arm in ARMS:
        for c in d["covers"][arm]["chains"]:
            for i, p in enumerate(c):
                r = 0.030 + shift + 0.004 * i
                obs[p.name] = {k: ObservedLoss(r, 900, membership_digest([f"u{k}"])) for k in RISKS}
    return obs


def test_chain_terminal_refuses_the_whole_study():
    D = synth()
    d = design(D)
    rec = fake_rec(d)
    caps_cert = chain_envelope_capacity(D, d["cert"], d["stage"], d["covers"], "realized", "cert")
    r_cert = {k: certification_boundary(900, 0.05, d["branch"]["delta_m"]) for k in RISKS}
    preds = evaluate_predictions(rec, d["covers"], _obs_for(d), r_cert, caps_cert)

    clean = p45_verdict(rec, preds, set())
    assert clean["verdict"] == rec["preconditions"]["R1_rule_R2"]["verdict"]
    assert clean["R2_recomputed"] is False and clean["denominator_shrunk"] is False

    one = sorted(preds["pairs"])[0]
    bad = p45_verdict(rec, preds, {one})
    assert bad["verdict"] == "REFUSED" and bad["terminal_pairs"] == [one]
    assert bad["T_verdict"] is None and bad["S_verdict"] is None, "a verdict survived a terminal chain"
    assert bad["R2_recomputed"] is False and bad["denominator_shrunk"] is False
    assert bad["R2_registered"] == rec["preconditions"]["R1_rule_R2"]["verdict"]


def _chain_names(c):
    return tuple(p.name for p in c)


def _v248_extraction(arm, res, covers):
    """The SUPERSEDED v2.4.8 extraction, kept only so the regression can prove it fails.

    Two faults: it iterates `res.chains`, which `certify_family` has already removed the terminal chains
    from, and it is gated on the reason string, which a stage-(2) drop does not set.
    """
    pairs = set()
    if "CHAIN_TERMINAL" in res.reasons:
        for c in res.chains:
            if any(p.name in res.inadmissible for p in c):
                for k in RISKS:
                    pairs.add(f"{arm}|{c[0].acq}|{k}")
    return pairs


def _family_contracts(policies, broken=None, counts=900, n_units=1000):
    """One contract per member. `broken` fails C1 only, so the family-wide C5 audit-drift check is untouched:
    the counts and membership digests stay IDENTICAL across the family, which is what C5 requires."""
    digs = {k: membership_digest([f"u{k}"]) for k in RISKS}
    cnt = {k: counts for k in RISKS}
    assert broken is None or any(p.name == broken for p in policies)

    def f(p):
        return Contract(p.name != broken, True, cnt, n_units, digs, 0.0, float("inf"))
    return f


def _multi_member_chain(d):
    for arm in ARMS:
        for c in d["covers"][arm]["chains"]:
            if len(c) >= 2:
                return arm, c
    raise AssertionError("the synthetic design produced no multi-member chain; the test needs one")


def test_chain_terminal_extraction_sees_chains_the_engine_dropped():
    """The defect this test exists for: `certify_family` DROPS a terminal chain from `Certification.chains`.

    Any extraction that walks the engine's surviving chains therefore cannot see the chains the
    CHAIN_TERMINAL rule is about, and P-4.5 silently returns the registered R2 verdict instead of REFUSED —
    with the full denominator. Two shapes are exercised:

    (A) real `certify_family` output in which a registered chain genuinely lost a member, so the registered
        chain is absent from `res.chains`. (In `run()` this particular shape also moves the cover id and is
        refused one step earlier; the extraction must not depend on that distant guard for its soundness.)
    (B) the stage-(5) shape, constructed on top of real engine output: `reasons` carries CHAIN_TERMINAL, the
        chain is absent from `res.chains`, and the offending member sits in `inadmissible` — exactly the
        tuple `certify_family` returns when `verify_chain` refuses a chain after the cover is fixed.

    In both, the superseded extraction returns the empty set and the new one returns the pair.
    """
    D = synth()
    d = design(D)
    rec = fake_rec(d)
    caps_cert = chain_envelope_capacity(D, d["cert"], d["stage"], d["covers"], "realized", "cert")
    r_cert = {k: certification_boundary(900, 0.05, d["branch"]["delta_m"]) for k in RISKS}
    obs = _obs_for(d)
    preds = evaluate_predictions(rec, d["covers"], obs, r_cert, caps_cert)
    arm, target = _multi_member_chain(d)
    pol = d["covers"][arm]["policies"]
    want = {f"{arm}|{target[0].acq}|{k}" for k in RISKS}

    # ---- (A) a member really is dropped, so the registered chain is not among the survivors -----------
    victim = target[-1].name
    resA = certify_family(pol, d["losses"], delta=DELTA_ARM,
                          contracts=_family_contracts(pol, broken=victim),
                          certify_eval=lambda p: obs[p.name], plan=None, inference_registration=IREG)
    assert victim in resA.inadmissible, "the planted C1 failure did not drop the member"
    assert _chain_names(target) not in {_chain_names(c) for c in resA.chains}, \
        "the registered chain survived intact; this shape is not the one under test"

    pairsA, seenA = chain_terminal_of(arm, resA, d["covers"])
    assert want <= pairsA and seenA, "the new extraction missed a chain the engine dropped"
    assert _v248_extraction(arm, resA, d["covers"]) == set(), \
        "the superseded extraction was expected to miss this shape; the test is not testing what it claims"

    # ---- (B) the stage-(5) shape: CHAIN_TERMINAL in reasons AND the chain removed ---------------------
    resB = certify_family(pol, d["losses"], delta=DELTA_ARM, contracts=_family_contracts(pol),
                          certify_eval=lambda p: obs[p.name], plan=None, inference_registration=IREG)
    assert _chain_names(target) in {_chain_names(c) for c in resB.chains}, "expected a clean run first"
    assert not resB.inadmissible, "expected no inadmissible member in the clean run"
    resB = replace(
        resB,
        chains=tuple(c for c in resB.chains if _chain_names(c) != _chain_names(target)),
        reasons=tuple(dict.fromkeys([*resB.reasons, "C6_CHAIN_CERTIFICATE", "CHAIN_TERMINAL"])),
        inadmissible={**resB.inadmissible, target[0].name: AdmissibilityResult(
            False, ("C6_CHAIN_CERTIFICATE",), ("chain dropped after the cover was fixed",))})

    assert "CHAIN_TERMINAL" in resB.reasons
    pairsB, seenB = chain_terminal_of(arm, resB, d["covers"])
    assert want <= pairsB and seenB, "the new extraction missed the stage-(5) shape"
    assert _v248_extraction(arm, resB, d["covers"]) == set(), \
        "the superseded extraction was expected to miss the stage-(5) shape"

    # ---- what each extraction then makes P-4.5 say ---------------------------------------------------
    good = p45_verdict(rec, preds, pairsB, seenB)
    assert good["verdict"] == "REFUSED" and good["chain_terminal_observed"] is True
    assert want <= set(good["terminal_pairs"])
    assert good["T_verdict"] is None and good["S_verdict"] is None
    assert good["R2_recomputed"] is False and good["denominator_shrunk"] is False
    assert good["R2_registered"] == rec["preconditions"]["R1_rule_R2"]["verdict"]

    old = p45_verdict(rec, preds, _v248_extraction(arm, resB, d["covers"]))
    assert old["verdict"] == rec["preconditions"]["R1_rule_R2"]["verdict"], \
        "the superseded path was expected to return the registered verdict; that IS the defect"
    assert old["verdict"] != "REFUSED"


def test_the_refusal_authority_is_the_observation_not_the_mapping():
    """A terminal observation with an EMPTY pair mapping must still REFUSE.

    The mapping from a terminal chain to its registered pairs is code, and code is where the v2.4.8 defect
    lived, so the mapping may not be the thing the refusal is conditioned on: the observation is.
    """
    D = synth()
    d = design(D)
    rec = fake_rec(d)
    caps_cert = chain_envelope_capacity(D, d["cert"], d["stage"], d["covers"], "realized", "cert")
    r_cert = {k: certification_boundary(900, 0.05, d["branch"]["delta_m"]) for k in RISKS}
    preds = evaluate_predictions(rec, d["covers"], _obs_for(d), r_cert, caps_cert)

    v = p45_verdict(rec, preds, set(), True)
    assert v["verdict"] == "REFUSED", "a terminal observation with no mapped pair did not refuse"
    assert v["terminal_pairs"] == [] and v["chain_terminal_observed"] is True
    assert v["T_verdict"] is None and v["S_verdict"] is None
    assert v["R2_recomputed"] is False and v["denominator_shrunk"] is False

    clean = p45_verdict(rec, preds, set(), False)
    assert clean["verdict"] == rec["preconditions"]["R1_rule_R2"]["verdict"]
    assert clean["chain_terminal_observed"] is False


def test_reference_and_cert_boundaries_do_not_mix():
    D = synth()
    d = design(D)
    rec = fake_rec(d)
    caps_cert = chain_envelope_capacity(D, d["cert"], d["stage"], d["covers"], "realized", "cert")
    obs = _obs_for(d)
    alpha = 0.05

    lo = {k: certification_boundary(300, alpha, d["branch"]["delta_m"]) for k in RISKS}
    hi = {k: certification_boundary(6000, alpha, d["branch"]["delta_m"]) for k in RISKS}
    assert lo["fire"] != hi["fire"], "the two certify counts gave the same boundary; pick others"

    A = evaluate_predictions(rec, d["covers"], obs, lo, caps_cert)
    B = evaluate_predictions(rec, d["covers"], obs, hi, caps_cert)

    # h*_ref is the REGISTERED margin, not a margin recomputed from n_cert
    for res in (A, B):
        assert res["h_star_ref_registered"] == rec["preconditions"]["R1_statuses"]["h_star_ref"]
        for k in RISKS:
            assert abs(res["h_star_ref_registered"][k]
                       - max(0.0, alpha - float(rec["preconditions"]["R1_family"]["r_star_ref"][k]))) < 1e-15
            assert abs(res["h_star_ref_registered"][k] - max(0.0, alpha - float(lo[k]))) > 1e-12 \
                or abs(float(lo[k]) - float(rec["preconditions"]["R1_family"]["r_star_ref"][k])) < 1e-12

    fields = ("gamma_sel_registered", "h_star_ref_registered", "capable_registered",
              "knife_edge_registered", "no_reach_registered", "sp_cert", "T", "S", "N")
    for key in A["pairs"]:
        for f in fields:
            assert A["pairs"][key][f] == B["pairs"][key][f], f"{f} moved with r*_cert: {key}"
        assert A["pairs"][key]["r_star_cert"] != B["pairs"][key]["r_star_cert"]
    moved = [k for k in A["pairs"] if A["pairs"][k]["X"] != B["pairs"][k]["X"]]
    assert moved, "r*_cert moved but X did not respond anywhere; X is not reading it"


# ------------------------------------------------------------------ 8

SENTINEL = "DO_NOT_READ_CERTIFY"


def _write_fixture(td, n_units=40, poison_certify=True):
    """A dataset + protocol on disk whose CERTIFY-side numeric cells are unparseable.

    Every numeric cell a loader must parse - the manifest's n_smoke / n_fire, the detector's s, s_fire,
    s_smoke, and the agent's g - carries the literal string DO_NOT_READ_CERTIFY on certify rows. Any code
    path that reads a certify row before R1 raises ValueError on the spot. Nothing can "read it and ignore
    it": the read itself is what fails.
    """
    import hashlib
    ds, proto = Path(td) / "ds", Path(td) / "P"
    ds.mkdir(parents=True)
    proto.mkdir()
    units = [f"g{i}" for i in range(n_units)]
    cert_u = certify_mask(np.asarray(sorted(units)), SF, SEED)
    certify_units = {u for u, c in zip(sorted(units), cert_u) if c}
    rows = []
    for i, u in enumerate(units):
        for f in range(3):
            rows.append({"uid": f"{u}f{f}", "unit": u, "event": f"{u}e0" if f < 2 else f"{u}n{f}",
                         "certify": u in certify_units})
    def cell(r, val):
        return SENTINEL if (poison_certify and r["certify"]) else val
    with open(ds / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "split", "source", "image", "n_smoke", "n_fire", "image_class"])
        for r in rows:
            w.writerow([r["uid"], "test", "src", f"img/{r['uid']}.jpg",
                        cell(r, 1), cell(r, 0), "fire"])
    with open(proto / "splits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "split", "role", "source", "unit", "event"])
        for r in rows:
            w.writerow([r["uid"], "test", ROLE, "src", r["unit"], r["event"]])
    (proto / "protocol.yaml").write_text("s_min: 0.02\nalpha: 0.05\n")
    det = proto / "det.csv"
    with open(det, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "s", "s_fire", "s_smoke", "boxes"])
        for r in rows:
            w.writerow([r["uid"], cell(r, 0.40), cell(r, 0.30), cell(r, 0.50),
                        "0:0.5:0.1:0.1:0.2:0.2"])
    sha = hashlib.sha256(det.read_bytes()).hexdigest()
    agents = {}
    for name in ("crop", "overlay"):
        ag = proto / f"agent_{name}.csv"
        with open(ag, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["uid", "g"])
            for r in rows:
                w.writerow([r["uid"], cell(r, 0.6)])
        ag.with_suffix(".meta.json").write_text(json.dumps(
            {"dry": False, "s_min": 0.02, "det_csv_sha256": sha, "view": name}))
        agents[name] = str(ag)
    return str(ds), str(proto), str(det), agents, certify_units


def test_certify_rows_are_not_read_before_R1():
    """The strong invariant: certify labels, scores and features are NOT READ before R1.

    The weaker invariant - certify data cannot AFFECT the design - is test 1. It is satisfied by a loader
    that materialises the whole role and then applies a select mask, which is precisely the access-boundary
    defect this test exists to forbid. Here every certify numeric cell is an unparseable sentinel, so:

      * stage 1 (resolve_role_split) must succeed: splits.csv carries no numeric cell at all;
      * stage 2 for the SELECT uids must succeed and return only select units;
      * loading the CERTIFY uids must FAIL on the sentinel - which proves the poison is real, and therefore
        that the select path's success means those rows were never parsed.
    """
    with tempfile.TemporaryDirectory() as td:
        ds, proto, det, agents, certify_units = _write_fixture(td)

        rs = resolve_role_split(proto, SF, SEED, role=ROLE)          # metadata only
        assert set(rs.certify_units) == certify_units
        assert rs.select_units and rs.certify_units
        assert not (rs.select_units & rs.certify_units)
        assert not (rs.select_uids & rs.certify_uids)

        D, P = load_select(ds, proto, det, agents, rs)                # must not touch a certify row
        assert D["_side"] == "select"
        assert set(np.asarray(D["unit"]).tolist()) == set(rs.select_units)
        assert not (set(np.asarray(D["uid"]).tolist()) & set(rs.certify_uids))
        assert np.isfinite(np.asarray(D["s"], float)).all(), "a sentinel reached the select fold"
        assert float(P["s_min"]) == 0.02

        # the poison is real: reading the certify side raises, so the success above is meaningful
        try:
            load_certify(ds, proto, det, agents, rs)
            raise AssertionError("the certify sentinel was parsed without error; the fixture is not poisoned "
                                 "and this test proves nothing")
        except ValueError:
            pass

        # and with a clean fixture the certify loader works, so the failure above is the sentinel and not
        # a broken certify path
        with tempfile.TemporaryDirectory() as td2:
            ds2, proto2, det2, agents2, _ = _write_fixture(td2, poison_certify=False)
            rs2 = resolve_role_split(proto2, SF, SEED, role=ROLE)
            Dc, _ = load_certify(ds2, proto2, det2, agents2, rs2)
            assert Dc["_side"] == "certify"
            assert set(np.asarray(Dc["unit"]).tolist()) == set(rs2.certify_units)
            assert len(all_rows(Dc)) == len(np.asarray(Dc["uid"]))


def test_role_split_is_metadata_only_and_matches_the_frozen_draw():
    """Stage 1 reproduces `certify_mask`'s draw exactly, and needs nothing but splits.csv.

    The whole two-stage design rests on this: the select/certify partition must be the SAME partition the
    frozen per-frame `certify_mask` produces, or the design fold would not be the registered one.
    """
    with tempfile.TemporaryDirectory() as td:
        ds, proto, det, agents, _ = _write_fixture(td, poison_certify=False)
        rs = resolve_role_split(proto, SF, SEED, role=ROLE)
        with open(Path(proto) / "splits.csv", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["role"] == ROLE]
        per_frame_units = np.asarray([r["unit"] for r in rows])
        cert = certify_mask(per_frame_units, SF, SEED)
        assert set(per_frame_units[~cert].tolist()) == set(rs.select_units)
        assert set(per_frame_units[cert].tolist()) == set(rs.certify_units)
        # and stage 1 does not need the dataset or the score files at all
        del ds, det, agents
        assert resolve_role_split(proto, SF, SEED, role=ROLE).to_dict() == rs.to_dict()
        try:
            resolve_role_split(proto, SF, SEED, role="no_such_role")
            raise AssertionError("an absent role was not refused")
        except SystemExit as e:
            assert "nothing to register" in str(e) and ROLE in str(e), str(e)


# ------------------------------------------------------------------ 10

def _ondisk_fold(td, n_units=1400, seed=7):
    """Write `synth()` out as a real dataset + protocol + score files, so the loaders are exercised."""
    import hashlib
    D = synth(n_units=n_units, seed=seed)
    ds, proto = Path(td) / "ds", Path(td) / "P"
    ds.mkdir(parents=True)
    proto.mkdir()
    uids = [f"f{i}" for i in range(len(D["s"]))]
    F = D["features"]
    with open(ds / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "split", "source", "image", "n_smoke", "n_fire", "image_class"])
        for i, u in enumerate(uids):
            w.writerow([u, "test", "src", f"img/{u}.jpg", int(D["y_smoke"][i]), int(D["y_fire"][i]), "fire"])
    with open(proto / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "split", "role", "source", "unit", "event"])
        for i, u in enumerate(uids):
            w.writerow([u, "test", ROLE, "src", D["unit"][i], D["event"][i] or f"{D['unit'][i]}n{i}"])
    (proto / "protocol.yaml").write_text(
        "s_min: 0.02\nalpha: 0.05\nselect_fraction: 0.3\ncalibrate_seed: 1\n"
        "costs: {calls: 1.0, fa: 10.0, handoff: 20.0}\n"
        f"dataset: {ds}\nmanifest_sha256: x\ndetector: {{weights_sha256: \"{'ab' * 32}\"}}\n")
    det = proto / "det.csv"
    with open(det, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "s", "s_fire", "s_smoke", "boxes"])
        for i, u in enumerate(uids):
            a = float(F["smoke_area"][i])
            w.writerow([u, D["s"][i], F["s_fire"][i], F["s_smoke"][i], f"0:0.5:0:0:{a:.6f}:1.0"])
    sha = hashlib.sha256(det.read_bytes()).hexdigest()
    agents = {}
    for name in ("crop", "overlay"):
        ag = proto / f"agent_{name}.csv"
        with open(ag, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uid", "g"])
            for i, u in enumerate(uids):
                w.writerow([u, D[f"g_{name}"][i]])
        ag.with_suffix(".meta.json").write_text(json.dumps(
            {"dry": False, "s_min": 0.02, "det_csv_sha256": sha, "view": name}))
        agents[name] = str(ag)
    return str(ds), str(proto), str(det), agents


@contextlib.contextmanager
def _lock_gate_bypassed():
    """Run `verify` without the frozen-protocol gate, which has its own coverage and needs a real lock."""
    import cascade.ace3_register as M
    keep = M.p0_protocol_state
    M.p0_protocol_state = lambda protocol: {"ok": True, "problems": [],
                                            "gate": "bypassed by test_ace3 (no lock in a fixture)"}
    try:
        yield
    finally:
        M.p0_protocol_state = keep


def test_verify_runs_end_to_end_from_files():
    """`verify` against files on disk, through both loader stages. The test whose absence let the loader
    defect reach a real run: every earlier test built `D` in memory, so nothing exercised the path from
    splits.csv and the score CSVs to R1.
    """
    import cascade.ace3_register as M
    with tempfile.TemporaryDirectory() as td:
        ds, proto, det, agents = _ondisk_fold(td)
        rs = M.resolve_role_split(proto, SF, SEED, role=ROLE)
        Ds, P = M.load_select(ds, proto, det, agents, rs)
        assert Ds["_side"] == "select"
        assert set(np.asarray(Ds["unit"]).tolist()) == set(rs.select_units)
        with _lock_gate_bypassed():
            rep, ok = M.verify(proto, Ds, P, rs, log=lambda *_: None)
        # The runtime-tree item compares the installed files with MANIFEST.sha256, so it is False in a
        # working tree with uncommitted edits. That is the provenance guard doing its job, not a design
        # failure, so it is asserted separately: if `ok` is False, the tree must be the only reason.
        tree_ok = rep["R1_runtime_tree"]["ok"]
        bad = [f"{arm}/{r['item']}" for arm, v in rep["R1_checklist_design_proxy"].items()
               for r in v if not r["ok"]]
        assert not bad, f"checklist items open: {bad}"
        assert ok or not tree_ok, "verify() refused for a reason other than the runtime tree"
        assert rep["R0_design_reference"]["certify_counts_read"] is False
        assert rep["R0_design_reference"]["branch"] == "full"
        assert rep["R1_cover"]["M_total"] == 4
        assert not rep["R1_cover"]["arm_purity_violations"]
        assert rep["R1_declared_vs_realized"]["ok"]
        assert rep["P0_set_operations"]["disjoint"]
        assert rep["P0_role_split"]["n_select_units"] == rs.n_select_units
        assert set(rep["R1_statuses"]["pairs"]) and "h_star_ref" in rep["R1_statuses"]
        assert rep["R1_rule_R2"]["verdict"] in ("INFORMATIVE", "UNINFORMATIVE")
        # verify() refuses a fold that is not the select one
        Dc, _ = M.load_certify(ds, proto, det, agents, rs)
        with _lock_gate_bypassed():
            try:
                M.verify(proto, Dc, P, rs, log=lambda *_: None)
                raise AssertionError("verify accepted the certify fold")
            except SystemExit as e:
                assert "select fold only" in str(e), str(e)


# ------------------------------------------------------------------ 11, 12, 13: the two bars

def test_the_checklist_bar_is_the_selection_guarantee_not_alpha():
    """`preflight_checklist`'s select_bar must be r*_ref, and the capacity item must be vacuous.

    Two bars exist in this study and they are not interchangeable. The checklist asks whether the coordinate
    can span the margin the SELECTION RULE leaves; Study 3's rule guarantees the safe endpoint at r*_ref, so
    that margin is max(0, r*_ref - r*_ref) = 0. bar = alpha is the PREDICTION-reference bar of R_0 §5 and
    belongs to R1_statuses. Feeding alpha here makes a family-design guard forbid the incapable coordinate
    that R2 needs to be INFORMATIVE - a gate cannot be the object under test.
    """
    from .ace_ltt import certification_boundary, chain_capacity
    with tempfile.TemporaryDirectory() as td:
        ds, proto, det, agents = _ondisk_fold(td)
        import cascade.ace3_register as M
        rs = M.resolve_role_split(proto, SF, SEED, role=ROLE)
        Ds, P = M.load_select(ds, proto, det, agents, rs)
        with _lock_gate_bypassed():
            rep, ok = M.verify(proto, Ds, P, rs, log=lambda *_: None)
        b = rep["R1_checklist_bars"]
        assert b["checklist_select_bar"] == rep["R1_family"]["r_star_ref"], b
        assert b["checklist_counts"] == rep["R0_design_reference"]["n_ref"], b
        assert all(v == 0.0 for v in b["checklist_h_star"].values()), b
        assert b["prediction_reference_bar"] == {k: float(DECLARATION["alpha"]) for k in RISKS}
        # and the non-vacuous margin is still alpha - r*_ref in the statuses
        for k in RISKS:
            assert abs(rep["R1_statuses"]["h_star_ref"][k]
                       - (float(DECLARATION["alpha"]) - float(rep["R1_family"]["r_star_ref"][k]))) < 1e-15
        # the capacity item closed, and it closed because h* = 0
        for arm, rows in rep["R1_checklist_design_proxy"].items():
            row = [r for r in rows if r["item"] == "chain_margin_capacity"][0]
            assert row["ok"], (arm, row["detail"])
            assert "h*=0.0000" in row["detail"], (arm, row["detail"])
        # bar = alpha WOULD have failed on this fold if any arm is incapable there; assert the mechanism
        losses = declared_losses()
        caps = M.chain_envelope_capacity(Ds, all_rows(Ds), rep["R1_family"]["thresholds"],
                                        arm_covers(rep["R1_family"]["thresholds"],
                                                   rep["R0_design_reference"]["modes"], losses),
                                        "design", "t")
        dm = float(rep["R0_design_reference"]["delta_m_declared"])
        nref = rep["R0_design_reference"]["n_ref"]
        rref = {k: certification_boundary(int(nref[k]), 0.05, dm) for k in RISKS}
        at_ref = chain_capacity(arm_min_capacity(caps, "L"), nref, losses, dm, rref)
        assert all(v["status"] == "MARGIN_CAPABLE" for v in at_ref.values()), at_ref
        assert all(v["required_margin"] == 0.0 for v in at_ref.values()), at_ref
        at_alpha = chain_capacity(arm_min_capacity(caps, "L"), nref, losses, dm,
                                  {k: 0.05 for k in RISKS})
        assert all(v["required_margin"] > 0 for v in at_alpha.values()), at_alpha


def test_capacity_minimum_is_taken_within_one_arm():
    """A tiny Gamma in arm L must not fail arm B's checklist. Otherwise the arm-wise cover buys nothing."""
    from .ace_ltt import Capacity
    caps = {"B|crop": {k: Capacity(k, 0.40, "design", "b") for k in RISKS},
            "B|overlay": {k: Capacity(k, 0.30, "design", "b") for k in RISKS},
            "L|crop": {k: Capacity(k, 1e-09, "design", "l") for k in RISKS},
            "L|overlay": {k: Capacity(k, 1e-09, "design", "l") for k in RISKS}}
    b = arm_min_capacity(caps, "B")
    l = arm_min_capacity(caps, "L")
    for k in RISKS:
        assert b[k].value == 0.30, b[k].value          # min over ARM B only, not 1e-09
        assert l[k].value == 1e-09
        assert "arm B" in b[k].basis and "arm L" in l[k].basis
    try:
        arm_min_capacity({kk: v for kk, v in caps.items() if kk.startswith("B")}, "L")
        raise AssertionError("an empty arm was not refused")
    except SystemExit as e:
        assert "capacity minimum is undefined" in str(e)


def test_formal_c3_changes_nothing_but_its_own_verdict():
    """Varying n_cert may move PASS/REFUSE and the C3 checklist detail, and nothing else.

    Not the family, not the cover, not the order, not the shares, not the R1 statuses, not R2 - those are
    registered quantities of the design fold at the prediction bar. This is the executable form of
    "audit metadata may refuse, never choose".
    """
    D = synth()
    d = design(D)
    before = json.dumps(d["frozen"], sort_keys=True)
    need = d["branch"]["n_min"]
    seen = {}
    for n in (need, need + 1, need + 2000, 20000):   # 20000, not 1e6: certification_boundary is O(n)
        r = _c3(d, D, {k: n for k in RISKS})
        seen[n] = r["verdict"]
        assert r["branch"]["M"] == d["branch"]["M"], "formal C3 moved the branch"
        assert r["bars"]["checklist_select_bar_r_star_ref"] == {k: float(v) for k, v in
                                                               d["r_star_ref"].items()}, r["bars"]
        for k in RISKS:
            expect = max(0.0, float(d["r_star_ref"][k]) - float(r["bars"]["r_star_cert"][k]))
            assert abs(r["bars"]["h_star_C3"][k] - expect) < 1e-15, (k, r["bars"])
        assert json.dumps(d["frozen"], sort_keys=True) == before, "a design quantity moved"
    assert seen[20000] == "PASS", seen
    # a realized support far worse than the design reference count makes h*_C3 positive
    tiny = _c3(d, D, {k: need for k in RISKS})
    assert all(v > 0 for v in tiny["bars"]["h_star_C3"].values()), tiny["bars"]
    huge = _c3(d, D, {k: 20000 for k in RISKS})
    assert all(v == 0.0 for v in huge["bars"]["h_star_C3"].values()), huge["bars"]


def run_tests():
    test_design_stage_never_sees_the_audit()
    test_formal_c3_refuses_insufficient_actual_support()
    test_no_full_to_fallback_switch()
    test_arms_never_share_a_chain()
    test_declared_realized_mismatch_refuses()
    test_chain_terminal_refuses_the_whole_study()
    test_chain_terminal_extraction_sees_chains_the_engine_dropped()
    test_the_refusal_authority_is_the_observation_not_the_mapping()
    test_reference_and_cert_boundaries_do_not_mix()
    test_certify_rows_are_not_read_before_R1()
    test_role_split_is_metadata_only_and_matches_the_frozen_draw()
    test_verify_runs_end_to_end_from_files()
    test_the_checklist_bar_is_the_selection_guarantee_not_alpha()
    test_capacity_minimum_is_taken_within_one_arm()
    test_formal_c3_changes_nothing_but_its_own_verdict()
    print("ACE-LTT STUDY 3 TESTS OK")


if __name__ == "__main__":
    run_tests()
