"""Tests for ace_ltt.py - the specification's validation items plus the six pre-freeze checklist guards.

 1  chain certificate: the inclusion proved from the thresholds really holds pointwise (for BOTH decision
    sets), and the certify-fold p-values are non-decreasing AND non-degenerate along the chain;
 2  a deliberately broken monotonicity (reversed chain, incomparable pair, non-antitone loss) is REJECTED;
 3  the strict-front counterexample: Pi_strict = {} while Pi_ACE != {} on the same data;
 4  every NOT CERTIFIED path is reachable and names its clause;
 5  certify-fold information in the ordering raises: observations to the constructors, raw delta shares, and
    a delta plan written against a different chain cover;
 6  loss-by-loss monotonicity: a loss cannot claim antitonicity without a listed aggregation, a
    policy-independent weighting and its own written lemma;
 7  the chain cover is deterministic, permutation-invariant and minimal, and the delta plan is bound to it;
 8  the audit is policy-independent in COUNT and in MEMBERSHIP, family-wide;
 9  payload semantics (B0 grid, beta, unit, normalization) are frozen before the run;
10  the executable pre-freeze checklist itself;
11  the inference contract is EXECUTED, not merely declared: an unimplemented construction is refused and
    the p-value is looked up from the loss's declaration rather than hard-wired to Hoeffding-Bentkus;
12  the audited unit SET is re-verified at certify time - same count, different units is a failure - and an
    observation without its membership digest is refused;
13  a post-cover failure is terminal for the WHOLE registered chain, including members that would pass;
14  candidates that fail C before registration never enter the cover, so no registered chain is partially
    populated.
"""
from __future__ import annotations

import contextlib
from types import MappingProxyType

import numpy as np

from .ace_ltt import (CHECKLIST, COVER_ALGORITHM, EMPTY_DENOMINATOR, MONOTONE_AGGREGATIONS, NOT_CERTIFIED,
                      BOUNDARY_SPANNING, Capacity, CertifyDataLeak, MARGIN_CAPABLE, MARGIN_INCAPABLE,
                      boundary_spanning, certification_boundary, certification_slack,
                      certification_margin, chain_capacity, unreachable_from,
                      Contract, DeltaPlan, LossSpec, PayloadSemantics, Policy, allocate_delta, audit_drift,
                      build_chains, certify_chain_relation, certify_family, chain_cover_id, chain_signature,
                      choose_operating_point, decision_sets, fixed_sequence, hb_pvalue, joint_pvalue,
                      BUILTIN_INFERENCE, INFERENCE_REGISTRY, ObservedLoss, as_observation, inference_spec,
                      InferenceRegistration, callable_identity, declare_inference, inference_registry_id,
                      membership_digest,
                      miss_loss, n_min, payload_loss, preflight_checklist, select_front, structural_order,
                      verify_chain)

LEMMA = "ACE-LTT spec Sec. 4.3 Lemma, instance for event-miss losses (2026-09-23)"
MISS = (miss_loss("fire", 0.05, LEMMA), miss_loss("smoke", 0.05, LEMMA))
HANDOFF = (miss_loss("smoke", 0.05, LEMMA, catch_set="response"),)   # operator hand-off counts as a catch
FA = LossSpec("false_alarm", 0.05, False)                            # NOT antitone under alarm inclusion
BUDGET = payload_loss(0.05)
SEM = PayloadSemantics("payload-semantics-2026-09-23", "split_group (global dedup component)",
                       budgets=(16.0, 32.0), beta=0.05, b_max=64.0)
N_POS = 80
IREG = InferenceRegistration("ace-inference-tests-2026-09-25",
                             inference_registry_id([BUILTIN_INFERENCE]), (BUILTIN_INFERENCE,))
MEMBERSHIP = {"fire": "sha256:fire-units", "smoke": "sha256:smoke-units", "budget": "sha256:all-units",
              "false_alarm": "sha256:all-units"}


def pol(A, b=None, vS=0.0, acq="crop", t=(0.02, 0.90), B0=16.0, name=""):
    return Policy(acq, t[0], t[1], A / 2 if b is None else b, A, {"S": vS, "rest": 0.0}, B0, name)


# --------------------------------------------------------------------------- a deterministic cascade

def sample():
    """One positive frame per positive unit, with g on a regular grid, plus negatives for the FA loss."""
    g = (np.arange(N_POS) + 0.5) / N_POS
    st = np.where(np.arange(N_POS) % 4 == 0, "S", "rest")
    return {"unit": np.r_[np.arange(N_POS), N_POS + np.arange(40)],
            "y": np.r_[np.ones(N_POS, bool), np.zeros(40, bool)],
            "s": np.full(N_POS + 40, 0.5), "g": np.r_[g, np.linspace(0.0, 1.0, 40)],
            "stratum": np.r_[st, np.full(40, "rest")]}


def risk_of(p, which="alarm", D=None):
    D = D or sample()
    caught = decision_sets(p, D["s"], D["g"], D["stratum"])[which]
    return float((~caught[D["y"]]).mean()), int(D["y"].sum())


def fa_of(p, D=None):
    D = D or sample()
    return float(decision_sets(p, D["s"], D["g"], D["stratum"])["alarm"][~D["y"]].mean())


@contextlib.contextmanager
def _patched_registry(entries):
    """Add constructions to the module's PRIVATE registry for the duration of a test, then restore it.

    The public `INFERENCE_REGISTRY` is a read-only mapping; tests that need a second construction (there is
    only one in the release) reach past it deliberately, and always restore the table.
    """
    import cascade.ace_ltt as m
    before = dict(m._INFERENCE_REGISTRY)
    m._INFERENCE_REGISTRY.update({k: MappingProxyType(dict(v)) for k, v in entries.items()})
    try:
        yield
    finally:
        m._INFERENCE_REGISTRY.clear()
        m._INFERENCE_REGISTRY.update(before)


def obs(**kw):
    """Observations in the v2.4.1 shape: (risk, n_units, membership digest) per loss.

    The digest is the one the contract declares, so a well-behaved evaluation matches the audit; a test
    that wants a mismatch passes its own third field.
    """
    return {k: (float(r), int(n), MEMBERSHIP[k]) for k, (r, n) in kw.items()}


def contract_ok(**kw):
    base = {"measurement_coherent": True, "score_contract_ok": True,
            "positive_units": {"fire": N_POS, "smoke": N_POS}, "budget_units": 120,
            "audit_membership": MEMBERSHIP, "max_payload_bytes": 12.0, "b_max": 64.0}
    return lambda _p: Contract(**{**base, **kw})


# --------------------------------------------------------------------------- 1. chain certificate / Lemma

def test_chain_certificate():
    D = sample()
    chain = [pol(0.001, 0.0005, 0.30, name="p1"), pol(0.020, 0.0100, 0.20, name="p2"),
             pol(0.040, 0.0200, 0.10, name="p3"), pol(0.100, 0.0500, 0.00, name="p4")]
    ok, certs = verify_chain(chain, MISS)
    assert ok and len(certs) == 3
    assert all(c["antitone_under_alarm_inclusion"] for c in certs)
    assert all(c["losses"]["fire"]["lemma"] == LEMMA for c in certs)
    assert all("alarm_inclusion" in c and "response_inclusion" in c and "payload" in c for c in certs)

    # the certificate's claim, checked pointwise on data it never saw - for BOTH decision sets
    for which in ("alarm", "response"):
        masks = [decision_sets(p, D["s"], D["g"], D["stratum"])[which] for p in chain]
        assert all(np.all(a | ~b) for a, b in zip(masks[:-1], masks[1:])), f"{which} inclusion violated"

    rs = [risk_of(p, "alarm", D) for p in chain]
    assert [r for r, _ in rs] == [0.0, 1 / N_POS, 2 / N_POS, 8 / N_POS]
    assert all(rs[i][0] <= rs[i + 1][0] for i in range(3)), "miss risk not antitone along the chain"
    ps = [joint_pvalue(obs(fire=(r, n), smoke=(r, n)), MISS)[0] for r, n in rs]
    assert all(ps[i] <= ps[i + 1] + 1e-12 for i in range(3)), "p_joint not non-decreasing along the chain"
    assert len(set(ps)) == 4 and ps[0] < 0.10 < ps[-1], f"degenerate p-values prove nothing: {ps}"

    # the same, for a loss whose catch set is alarm u operator (the hand-off cascade): b carries it, not A
    rh = [risk_of(p, "response", D)[0] for p in chain]
    assert all(rh[i] <= rh[i + 1] for i in range(3)) and rh != [r for r, _ in rs]

    assert build_chains(chain, MISS) == (tuple(chain),)                 # one chain, safest element first
    assert build_chains(chain, HANDOFF) == (tuple(chain),)
    assert n_min(0.05, 0.10) == 45 and n_min(0.05, 0.05) == 59


def test_structural_order_is_only_bookkeeping():
    """Sec. 4.1 asks for alarm AND escalate inclusion; widening the band breaks the first one."""
    wide, narrow = pol(0.30, t=(0.00, 0.95), name="wide"), pol(0.30, t=(0.02, 0.50), name="narrow")
    assert structural_order(wide, narrow, "escalate") and not structural_order(wide, narrow, "alarm")
    assert not structural_order(wide, narrow)                           # both required -> False
    D = sample()
    s, g = np.full(len(D["s"]), 0.60), np.zeros(len(D["g"]))
    assert decision_sets(narrow, s, g, D["stratum"])["alarm"].all()
    assert not decision_sets(wide, s, g, D["stratum"])["alarm"].any()   # exactly the inclusion that fails
    assert structural_order(pol(0.10, 0.05, 0.2, t=(0.00, 0.90)), pol(0.30, 0.20, 0.0, t=(0.02, 0.90)))


# --------------------------------------------------------------------------- 2. broken monotonicity

def test_broken_monotonicity_rejected():
    hi, lo = pol(0.30, name="safe"), pol(0.70, name="loose")
    assert certify_chain_relation(hi, lo, MISS) is not None
    assert certify_chain_relation(lo, hi, MISS) is None                 # reversed: no certificate
    assert verify_chain([lo, hi], MISS)[0] is False                     # clause C6

    mixed = Policy("crop", 0.02, 0.90, 0.20, 0.20, {"S": 0.0, "rest": 0.0}, 16.0, "mixed")
    assert certify_chain_relation(hi, mixed, MISS) is None and certify_chain_relation(mixed, hi, MISS) is None
    assert len(build_chains([hi, lo, mixed], MISS)) == 2

    assert certify_chain_relation(hi, pol(0.70, acq="overlay"), MISS) is None
    assert certify_chain_relation(hi, pol(0.70, B0=32.0), MISS) is None

    assert certify_chain_relation(hi, lo, (*MISS, FA)) is None
    assert all(len(c) == 1 for c in build_chains([hi, lo], (*MISS, FA)))
    assert fa_of(hi) >= fa_of(lo)

    for bad in (dict(b=0.9, A=0.1), dict(veto={"S": 1.4}), dict(t_low=0.9, t_high=0.1)):
        try:
            Policy(**{"acq": "crop", "t_low": 0.02, "t_high": 0.90, "b": 0.0, "A": 1.0,
                      "veto": {"S": 0.0}, **bad})
            raise AssertionError("invalid policy accepted")
        except ValueError:
            pass


# --------------------------------------------------------------------------- 6. loss-by-loss monotonicity

def test_loss_by_loss_monotonicity_contract():
    """catch_set alone is not enough: the aggregation must have its own written lemma."""
    for bad in (dict(aggregation="unit_max_over_frames_weighted_by_response_quality"),
                dict(aggregation="unit_any_positive_frame", lemma=""),
                dict(weighting="tuned_on_select_fold"),
                dict(empty_denominator=""),                      # a partial function is not a loss
                dict(empty_denominator="decide_later"),
                dict(catch_set="whatever")):
        try:
            LossSpec(**{"name": "smoke", "alpha": 0.05, "antitone_under_alarm_inclusion": True,
                        "aggregation": "unit_any_positive_frame", "lemma": LEMMA,
                        "empty_denominator": "exclude_unit", **bad})
            raise AssertionError(f"loss accepted without a monotonicity contract: {bad}")
        except ValueError:
            pass
    # a loss penalising the CONTENT of a response is not antitone; it is allowed only as non-antitone
    wrong = LossSpec("wrong_class_response", 0.05, False, aggregation="unit_any_positive_frame")
    assert certify_chain_relation(pol(0.3), pol(0.7), (*MISS, wrong)) is None
    # every listed aggregation is usable, and each carries a one-line lemma statement
    for agg in MONOTONE_AGGREGATIONS:
        for empty in EMPTY_DENOMINATOR:                  # every aggregation is a TOTAL function, declared
            L = miss_loss("smoke", 0.05, LEMMA, aggregation=agg, empty_denominator=empty)
            c = certify_chain_relation(pol(0.3), pol(0.7), (L,))
            assert c is not None and c["losses"]["smoke"]["empty_denominator"] == empty
        assert MONOTONE_AGGREGATIONS[agg]
    for bad in (dict(kind="payload", antitone_under_alarm_inclusion=True), dict(kind="typo"), dict(alpha=0.0),
                dict(kind="payload", aggregation="unit_any_positive_frame")):
        try:
            LossSpec(**{"name": "x", "alpha": 0.05, "antitone_under_alarm_inclusion": False, **bad})
            raise AssertionError("invalid loss accepted")
        except ValueError:
            pass


# --------------------------------------------------------------------------- 7. chain cover

def test_chain_cover_is_deterministic_and_minimal():
    """M chains means delta/M per chain, so an inflated cover silently costs power (Dilworth)."""
    raw = [(0.68, 0.19, 0.31), (0.20, 0.08, 0.55), (0.06, 0.01, 0.40),
           (0.55, 0.14, 0.09), (0.16, 0.07, 0.41), (0.28, 0.03, 0.95)]
    ps = [pol(A, b, v, name=f"q{i}") for i, (A, b, v) in enumerate(raw)]
    chains = build_chains(ps, MISS)
    assert len(chains) == 4 and sum(len(c) for c in chains) == len(ps)   # greedy gives 5 here
    assert all(verify_chain(c, MISS)[0] for c in chains)
    assert len({p.name for c in chains for p in c}) == len(ps)           # a partition, not a cover with repeats
    names = [[p.name for p in c] for c in chains]
    for perm in ([ps[i] for i in (3, 0, 5, 1, 4, 2)], list(reversed(ps))):
        assert [[p.name for p in c] for c in build_chains(perm, MISS)] == names, "cover is input-order dependent"
    assert chain_cover_id(chains) == chain_cover_id(build_chains(list(reversed(ps)), MISS))
    assert chain_cover_id(chains) != chain_cover_id(build_chains(ps[:-1], MISS))
    # the ORDER inside a chain is part of the identity: the fixed sequence is order-sensitive
    long_chain = max(chains, key=len)
    assert len(long_chain) >= 2
    swapped = (long_chain[1], long_chain[0], *long_chain[2:])
    assert chain_cover_id([long_chain]) != chain_cover_id([swapped]), "cover id ignores intra-chain order"
    assert sorted(p.name for p in long_chain) == sorted(p.name for p in swapped)
    assert COVER_ALGORITHM.startswith("dilworth")


# --------------------------------------------------------------------------- 3. strict-front counterexample

def _counterexample_family():
    """Three tied zero-loss candidates of ONE chain; only the conservative ones survive the certify fold."""
    cons, mid, cheap = pol(0.30, name="cons"), pol(0.50, name="mid"), pol(0.70, name="cheap")
    select = [(cons, 0.0, 3.0), (mid, 0.0, 2.0), (cheap, 0.0, 1.0)]     # select fold (risk_hat, cost_hat)
    truth = {"cons": 0.0, "mid": 0.0, "cheap": 16 / N_POS}              # certify fold
    return select, (lambda p: obs(fire=(truth[p.name], N_POS), smoke=(truth[p.name], N_POS)))


def test_strict_front_counterexample():
    select, ev = _counterexample_family()
    strict, weak = select_front(select, strict=True), select_front(select, strict=False)
    assert [p.name for p in strict] == ["cheap"]                        # classical Pareto drops the ties
    assert [p.name for p in weak] == ["cons", "mid", "cheap"]           # ACE keeps them

    kw = dict(delta=0.10, contracts=contract_ok(), certify_eval=ev, inference_registration=IREG)
    ace, strictly = certify_family(weak, MISS, **kw), certify_family(strict, MISS, **kw)
    assert len(strictly.certified) == 0 and "NO_REJECTION" in strictly.reasons       # Pi_strict = {}
    assert [p.name for p in ace.certified] == ["cons", "mid"]                        # Pi_ACE != {}
    assert ace.ok and not strictly.ok and not ace.partial
    cost = lambda p: 1.0 - p.A                      # pre-registered utility: a looser alarm costs less
    best, info = choose_operating_point(ace, cost)  # cheapest INSIDE the certified set, not outside it
    assert best.name == "mid" and info["n_certified"] == 2
    assert choose_operating_point(strictly, cost)[0] is None
    shuffled = select_front([select[2], select[0], select[1]], strict=False)
    assert [p.name for p in shuffled] == [p.name for p in weak]         # order is inclusion, never cost


# --------------------------------------------------------------------------- 9. frozen payload semantics

def test_payload_semantics_are_frozen():
    losses = (*MISS, BUDGET)
    fam = [pol(0.30, name="a"), pol(0.60, name="b")]
    ev = lambda p: obs(fire=(0.0, N_POS), smoke=(0.0, N_POS),
                       budget=(0.0 if p.name == "a" else 0.5, 120))
    base = dict(delta=0.10, contracts=contract_ok(), certify_eval=ev, inference_registration=IREG)

    r = certify_family(fam, losses, payload_semantics=SEM, **base)
    assert [p.name for p in r.certified] == ["a"] and r.pvalues["b"]["per_loss"]["budget"] == 1.0
    assert r.to_dict()["payload_semantics"]["normalization"] == "KiB_per_evaluated_frame"

    nosem = certify_family(fam, losses, **base)                          # a payload risk without semantics
    assert not nosem.ok and "C7_FROZEN_PAYLOAD_SEMANTICS" in nosem.reasons

    off = certify_family([pol(0.30, B0=19.5, name="a")], losses, payload_semantics=SEM, **base)
    assert not off.ok and "C7_FROZEN_PAYLOAD_SEMANTICS" in off.reasons   # B0 fitted after the fact
    mism = certify_family(fam, (*MISS, payload_loss(0.20)), payload_semantics=SEM, **base)
    assert not mism.ok and "C7_FROZEN_PAYLOAD_SEMANTICS" in mism.reasons  # beta != registered beta

    thin = certify_family(fam, losses, payload_semantics=SEM, delta=0.10, certify_eval=ev,
                          inference_registration=IREG,
                          contracts=contract_ok(budget_units=20))
    assert not thin.ok and "C3_CERTIFIABILITY_AUDIT" in thin.reasons      # R_B is audited, not exempt
    for bad in (dict(normalization="bit_per_second"), dict(budgets=()), dict(registration=" "),
                dict(unit_definition=""), dict(beta=1.0)):
        try:
            PayloadSemantics(**{"registration": "r", "unit_definition": "u", "budgets": (16.0,), **bad})
            raise AssertionError("invalid payload semantics accepted")
        except ValueError:
            pass


# --------------------------------------------------------------------------- 4. every NOT CERTIFIED path

def test_not_certified_paths():
    ok_ev = lambda _p: obs(fire=(0.0, N_POS), smoke=(0.0, N_POS))
    base = dict(delta=0.10, certify_eval=ok_ev, inference_registration=IREG)
    fam = [pol(0.30, name="a")]

    assert certify_family([], MISS, contracts=contract_ok(), **base).reasons == ("EMPTY_FAMILY",)

    cases = {"C1_MEASUREMENT_COHERENCE": contract_ok(measurement_coherent=False),
             "C2_SCORE_CONTRACT": contract_ok(score_contract_ok=False),
             "C3_CERTIFIABILITY_AUDIT": contract_ok(positive_units={"fire": N_POS, "smoke": 17}),
             "C4_PAYLOAD_CAP": contract_ok(max_payload_bytes=96.0),
             "C5_SPLIT_INDEPENDENCE": contract_ok(split_independent=False)}
    for code, c in cases.items():
        r = certify_family(fam, MISS, contracts=c, **base)
        assert not r.ok and code in r.reasons and code in NOT_CERTIFIED, code
        assert not r.certified and "NO_REJECTION" in r.reasons and not r.partial

    # C3 is evaluated at the share the candidate is actually tested at, not at delta
    tight = certify_family([pol(0.3, name="a"), pol(0.5, acq="overlay", name="b")], MISS,
                           contracts=contract_ok(positive_units={"fire": N_POS, "smoke": 46}), **base)
    assert "C3_CERTIFIABILITY_AUDIT" in tight.reasons and n_min(0.05, 0.05) > 46 >= n_min(0.05, 0.10)

    # an undeclared audit membership is a failure, not a default
    r = certify_family(fam, MISS, contracts=contract_ok(audit_membership={"fire": "x"}), **base)
    assert "C3_CERTIFIABILITY_AUDIT" in r.reasons and not r.ok

    assert verify_chain([pol(0.70), pol(0.30)], MISS)[0] is False        # C6, hand-built mis-ordered chain

    bad = certify_family(fam, MISS, contracts=contract_ok(), delta=0.10, inference_registration=IREG,
                         certify_eval=lambda _p: obs(fire=(0.30, N_POS), smoke=(0.0, N_POS)))
    assert bad.reasons == ("NO_REJECTION",) and bad.pvalues["a"]["p_joint"] == 1.0

    assert joint_pvalue(obs(fire=(0.0, N_POS)), MISS)[0] == 1.0
    assert hb_pvalue(0.0, 0, 0.05) == 1.0 and fixed_sequence([0.2, 0.01], 0.10) == 0
    for r_, n_ in ((-0.02, 60), (1.5, 60), (float("nan"), 60), (0.0, -1)):
        try:
            hb_pvalue(r_, n_, 0.05)
            raise AssertionError(f"out-of-range ({r_}, {n_}) accepted")
        except ValueError:
            pass

    part = certify_family([pol(0.30, name="a"), pol(0.60, acq="overlay", name="b")], MISS, delta=0.10,
                          inference_registration=IREG,
                          certify_eval=ok_ev,
                          contracts=lambda p: Contract(p.name == "a", True, {"fire": N_POS, "smoke": N_POS},
                                                       120, MEMBERSHIP, 12.0, 64.0))
    assert part.ok and part.partial and "C1_MEASUREMENT_COHERENCE" in part.reasons
    assert part.to_dict()["partial"] is True


# --------------------------------------------------------------------------- 8. policy-independent audit

def test_audit_membership_is_policy_independent():
    ok_ev = lambda _p: obs(fire=(0.0, N_POS), smoke=(0.0, N_POS))
    fam = [pol(0.30, name="a"), pol(0.60, name="b")]

    def drift_count(p):
        return Contract(True, True, {"fire": N_POS, "smoke": N_POS - (p.name == "b")}, 120, MEMBERSHIP,
                        12.0, 64.0)

    def drift_member(p):                                # same COUNT, different units: the subtler failure
        m = dict(MEMBERSHIP, smoke="sha256:units-this-policy-escalated" if p.name == "b" else MEMBERSHIP["smoke"])
        return Contract(True, True, {"fire": N_POS, "smoke": N_POS}, 120, m, 12.0, 64.0)

    for c, word in ((drift_count, "COUNT"), (drift_member, "MEMBERSHIP")):
        r = certify_family(fam, MISS, delta=0.10, contracts=c, certify_eval=ok_ev,
                           inference_registration=IREG)
        assert "C5_SPLIT_INDEPENDENCE" in r.reasons and not r.ok
        assert any(word in d for v in r.inadmissible.values() for d in v.detail)
    assert audit_drift(fam, {p.name: contract_ok()(p) for p in fam}, MISS) == []

    # and the evaluated unit count must be the audited one
    r = certify_family(fam[:1], MISS, delta=0.10, contracts=contract_ok(), inference_registration=IREG,
                       certify_eval=lambda _p: obs(fire=(0.0, N_POS), smoke=(0.0, 12)))
    assert "C3_CERTIFIABILITY_AUDIT" in r.reasons and not r.ok


# --------------------------------------------------------------------------- 5. certify data in the ordering

def test_no_certify_data_in_ordering():
    ps = [pol(0.30, name="a"), pol(0.60, name="b")]
    y = np.array([True, False, True])
    for call in (lambda: build_chains(ps, MISS, y=y),
                 lambda: certify_chain_relation(ps[0], ps[1], MISS, g=y),
                 lambda: verify_chain(ps, MISS, observations=y),
                 lambda: allocate_delta(build_chains(ps, MISS), 0.10, None, certify_risk=y),
                 lambda: select_front([(ps[0], 0.0, 1.0)], fold="certify")):
        try:
            call()
            raise AssertionError("certify data accepted by an ordering constructor")
        except CertifyDataLeak:
            pass

    two = build_chains([pol(0.30, name="a"), pol(0.60, acq="overlay", name="b")], MISS)
    assert len(two) == 2 and allocate_delta(two, 0.10) == (0.05, 0.05)
    try:
        allocate_delta(two, 0.10, [0.9, 0.1])
        raise AssertionError("raw post-hoc shares accepted")
    except CertifyDataLeak:
        pass
    shares = {chain_signature(two[0]): 3.0, chain_signature(two[1]): 1.0}
    plan = DeltaPlan("reg-2026-09-23T00:00:00Z", shares, chain_cover_id(two))
    got = allocate_delta(two, 0.10, plan)
    assert np.allclose(got, (0.075, 0.025)) and sum(got) <= 0.10 + 1e-12
    # a plan written against another cover cannot be re-pointed at this one
    try:
        allocate_delta(two, 0.10, DeltaPlan("reg", shares, chain_cover_id(two[:1])))
        raise AssertionError("plan from a different cover accepted")
    except CertifyDataLeak:
        pass
    try:
        allocate_delta(two, 0.10, DeltaPlan("reg", {chain_signature(two[0]): 1.0}, chain_cover_id(two)))
        raise AssertionError("chain with no registered share accepted")
    except KeyError:
        pass
    for bad in (dict(registration="  "), dict(cover_id="")):
        try:
            DeltaPlan(**{"registration": "r", "shares": shares, "cover_id": "c", **bad})
            raise AssertionError("unregistered plan accepted")
        except ValueError:
            pass

    runs = [certify_family(ps, MISS, delta=0.10, contracts=contract_ok(), inference_registration=IREG,
                           certify_eval=lambda _p, r=r: obs(fire=(r, N_POS), smoke=(r, N_POS)))
            for r in (0.0, 0.5)]
    assert [[p.name for p in c] for c in runs[0].chains] == [[p.name for p in c] for c in runs[1].chains]
    assert runs[0].deltas == runs[1].deltas == (0.10,) and runs[0].cover_id == runs[1].cover_id
    assert len(runs[0].certified) == 2 and len(runs[1].certified) == 0


# --------------------------------------------------------------------------- 10. the checklist itself

def CHECKLIST_FAMILY():
    """The declared example family used by `python -m cascade.ace_ltt checklist` (no data of any kind)."""
    fam = [pol(0.30, name="cons"), pol(0.50, name="mid"), pol(0.70, name="cheap"),
           pol(0.40, acq="overlay", B0=32.0, name="ov")]
    losses = (*MISS, BUDGET)
    chains = build_chains(fam, losses)
    plan = DeltaPlan("ace-delta-plan-2026-09-23", {chain_signature(c): 1.0 for c in chains},
                     chain_cover_id(chains))
    cap = {k: Capacity(k, 0.05, "design", "select fold") for k in ("fire", "smoke")}
    slk = {"crop": [0.004, 0.001, -0.002], "overlay": [0.003, -0.001]}   # boundary spanning on select
    ireg = declare_inference("ace-inference-2026-09-25", losses)
    return fam, losses, contract_ok(), plan, SEM, cap, slk, ireg


def test_preflight_checklist():
    fam, losses, con, plan, sem, cap, slk, ireg = CHECKLIST_FAMILY()
    kw = dict(contracts=con, plan=plan, payload_semantics=sem, capacities=cap, joint_slacks=slk,
              weights_sha256="a" * 64, inference_registration=ireg)
    rows = preflight_checklist(fam, losses, **kw)
    assert [r["item"] for r in rows] == list(CHECKLIST) and len(rows) == 9
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]

    # each item must actually be able to fail
    assert not preflight_checklist(fam, losses, **{**kw, "weights_sha256": ""})[5]["ok"]
    assert not preflight_checklist(fam, losses, **{**kw, "payload_semantics": None})[4]["ok"]
    assert not preflight_checklist(fam, losses, **{**kw, "contracts": None})[3]["ok"]
    assert not preflight_checklist(fam, (*MISS, FA, BUDGET), **kw)[0]["ok"]
    other = DeltaPlan("r", dict(plan.shares), chain_cover_id(build_chains(fam[:2], losses)))
    assert not preflight_checklist(fam, losses, **{**kw, "plan": other})[2]["ok"]
    assert not preflight_checklist(fam, losses, **{**kw, "capacities": None})[6]["ok"]
    assert not preflight_checklist(fam, losses, **{**kw, "joint_slacks": None})[7]["ok"]
    assert not preflight_checklist(fam, losses, **{**kw, "inference_registration": None})[8]["ok"]
    stale = InferenceRegistration(ireg.registration, "b" * 64, ireg.inferences)
    assert not preflight_checklist(fam, losses, **{**kw, "inference_registration": stale})[8]["ok"]
    allpos = {"crop": [0.004, 0.001], "overlay": [0.003]}
    assert not preflight_checklist(fam, losses, **{**kw, "joint_slacks": allpos})[7]["ok"]


def test_chain_margin_capacity():
    """Study 1's failure mode as an executable FAMILY-DESIGN condition (never a validity condition)."""
    assert abs(certification_boundary(1138, 0.05, 0.05) - 0.0360) < 5e-4
    assert abs(certification_margin(1138, 0.05, 0.05) - 0.0140) < 5e-4
    assert certification_boundary(0, 0.05, 0.05) == 0.0
    assert certification_boundary(10, 0.05, 0.05) == 0.0          # below n_min: nothing is certifiable
    b = [certification_boundary(n, 0.05, 0.05) for n in (100, 500, 1000, 5000, 50000)]
    assert b == sorted(b) and b[-1] < 0.05                        # rises towards alpha, never reaches it

    counts = {"fire": 1138, "smoke": 1079}
    lam = {"fire": Capacity("fire", 4 / 1138, "design", "select"),                        # the lambda chain
           "smoke": Capacity("smoke", 12 / 1079, "design", "select")}
    cap = chain_capacity(lam, counts, MISS, 0.05)
    assert all(v["status"] == MARGIN_INCAPABLE for v in cap.values())
    assert all(0.013 < v["required_margin"] < 0.015 for v in cap.values())
    cap2 = chain_capacity({k: Capacity(k, 0.05, "design", "select") for k in ("fire", "smoke")},
                          counts, MISS, 0.05)                                             # a b-threshold chain
    assert all(v["status"] == MARGIN_CAPABLE and v["usable_for_registration"] for v in cap2.values())

    # provenance: a capacity realized on the certify fold is diagnostic, never usable for registration
    real = chain_capacity({k: Capacity(k, 0.05, "realized", "certify fold") for k in ("fire", "smoke")},
                          counts, MISS, 0.05)
    assert all(v["status"] == MARGIN_CAPABLE and not v["usable_for_registration"] for v in real.values())
    assert all(not v["usable_for_registration"] for v in
               chain_capacity({"fire": 0.05, "smoke": 0.05}, counts, MISS, 0.05).values())
    for bad in (dict(source="guessed"), dict(basis=" "), dict(value=-0.1)):
        try:
            Capacity(**{"loss": "fire", "value": 0.05, "source": "design", "basis": "select", **bad})
            raise AssertionError("invalid capacity accepted")
        except ValueError:
            pass
    assert chain_capacity({}, counts, MISS, 0.05)["fire"]["status"] == "NOT_DECLARED"

    # the condition IS the anchored statement evaluated at the worst select-feasible anchor, R = alpha
    for gamma in (0.002, 0.010, 0.0139, 0.020):
        incap = gamma < certification_margin(1079, 0.05, 0.05)
        assert unreachable_from(0.05, gamma, 1079, 0.05, 0.05) == incap

    # and it does NOT say all-or-none: one capacity, two members straddling r*
    r = certification_boundary(1138, 0.05, 0.05)
    lo, hi = r - 0.002, r + 0.004                                  # 0.034 and 0.040, within Gamma = 0.010
    assert abs(hi - lo) <= 0.010
    assert hb_pvalue(lo, 1138, 0.05) <= 0.05 < hb_pvalue(hi, 1138, 0.05)


def test_boundary_spanning():
    """Sec. 4.6: the chain must see the joint certification boundary on the SELECT fold."""
    assert boundary_spanning([0.004, 0.001, -0.002, -0.01])[0] == BOUNDARY_SPANNING
    st, info = boundary_spanning([0.004, 0.003])                      # nothing to reject
    assert st != BOUNDARY_SPANNING and "nothing to reject" in info["reason"]
    st, info = boundary_spanning([-0.001, -0.01])                     # nothing certifiable
    assert st != BOUNDARY_SPANNING and "certifiable side" in info["reason"]
    assert boundary_spanning([])[0] != BOUNDARY_SPANNING
    assert boundary_spanning([0.0, -0.1])[0] == BOUNDARY_SPANNING     # slack exactly 0 counts as feasible

    # the joint slack is the binding loss, and it is computed from r*, not from alpha
    per, sj = certification_slack({"fire": 0.030, "smoke": 0.034}, {"fire": 1141, "smoke": 1043}, MISS, 0.05)
    assert abs(sj - min(per.values())) < 1e-12 and per["smoke"] < per["fire"]
    assert all(abs(v) < 0.05 for v in per.values())

    # Gamma < g* does NOT imply the chain cannot straddle r* - the counterexample, as a test
    from .ace_ltt import certification_boundary as cb
    r = cb(1141, 0.05, 0.05)
    lo, hi = r - 0.002, r + 0.004                                     # span 0.006, inside a Gamma of 0.007
    assert boundary_spanning([r - lo, r - hi])[0] == BOUNDARY_SPANNING


# ------------------------------------------------------------- 11. v2.4.1: the inference contract EXECUTES

def test_inference_contract_is_executed():
    """Regression for the v2.4 bypass: a named construction was accepted and then tested with the built-in
    Hoeffding-Bentkus p-value anyway (measured p_joint = 0.2969... for `weighted_eb_bound/v1`)."""
    # (1) a non-uniform weighting cannot keep the built-in pair: the estimand is a different functional
    try:
        miss_loss("fire", 0.05, LEMMA, weighting="declared_policy_independent")
        raise AssertionError("weighted estimand accepted with the unweighted HB p-value")
    except ValueError as e:
        assert "UNWEIGHTED" in str(e)

    # (2) and it cannot escape by NAMING a construction this release does not implement
    for bad in ("weighted_eb_bound/v1", "", "   ", "hb_unweighted_unit_mean/v2"):
        try:
            LossSpec("fire", 0.05, True, aggregation="unit_any_positive_frame", lemma=LEMMA,
                     empty_denominator="exclude_unit", weighting="declared_policy_independent",
                     inference=bad)
            raise AssertionError(f"unimplemented inference {bad!r} accepted")
        except ValueError:
            pass
    assert LossSpec("fire", 0.05, False, inference=BUILTIN_INFERENCE).inference == BUILTIN_INFERENCE
    try:
        inference_spec("weighted_eb_bound/v1")
        raise AssertionError("inference_spec fell back to the built-in construction")
    except ValueError as e:
        assert "not implemented by this release" in str(e)

    # (3) the p-value really is looked up, not hard-wired: a sentinel construction must be the one used.
    #     The public table is read-only (test 15), so this reaches past it into the module's private dict -
    #     which is the point: an attacker who does the same is caught by the REGISTRATION check, not by the
    #     dispatch.
    sentinel = "unit-test/constant-0.5"
    with _patched_registry({sentinel: {"pvalue": lambda o, L: 0.5, "estimand": "unit test",
                                       "statistic": ("risk", "n", "membership"),
                                       "weightings": ("uniform",), "chain_compatible": True,
                                       "version": "v0", "reference": "test only"}}):
        L = LossSpec("fire", 0.05, False, inference=sentinel)
        pj, parts = joint_pvalue(obs(fire=(0.0, N_POS)), (L,))
        assert pj == 0.5 and parts["fire"] == 0.5, (pj, parts)
        assert hb_pvalue(0.0, N_POS, 0.05) < 0.5                      # i.e. the built-in would differ
    assert sentinel not in INFERENCE_REGISTRY


# ------------------------------------------------------- 12. v2.4.1: the audit population is re-verified

def test_certify_membership_identity():
    fam = [pol(0.30, name="a")]
    base = dict(delta=0.10, contracts=contract_ok(), inference_registration=IREG)

    # same COUNT, different units: invisible to the v2.4 check, terminal from v2.4.1
    r = certify_family(fam, MISS, certify_eval=lambda _p: {
        "fire": (0.0, N_POS, "sha256:some-other-80-units"), "smoke": (0.0, N_POS, MEMBERSHIP["smoke"])},
        **base)
    assert not r.certified and "C3_CERTIFIABILITY_AUDIT" in r.reasons and "CHAIN_TERMINAL" in r.reasons
    assert any("membership" in d for v in r.inadmissible.values() for d in v.detail)

    # an observation without its membership digest is refused outright, not padded with a blank
    for shape in ((0.0, N_POS), (0.0, N_POS, ""), (0.0, N_POS, "   ")):
        try:
            certify_family(fam, MISS, certify_eval=lambda _p, x=shape: {"fire": x, "smoke": x}, **base)
            raise AssertionError(f"observation {shape!r} accepted without a membership digest")
        except ValueError:
            pass

    assert ObservedLoss(0.0, N_POS, "d").to_dict()["membership_sha256"] == "d"
    assert as_observation("fire", ObservedLoss(0.1, 5, "d")).risk == 0.1
    assert membership_digest(["u2", "u1", "u1"]) == membership_digest(["u1", "u2"])   # sorted, de-duplicated


# --------------------------------------- 13. v2.4.1: a post-cover failure is terminal for the WHOLE chain

def test_post_cover_failure_is_chain_terminal():
    """The registered chain is [a, b]; a would certify on its own. Under v2.3 semantics b's audit failure
    dropped only b and a was still certified - which is not an up-set of the REGISTERED chain."""
    fam = [pol(0.30, name="a"), pol(0.60, name="b")]
    assert [[q.name for q in c] for c in build_chains(fam, MISS)] == [["a", "b"]]

    good = certify_family(fam, MISS, delta=0.10, contracts=contract_ok(), inference_registration=IREG,
                          certify_eval=lambda _p: obs(fire=(0.0, N_POS), smoke=(0.0, N_POS)))
    assert [p.name for p in good.certified] == ["a", "b"]              # the control

    def ev(p):
        if p.name == "a":
            return obs(fire=(0.0, N_POS), smoke=(0.0, N_POS))
        return {"fire": (0.0, N_POS, "sha256:units-b-evaluated"), "smoke": (0.0, N_POS, MEMBERSHIP["smoke"])}

    r = certify_family(fam, MISS, delta=0.10, contracts=contract_ok(), certify_eval=ev,
                       inference_registration=IREG)
    assert not r.certified and not r.ok, [p.name for p in r.certified]
    assert {"C3_CERTIFIABILITY_AUDIT", "CHAIN_TERMINAL", "NO_REJECTION"} <= set(r.reasons)
    assert [[q.name for q in c] for c in r.chains] == [["a", "b"]]     # the chain stays in the record
    assert "a" in r.pvalues and r.pvalues["a"]["p_joint"] < 0.10       # a WOULD have passed; it is not kept


# --------------------------------- 14. v2.4.1: the registered cover is built over the SURVIVORS only

def test_precover_inadmissibility_leaves_the_cover():
    """The safe member is inadmissible BEFORE registration, so it never enters a chain; the aggressive one
    is certified as a singleton chain of its own. This is the fix's point: no registered chain is ever
    partially populated."""
    fam = [pol(0.30, name="safe"), pol(0.60, name="aggr")]
    assert [[q.name for q in c] for c in build_chains(fam, MISS)] == [["safe", "aggr"]]

    contracts = lambda p: Contract(p.name != "safe", True, {"fire": N_POS, "smoke": N_POS}, 120,
                                   MEMBERSHIP, 12.0, 64.0)
    r = certify_family(fam, MISS, delta=0.10, contracts=contracts, inference_registration=IREG,
                       certify_eval=lambda _p: obs(fire=(0.0, N_POS), smoke=(0.0, N_POS)))
    assert [p.name for p in r.certified] == ["aggr"] and r.partial
    assert [[q.name for q in c] for c in r.chains] == [["aggr"]]       # survivors only, not ["safe","aggr"]
    assert r.deltas == (0.10,) and r.cover_id == chain_cover_id(r.chains)
    assert "C1_MEASUREMENT_COHERENCE" in r.reasons and "CHAIN_TERMINAL" not in r.reasons
    assert "safe" in r.inadmissible and "safe" not in r.pvalues        # dropped before any p-value existed


# ------------------------------------- 15. v2.4.2: the registry is frozen and pinned by CODE identity

def test_inference_registry_is_frozen():
    import cascade.ace_ltt as m

    # the public table cannot be extended, removed from, or re-pointed
    def _set_entry():
        INFERENCE_REGISTRY["x"] = {}

    def _del_entry():
        del INFERENCE_REGISTRY[BUILTIN_INFERENCE]

    def _repoint():
        INFERENCE_REGISTRY[BUILTIN_INFERENCE]["pvalue"] = lambda o, L: 0.0

    for op in (_set_entry, _del_entry, _repoint):
        try:
            op()
            raise AssertionError("the inference registry is mutable through its public name")
        except (TypeError, AttributeError):
            pass
    assert len(INFERENCE_REGISTRY) == 1                      # this release implements exactly one pair

    # the v2.4.1 attack: substitute the callable but keep __module__ / __qualname__. The NAME-based digest
    # was identical; the code-identity digest must not be.
    before = inference_registry_id([BUILTIN_INFERENCE])
    original = INFERENCE_REGISTRY[BUILTIN_INFERENCE]["pvalue"]

    def forged(obs_, loss):
        return 0.0

    forged.__module__, forged.__qualname__ = original.__module__, original.__qualname__
    assert (f"{forged.__module__}.{forged.__qualname__}"
            == f"{original.__module__}.{original.__qualname__}")     # indistinguishable by name
    with _patched_registry({BUILTIN_INFERENCE: {**dict(INFERENCE_REGISTRY[BUILTIN_INFERENCE]),
                                                "pvalue": forged}}):
        after = inference_registry_id([BUILTIN_INFERENCE])
        assert after != before, "a substituted callable kept the same registry id"
        assert joint_pvalue(obs(fire=(0.02, N_POS)), MISS[:1])[0] == 0.0    # the substitution IS live
        # and with the registration in hand, nothing is certified while it is live
        r = certify_family([pol(0.30, name="a")], MISS, delta=0.10, contracts=contract_ok(),
                           certify_eval=lambda _p: obs(fire=(0.0, N_POS), smoke=(0.0, N_POS)),
                           inference_registration=IREG)
        assert not r.certified and "I_INFERENCE_REGISTRATION" in r.reasons
        assert any("registry_id changed" in d for v in r.inadmissible.values() for d in v.detail)
    assert inference_registry_id([BUILTIN_INFERENCE]) == before       # restored

    # what the digest is made of, and what it refuses to pin
    ident = callable_identity(original)
    assert ident["basis"] == "source" and len(ident["sha256"]) == 64
    try:
        callable_identity(min)                                        # no readable source
        raise AssertionError("an unpinnable callable was accepted")
    except ValueError as e:
        assert "no readable source" in str(e)
    assert declare_inference("r", MISS).registry_id == before
    _ = m                                                             # (module handle kept for clarity)


# ------------------------------- 16. v2.4.2: (O) is ENFORCED, not just recorded in the registry entry

def test_chain_compatibility_is_enforced():
    """A construction that is valid but NOT chain-compatible may not sit in a multi-member chain: the
    fixed-sequence stop relies on p_joint being non-decreasing along the chain, which is a property of the
    p-value construction, not of the policy order."""
    name = "unit-test/valid-but-not-chain-compatible"
    entry = {"pvalue": lambda o, L: hb_pvalue(o.risk, o.n, L.alpha), "estimand": "unit test",
             "statistic": ("risk", "n", "membership"), "weightings": ("uniform",),
             "chain_compatible": False, "version": "v0", "reference": "test only"}
    fam = [pol(0.30, name="a"), pol(0.60, name="b")]
    assert [[q.name for q in c] for c in build_chains(fam, MISS)] == [["a", "b"]]   # the control

    with _patched_registry({name: entry}):
        losses = tuple(miss_loss(L.name, L.alpha, LEMMA) for L in MISS)
        losses = tuple(LossSpec(L.name, L.alpha, True, aggregation=L.aggregation, lemma=LEMMA,
                                empty_denominator=L.empty_denominator, inference=name) for L in losses)
        assert certify_chain_relation(fam[0], fam[1], losses) is None    # no certificate at all
        assert [[q.name for q in c] for c in build_chains(fam, losses)] == [["a"], ["b"]]
        assert verify_chain(fam, losses)[0] is False                     # hand-built chain refused (C6)

        ireg = declare_inference("chain-incompatible-test", losses)
        r = certify_family(fam, losses, delta=0.10, contracts=contract_ok(),
                           certify_eval=lambda _p: obs(fire=(0.0, N_POS), smoke=(0.0, N_POS)),
                           inference_registration=ireg)
        # two singleton chains at delta/2 each, no chain longer than one member
        assert all(len(c) == 1 for c in r.chains) and r.deltas == (0.05, 0.05)


# ----------------------------------- 17. v2.4.2: the registration is required and verified before use

def test_inference_registration_is_required():
    fam = [pol(0.30, name="a")]
    ev = lambda _p: obs(fire=(0.0, N_POS), smoke=(0.0, N_POS))
    base = dict(delta=0.10, contracts=contract_ok(), certify_eval=ev)

    none = certify_family(fam, MISS, **base)                            # omitted: a refusal, not a default
    assert not none.certified and none.reasons == ("I_INFERENCE_REGISTRATION", "NO_REJECTION")
    assert "I_INFERENCE_REGISTRATION" in NOT_CERTIFIED

    ok = certify_family(fam, MISS, inference_registration=IREG, **base)
    assert ok.certified and ok.to_dict()["inference_registration"]["registry_id"] == IREG.registry_id
    assert ok.to_dict()["inference_registry_id"] == IREG.registry_id

    wrong = InferenceRegistration("r", "0" * 64, (BUILTIN_INFERENCE,))
    r = certify_family(fam, MISS, inference_registration=wrong, **base)
    assert not r.certified and "I_INFERENCE_REGISTRATION" in r.reasons

    with _patched_registry({"unit-test/second": {**dict(INFERENCE_REGISTRY[BUILTIN_INFERENCE])}}):
        extra = InferenceRegistration("r", inference_registry_id([BUILTIN_INFERENCE, "unit-test/second"]),
                                      (BUILTIN_INFERENCE, "unit-test/second"))
        r = certify_family(fam, MISS, inference_registration=extra, **base)   # registered but unused
        assert not r.certified and "I_INFERENCE_REGISTRATION" in r.reasons
        assert any("not used by any loss" in d for v in r.inadmissible.values() for d in v.detail)

    for bad in (dict(registration=" "), dict(registry_id="abc"), dict(inferences=())):
        try:
            InferenceRegistration(**{"registration": "r", "registry_id": "a" * 64,
                                     "inferences": (BUILTIN_INFERENCE,), **bad})
            raise AssertionError(f"malformed inference registration accepted: {bad}")
        except ValueError:
            pass


def run_tests():
    test_chain_certificate()
    test_structural_order_is_only_bookkeeping()
    test_broken_monotonicity_rejected()
    test_loss_by_loss_monotonicity_contract()
    test_chain_cover_is_deterministic_and_minimal()
    test_strict_front_counterexample()
    test_payload_semantics_are_frozen()
    test_not_certified_paths()
    test_audit_membership_is_policy_independent()
    test_no_certify_data_in_ordering()
    test_preflight_checklist()
    test_chain_margin_capacity()
    test_boundary_spanning()
    test_inference_contract_is_executed()
    test_certify_membership_identity()
    test_post_cover_failure_is_chain_terminal()
    test_precover_inadmissibility_leaves_the_cover()
    test_inference_registry_is_frozen()
    test_chain_compatibility_is_enforced()
    test_inference_registration_is_required()
    print("ACE-LTT TESTS OK")


if __name__ == "__main__":
    run_tests()
