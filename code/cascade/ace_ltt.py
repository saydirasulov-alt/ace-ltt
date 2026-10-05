"""ACE-LTT - Admissible Cascade Envelope via Learn-then-Test.

Executable copy of the frozen specification ACE_LTT_spec (Sec. references below). Any change to the semantics
requires a new spec version and a new registration; this module deliberately contains no tuning knobs.

What is certified is ONE policy object

    pi = (a, t_low, t_high, b, A, v, B0)

a          acquisition operator (which measurement is produced AND transmitted, encoder included),
t_low/high edge thresholds on the detector score s(x),
b, A       verifier thresholds on g_v = min(1, g + v(x)):  g_v < b dismiss, g_v >= A alarm, else operator,
v          veto authority: label-free map to [0, 1], piecewise constant on a DECLARED stratum partition
           (v == 0 full veto, v == 1 no veto),
B0         payload budget (part of the policy, not of the evaluation).

Two decision sets matter, and they are kept apart because different losses use different ones:

    alarm(pi)    = {s >= t_high} u {escalated and g_v >= A}                    governed by A
    response(pi) = {s >= t_high} u {escalated and g_v >= b}   (alarm u operator)   governed by b

A miss loss declares which set catches an event (`LossSpec.catch_set`): a cascade that counts a hand-off to
the human operator as a catch uses `response`, one that does not uses `alarm`. The chain certificate proves
inclusion for BOTH sets, so it covers either declaration.

Three stages, strictly in this order:

  1. Admissibility contract C - deterministic, fail-closed. Failure returns NOT CERTIFIED with the failing
     clause; never a weaker guarantee.
  2. Ordering - chains, chain order and the shares delta_m are functions of the policy algebra and of
     select-fold quantities ONLY; they touch no certify data, not even unlabelled inputs (P1). Observations
     passed to an ordering constructor raise CertifyDataLeak, and a non-uniform delta_m split is accepted
     only from a registered DeltaPlan keyed by data-free chain signatures.
  3. Testing - HB/binomial p-values on the certify fold, intersection-union over the constrained losses,
     fixed sequence inside each chain at its share delta_m.

The chain certificate (P2) is label-free: it is proved from the thresholds, for every possible input,

    forall x: alarm_p(x) >= alarm_q(x) and response_p(x) >= response_q(x)
        <== same a, same (t_low, t_high), same B0, b_p <= b_q, A_p <= A_q, v_p(.) >= v_q(.)

and it transfers to a constrained loss ONLY if that loss is antitone under alarm-set inclusion (miss losses
are; a false-alarm loss is not and therefore belongs in the operating-point utility, not in the constrained
family). Same a and same (t_low, t_high, B0) also make the transmitted bytes identical, hence Z_B equal.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from .risk import hb_pvalue as _hb_grid

# ---------------------------------------------------------------- reason codes (fail-closed vocabulary)

NOT_CERTIFIED = {
    "C1_MEASUREMENT_COHERENCE": "scores and payload bytes do not come from the same transmitted representation",
    "C2_SCORE_CONTRACT": "detector/agent contract broken (csv hash, s_min, model id/revision/quantization, dry)",
    "C3_CERTIFIABILITY_AUDIT": "a constrained risk has fewer certify units than n_min(alpha, delta_chain), "
                               "or the evaluated unit set disagrees with the audited one in count or "
                               "membership",
    "C4_PAYLOAD_CAP": "deterministic per-frame cap |r_a(x)| <= B_max violated by construction",
    "C5_SPLIT_INDEPENDENCE": "family, chains, order or delta shares depend on a candidate-dependent "
                             "certify-fold quantity; audit metadata may verify or refuse, never reshape",
    "C6_CHAIN_CERTIFICATE": "a chain is not totally ordered by >=_cert, or a constrained loss is not antitone",
    "CHAIN_TERMINAL": "a failure after the cover was fixed: the whole chain is dropped, its share is not "
                      "redistributed, and no member of it is tested or certified (v2.4)",
    "C7_FROZEN_PAYLOAD_SEMANTICS": "B0, beta, the unit definition or the C_U normalization were not frozen "
                                   "before the run, or a candidate uses a budget outside the registered grid",
    "I_INFERENCE_REGISTRATION": "the executable inference table does not match the registered one - or no "
                                "inference registration was supplied - so the p-value construction that "
                                "would be used is not the one the study registered (contract I)",
    "I_CHAIN_COMPATIBILITY": "a declared p-value construction is not chain-compatible (A2-O), so no "
                             "multi-member chain may be certified with it",
    "NO_REJECTION": "the fixed sequence stopped before rejecting anything; the certified set is empty",
    "EMPTY_FAMILY": "no admissible candidate",
}

SPEC_VERSION = "v2.4.3"  # v2.4.2 + an explicit provenance threat model; no new mechanism.
#   v2.4.1 dispatched on the declared construction but recorded only its qualified NAME, so a callable
#   swapped at run time under the same module.qualname produced an identical registry id and a different
#   p-value; and `chain_compatible` was stored and never enforced, so a chain-incompatible construction
#   could still land in a multi-member chain, which is exactly what A2-O forbids. v2.4.2 makes the registry
#   immutable, digests each construction's own SOURCE, binds the digest to the registration and verifies it
#   before any p-value, and refuses a >=_cert certificate when any declared construction is not
#   chain-compatible (so such a family has singleton chains and no fail-safe path).
#   v2.4.3 adds no mechanism: it states PROVENANCE_THREAT_MODEL, the boundary these checks do and do not
#   claim, and surfaces the contract-I refusal in the human-readable report of the generic instance.
#   v2.4 declared `inference` but never dispatched on it: a named construction was accepted and then tested
#   with the built-in Hoeffding-Bentkus p-value anyway, so the contract was a label, not a contract. v2.4.1
#   routes every p-value through INFERENCE_REGISTRY and refuses any construction this release does not
#   implement, and requires certify_eval to report WHICH units it evaluated, so the audited population is
#   re-verified on membership and not on the count alone.
#   v2.2 stays frozen for Study 1, v2.3 for Study 2; both remain verifiable against their own packages.
GREEDY_ABOVE = 400          # above this many candidates per group, fall back to the greedy chain cover


class CertifyDataLeak(ValueError):
    """Raised when certify-fold observations reach an ordering constructor (clause C5, patch P1)."""


def _assert_data_free(**observations):
    bad = sorted(k for k, v in observations.items() if v is not None)
    if bad:
        raise CertifyDataLeak(
            f"ordering must be a function of the policy algebra and select-fold quantities only; got {bad}")


# ---------------------------------------------------------------- policy object and loss declaration

@dataclass(frozen=True)
class Policy:
    """A deployable policy. `veto` is piecewise constant on `strata`, stored as ((name, lam), ...) sorted."""
    acq: str
    t_low: float
    t_high: float
    b: float
    A: float
    veto: tuple = ()
    B0: float = float("inf")
    name: str = ""

    def __post_init__(self):
        object.__setattr__(self, "veto", tuple(sorted((str(k), float(x)) for k, x in dict(self.veto).items())))
        if not (0.0 <= self.t_low <= self.t_high):
            raise ValueError("require 0 <= t_low <= t_high")
        if not (self.b <= self.A):
            raise ValueError("require b <= A: dismiss / operator / alarm must be an ordered partition")
        if not all(0.0 <= x <= 1.0 for _, x in self.veto):
            raise ValueError("veto authority must lie in [0, 1]")
        object.__setattr__(self, "name", self.name or self.auto_name())

    @property
    def strata(self):
        return tuple(k for k, _ in self.veto)

    def veto_at(self, stratum):
        d = dict(self.veto)
        if stratum not in d:
            raise KeyError(f"stratum {stratum!r} is not part of the declared partition {self.strata}")
        return d[stratum]

    def auto_name(self):
        lam = ",".join(f"{k}={x:g}" for k, x in self.veto) or "-"
        return f"{self.acq}|t[{self.t_low:g},{self.t_high:g}]|b{self.b:g}|A{self.A:g}|v[{lam}]|B{self.B0:g}"

    def to_dict(self):
        return {"acq": self.acq, "t_low": self.t_low, "t_high": self.t_high, "b": self.b, "A": self.A,
                "veto": {k: x for k, x in self.veto}, "B0": self.B0, "name": self.name}


CATCH_SETS = ("alarm", "response")

# ---- loss-by-loss monotonicity contract -------------------------------------------------------------
# A catch-set inclusion alone does NOT make a unit-level loss antitone: the aggregation from frame
# indicators to a unit number must itself be monotone, with non-negative weights that do not depend on the
# policy. Only the aggregations listed here have a written lemma; anything else must declare
# antitone_under_alarm_inclusion = False and is then tested alone, in a singleton chain.
#
# Each entry is of the form  l_u = f( (1{x in catch_set})_{x in u} )  with f non-increasing in every
# argument and the weights fixed before the run - which is exactly what the lemma needs.
MONOTONE_AGGREGATIONS = {
    "unit_any_positive_frame": "unit missed iff no positive frame of the unit is in the catch set",
    "unit_mean_over_positives": "mean over the unit's positives of 1{frame not in the catch set}",
    "unit_mean_over_events": "per event: missed iff no positive frame of the event is in the catch set; "
                             "then averaged over the unit's events with weights fixed before the run",
}
# Losses that are NOT covered, listed so the refusal is explicit rather than an omission: any loss that
# penalises the CONTENT of a response (wrong class, wrong box, operator overload, false alarm) is not
# monotone under catch-set inclusion - adding responses can make it worse.
NON_MONOTONE_EXAMPLES = ("false_alarm", "wrong_class_response", "operator_load", "precision")
PAYLOAD_AGGREGATION = "unit_mean_bytes_per_evaluated_frame_over_B0"
WEIGHTINGS = ("uniform", "declared_policy_independent")
# The built-in half of the inference contract I (Sec. 2): the estimand the engine computes and the p-value
# construction proved valid for it. Anything else must be named by the loss AND implemented here.
BUILTIN_INFERENCE = "hb_unweighted_unit_mean"


def _pvalue_hb_unweighted_unit_mean(obs, loss):
    """The built-in pair: estimand = unweighted mean of the unit losses over the audit population;
    p-value = the Hoeffding-Bentkus bound for H0: R > alpha at the audited unit count."""
    return hb_pvalue(obs.risk, obs.n, loss.alpha)


# Every construction this release can EXECUTE, with the estimand it tests, the statistic it consumes, the
# weightings it is proved for, and whether it is chain-compatible in the sense of A2-O. A construction that
# is not in this table is refused at declaration time: v2.4 accepted an arbitrary name and then tested with
# the built-in p-value regardless, which is a silent substitution of the estimand and a bypass of I.
#
# This release implements exactly ONE pair. The table is exposed as a READ-ONLY mapping (v2.4.2): a
# construction cannot be added, removed or re-pointed through the public name, and any entry that did
# appear would change `inference_registry_id`, which the registration pins.
_INFERENCE_REGISTRY = {
    BUILTIN_INFERENCE: MappingProxyType({
        "pvalue": _pvalue_hb_unweighted_unit_mean,
        "estimand": "E[ l_k(pi, U) | Z_k(U) = 1 ], the unweighted unit mean over the audit population",
        "statistic": ("risk", "n", "membership"),
        "weightings": ("uniform",),
        "chain_compatible": True,       # monotone in the empirical risk at fixed (n, alpha): Lemma 3 (O)
        "version": "v1",
        "reference": "Hoeffding-Bentkus, as in LTT; proof in the supplement (M4)",
    }),
}
INFERENCE_REGISTRY = MappingProxyType(_INFERENCE_REGISTRY)


def inference_spec(name):
    """The registry entry for a declared construction, or a typed refusal (never a silent fallback)."""
    spec = INFERENCE_REGISTRY.get(str(name))
    if spec is None:
        raise ValueError(
            f"inference {name!r}: custom inference is not implemented by this release. This engine can "
            f"execute only {tuple(INFERENCE_REGISTRY)}; a named-but-absent construction would be tested "
            f"with the built-in Hoeffding-Bentkus p-value for a DIFFERENT estimand, which is not valid. "
            f"Register the construction in INFERENCE_REGISTRY - a callable, the statistic it consumes and "
            f"a proof of (V) and (O) - before declaring it.")
    return spec


PROVENANCE_THREAT_MODEL = (
    "The provenance checks of this engine - MANIFEST.sha256 over the runtime tree, analysis_code_sha256 "
    "over ANALYSIS_CODE, the registered declaration, the audit membership digests and "
    "inference_registry_id over the executable inference table - detect changes to registered "
    "declarations, to package code and to the executable inference registration between registration and "
    "execution. They are NOT intended to defend against arbitrary mutation of the running Python process "
    "or against a malicious execution environment: code executing inside the interpreter can rebind a "
    "module global that a registered callable calls, or the verifier itself, and no digest computed by "
    "that same process can exclude this. The guarantee these checks support is reproducibility and the "
    "detection of accidental or package-level drift, not tamper-resistance.")

def callable_identity(f):
    """Code identity of a p-value callable: its own SOURCE, not its name.

    v2.4.1 hashed `module.qualname`, which is forgeable - a substituted callable carrying the original
    __module__ and __qualname__ produced an identical digest and a different p-value. What is hashed here is
    the text of the callable itself (version-independent, unlike bytecode, and available because the engine
    ships as source), plus its defaults and the number of closure cells it captures. Source is required: a
    callable whose source cannot be read cannot be pinned, and an unpinnable construction is refused rather
    than recorded with a weaker digest.

    SCOPE. What this pins is the registered SOURCE of the construction, not its transitive dependency
    graph: rebinding a module global that this source calls leaves the digest unchanged. That is inside
    PROVENANCE_THREAT_MODEL above and outside what any in-process digest can cover.
    """
    try:
        src = inspect.getsource(f)
    except (OSError, TypeError) as e:                              # C-level, exec'd or built-in callable
        raise ValueError(f"inference callable {f!r} has no readable source, so its code identity cannot be "
                         f"pinned to a registration ({e}). Ship the construction as source.") from e
    body = "\n".join(line.rstrip() for line in src.strip().splitlines())
    return {"basis": "source",
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "defaults": repr(getattr(f, "__defaults__", None)) + "|" + repr(getattr(f, "__kwdefaults__", None)),
            "closure_cells": len(getattr(f, "__closure__", ()) or ())}


def inference_registry_id(names=None):
    """Digest of the EXECUTABLE inference table: which constructions exist and what CODE implements them.

    A loss declares a construction by NAME. The name alone is a weak record, and so is a qualified name:
    the table could be extended or re-pointed between registration and run, and neither the certificate nor
    a name-based digest would show it. This digest covers, per construction, the estimand, the statistic it
    consumes, the weightings it is proved for, its chain compatibility, its version, its qualified name AND
    the code identity of the callable that computes it (`callable_identity`). Bound into the registration
    and re-verified before any p-value is computed (see InferenceRegistration), it makes the chain
    declared -> implemented -> registered -> unchanged at execution.
    """
    keys = sorted(INFERENCE_REGISTRY) if names is None else sorted({str(n) for n in names})
    rows = []
    for k in keys:
        sp = inference_spec(k)
        f = sp["pvalue"]
        ident = callable_identity(f)
        rows.append("|".join((k, str(sp["estimand"]), ",".join(sp["statistic"]), ",".join(sp["weightings"]),
                              str(bool(sp["chain_compatible"])), str(sp["version"]),
                              f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', repr(f))}",
                              ident["basis"], ident["sha256"], ident["defaults"],
                              str(ident["closure_cells"]))))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


@dataclass(frozen=True)
class InferenceRegistration:
    """The registered identity of the p-value machinery, checked before any p-value is computed (v2.4.2).

    `registry_id` is `inference_registry_id(inferences)` as recorded in the registration; `inferences` are
    the construction names the study declared. certify_family recomputes both from the code that is about
    to run and refuses on any difference, so a construction cannot be swapped, re-pointed or silently
    upgraded between registration and execution.
    """
    registration: str
    registry_id: str
    inferences: tuple = ()

    def __post_init__(self):
        if not str(self.registration).strip():
            raise ValueError("an inference registration must name the registration that declared it")
        if len(str(self.registry_id)) != 64:
            raise ValueError("registry_id must be the 64-hex inference_registry_id of the declared "
                             "constructions, recorded at registration time")
        object.__setattr__(self, "inferences", tuple(sorted({str(x) for x in self.inferences})))
        if not self.inferences:
            raise ValueError("an inference registration must list the constructions it covers")

    def problems(self, losses):
        """Every reason this registration does not describe the code that is about to run."""
        declared = sorted({L.inference for L in losses})
        bad = []
        undeclared = [x for x in declared if x not in self.inferences]
        if undeclared:
            bad.append(f"constructions used but not registered: {undeclared}")
        unused = [x for x in self.inferences if x not in declared]
        if unused:
            bad.append(f"constructions registered but not used by any loss: {unused}")
        try:
            now = inference_registry_id(declared)
        except ValueError as e:                                     # unpinnable callable
            bad.append(str(e))
            return bad
        if now != self.registry_id:
            bad.append(f"inference_registry_id changed since registration: registered "
                       f"{self.registry_id[:16]}... != executable {now[:16]}...")
        return bad

    def to_dict(self):
        return {"registration": self.registration, "registry_id": self.registry_id,
                "inferences": list(self.inferences)}


def declare_inference(registration, losses):
    """Helper for the REGISTRATION side: the InferenceRegistration for a declared loss family.

    Called once, when the family is frozen, and its output is what goes into the registration record. It is
    deliberately not called at execution time - that is the whole point of pinning it.
    """
    names = sorted({L.inference for L in losses})
    return InferenceRegistration(registration, inference_registry_id(names), tuple(names))

# An aggregation with a denominator is a partial function until the empty case is fixed: what happens to a
# unit with no positive frame (unit_mean_over_positives) or no event (unit_mean_over_events)? Choosing that
# after seeing the label pattern would move units in and out of the audit population post hoc, so the rule is
# part of the loss declaration and must be one of these three. All three are policy-independent - whether a
# unit has a positive does not depend on pi - so the monotonicity lemma survives each of them.
EMPTY_DENOMINATOR = {
    "exclude_unit": "the unit is not part of this risk's audit population (denominator = units with >= 1)",
    "loss_zero": "the unit stays in the population and contributes l_u = 0",
    "not_certified": "an empty denominator is a failure of the declaration (fail closed)",
}


@dataclass(frozen=True)
class LossSpec:
    """A constrained loss and its OWN monotonicity contract.

    antitone_under_alarm_inclusion   the chain certificate covers this loss only if True (P2). Claiming it
                                     requires a listed aggregation, a policy-independent weighting and the
                                     id of the written lemma for THIS loss - a generic flag is not enough.
    kind                             'miss' (label-dependent) or 'payload' (Z_B, label-free).
    catch_set                        for a miss loss: which decision set counts as catching the event,
                                     'alarm' or 'response' (= alarm u operator, i.e. a hand-off counts).
    aggregation                      how frame indicators become the unit loss (see MONOTONE_AGGREGATIONS).
    weighting                        'uniform', or weights declared before the run and independent of the
                                     policy's output.
    empty_denominator                what a unit with an empty denominator does (see EMPTY_DENOMINATOR);
                                     declared here so the loss is a TOTAL function before any label is seen.
    lemma                            registration id of the written monotonicity lemma for this loss.
    inference                        the SECOND half of the inference contract I: the p-value construction
                                     proved valid for THIS loss's estimand, and chain-compatible with the
                                     declared policy order. The built-in pair is
                                     BUILTIN_INFERENCE = unweighted unit mean tested by Hoeffding-Bentkus,
                                     and it is available ONLY for uniform unit weighting: a weighted
                                     estimand is a different functional and the HB p-value does not
                                     transfer to it. A non-uniform weighting must therefore NAME its own
                                     construction here, and a loss that does not is refused at declaration
                                     time rather than silently certified by the wrong p-value. (v2.4.)
    """
    name: str
    alpha: float
    antitone_under_alarm_inclusion: bool
    kind: str = "miss"
    catch_set: str = "alarm"
    aggregation: str = "unit_any_positive_frame"
    weighting: str = "uniform"
    empty_denominator: str = ""
    lemma: str = ""
    inference: str = BUILTIN_INFERENCE

    def __post_init__(self):
        if self.kind not in ("miss", "payload"):
            raise ValueError("kind must be 'miss' or 'payload'")
        if self.catch_set not in CATCH_SETS:
            raise ValueError(f"catch_set must be one of {CATCH_SETS}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS} (never a function of the policy output)")
        if not str(self.inference).strip():
            raise ValueError(f"loss {self.name!r} must name its p-value construction (inference contract I)")
        if self.inference == BUILTIN_INFERENCE and self.weighting != "uniform":
            raise ValueError(
                f"loss {self.name!r} declares weighting {self.weighting!r} but keeps the built-in inference "
                f"{BUILTIN_INFERENCE!r}. The built-in pair is the UNWEIGHTED unit mean tested by "
                f"Hoeffding-Bentkus; a weighted estimand is a different functional and that p-value does not "
                f"transfer to it. Name a construction proved valid for the weighted estimand, and "
                f"chain-compatible with >=_cert, in `inference`.")
        spec = inference_spec(self.inference)              # v2.4.1: refuses anything not EXECUTABLE here
        if self.weighting not in spec["weightings"]:
            raise ValueError(
                f"loss {self.name!r}: inference {self.inference!r} is proved for weightings "
                f"{spec['weightings']}, not for {self.weighting!r}")
        if self.kind == "payload":
            if self.antitone_under_alarm_inclusion:
                raise ValueError("a payload loss is constant along a chain, not antitone; declare it False")
            if self.aggregation != PAYLOAD_AGGREGATION:
                raise ValueError(f"a payload loss must use the frozen aggregation {PAYLOAD_AGGREGATION!r}")
        elif self.antitone_under_alarm_inclusion:
            if self.aggregation not in MONOTONE_AGGREGATIONS:
                raise ValueError(
                    f"aggregation {self.aggregation!r} has no monotonicity lemma; declare "
                    f"antitone_under_alarm_inclusion=False, or add the lemma to MONOTONE_AGGREGATIONS. "
                    f"Content-of-response losses such as {NON_MONOTONE_EXAMPLES} are never antitone.")
            if not str(self.lemma).strip():
                raise ValueError(f"loss {self.name!r} claims antitonicity without naming its written lemma")
            if self.empty_denominator not in EMPTY_DENOMINATOR:
                raise ValueError(
                    f"loss {self.name!r} must declare what an empty denominator does, one of "
                    f"{tuple(EMPTY_DENOMINATOR)}; it cannot be decided after the label pattern is seen")

    def to_dict(self):
        return {"name": self.name, "alpha": self.alpha, "kind": self.kind, "catch_set": self.catch_set,
                "aggregation": self.aggregation, "weighting": self.weighting, "lemma": self.lemma,
                "inference": self.inference,
                "empty_denominator": self.empty_denominator,
                "antitone_under_alarm_inclusion": self.antitone_under_alarm_inclusion}


def miss_loss(name, alpha, lemma, *, catch_set="alarm", aggregation="unit_any_positive_frame",
              weighting="uniform", empty_denominator="exclude_unit"):
    """A constrained miss loss that carries its own monotonicity lemma (the only way to claim antitonicity).

    `empty_denominator` defaults to 'exclude_unit' - the convention the cascade already uses, where a risk is
    averaged over the units that carry a positive - but it is written into the loss and into the chain
    certificate, so it is a declaration, never an implicit fallback.
    """
    return LossSpec(name, alpha, True, kind="miss", catch_set=catch_set, aggregation=aggregation,
                    weighting=weighting, empty_denominator=empty_denominator, lemma=lemma)


def payload_loss(beta=0.05, name="budget"):
    """The frozen budget loss Z_{B,U} = 1{C_U > B0}; identical along a chain (same a, band and B0)."""
    return LossSpec(name, beta, False, kind="payload", aggregation=PAYLOAD_AGGREGATION)


PAYLOAD_LOSS = payload_loss()


@dataclass(frozen=True)
class PayloadSemantics:
    """B0, beta, the unit definition and the C_U normalization, frozen BEFORE the run (clause C7).

    Fitting B0 to the observed payload distribution afterwards would invalidate p_B, so the budget grid is
    declared here and a candidate whose B0 is not on that grid is refused.
    """
    registration: str
    unit_definition: str
    budgets: tuple
    beta: float = 0.05
    normalization: str = "KiB_per_evaluated_frame"
    b_max: float = float("inf")

    def __post_init__(self):
        if not str(self.registration).strip() or not str(self.unit_definition).strip():
            raise ValueError("payload semantics must name its registration and its unit definition")
        if self.normalization != "KiB_per_evaluated_frame":
            raise ValueError("only the frozen normalization is available; the bit/s variant requires v3")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must lie in (0, 1)")
        b = tuple(sorted({float(x) for x in self.budgets}))
        if not b:
            raise ValueError("the budget grid must be declared before the run")
        object.__setattr__(self, "budgets", b)

    def problems(self, policy: Policy, losses):
        out = []
        pay = [L for L in losses if L.kind == "payload"]
        if pay and policy.B0 not in self.budgets:
            out.append(f"B0={policy.B0:g} is not on the registered grid {list(self.budgets)}")
        for L in pay:
            if float(L.alpha) != float(self.beta):
                out.append(f"{L.name}: alpha {L.alpha:g} != registered beta {self.beta:g}")
        return out

    def to_dict(self):
        return {"registration": self.registration, "unit_definition": self.unit_definition,
                "budgets": list(self.budgets), "beta": self.beta, "normalization": self.normalization,
                "b_max": self.b_max}


def n_min(alpha, delta):
    """Smallest number of units for which zero observed loss can certify risk <= alpha at level delta."""
    return int(math.ceil(math.log(delta) / math.log(1.0 - float(alpha))))


# ---------------------------------------------------------------- Sec. 2: admissibility contract C

@dataclass
class Contract:
    """Deterministic facts about a candidate, established before any p-value is computed.

    `positive_units[k]` is the number of certify units carrying a positive of type k, and `budget_units` the
    number of certify units the payload risk is averaged over.

    `audit_membership[k]` identifies WHICH units those are - a digest of the sorted certify unit ids that
    enter risk k (e.g. sha256 of the joined ids). Both the count and the membership must be functions of the
    split and the labels ONLY, never of the candidate's behaviour on the certify fold: otherwise the chain
    truncation they cause correlates with the p-values and the conditional argument of C3 fails. Counts and
    membership are therefore required to be IDENTICAL across the whole family, and a missing membership
    digest is a failure, not a default.
    """
    measurement_coherent: bool                      # C1: scores and bytes from ONE transmitted representation
    score_contract_ok: bool                         # C2: shared csv hash / s_min / model id / no dry scores
    positive_units: Mapping[str, int] = field(default_factory=dict)   # C3, miss losses
    budget_units: int = 0                                             # C3, payload loss
    audit_membership: Mapping[str, str] = field(default_factory=dict)  # C3/C5, per loss: which units
    max_payload_bytes: float = 0.0                  # C4
    b_max: float = float("inf")                     # C4
    split_independent: bool = True                  # C5 (self-attested; the executable part is below)
    note: str = ""

    def units_for(self, loss: LossSpec):
        return int(self.budget_units) if loss.kind == "payload" else int(self.positive_units.get(loss.name, 0))

    def membership_for(self, loss: LossSpec):
        return str(self.audit_membership.get(loss.name, ""))


@dataclass(frozen=True)
class AdmissibilityResult:
    admissible: bool
    reasons: tuple = ()
    detail: tuple = ()

    def to_dict(self):
        return {"admissible": self.admissible, "reasons": list(self.reasons), "detail": list(self.detail)}


def check_admissibility(policy: Policy, contract: Contract, losses: Sequence[LossSpec], delta_chain: float):
    """Clauses C1-C5 for one candidate, audited at the share delta_chain it is actually tested at.

    C6 is verified per chain (verify_chain); the chain-wise part of C5 is enforced in certify_family.
    """
    reasons, detail = [], []
    if not contract.measurement_coherent:
        reasons.append("C1_MEASUREMENT_COHERENCE")
    if not contract.score_contract_ok:
        reasons.append("C2_SCORE_CONTRACT")
    for L in losses:                                  # every constrained loss, payload included
        need = n_min(L.alpha, delta_chain)
        have = contract.units_for(L)
        if have < need:
            reasons.append("C3_CERTIFIABILITY_AUDIT")
            detail.append(f"{L.name}: {have} certify units < n_min({L.alpha:g}, {delta_chain:g})={need}")
        if not contract.membership_for(L):
            reasons.append("C3_CERTIFIABILITY_AUDIT")
            detail.append(f"{L.name}: the audit unit membership is not declared")
    if not (contract.max_payload_bytes <= contract.b_max):
        reasons.append("C4_PAYLOAD_CAP")
        detail.append(f"max payload {contract.max_payload_bytes:g} > B_max {contract.b_max:g}")
    if not contract.split_independent:
        reasons.append("C5_SPLIT_INDEPENDENCE")
    if contract.note:
        detail.append(contract.note)
    return AdmissibilityResult(not reasons, tuple(dict.fromkeys(reasons)), tuple(detail))


# ---------------------------------------------------------------- Sec. 4.1 / 4.3: the two orders

def decision_sets(p: Policy, s, g, stratum):
    """Reference implementation of the two decision sets; used to CHECK certificates, never to build them."""
    s = np.asarray(s, float)
    gv = np.minimum(1.0, np.asarray(g, float) + np.array([p.veto_at(k) for k in stratum]))
    esc = (s >= p.t_low) & (s < p.t_high)
    edge = s >= p.t_high
    return {"alarm": edge | (esc & (gv >= p.A)), "response": edge | (esc & (gv >= p.b))}


def structural_order(p: Policy, q: Policy, part="both"):
    """p >=_str q (bookkeeping ONLY, no statistical claim).

    Sec. 4.1 asks for BOTH alarm(p) >= alarm(q) and escalate(p) >= escalate(q). Raising t_high enlarges the
    uncertain band but SHRINKS the edge-alarm set, so the two inclusions can hold together only at equal
    t_high - which is exactly why >=_str says nothing about p_joint and is never used for testing.
    `part` selects one inclusion ('alarm', 'escalate') or requires both.
    """
    if p.acq != q.acq or p.strata != q.strata:
        return False
    veto = all(p.veto_at(k) >= q.veto_at(k) for k in p.strata)
    alarm = (p.t_high <= q.t_high) and veto and p.b <= q.b and p.A <= q.A
    escalate = (p.t_low <= q.t_low) and (p.t_high >= q.t_high)
    return {"alarm": bool(alarm), "escalate": bool(escalate), "both": bool(alarm and escalate)}[part]


def certify_chain_relation(p: Policy, q: Policy, losses: Sequence[LossSpec], **observations):
    """Label-free certificate that p >=_cert q, proved from the thresholds for EVERY possible input (P1/P2).

    Returns a dict certificate, or None if the premises do not hold. No data of any kind is read: passing
    observations raises CertifyDataLeak. A certificate is issued only when every constrained label-dependent
    loss is antitone under alarm-set inclusion (P2); a false-alarm-type loss is not, so a family containing
    one has singleton chains and no fail-safe path.
    """
    _assert_data_free(**observations)
    if p.acq != q.acq:
        return None                                  # different transmitted representation: bytes differ
    if (p.t_low, p.t_high) != (q.t_low, q.t_high) or p.B0 != q.B0:
        return None                                  # payload equality (hence Z_B equality) would fail
    if p.strata != q.strata:
        return None                                  # v is comparable only on a common declared partition
    if not (p.b <= q.b and p.A <= q.A and all(p.veto_at(k) >= q.veto_at(k) for k in p.strata)):
        return None
    bad = [L.name for L in losses if L.kind == "miss" and not L.antitone_under_alarm_inclusion]
    if bad:
        return None                                  # certificate does not cover these losses (P2)
    # (O) of contract I, enforced and not merely recorded (v2.4.2): the chain certificate is only useful
    # because p_joint is non-decreasing along the chain, and that is a property of the p-value CONSTRUCTION.
    # A construction declared chain-incompatible therefore gets no certificate at all, so such a family has
    # singleton chains - the same treatment as a loss without a monotonicity lemma.
    if any(not inference_spec(L.inference)["chain_compatible"] for L in losses):
        return None
    # loss-by-loss: the catch-set inclusion transfers only through an aggregation that has its own lemma
    if any(L.aggregation not in MONOTONE_AGGREGATIONS or not str(L.lemma).strip()
           or L.empty_denominator not in EMPTY_DENOMINATOR
           for L in losses if L.kind == "miss"):
        return None
    return {"ge": p.name, "le": q.name,
            "premises": [f"acq {p.acq!r} identical", f"(t_low, t_high) = ({p.t_low:g}, {p.t_high:g}) identical",
                         f"B0 = {p.B0:g} identical", f"b {p.b:g} <= {q.b:g}", f"A {p.A:g} <= {q.A:g}",
                         "veto " + ", ".join(f"{k}: {p.veto_at(k):g} >= {q.veto_at(k):g}" for k in p.strata)],
            "alarm_inclusion": "forall x: alarm_p(x) superset-of-or-equal alarm_q(x)     [from A_p <= A_q]",
            "response_inclusion": "forall x: response_p(x) superset-of-or-equal response_q(x)  [from b_p <= b_q]",
            "payload": "C_U and Z_{B,U} identical on every unit (same a, same band, same B0)",
            "losses": {L.name: {"catch_set": L.catch_set, "aggregation": L.aggregation,
                                "weighting": L.weighting, "empty_denominator": L.empty_denominator,
                                "lemma": L.lemma}
                       for L in losses if L.kind == "miss"},
            "inference": {L.name: L.inference for L in losses},
            "antitone_under_alarm_inclusion": True}


# ---------------------------------------------------------------- Sec. 4.4: chains and multiplicity

def _inclusion_key(p: Policy):
    """Most inclusive decision sets first: low A, low b, large veto authority."""
    return (p.A, p.b, tuple(-p.veto_at(k) for k in p.strata), p.name)


def _min_chain_cover(items, comparable):
    """Dilworth minimum chain cover of a finite poset (maximum bipartite matching, Kuhn).

    `items` are sorted so that i < j whenever items[i] >=_cert items[j]; `comparable(i, j)` is the relation.
    A minimum cover matters statistically: M chains means delta/M per chain, so an inflated M costs power.
    """
    n = len(items)
    adj = [[j for j in range(i + 1, n) if comparable(i, j)] for i in range(n)]
    succ, pred = [-1] * n, [-1] * n

    def augment(i, seen):
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if pred[j] == -1 or augment(pred[j], seen):
                pred[j], succ[i] = i, j
                return True
        return False

    for i in range(n):
        augment(i, [False] * n)
    chains = []
    for j in range(n):
        if pred[j] == -1:
            c = [j]
            while succ[c[-1]] != -1:
                c.append(succ[c[-1]])
            chains.append(tuple(items[k] for k in c))
    return chains


def _greedy_chain_cover(items, comparable):
    """Deterministic fallback for very large groups (not minimal, hence conservative: M is larger)."""
    rest, chains = list(range(len(items))), []
    while rest:
        chain, left = [rest[0]], []
        for j in rest[1:]:
            if comparable(chain[-1], j):
                chain.append(j)
            else:
                left.append(j)
        chains.append(tuple(items[k] for k in chain))
        rest = left
    return chains


def build_chains(policies: Sequence[Policy], losses: Sequence[LossSpec], **observations):
    """Partition candidates into chains totally ordered by >=_cert, largest (safest) element first.

    Incomparable candidates are never put in the same chain, and cost is never used as a fallback ordering
    inside a chain - that would destroy the fail-safe proposition. Data-free: observations raise.
    """
    _assert_data_free(**observations)
    losses = tuple(losses)
    groups = {}
    for p in policies:
        groups.setdefault((p.acq, p.t_low, p.t_high, p.B0, p.strata), []).append(p)
    out = []
    for key in sorted(groups, key=lambda k: (str(k[0]), k[1], k[2], k[3], k[4])):
        items = sorted(groups[key], key=_inclusion_key)
        comparable = lambda i, j: certify_chain_relation(items[i], items[j], losses) is not None  # noqa: E731
        cover = _greedy_chain_cover if len(items) > GREEDY_ABOVE else _min_chain_cover
        out.extend(sorted(cover(items, comparable), key=lambda c: _inclusion_key(c[0])))
    return tuple(out)


def chain_signature(chain: Sequence[Policy]):
    """Data-free identifier of a chain: its group key plus the name of its >=_cert-largest element."""
    p = chain[0]
    return (p.acq, p.t_low, p.t_high, p.B0, p.strata, p.name)


COVER_ALGORITHM = "dilworth_min_cover_kuhn/v1+greedy_above_400"


def chain_cover_id(chains):
    """Digest of the cover ITSELF: algorithm, version and the exact ORDERED chains it produced.

    The order inside a chain is hashed, not only its membership: the fixed sequence is order-sensitive, so
    [pi1, pi2, pi3] and [pi2, pi1, pi3] are different certification procedures even with the same members,
    and a DeltaPlan bound to one must not validate against the other.

    build_chains is a deterministic function of the candidate set alone (sorted by the inclusion key, name
    as final tie-break), so this id can be computed - and a DeltaPlan written against it - before any data
    is touched. allocate_delta refuses a plan whose id does not match, which removes the last freedom
    between 'find the cover' and 'split delta': the split is bound to one specific cover.
    """
    h = hashlib.sha256(COVER_ALGORITHM.encode())
    for c in chains:
        h.update(b"\0chain\0" + "\0".join(p.name for p in c).encode())
    return h.hexdigest()


def verify_chain(chain: Sequence[Policy], losses: Sequence[LossSpec], **observations):
    """Clause C6: every consecutive pair carries a certificate. Returns (ok, certificates)."""
    _assert_data_free(**observations)
    certs = []
    for a, b in zip(chain[:-1], chain[1:]):
        c = certify_chain_relation(a, b, losses)
        if c is None:
            return False, tuple(certs)
        certs.append(c)
    return True, tuple(certs)


@dataclass(frozen=True)
class DeltaPlan:
    """A PRE-REGISTERED non-uniform split of delta across chains (Sec. 4.4, 'uniform unless declared').

    Shares are keyed by chain_signature - a function of the policy algebra alone - and carry the id of the
    registration that declared them AND the id of the chain cover they were written against, so a split can
    be neither chosen after the p-values are seen nor re-pointed at a different cover. A chain with no
    declared share is a failure, not a default.
    """
    registration: str
    shares: tuple = ()
    cover_id: str = ""

    def __post_init__(self):
        if not str(self.registration).strip():
            raise ValueError("a non-uniform delta split must name the registration that declared it")
        if not str(self.cover_id).strip():
            raise ValueError("a non-uniform delta split must name the chain cover it was written against "
                             "(chain_cover_id of the candidate family)")
        object.__setattr__(self, "shares", tuple((tuple(k), float(w)) for k, w in dict(self.shares).items()))
        if any(w < 0 for _, w in self.shares) or not sum(w for _, w in self.shares) > 0:
            raise ValueError("shares must be non-negative with a positive sum")

    def weight(self, sig):
        d = dict(self.shares)
        if tuple(sig) not in d:
            raise KeyError(f"no registered share for chain {sig}")
        return d[tuple(sig)]


def allocate_delta(chains, delta, plan: DeltaPlan | None = None, **observations):
    """Pre-registered split of the error budget across chains (uniform Bonferroni unless a plan is given).

    Data-free: observations raise, and a non-uniform split is accepted only from a registered DeltaPlan.
    sum(delta_m) <= delta, so FWER over the whole family is <= delta.
    """
    _assert_data_free(**observations)
    M = len(chains)
    if M == 0:
        return ()
    if plan is None:
        return tuple(float(delta) / M for _ in range(M))
    if not isinstance(plan, DeltaPlan):
        raise CertifyDataLeak("a non-uniform delta split must be a registered DeltaPlan, not raw numbers")
    if plan.cover_id != chain_cover_id(chains):
        raise CertifyDataLeak("the registered delta split was written against a different chain cover "
                              f"({plan.cover_id[:12]}... != {chain_cover_id(chains)[:12]}...)")
    w = np.array([plan.weight(chain_signature(c)) for c in chains], float)
    return tuple(float(delta) * x for x in (w / w.sum()))


# ---------------------------------------------------------------- Sec. 3: p-values

def hb_pvalue(rhat, n, alpha):
    """Scalar Hoeffding-Bentkus p-value for H0: R > alpha (same implementation as cascade.risk).

    Fails CLOSED on out-of-range input: a loss outside [0, 1] or a negative count is an error, not a p of 0.
    """
    r, n = float(rhat), int(n)
    if not np.isfinite(r) or not (0.0 <= r <= 1.0):
        raise ValueError(f"empirical risk must be finite in [0, 1]; got {rhat!r}")
    if n < 0:
        raise ValueError("number of units must be non-negative")
    if n == 0:
        return 1.0
    return float(np.asarray(_hb_grid(r, n, float(alpha))).ravel()[0])


@dataclass(frozen=True)
class ObservedLoss:
    """What certify_eval reports for one constrained loss: the statistic AND the population it came from.

    `membership` is the digest of the units actually evaluated - `membership_digest(ids)` - and it is
    required, not optional. Up to v2.4 the engine compared only the COUNT against the audit record, so an
    evaluation on a different set of the same size passed unnoticed; the conditional argument of C3/Lemma 0
    is about the SET, so the set is what must be re-verified (v2.4.1).
    """
    risk: float
    n: int
    membership: str

    def __post_init__(self):
        if not np.isfinite(float(self.risk)):
            raise ValueError("an observed risk must be finite")
        if int(self.n) < 0:
            raise ValueError("an observed unit count must be non-negative")
        if not str(self.membership).strip():
            raise ValueError("an observation must name WHICH units it was evaluated on (membership digest)")

    def to_dict(self):
        return {"risk": float(self.risk), "n_units": int(self.n), "membership_sha256": str(self.membership)}


def membership_digest(unit_ids):
    """The canonical audit-membership digest: sha256 of the newline-joined sorted DISTINCT unit ids.

    One function for the registration side and the certify side, so the two digests are comparable by
    construction rather than by convention.
    """
    ids = sorted({str(u) for u in unit_ids})
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def as_observation(name, value):
    """Coerce what certify_eval returned into an ObservedLoss, refusing the pre-v2.4.1 shape.

    A bare (risk, n) pair is REFUSED rather than padded with a blank digest: accepting it would restore
    exactly the count-only check that v2.4.1 exists to remove.
    """
    if isinstance(value, ObservedLoss):
        return value
    if hasattr(value, "keys"):
        return ObservedLoss(float(value["risk"]), int(value.get("n", value.get("n_units"))),
                            str(value.get("membership", value.get("membership_sha256", ""))))
    seq = tuple(value)
    if len(seq) == 3:
        return ObservedLoss(float(seq[0]), int(seq[1]), str(seq[2]))
    raise ValueError(
        f"loss {name!r}: certify_eval must report (risk, n_units, membership_sha256) - or an ObservedLoss - "
        f"not {len(seq)} field(s). From v2.4.1 the evaluated unit SET is re-verified against the audited "
        f"one, so an observation without its membership digest cannot be checked and is refused.")


def joint_pvalue(observed, losses: Sequence[LossSpec]):
    """Intersection-union p-value p = max_k p_k over the constrained losses.

    `observed` maps loss name -> ObservedLoss (or (risk, n, membership_sha256)). Each p_k is computed by the
    construction the loss DECLARED, looked up in INFERENCE_REGISTRY - not by the built-in p-value regardless
    of the declaration, which is what v2.4 did. A missing loss is a failure of the audit, not a licence to
    drop the constraint, so it yields p = 1.
    """
    parts = {}
    for L in losses:
        raw = observed.get(L.name)
        if raw is None:
            parts[L.name] = 1.0
            continue
        obs = as_observation(L.name, raw)
        parts[L.name] = float(inference_spec(L.inference)["pvalue"](obs, L))
    return (max(parts.values()) if parts else 1.0), parts


MARGIN_CAPABLE, MARGIN_INCAPABLE, MARGIN_NOT_DECLARED = "MARGIN_CAPABLE", "MARGIN_INCAPABLE", "NOT_DECLARED"
BOUNDARY_SPANNING, NOT_SPANNING = "BOUNDARY_SPANNING", "NOT_SPANNING"
CAPACITY_SOURCES = ("design", "realized")


@dataclass(frozen=True)
class Capacity:
    """Gamma_k(S) with its PROVENANCE, because Gamma is label-dependent.

    source = 'design'   computed from select/design data, or from an independent prior audit. May inform
                        family registration.
    source = 'realized' computed on the certify or sealed test fold after it was opened. DIAGNOSTIC ONLY:
                        it never accepts a family, never changes a chain or an order. Admitting a realized
                        capacity into family design would let the family adapt to the test labels, which is
                        exactly what the pre-registration exists to prevent. (A future theory of
                        label-pattern-conditional validity could relax this; it does not exist yet.)
    """
    loss: str
    value: float
    source: str
    basis: str = ""

    def __post_init__(self):
        if self.source not in CAPACITY_SOURCES:
            raise ValueError(f"capacity source must be one of {CAPACITY_SOURCES}")
        if not (self.value >= 0.0):
            raise ValueError("a capacity is non-negative")
        if not str(self.basis).strip():
            raise ValueError("a capacity must name the data it was computed from")

    def to_dict(self):
        return {"loss": self.loss, "value": self.value, "source": self.source, "basis": self.basis}


def certification_boundary(n, alpha, delta, tol=1e-9):
    """r*(n, alpha, delta) = sup{ r <= alpha : p_HB(r; n, alpha) <= delta }.

    The largest empirical risk on n units that a valid HB p-value can still reject at level delta. It is a
    function of the PRE-REGISTERED (n, alpha, delta) only; no observed candidate risk enters, so a selection
    rule may use it without becoming post-hoc.
    """
    if n <= 0:
        return 0.0
    if hb_pvalue(0.0, n, alpha) > delta:
        return 0.0                                   # below n_min: nothing at this n is certifiable
    lo, hi = 0.0, float(alpha)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if hb_pvalue(mid, n, alpha) <= delta else (lo, mid)
    return lo


def certification_margin(n, alpha, delta):
    """g*(n, alpha, delta) = alpha - r*: the finite-sample gap between nominal feasibility and certifiability.

    Feasible-on-select is not certifiable-on-test: a rule bounded by `R_select <= alpha` can hand the testing
    stage a candidate that no amount of data at this n can certify.
    """
    return float(alpha) - certification_boundary(n, alpha, delta)


def chain_capacity(capacities, unit_counts, losses, delta_m, select_bar=None):
    """Per loss: the chain's LOSS CAPACITY against the certification margin it would have to span.

    `capacities[k]` is a Capacity carrying Gamma_k(S) and its provenance (a bare float is accepted but has
    no provenance and is never usable for registration). Gamma_k(S) is

        Gamma_k(S) = sum_u w_{k,u} * |E_{k,u} n S| / |E_{k,u}|

    computed with the same unit weights, event definition, aggregation and empty-denominator rule the
    LossSpec declares, where S is a declared INFLUENCE ENVELOPE for the chain's varying coordinate: any
    event set that CONTAINS every event whose catch status that coordinate can change. It bounds the
    chain's reach: |R_k(pi) - R_k(pi')| <= Gamma_k(S) for every two members.

    S need NOT be minimal. The minimal choice - the exact acting domain - gives the tightest bound; any
    superset is admissible and gives a looser one, and the bound direction is what the guarantee needs. A
    LOOSER envelope is not uniformly conservative at the study level, and the two directions must not be
    conflated: a larger Gamma_k(S) makes the reach bound EASIER to satisfy and makes MARGIN_CAPABLE easier
    to trigger, so it makes a sufficiency claim built on MARGIN_CAPABLE HARDER. Nothing here is a validity
    quantity either way (see the last paragraph).

    The margin a chain must supply depends on the SELECT BAR the rule actually used - the bound the
    selection stage guarantees for the candidates it hands over:

        h*_k(bar_k) = max(0, bar_k - r*_k)

    With bar = alpha this is the old g* = alpha - r*. With bar = r* (a rule that already selects inside the
    certification boundary) it is 0, and the margin condition is vacuous - which is correct, not a loophole:
    there is nothing to pull the anchor across. `select_bar` defaults to alpha_k.

    A chain is MARGIN_CAPABLE for k when Gamma_k(S) >= h*_k.

    What the two directions mean, exactly:
      Gamma_k < h*_k : the coordinate structurally cannot pull the WORST-CASE select-feasible anchor inside
                       the boundary. It does NOT say the chain cannot straddle r*: with r* = 0.036 and
                       Gamma = 0.007, members at 0.034 and 0.040 differ by 0.006 <= Gamma and still fall on
                       opposite sides.
      Gamma_k >= h*_k: "not ruled out". NOT informative, NOT boundary-crossing, NOT a guarantee that any
                       member certifies. Whether the chain actually sees the boundary is Sec. 4.6
                       (boundary spanning), and whether a member crosses it is settled on the certify fold.

    MARGIN_CAPABLE does NOT mean a candidate is guaranteed to cross the boundary. It means only that the
    coordinate carries, in principle, as much capacity as the finite-sample margin requires. Whether any
    member actually crosses r* stays an empirical question settled on the certify fold.

    This is a FAMILY-DESIGN condition, not a validity condition. Failing it invalidates nothing - not LTT,
    not the p-values, not the fixed sequence, not the ACE guarantee. It says only that the declared
    coordinate cannot span the full finite-sample gap between alpha_k and r*_k, so a select-feasible policy
    sitting near alpha_k is not guaranteed to be moved into the certifiable region by varying that
    coordinate alone. It does NOT imply that the chain certifies all of its members or none of them: with
    Gamma = 0.010 and r* = 0.036, members at 0.034 and 0.040 still fall on opposite sides.
    """
    out = {}
    for L in losses:
        if L.kind != "miss":
            continue
        n = int(unit_counts.get(L.name, 0))
        r = certification_boundary(n, L.alpha, delta_m)
        bar = float(L.alpha if select_bar is None else
                    (select_bar.get(L.name, L.alpha) if hasattr(select_bar, "get") else select_bar))
        need = max(0.0, bar - r)
        c = capacities.get(L.name)
        # duck-typed: `python -m cascade.ace_ltt` loads this module twice (__main__ and cascade.ace_ltt),
        # so isinstance() against Capacity is not reliable across that boundary.
        has = c is not None and hasattr(c, "value") and hasattr(c, "source")
        val = None if c is None else (float(c.value) if has else float(c))
        src = None if c is None else (str(c.source) if has else "unspecified")
        out[L.name] = {
            "n_units": n, "boundary": r, "select_bar": bar, "required_margin": need,
            "capacity": val, "source": src,
            "basis": (str(getattr(c, "basis", "")) if has else ""),
            "usable_for_registration": src == "design",
            "status": MARGIN_NOT_DECLARED if val is None else
            (MARGIN_CAPABLE if val >= need else MARGIN_INCAPABLE)}
    return out


def certification_slack(select_risks, unit_counts, losses, delta_m):
    """Per loss, the SELECT-fold certification slack s_k(pi) = r*_k - R_sel_k(pi), and the joint slack.

    Positive = this candidate sits on the certifiable side of the boundary as the select fold sees it;
    negative = it does not. Computed from select-fold risks only, so it may inform family construction.
    It predicts nothing about the certify fold: a positive select slack is not a certificate.
    """
    per = {}
    for L in losses:
        if L.kind != "miss":
            continue
        r = certification_boundary(int(unit_counts.get(L.name, 0)), L.alpha, delta_m)
        per[L.name] = float(r) - float(select_risks[L.name])
    return per, (min(per.values()) if per else float("-inf"))


def boundary_spanning(joint_slacks):
    """Sec. 4.6: does the chain's coordinate actually cross the joint certification boundary on the select
    fold? True iff at least one member is on the certifiable side and at least one is not:

        max_pi s_joint(pi) >= 0   and   min_pi s_joint(pi) < 0

    This is the informativeness criterion. A chain that fails it gives the fixed sequence nothing to select
    between - every member falls on the same side as the select fold sees it. It is NOT a capacity bound and
    NOT a prediction: on the certify fold the chain may still certify all of its members or none.
    """
    s = [float(x) for x in joint_slacks]
    if not s:
        return NOT_SPANNING, {"n": 0}
    hi, lo = max(s), min(s)
    ok = hi >= 0.0 > lo
    return (BOUNDARY_SPANNING if ok else NOT_SPANNING), {
        "n": len(s), "max_joint_slack": hi, "min_joint_slack": lo,
        "safe_endpoint_feasible": s[0] >= 0.0,
        "reason": "" if ok else ("no member on the certifiable side" if hi < 0 else
                                 "every member on the certifiable side: the sequence has nothing to reject")}


def unreachable_from(anchor_risk, capacity, n, alpha, delta):
    """Post-run diagnostic (uses an observed risk, so never a pre-registration check).

    If the anchor's empirical risk minus the chain's whole capacity still exceeds r*, then no member
    reachable from that anchor through the chain's coordinate can pass the HB threshold for this loss - the
    outcome was settled by the anchor, not by the ordering.
    """
    return float(anchor_risk) - float(capacity) > certification_boundary(n, alpha, delta)


def fixed_sequence(pvalues, delta_m):
    """Rejected prefix of one chain: stop at the first p > delta_m (Sec. 4.4)."""
    k = 0
    for p in pvalues:
        if p > delta_m:
            break
        k += 1
    return k


# ---------------------------------------------------------------- Sec. 4.4: select-fold front construction

def _dominates(z, w, strict):
    """strict=True: the classical Pareto rule (weakly better in both, strictly in one) - it removes ties.
    strict=False: ACE's rule - only a candidate that is strictly better in BOTH coordinates dominates."""
    dr, dc = float(z[1]) - float(w[1]), float(z[2]) - float(w[2])       # how much worse z is than w
    return (dr > 0 and dc > 0) if not strict else (dr >= 0 and dc >= 0 and (dr > 0 or dc > 0))


def select_front(items, strict=False, fold="select"):
    """Front on select-fold (risk_hat, cost_hat). ACE keeps ties; the strict variant is the failure mode.

    items: sequence of (policy, risk_hat, cost_hat) measured on the SELECT fold. The returned order is the
    canonical >=_cert-inclusion order and carries no cost information: the testing order comes from
    build_chains, never from cost (Sec. 4.4). The weak front itself is classical; what ACE claims is its
    interaction with the testing order - a strict front keeps only the cheapest member of a zero-loss tie
    group, and if that member fails on the certify fold the sequence stops with an empty certified set
    although a more conservative member of the same chain would have passed.
    """
    if fold != "select":
        raise CertifyDataLeak("the candidate family is built on the select fold only (clause C5)")
    items = list(items)
    keep = [z for z in items if not any(_dominates(z, w, strict) for w in items if w[0].name != z[0].name)]
    return tuple(z[0] for z in sorted(keep, key=lambda z: _inclusion_key(z[0])))


# ---------------------------------------------------------------- Sec. 5: certification

@dataclass
class Certification:
    certified: tuple = ()
    chains: tuple = ()
    deltas: tuple = ()
    pvalues: dict = field(default_factory=dict)
    certificates: tuple = ()
    inadmissible: dict = field(default_factory=dict)
    reasons: tuple = ()
    alpha: dict = field(default_factory=dict)
    delta: float = 0.10
    registration: str = "uniform"
    cover_id: str = ""
    losses: tuple = ()
    payload_semantics: dict = field(default_factory=dict)
    inference_registration: object = None

    @property
    def ok(self):
        return bool(self.certified)

    @property
    def partial(self):
        """True when the certified set is non-empty but some candidate was dropped by a clause of C."""
        return bool(self.certified and self.inadmissible)

    def guarantee(self):
        return ("P[ for all pi in certified, for all k: R_k(pi) <= alpha_k ] >= 1 - delta, conditional on the "
                "admissibility contract C (chain certificates included), on split independence and on "
                "candidate-wise valid p-values.")

    def to_dict(self):
        return {"certified": [p.to_dict() for p in self.certified], "ok": self.ok, "partial": self.partial,
                "delta": self.delta, "alpha": self.alpha, "reasons": list(self.reasons),
                "delta_registration": self.registration, "cover_id": self.cover_id,
                "cover_algorithm": COVER_ALGORITHM, "losses": [L.to_dict() for L in self.losses],
                "inference_registry_id": (inference_registry_id(L.inference for L in self.losses)
                                          if self.losses else ""),
                "inference_registration": (self.inference_registration.to_dict()
                                           if self.inference_registration else None),
                "payload_semantics": self.payload_semantics,
                "chains": [[p.name for p in c] for c in self.chains], "deltas": list(self.deltas),
                "pvalues": self.pvalues, "certificates": list(self.certificates),
                "inadmissible": {k: v.to_dict() for k, v in self.inadmissible.items()},
                "guarantee": self.guarantee()}


def audit_drift(policies, facts, losses):
    """Executable part of C5: the audit must be a property of the split and the labels, not of the candidate.

    Checked FAMILY-wide, not only inside a chain, and on the membership as well as the count: if two
    candidates disagree on how many units - or on WHICH units - carry a risk, then the audit, and hence the
    truncation it causes, is a function of the candidates' own behaviour on the certify fold, and the
    conditional argument behind C3 no longer holds.
    """
    bad = []
    for L in losses:
        n_seen = {facts[p.name].units_for(L) for p in policies}
        m_seen = {facts[p.name].membership_for(L) for p in policies}
        if len(n_seen) > 1:
            bad.append(f"{L.name}: audited unit COUNT differs across candidates {sorted(n_seen)}")
        if len(m_seen) > 1:
            bad.append(f"{L.name}: audited unit MEMBERSHIP differs across candidates "
                       f"{sorted(x[:12] for x in m_seen)}")
    return bad


def certify_family(policies: Sequence[Policy], losses: Sequence[LossSpec], *, delta: float,
                   contracts: Callable[[Policy], Contract], certify_eval: Callable[[Policy], Mapping],
                   plan: DeltaPlan | None = None, payload_semantics: PayloadSemantics | None = None,
                   inference_registration: InferenceRegistration | None = None):
    """Full ACE-LTT path: admissibility -> chains + delta_m (data-free) -> fixed-sequence certification.

    v2.4 ORDER OF OPERATIONS, and why it changed. Up to v2.3 the cover was built over ALL candidates and
    inadmissible members were then dropped from inside their chain. That can leave a chain whose safest
    member was removed while a more aggressive one is certified - so the returned set is not an up-set of
    the REGISTERED chain, and Theorem 1(ii) does not hold as stated. From v2.4:

      1. family-wide preconditions (C7 semantics, C5 audit drift) are checked first;
      2. every candidate-level clause of C is discharged BEFORE any cover exists, at a provisional uniform
         share; inadmissible candidates leave the family entirely;
      3. the REGISTERED cover, its order and (delta_m) are built over the SURVIVORS only;
      4. each survivor is re-checked at the share it will actually be tested at, and THIS pass is the
         authoritative one. For an EXACT minimum cover, deleting candidates cannot increase M, so under
         uniform allocation delta_m cannot fall and n_min cannot rise, and the pass is a verification. No
         such monotonicity is assumed for the greedy fallback above GREEDY_ABOVE, where deletion can
         increase M (measured: 3 -> 4 on a 402-candidate group), nor under a registered non-uniform plan.
         The re-check therefore may genuinely bind even under uniform allocation, and when it does the
         whole chain is terminal rather than partially tested;
      5. after the cover is fixed, ANY failure - chain certificate, or an evaluated unit set that does not
         match the audited one - is TERMINAL FOR THE WHOLE CHAIN. Nothing from that chain is certified and
         its share is not redistributed.

    The returned set is therefore a prefix of a chain of the registered cover, which is what the fail-safe
    statement of Theorem 1(ii) claims.

    `contracts(pi)` returns the deterministic facts of Sec. 2; `certify_eval(pi)` returns
    {loss name: (empirical risk, n units, audit membership digest)} - or ObservedLoss - on the CERTIFY fold,
    and is called only after the chains, their order and the shares delta_m are fixed; that is what makes
    clause C5 structural rather than a promise. A bare (risk, n) pair is refused: from v2.4.1 the evaluated
    unit SET is re-verified against the audited one.

    `inference_registration` is REQUIRED (v2.4.2). It carries the registered `inference_registry_id`, which
    is recomputed here from the code about to run; a mismatch certifies nothing. Passing None is not a
    default but a refusal: without it, the p-value construction that will execute is unpinned.
    """
    losses, policies = tuple(losses), tuple(policies)
    reg = "uniform" if plan is None else plan.registration
    sem = payload_semantics.to_dict() if payload_semantics else {}
    out = dict(delta=float(delta), registration=reg, losses=losses, payload_semantics=sem)
    if not policies:
        return Certification(reasons=("EMPTY_FAMILY",), **out)

    # ---- (1) family-wide preconditions -------------------------------------------------------------
    if any(L.kind == "payload" for L in losses) and payload_semantics is None:
        return Certification(reasons=("C7_FROZEN_PAYLOAD_SEMANTICS", "NO_REJECTION"), **out)
    # contract I, before anything else touches a p-value: is the machinery the registered machinery?
    if inference_registration is None:
        return Certification(reasons=("I_INFERENCE_REGISTRATION", "NO_REJECTION"),
                             inadmissible={policies[0].name: AdmissibilityResult(
                                 False, ("I_INFERENCE_REGISTRATION",),
                                 ("no inference registration supplied; the p-value construction that would "
                                  "execute is not pinned to any registration",))}, **out)
    ibad = inference_registration.problems(losses)
    if ibad:
        return Certification(reasons=("I_INFERENCE_REGISTRATION", "NO_REJECTION"),
                             inadmissible={policies[0].name: AdmissibilityResult(
                                 False, ("I_INFERENCE_REGISTRATION",), tuple(ibad))}, **out)
    out["inference_registration"] = inference_registration
    facts = {p.name: contracts(p) for p in policies}
    drift = audit_drift(policies, facts, losses)
    if drift:                                                    # family-wide: nothing can be certified
        return Certification(reasons=("C5_SPLIT_INDEPENDENCE", "NO_REJECTION"),
                             inadmissible={policies[0].name: AdmissibilityResult(
                                 False, ("C5_SPLIT_INDEPENDENCE",), tuple(drift))}, **out)

    # ---- (2) candidate-level admissibility, BEFORE the cover exists --------------------------------
    provisional = build_chains(policies, losses)                 # only to obtain an audit share
    prov_share = {p.name: dm for c, dm in zip(provisional, allocate_delta(provisional, float(delta), None))
                  for p in c}
    reasons, inadmissible, survivors = [], {}, []
    for p in policies:
        res = check_admissibility(p, facts[p.name], losses, prov_share[p.name])
        bad_pay = payload_semantics.problems(p, losses) if payload_semantics else []
        if bad_pay:
            res = AdmissibilityResult(False, (*res.reasons, "C7_FROZEN_PAYLOAD_SEMANTICS"),
                                      (*res.detail, *bad_pay))
        if res.admissible:
            survivors.append(p)
        else:
            inadmissible[p.name] = res
            reasons.extend(res.reasons)
    if not survivors:
        return Certification(reasons=tuple(dict.fromkeys([*reasons, "NO_REJECTION"])),
                             inadmissible=inadmissible, **out)

    # ---- (3) the REGISTERED cover is built over the survivors ---------------------------------------
    chains = build_chains(survivors, losses)                     # P1: policy algebra only
    deltas = allocate_delta(chains, float(delta), plan)          # P1: uniform, or a registered plan
    out["cover_id"] = chain_cover_id(chains)

    # ---- (4) re-check at the share actually used, and (5) chain-terminal failures -------------------
    keep_chains, keep_deltas, certificates = [], [], []
    for chain, dm in zip(chains, deltas):
        bad = []
        for p in chain:
            res = check_admissibility(p, facts[p.name], losses, dm)
            if not res.admissible:
                inadmissible[p.name] = res
                reasons.extend(res.reasons)
                bad.append(p.name)
        if bad:                                                  # terminal: no partial testing of a chain
            reasons.append("CHAIN_TERMINAL")
            inadmissible.setdefault(chain[0].name, AdmissibilityResult(
                False, ("CHAIN_TERMINAL",), (f"chain {[q.name for q in chain]} dropped: {bad} inadmissible "
                                             f"at its registered share delta_m={dm:g}",)))
            continue
        ok, certs = verify_chain(chain, losses)
        if not ok:
            reasons.extend(("C6_CHAIN_CERTIFICATE", "CHAIN_TERMINAL"))
            inadmissible[chain[0].name] = AdmissibilityResult(
                False, ("C6_CHAIN_CERTIFICATE",), (f"chain {[q.name for q in chain]} is not ordered",))
            continue
        certificates.extend(certs)
        keep_chains.append(tuple(chain))
        keep_deltas.append(dm)

    # ---- testing: a prefix of each registered chain --------------------------------------------------
    certified, pvals = [], {}
    for chain, dm in zip(keep_chains, keep_deltas):
        ps, terminal = [], False
        for p in chain:
            obs = {k: as_observation(k, v) for k, v in dict(certify_eval(p)).items()}
            # v2.4.1: identity of the audit population, not just its size. The count alone cannot detect an
            # evaluation on a different set of the same cardinality, and it is the SET that Lemma 0
            # conditions on.
            mism = []
            for L in losses:
                if L.name not in obs:
                    continue
                o, c = obs[L.name], facts[p.name]
                if int(o.n) != c.units_for(L):
                    mism.append(f"{L.name}: evaluated on {o.n} units, audited {c.units_for(L)}")
                elif o.membership != c.membership_for(L):
                    mism.append(f"{L.name}: evaluated membership {o.membership[:12]}... != audited "
                                f"{c.membership_for(L)[:12]}...")
            if mism:      # evaluated on a different unit set than was audited: the chain is terminal (v2.4)
                reasons.extend(("C3_CERTIFIABILITY_AUDIT", "CHAIN_TERMINAL"))
                inadmissible[p.name] = AdmissibilityResult(
                    False, ("C3_CERTIFIABILITY_AUDIT",),
                    (f"evaluated units != audited units for {mism}; chain "
                     f"{[q.name for q in chain]} is terminal",))
                terminal = True
                break
            pj, parts = joint_pvalue(obs, losses)
            pvals[p.name] = {"p_joint": pj, "per_loss": parts, "delta_m": dm}
            ps.append(pj)
            if pj > dm:
                break                                  # fixed sequence: nothing after this can be rejected
        if not terminal:
            certified.extend(chain[:fixed_sequence(ps, dm)])
    if not certified:
        reasons.append("NO_REJECTION")
    return Certification(tuple(certified), tuple(keep_chains), tuple(keep_deltas), pvals, tuple(certificates),
                         inadmissible, tuple(dict.fromkeys(reasons)), {L.name: L.alpha for L in losses}, **out)


def choose_operating_point(cert: Certification, utility: Callable[[Policy], float]):
    """Minimise the PRE-REGISTERED utility over the certified set. False alarms live here, not in the
    constrained family (P2), so the choice never weakens the guarantee - it only picks inside it."""
    if not cert.certified:
        return None, {"reason": "NO_REJECTION", "detail": NOT_CERTIFIED["NO_REJECTION"]}
    scored = sorted(((float(utility(p)), p.name, p) for p in cert.certified), key=lambda z: (z[0], z[1]))
    best = scored[0][2]
    return best, {"utility": scored[0][0], "n_certified": len(cert.certified), "partial": cert.partial,
                  "front": [{"policy": n, "utility": u} for u, n, _ in scored],
                  "p_joint": cert.pvalues.get(best.name, {}).get("p_joint")}


# ---------------------------------------------------------------- pre-freeze checklist (executable)

CHECKLIST = ("loss_by_loss_monotonicity", "deterministic_chain_cover", "predeclared_delta_plan",
             "policy_independent_audit_membership", "fixed_payload_semantics", "weights_hash_populated",
             "chain_margin_capacity", "chain_boundary_spanning", "inference_contract_executable")


def preflight_checklist(policies, losses, *, contracts=None, plan=None, payload_semantics=None,
                        weights_sha256="", protocol_dir=None, capacities=None, delta=0.10, select_bar=None,
                        joint_slacks=None, inference_registration=None):
    """The items that must be closed before freeze. Returns one row per item; `ok` is computed, not
    asserted. Nothing here touches the certify fold - it is a declaration audit, run before any evaluation."""
    losses, policies = tuple(losses), tuple(policies)
    rows = []

    def add(item, ok, detail):
        rows.append({"item": item, "ok": bool(ok), "detail": detail})

    miss = [L for L in losses if L.kind == "miss"]
    claimed = [L for L in miss if L.antitone_under_alarm_inclusion]
    missing = [L.name for L in claimed if L.aggregation not in MONOTONE_AGGREGATIONS or not L.lemma.strip()
               or L.empty_denominator not in EMPTY_DENOMINATOR]
    uncovered = [L.name for L in miss if not L.antitone_under_alarm_inclusion]
    add(CHECKLIST[0], bool(miss) and not missing and not uncovered,
        "no constrained miss loss declared" if not miss else
        (f"losses claiming antitonicity without their own lemma: {missing}" if missing else
         (f"constrained but NOT antitone: {uncovered} -> the chain certificate covers nothing, chains are "
          f"singletons and the fail-safe path does not apply; move them to the operating-point utility"
          if uncovered else
          "; ".join(f"{L.name}: catch_set={L.catch_set}, agg={L.aggregation}, w={L.weighting}, "
                    f"empty={L.empty_denominator}, lemma={L.lemma}" for L in claimed))))

    chains = build_chains(policies, losses) if policies else ()
    shuffled = build_chains(tuple(reversed(policies)), losses) if policies else ()
    same = [[p.name for p in c] for c in chains] == [[p.name for p in c] for c in shuffled]
    add(CHECKLIST[1], bool(chains) and same,
        f"{COVER_ALGORITHM}: M={len(chains)}, permutation-invariant={same}, id={chain_cover_id(chains)[:16]}"
        if chains else "no candidate")

    if plan is None:
        add(CHECKLIST[2], True, "no chain" if not chains else
            f"uniform: M={len(chains)}, delta_m = delta/M (delta=0.10 -> {0.10 / len(chains):.4f})")
    else:
        bound = plan.cover_id == chain_cover_id(chains)
        covered = all(tuple(chain_signature(c)) in dict(plan.shares) for c in chains)
        add(CHECKLIST[2], bound and covered,
            f"registration={plan.registration}, bound to this cover={bound}, every chain declared={covered}")

    if contracts is None:
        add(CHECKLIST[3], False, "no contracts callable given; membership cannot be audited")
    else:
        facts = {p.name: contracts(p) for p in policies}
        drift = audit_drift(policies, facts, losses)
        undeclared = sorted({L.name for L in losses for p in policies if not facts[p.name].membership_for(L)})
        add(CHECKLIST[3], not drift and not undeclared,
            "; ".join(drift) or (f"membership not declared for {undeclared}" if undeclared else
                                 "count and membership identical across the whole family"))

    pay = [L for L in losses if L.kind == "payload"]
    if not pay:
        add(CHECKLIST[4], True, "no payload risk in the constrained family")
    elif payload_semantics is None:
        add(CHECKLIST[4], False, "a payload risk needs frozen semantics (B0 grid, beta, unit, normalization)")
    else:
        bad = [x for p in policies for x in payload_semantics.problems(p, losses)]
        add(CHECKLIST[4], not bad,
            "; ".join(sorted(set(bad))) or
            f"registration={payload_semantics.registration}, unit={payload_semantics.unit_definition}, "
            f"B0 grid={list(payload_semantics.budgets)}, beta={payload_semantics.beta:g}, "
            f"norm={payload_semantics.normalization}")

    w = str(weights_sha256 or "")
    if not w and protocol_dir:
        try:
            import yaml
            P = yaml.safe_load((__import__("pathlib").Path(protocol_dir).expanduser() / "protocol.yaml").read_text())
            w = str((P.get("detector") or {}).get("weights_sha256") or "")
        except Exception as e:                                      # noqa: BLE001
            w = f"!{e}"
    okw = len(w) == 64 and all(c in "0123456789abcdef" for c in w.lower())
    add(CHECKLIST[5], okw, f"detector.weights_sha256 = {w[:16] + '...' if okw else (w or 'EMPTY')}")

    # 7 (Sec. 4.5): can the coordinate span the margin the SELECT BAR leaves to be crossed?
    if contracts is None or not chains:
        add(CHECKLIST[6], False, "no contracts or no chain: the chain's loss capacity cannot be checked")
    elif capacities is None:
        add(CHECKLIST[6], False, "capacities not declared")
    else:
        dm = float(delta) / len(chains)
        counts = {L.name: contracts(policies[0]).units_for(L) for L in losses}
        cap = chain_capacity(capacities, counts, losses, dm, select_bar)
        bad = [k for k, v in cap.items()
               if v["status"] != MARGIN_CAPABLE or not v["usable_for_registration"]]
        add(CHECKLIST[6], not bad, "; ".join(
            f"{k}: n={v['n_units']}, r*={v['boundary']:.4f}, bar={v['select_bar']:.4f}, "
            f"h*={v['required_margin']:.4f}, Gamma="
            f"{'NOT DECLARED' if v['capacity'] is None else format(v['capacity'], '.4f')}"
            f" [{v['source']}] -> {v['status']}"
            f"{'' if v['usable_for_registration'] else ' (not design-time: diagnostic only)'}"
            for k, v in cap.items()))

    # 8 (Sec. 4.6): does the chain actually cross the joint boundary on the select fold?
    if joint_slacks is None:
        add(CHECKLIST[7], False,
            "joint select-fold slacks not declared: a study must show its chain sees the certification "
            "boundary, or the fixed sequence has nothing to select between")
    else:
        rows_ = {}
        okall = True
        for name, sl in joint_slacks.items():
            st, info = boundary_spanning(sl)
            rows_[name] = (st, info)
            okall &= st == BOUNDARY_SPANNING
        add(CHECKLIST[7], okall, "; ".join(
            f"{n}: {st} (max {i.get('max_joint_slack', float('nan')):+.4f}, "
            f"min {i.get('min_joint_slack', float('nan')):+.4f}, safe endpoint feasible="
            f"{i.get('safe_endpoint_feasible')})" + (f" - {i['reason']}" if i.get("reason") else "")
            for n, (st, i) in rows_.items()))

    # 9 (contract I, v2.4.2): declared -> implemented -> registered -> unchanged at execution
    if not losses:
        add(CHECKLIST[8], False, "no constrained loss declared")
    elif inference_registration is None:
        add(CHECKLIST[8], False,
            "no inference registration declared: the p-value construction each loss names is not pinned to "
            "this study, so a substituted or re-pointed implementation would not be detected at execution")
    else:
        bad = list(inference_registration.problems(losses))
        multi = [ [p.name for p in c] for c in chains if len(c) > 1 ]
        incompatible = sorted({L.inference for L in losses
                               if not inference_spec(L.inference)["chain_compatible"]})
        if multi and incompatible:
            bad.append(f"multi-member chains {multi} with chain-incompatible construction(s) "
                       f"{incompatible}: A2-O does not hold")
        add(CHECKLIST[8], not bad, "; ".join(bad) or
            f"registration={inference_registration.registration}, "
            f"registry_id={inference_registration.registry_id[:16]}..., "
            + "; ".join(f"{L.name} -> {L.inference} "
                        f"[{inference_spec(L.inference)['version']}, w={L.weighting}, chain_compatible="
                        f"{inference_spec(L.inference)['chain_compatible']}]" for L in losses))
    return rows


def _cli(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="python -m cascade.ace_ltt",
                                 description="ACE-LTT pre-freeze checklist (declaration audit; reads no labels)")
    ap.add_argument("cmd", choices=["checklist", "selftest"])
    ap.add_argument("--protocol", help="protocol directory, to read detector.weights_sha256")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        from .test_ace_ltt import run_tests
        return run_tests()
    from .test_ace_ltt import CHECKLIST_FAMILY           # the declared example family (no data)
    pol, los, con, plan, sem, cap, slk, ireg = CHECKLIST_FAMILY()
    rows = preflight_checklist(pol, los, contracts=con, plan=plan, payload_semantics=sem,
                               protocol_dir=a.protocol, capacities=cap, joint_slacks=slk,
                               inference_registration=ireg)
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"[{'PASS' if r['ok'] else 'OPEN'}] {r['item']:<38} {r['detail']}")
        print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} closed" +
              ("" if all(r["ok"] for r in rows) else "  -> NOT ready to freeze"))


__all__ = ["Policy", "LossSpec", "PAYLOAD_LOSS", "payload_loss", "miss_loss", "PayloadSemantics",
           "MONOTONE_AGGREGATIONS", "EMPTY_DENOMINATOR", "COVER_ALGORITHM", "CHECKLIST",
           "preflight_checklist", "audit_drift",
           "chain_cover_id", "certification_boundary", "certification_margin", "chain_capacity",
           "unreachable_from", "MARGIN_CAPABLE", "MARGIN_INCAPABLE", "BOUNDARY_SPANNING",
           "NOT_SPANNING", "certification_slack", "boundary_spanning", "Capacity",
           "SPEC_VERSION", "PROVENANCE_THREAT_MODEL",
           "Contract",
           "AdmissibilityResult",
           "Certification", "CertifyDataLeak", "DeltaPlan", "NOT_CERTIFIED", "n_min", "check_admissibility",
           "structural_order", "decision_sets", "certify_chain_relation", "build_chains", "verify_chain",
           "chain_signature", "allocate_delta", "hb_pvalue", "joint_pvalue", "fixed_sequence", "select_front",
           "BUILTIN_INFERENCE", "INFERENCE_REGISTRY", "inference_spec", "inference_registry_id",
           "InferenceRegistration", "declare_inference", "callable_identity",
           "ObservedLoss", "as_observation", "membership_digest",
           "certify_family", "choose_operating_point", "replace"]


if __name__ == "__main__":
    _cli()
