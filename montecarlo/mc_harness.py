#!/usr/bin/env python3
"""ACE-LTT controlled validation - Monte-Carlo harness for the frozen pre-registration R0.

Implements ACE_MC_preregistration_R0.md and nothing else. Written AFTER R0 was frozen (provenance order:
frozen engine -> frozen R0 -> this harness). The engine is imported from ~/Norqobil/v249/cascade as a
library; no engine file or module constant is modified. ace_toy.py is not imported and not modified.

    python mc_prereg/mc_harness.py verify
    python mc_prereg/mc_harness.py selftest                  # development check, off-grid, nothing kept
    python mc_prereg/mc_harness.py pilot --workers P         # R0 section 7 timing rule
    python mc_prereg/mc_harness.py run   --workers P         # MC-A/B, MC-C, MC-D  (needs the pilot record)
    python mc_prereg/mc_harness.py summarize                 # applies the R0 decision rules

All commands run from ~/Norqobil. Nothing is written outside ~/Norqobil/mc_prereg/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ identities pinned by R0 (section 11)
R0_SHA256 = "a8e0b6fca234f043d7ead1e4a6f7d00b605a455888c1101cf504274c6271cedd"
ENGINE_SHA256 = "ca93d2f3e4eafa85255f512318c0e7a350d259640518696ddbfed23e93960241"
GEN_JSON_SHA256 = "0f0ed400138b15a28b3408b323e7964f420d698dddb1fc3d5b5e5284cc902ab8"

ROOT = Path(os.environ.get("NORQOBIL", Path.home() / "Norqobil")).resolve()
HERE = ROOT / "mc_prereg"
ENGINE_DIR = ROOT / "v249"
RUN = HERE / "run_R0"
sys.path.insert(0, str(ENGINE_DIR))

from cascade.ace_ltt import (Contract, LossSpec, Policy, _greedy_chain_cover, _inclusion_key,  # noqa: E402
                             _min_chain_cover, allocate_delta, as_observation, audit_drift, build_chains,
                             certify_chain_relation, certify_family, check_admissibility, decision_sets,
                             declare_inference, fixed_sequence, joint_pvalue, miss_loss, n_min, verify_chain)

# ------------------------------------------------------------------ R0 sections 3, 4, 12
DELTA, ALPHA = 0.10, 0.05
N_GRID = (50, 80, 150, 400, 1000)
CONFIGS = ("spread", "least-favourable", "global-null")
D_GRID, L_GRID = (1, 2, 3), (1, 2)
ADM = ("present", "removed")
P1 = 0.3
BETA = {1: ((2.0, 5.0), (1.5, 4.0)), 2: ((1.3, 2.0), (1.3, 2.0))}   # k -> (stratum S0, stratum S1)
REGISTRATION = "ace_ltt_mc_prereg/R0"
MASTER_SEED = 20261001
REPS_A, DELSETS_D = 200, 1000
PILOT_SEED, PILOT_N, PILOT_REPS = 99, 300, 20
PILOT_LEVELS = {1: [[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]],
                2: [[0.030, 0.045, 0.060, 0.075, 0.090], [0.0, 0.01, 0.02, 0.03, 0.04]],
                3: [[0.04, 0.06, 0.08, 0.10], [0.0, 0.01, 0.02, 0.03], [0.0, 0.01, 0.02, 0.03]]}
D2_LEVELS = [[round(0.10 + 0.05 * i, 2) for i in range(8)], [round(0.02 * i, 2) for i in range(8)],
             [round(0.02 * i, 2) for i in range(8)]]
T_LIMIT_S = 259_200.0
S_EDGE = 0.5                                     # edge score of every event: band [0, 1) -> escalated


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_identities():
    got = {"R0": sha256(HERE / "ACE_MC_preregistration_R0.md"),
           "engine": sha256(ENGINE_DIR / "cascade" / "ace_ltt.py"),
           "generator_json": sha256(HERE / "mc_generator_params.json")}
    want = {"R0": R0_SHA256, "engine": ENGINE_SHA256, "generator_json": GEN_JSON_SHA256}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise SystemExit(f"identity mismatch - refusing to run: {bad}")
    return got


def levels_for(cfg, d, L):
    gen = json.load(open(HERE / "mc_generator_params.json"))
    return gen[f"{cfg}|d={d}|L={L}"]["levels"]


# ------------------------------------------------------------------ family, losses, truth
def make_family(d, lv):
    b = lv[0]
    v0 = lv[1] if d == 3 else [0.0]
    v1 = lv[-1] if d >= 2 else [0.0]
    pol = [Policy("a", 0.0, 1.0, float(bb), 1.0, (("S0", float(a0)), ("S1", float(a1))), float("inf"))
           for bb in b for a0 in v0 for a1 in v1]
    return tuple(sorted(pol, key=_inclusion_key))


def make_losses(L, adm):
    out = []
    for k in range(1, L + 1):
        if adm == "removed" and k == 1:
            out.append(LossSpec("m1", ALPHA, False, kind="miss", catch_set="response",
                                aggregation="unit_any_positive_frame", empty_denominator="exclude_unit"))
        else:
            out.append(miss_loss(f"m{k}", ALPHA, f"L-mc-{k}", catch_set="response",
                                 aggregation="unit_any_positive_frame"))
    return tuple(out)


def _F(ab, x):
    from scipy.special import betainc
    return betainc(ab[0], ab[1], np.clip(np.asarray(x, float), 0.0, 1.0))


def true_risk(p, k):
    return float((1 - P1) * _F(BETA[k][0], p.b - p.veto_at("S0")) + P1 * _F(BETA[k][1], p.b - p.veto_at("S1")))


def null_set(pol, L):
    return {p.name for p in pol if any(true_risk(p, k) > ALPHA for k in range(1, L + 1))}


# ------------------------------------------------------------------ one replication's data and engine hooks
def sample(rng, n, L):
    """R0 section 3 sampling order: sigma, then g_1, then g_2."""
    sigma = rng.random(n) < P1
    g = {}
    for k in range(1, L + 1):
        a = np.where(sigma, BETA[k][1][0], BETA[k][0][0])
        b = np.where(sigma, BETA[k][1][1], BETA[k][0][1])
        g[k] = rng.beta(a, b)
    ids = [f"u{i}" for i in range(n)]
    dig = hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()
    return {"n": n, "sigma": sigma, "g": g, "ids": ids, "dig": dig,
            "stratum": np.where(sigma, "S1", "S0")}


def caught(D, p, k):
    v = np.where(D["sigma"], p.veto_at("S1"), p.veto_at("S0"))
    return np.minimum(1.0, D["g"][k] + v) >= p.b


class Hooks:
    """contracts / certify_eval for certify_family, on the ace_toy pattern (R0 Q10)."""

    def __init__(self, D, losses):
        self.D, self.losses = D, losses
        self._miss = {}

    def miss(self, p, k):
        key = (p.name, k)
        if key not in self._miss:
            self._miss[key] = ~caught(self.D, p, k)
        return self._miss[key]

    def contracts(self, p):
        n, dig = self.D["n"], self.D["dig"]
        return Contract(True, True, {L.name: n for L in self.losses}, 0, {L.name: dig for L in self.losses},
                        0.0, float("inf"), True)

    def certify_eval(self, p):
        n, dig = self.D["n"], self.D["dig"]
        return {L.name: (float(self.miss(p, int(L.name[1:])).mean()), n, dig) for L in self.losses}


class JointMemo:
    """Per-replication memo of the ENGINE's joint_pvalue return value (identical inputs, identical output)."""

    def __init__(self, hooks, losses):
        self.h, self.losses, self.m = hooks, losses, {}

    def __call__(self, p):
        if p.name not in self.m:
            obs = {k: as_observation(k, v) for k, v in dict(self.h.certify_eval(p)).items()}
            self.m[p.name] = joint_pvalue(obs, self.losses)
        return self.m[p.name]


# ------------------------------------------------------------------ Path L (R0 section 6, bound by Q9)
def path_L(policies, losses, cover, hooks, ireg, memo, provisional):
    """Transcription of certify_family (ace_ltt.py 1240-1356) for a GIVEN cover; no added logic."""
    losses, policies = tuple(losses), tuple(policies)
    ibad = ireg.problems(losses)
    if ibad:
        return {"certified": (), "inadmissible": (policies[0].name,),
                "reasons": ("I_INFERENCE_REGISTRATION", "NO_REJECTION"), "path": "R1"}
    facts = {p.name: hooks.contracts(p) for p in policies}
    if audit_drift(policies, facts, losses):
        return {"certified": (), "inadmissible": (policies[0].name,),
                "reasons": ("C5_SPLIT_INDEPENDENCE", "NO_REJECTION"), "path": "R1"}
    prov_share = {p.name: dm for c, dm in zip(provisional, allocate_delta(provisional, float(DELTA), None))
                  for p in c}
    reasons, inadmissible, survivors = [], {}, []
    for p in policies:
        res = check_admissibility(p, facts[p.name], losses, prov_share[p.name])
        if res.admissible:
            survivors.append(p)
        else:
            inadmissible[p.name] = res
            reasons.extend(res.reasons)
    if not survivors:
        return {"certified": (), "inadmissible": tuple(sorted(inadmissible)),
                "reasons": tuple(dict.fromkeys([*reasons, "NO_REJECTION"])), "path": "R2"}
    if {p.name for c in cover for p in c} != {p.name for p in survivors}:
        raise RuntimeError("harness anomaly: partial survival under common n (R0 section 5 says all-or-none)")
    chains = tuple(tuple(c) for c in cover)
    deltas = allocate_delta(chains, float(DELTA), None)
    keep_chains, keep_deltas = [], []
    for chain, dm in zip(chains, deltas):
        bad = []
        for p in chain:
            res = check_admissibility(p, facts[p.name], losses, dm)
            if not res.admissible:
                inadmissible[p.name] = res
                reasons.extend(res.reasons)
                bad.append(p.name)
        if bad:
            reasons.append("CHAIN_TERMINAL")
            inadmissible.setdefault(chain[0].name, None)
            continue
        ok, _ = verify_chain(chain, losses)
        if not ok:
            reasons.extend(("C6_CHAIN_CERTIFICATE", "CHAIN_TERMINAL"))
            inadmissible[chain[0].name] = None
            continue
        keep_chains.append(chain)
        keep_deltas.append(dm)
    certified = []
    for chain, dm in zip(keep_chains, keep_deltas):
        ps, terminal = [], False
        for p in chain:
            obs = {k: as_observation(k, v) for k, v in dict(hooks.certify_eval(p)).items()}
            mism = []
            for L in losses:
                if L.name not in obs:
                    continue
                o, c = obs[L.name], facts[p.name]
                if int(o.n) != c.units_for(L) or o.membership != c.membership_for(L):
                    mism.append(L.name)
            if mism:
                reasons.extend(("C3_CERTIFIABILITY_AUDIT", "CHAIN_TERMINAL"))
                inadmissible[p.name] = None
                terminal = True
                break
            pj, _ = memo(p)
            ps.append(pj)
            if pj > dm:
                break
        if not terminal:
            certified.extend(chain[:fixed_sequence(ps, dm)])
    if not certified:
        reasons.append("NO_REJECTION")
    path = "T4" if "CHAIN_TERMINAL" in reasons else ("pass" if not inadmissible else "other")
    return {"certified": tuple(p.name for p in certified), "inadmissible": tuple(sorted(inadmissible)),
            "reasons": tuple(dict.fromkeys(reasons)), "path": path}


def engine_summary(C):
    path = ("R2" if (not C.cover_id and C.inadmissible) else          # refused before any cover (step 2)
            "T4" if "CHAIN_TERMINAL" in C.reasons else "pass" if not C.inadmissible else "other")
    return {"certified": tuple(p.name for p in C.certified), "inadmissible": tuple(sorted(C.inadmissible)),
            "reasons": tuple(C.reasons), "path": path}


# ------------------------------------------------------------------ covers (R0 section 6, MC-A)
def comparability(items, losses):
    n = len(items)
    return [[j > i and certify_chain_relation(items[i], items[j], losses) is not None for j in range(n)]
            for i in range(n)]


def refinements(cmin, n_items, rng, k=10):
    Mmin, out = len(cmin), []
    for j in range(1, k + 1):
        target = Mmin + int(round(j * (n_items - Mmin) / 11))
        chains = [list(c) for c in cmin]
        while len(chains) < target:
            long_ = [i for i, c in enumerate(chains) if len(c) >= 2]
            ci = long_[int(rng.integers(len(long_)))]
            c = chains[ci]
            cut = 1 + int(rng.integers(len(c) - 1))
            chains[ci:ci + 1] = [c[:cut], c[cut:]]
        out.append(tuple(tuple(c) for c in chains))
    return out


def random_cover(items, comp, rng):
    chains = []
    for i in rng.permutation(len(items)):
        ok = [c for c in chains if all(comp[min(i, j)][max(i, j)] for j in c)]
        if ok:
            ok[int(rng.integers(len(ok)))].append(int(i))
        else:
            chains.append([int(i)])
    return tuple(tuple(sorted((items[j] for j in c), key=_inclusion_key)) for c in chains)


def predicted_path(n, M_prov, M):
    if n < n_min(ALPHA, DELTA / M_prov):
        return "R2"
    return "T4" if n < n_min(ALPHA, DELTA / M) else "pass"


# ------------------------------------------------------------------ MC-A / MC-B: one cell
def cell_A(args):
    cfg, d, L, n, adm, first_index = args
    lv = levels_for(cfg, d, L)
    pol = make_family(d, lv)
    losses = make_losses(L, adm)
    ireg = declare_inference(REGISTRATION, losses)
    nulls = null_set(pol, L)
    cmin = build_chains(pol, losses)
    comp = comparability(pol, losses)
    edges = [(i, j) for i in range(len(pol)) for j in range(i + 1, len(pol)) if comp[i][j]]
    M_prov = len(cmin)
    rows = []
    for r in range(REPS_A):
        rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED, spawn_key=(0, first_index + r)))
        D = sample(rng, n, L)
        h = Hooks(D, losses)
        memo = JointMemo(h, losses)
        a0 = 0
        for p in pol:
            for k in range(1, L + 1):
                ref = decision_sets(p, np.full(n, S_EDGE), D["g"][k], D["stratum"])["response"]
                a0 += int(np.sum(ref != caught(D, p, k)))
        C = certify_family(pol, losses, delta=DELTA, contracts=h.contracts, certify_eval=h.certify_eval,
                           plan=None, payload_semantics=None, inference_registration=ireg)
        eng = engine_summary(C)
        covers = [("min", cmin)]
        if adm == "present":
            covers += [(f"ref{j + 1}", c) for j, c in enumerate(refinements(cmin, len(pol), rng))]
            covers += [(f"rnd{j + 1}", random_cover(pol, comp, rng)) for j in range(10)]
            covers += [("single", tuple((p,) for p in pol))]
        defects, res = [], []
        for kind, cov in covers:
            if not all(verify_chain(c, losses)[0] for c in cov):
                defects.append(kind)
                continue
            out = path_L(pol, losses, cov, h, ireg, memo, cmin)
            res.append((kind, len(cov), out))
        a6 = int(eng["certified"] != res[0][2]["certified"] or eng["inadmissible"] != res[0][2]["inadmissible"]
                 or tuple(eng["reasons"]) != tuple(res[0][2]["reasons"]))
        pj = {p.name: memo(p) for p in pol}
        a2 = a5 = 0
        cov_rows = []
        for kind, M, out in res:
            ok = out["path"] == "pass"
            if ok:
                thr = {p.name for p in pol if pj[p.name][0] <= DELTA / M}
                a2 += int(set(out["certified"]) != thr)
            obs_path = out["path"] if out["path"] in ("R2", "T4", "pass") else "other"
            a5 += int(obs_path != predicted_path(n, M_prov, M))
            cov_rows.append([kind, M, out["path"], len(out["certified"]),
                             len([x for x in out["certified"] if x not in nulls])])
        a1 = a4 = 0
        for (k1, M1, o1) in res:
            for (k2, M2, o2) in res:
                if M1 < M2:
                    if o1["path"] == "pass" and o2["path"] == "pass":
                        a1 += int(not set(o2["certified"]) <= set(o1["certified"]))
                    c3 = any(x == "C3_CERTIFIABILITY_AUDIT" for x in o1["reasons"])
                    a4 += int(o1["path"] in ("R2", "T4") and c3 and o2["path"] == "pass")
        a3 = 0
        for i, j in edges:
            pi, pjj = pj[pol[i].name][1], pj[pol[j].name][1]
            a3 += int(any(pi[L_.name] > pjj[L_.name] for L_ in losses))
        rows.append({"cell": [cfg, d, L, n, adm], "rep": r, "A0": a0, "A1": a1, "A2": a2, "A3": a3, "A4": a4,
                     "A5": a5, "A6": a6, "defects": defects, "engine_path": eng["path"],
                     "n_nonnull": len(pol) - len(nulls), "covers": cov_rows})
    return rows


# ------------------------------------------------------------------ MC-C: one cell
def mcC_rep(pol, losses, ireg, nulls, cmin, singles, D):
    h = Hooks(D, losses)
    memo = JointMemo(h, losses)
    C = certify_family(pol, losses, delta=DELTA, contracts=h.contracts, certify_eval=h.certify_eval,
                       plan=None, payload_semantics=None, inference_registration=ireg)
    eng = engine_summary(C)
    s = path_L(pol, losses, singles, h, ireg, memo, cmin)
    inv = int(any(x in nulls for x in eng["certified"]))
    inv_s = int(any(x in nulls for x in s["certified"]))
    miss = np.array([[h.miss(p, k).sum() for k in range(1, len(losses) + 1)] for p in pol], float)
    return inv, inv_s, eng["path"], miss


def cell_C(args):
    cfg, d, L, n, first_index, reps, seed_block = args
    lv = levels_for(cfg, d, L)
    pol = make_family(d, lv)
    losses = make_losses(L, "present")
    ireg = declare_inference(REGISTRATION, losses)
    nulls = null_set(pol, L)
    cmin = build_chains(pol, losses)
    singles = tuple((p,) for p in pol)
    inv, inv_s, paths = [], [], {}
    msum = np.zeros((len(pol), L))
    for r in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED, spawn_key=(seed_block, first_index + r)))
        D = sample(rng, n, L)
        a, b, pth, m = mcC_rep(pol, losses, ireg, nulls, cmin, singles, D)
        inv.append(a); inv_s.append(b); paths[pth] = paths.get(pth, 0) + 1; msum += m
    truth = [[true_risk(p, k) for k in range(1, L + 1)] for p in pol]
    return {"cell": [cfg, d, L, n], "reps": reps, "invalid": int(sum(inv)), "invalid_singleton": int(sum(inv_s)),
            "paths": paths, "invalid_bits": "".join(map(str, inv)), "miss_sum": msum.tolist(),
            "truth": truth, "policies": [p.name for p in pol]}


# ------------------------------------------------------------------ MC-D
def run_D():
    out, idx = [], 0
    grids = [("D1", d, make_family(d, levels_for("spread", d, 1))) for d in D_GRID]
    grids.append(("D2", 3, make_family(3, D2_LEVELS)))
    losses = make_losses(1, "present")
    for arm, d, pol in grids:
        comp = comparability(pol, losses)
        pos = {p.name: i for i, p in enumerate(pol)}

        def covers(items):
            ii = [pos[p.name] for p in items]
            cmp = lambda a, b: comp[ii[a]][ii[b]]  # noqa: E731  (the engine relation, precomputed)
            exact = _min_chain_cover(list(items), cmp)
            greedy = build_chains(items, losses) if arm == "D2" else _greedy_chain_cover(list(items), cmp)
            return exact, greedy

        ex0, gr0 = covers(pol)
        smax = 111 if arm == "D2" else len(pol) // 4
        for r in range(DELSETS_D):
            rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED, spawn_key=(3, idx)))
            idx += 1
            size = int(rng.integers(1, smax + 1))
            S = set(int(x) for x in rng.choice(len(pol), size=size, replace=False))
            items = tuple(p for i, p in enumerate(pol) if i not in S)
            ex, gr = covers(items)
            chk = int(len(ex) > len(ex0)) + int(len(gr) < len(ex)) + \
                int(not all(verify_chain(c, losses)[0] for c in (*ex, *gr)))
            out.append({"arm": arm, "d": d, "N": len(pol), "rep": r, "S": sorted(S), "M_exact_full": len(ex0),
                        "M_greedy_full": len(gr0), "M_exact": len(ex), "M_greedy": len(gr),
                        "E": int(len(gr) > len(gr0)), "consistency_violations": chk,
                        "greedy_full": [[p.name for p in c] for c in gr0] if r == 0 else None,
                        "greedy_after": [[p.name for p in c] for c in gr] if len(gr) > len(gr0) else None})
    return out


# ------------------------------------------------------------------ cell orders (R0 section 7)
def cells_A():
    return [(c, d, L, n, a) for c in CONFIGS for d in D_GRID for L in L_GRID for n in N_GRID for a in ADM]


def cells_C():
    return [(c, d, L, n) for c in CONFIGS for d in D_GRID for L in L_GRID for n in N_GRID]


# ------------------------------------------------------------------ commands
def versions():
    import scipy
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform()}


def cmd_verify(_):
    print(json.dumps(verify_identities(), indent=1))
    print("identities OK")


def cmd_selftest(_):
    """Development check on the OFF-GRID pilot families, seed 7; prints only code-level invariants."""
    verify_identities()
    for d in D_GRID:
        pol = make_family(d, PILOT_LEVELS[d])
        losses = make_losses(1, "present")
        ireg = declare_inference(REGISTRATION, losses)
        cmin = build_chains(pol, losses)
        comp = comparability(pol, losses)
        for r in range(2):
            rng = np.random.default_rng(np.random.SeedSequence(7, spawn_key=(d, r)))
            D = sample(rng, PILOT_N, 1)
            h = Hooks(D, losses)
            memo = JointMemo(h, losses)
            ref = decision_sets(pol[0], np.full(PILOT_N, S_EDGE), D["g"][1], D["stratum"])["response"]
            assert np.array_equal(ref, caught(D, pol[0], 1)), "A0-type mismatch"
            C = certify_family(pol, losses, delta=DELTA, contracts=h.contracts, certify_eval=h.certify_eval,
                               plan=None, payload_semantics=None, inference_registration=ireg)
            e, l_ = engine_summary(C), path_L(pol, losses, cmin, h, ireg, memo, cmin)
            assert (e["certified"], e["inadmissible"], tuple(e["reasons"])) == \
                   (l_["certified"], l_["inadmissible"], tuple(l_["reasons"])), "A6-type mismatch"
            for cov in [*refinements(cmin, len(pol), rng, 3), random_cover(pol, comp, rng)]:
                assert all(verify_chain(c, losses)[0] for c in cov), "cover generator defect"
                path_L(pol, losses, cov, h, ireg, memo, cmin)
    rm = make_losses(1, "removed")
    assert len(build_chains(make_family(2, PILOT_LEVELS[2]), rm)) == 25
    ss = np.random.SeedSequence(MASTER_SEED).spawn(4)[0].spawn(3)[2]
    assert list(ss.generate_state(2)) == list(np.random.SeedSequence(MASTER_SEED, spawn_key=(0, 2)).generate_state(2))
    print("selftest OK (off-grid pilot families, seed 7; no registered cell touched, nothing written)")


def cmd_pilot(a):
    verify_identities()
    rec_path = RUN / "pilot_record.json"
    if rec_path.exists():
        raise SystemExit("pilot record exists; the pilot runs once")
    RUN.mkdir(parents=True, exist_ok=True)
    rec = {"P": int(a.workers), "written_before_pilot": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(rec, open(rec_path, "w"), indent=1)
    tbar = {}
    for d in D_GRID:
        pol = make_family(d, PILOT_LEVELS[d])
        losses = make_losses(1, "present")
        ireg = declare_inference(REGISTRATION, losses)
        cmin = build_chains(pol, losses)
        singles = tuple((p,) for p in pol)
        nulls = null_set(pol, 1)
        ts = []
        for r in range(PILOT_REPS):
            t0 = time.perf_counter()
            rng = np.random.default_rng(np.random.SeedSequence(PILOT_SEED, spawn_key=(d, r)))
            D = sample(rng, PILOT_N, 1)
            mcC_rep(pol, losses, ireg, nulls, cmin, singles, D)       # outputs discarded unread
            ts.append(time.perf_counter() - t0)
        tbar[d] = float(np.mean(ts))
    T = 5000.0 / a.workers * 50.4 * sum(tbar.values())
    rec.update({"tbar_s": tbar, "T_hat_s": T, "limit_s": T_LIMIT_S, "mcC_reps": 2000 if T > T_LIMIT_S else 5000,
                **versions()})
    json.dump(rec, open(rec_path, "w"), indent=1)
    print(json.dumps(rec, indent=1))


def cmd_run(a):
    ids = verify_identities()
    rec = json.load(open(RUN / "pilot_record.json"))
    if int(rec["P"]) != int(a.workers):
        raise SystemExit(f"worker count {a.workers} != the P={rec['P']} recorded before the pilot")
    for f in ("mcA.jsonl", "mcC.jsonl", "mcD.jsonl", "manifest.json"):
        if (RUN / f).exists():
            raise SystemExit(f"{f} exists; the registered run is never overwritten")
    R = int(rec["mcC_reps"])
    man = {"R0_sha256": ids["R0"], "engine_sha256": ids["engine"], "generator_json_sha256": ids["generator_json"],
           "harness_sha256": sha256(Path(__file__)), "workers": a.workers, "mcC_reps": R, "pilot_record": rec,
           "master_seed": MASTER_SEED, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **versions()}
    json.dump(man, open(RUN / "manifest.json", "w"), indent=1)
    argsA = [(*c, i * REPS_A) for i, c in enumerate(cells_A())]
    argsC = [(*c, i * R, R, 2) for i, c in enumerate(cells_C())]
    with ProcessPoolExecutor(a.workers) as ex, open(RUN / "mcA.jsonl", "w") as fA, open(RUN / "mcC.jsonl", "w") as fC:
        for rows in ex.map(cell_A, argsA):
            for row in rows:
                fA.write(json.dumps(row) + "\n")
            fA.flush()
        for row in ex.map(cell_C, argsC):
            fC.write(json.dumps(row) + "\n")
            fC.flush()
    with open(RUN / "mcD.jsonl", "w") as fD:
        for row in run_D():
            fD.write(json.dumps(row) + "\n")
    wit = [r for r in (json.loads(x) for x in open(RUN / "mcD.jsonl")) if r["E"]]
    if wit:
        (RUN / "witness").mkdir(exist_ok=True)
        w = sorted(wit, key=lambda r: (r["arm"] != "D1", r["N"], len(r["S"]), r["rep"]))[0]
        json.dump(w, open(RUN / "witness" / "mcD_witness.json", "w"), indent=1)
    man["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(man, open(RUN / "manifest.json", "w"), indent=1)
    print("run complete; next: summarize")


def cp(x, N, conf=0.95):
    from scipy.stats import beta
    lo = 0.0 if x == 0 else float(beta.ppf((1 - conf) / 2, x, N - x + 1))
    hi = 1.0 if x == N else float(beta.ppf(1 - (1 - conf) / 2, x + 1, N - x))
    return lo, hi


def cmd_summarize(_):
    from scipy.stats import beta
    A = [json.loads(x) for x in open(RUN / "mcA.jsonl")]
    C = [json.loads(x) for x in open(RUN / "mcC.jsonl")]
    Dd = [json.loads(x) for x in open(RUN / "mcD.jsonl")]
    out = {"MC-A": {k: int(sum(r[k] for r in A)) for k in ("A0", "A1", "A2", "A3", "A4", "A5", "A6")}}
    out["MC-A"]["cover_defects"] = int(sum(len(r["defects"]) for r in A))
    out["MC-A"]["replications"] = len(A)
    # MC-B
    mb = {}
    for r in A:
        cov = {c[0]: c for c in r["covers"]}
        if "min" not in cov:
            continue
        cm = cov["min"]
        key = "|".join(map(str, r["cell"]))
        e = mb.setdefault(key, {"delta_by_M": {}, "yield_min": [], "yield_single": [], "noninterpretable": 0})
        for c in r["covers"][1:]:
            if c[2] == "pass" and cm[2] == "pass":
                e["delta_by_M"].setdefault(str(c[1] / cm[1]), []).append(cm[3] - c[3])
            elif c[2] == "pass" and cm[2] != "pass":
                e["noninterpretable"] += 1
        if r["n_nonnull"] > 0 and r["cell"][0] != "global-null":
            e["yield_min"].append(cm[4] / r["n_nonnull"])
            if "single" in cov:
                e["yield_single"].append(cov["single"][4] / r["n_nonnull"])
    for e in mb.values():
        e["delta_by_M"] = {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                               "P_pos": float(np.mean(np.array(v) > 0)), "count": len(v)}
                           for k, v in e["delta_by_M"].items()}
        e["yield_min"] = float(np.mean(e["yield_min"])) if e["yield_min"] else None
        e["yield_single"] = float(np.mean(e["yield_single"])) if e["yield_single"] else None
    out["MC-B"] = mb
    # MC-C
    vac = {("2", "50"), ("3", "50"), ("3", "80")}
    cells, pooled_x, pooled_N, fails, g1 = [], 0, 0, [], []
    for r in C:
        x, N = r["invalid"], r["reps"]
        lo99 = 0.0 if x == 0 else float(beta.ppf(0.01, x, N - x + 1))
        vacuous = (str(r["cell"][1]), str(r["cell"][3])) in vac
        cells.append({"cell": r["cell"], "fwer": x / N, "ci95": cp(x, N), "lower99": lo99,
                      "fail": lo99 > DELTA, "vacuous": vacuous, "paths": r["paths"],
                      "fwer_singleton": r["invalid_singleton"] / N})
        if lo99 > DELTA:
            fails.append(r["cell"])
        if not vacuous:
            pooled_x += x; pooled_N += N
        n = r["cell"][3]
        for i, row in enumerate(r["miss_sum"]):
            for k, s in enumerate(row):
                R_ = r["truth"][i][k]
                z = (s / (n * N) - R_) / math.sqrt(R_ * (1 - R_) / (n * N))
                if abs(z) > 5:
                    g1.append({"cell": r["cell"], "policy": r["policies"][i], "k": k + 1, "z": z})
    worst = max((c for c in cells if not c["vacuous"]), key=lambda c: c["fwer"])
    out["MC-C"] = {"cells": cells, "failures": fails, "pooled_fwer": pooled_x / pooled_N,
                   "pooled_ci95": cp(pooled_x, pooled_N), "worst_cell": worst, "G1_failures": g1}
    # MC-D
    md = {}
    for arm_d in sorted({(r["arm"], r["d"]) for r in Dd}):
        rs = [r for r in Dd if (r["arm"], r["d"]) == arm_d]
        x, N = sum(r["E"] for r in rs), len(rs)
        md[f"{arm_d[0]}|d={arm_d[1]}"] = {"M_greedy_minus_exact_full": rs[0]["M_greedy_full"] - rs[0]["M_exact_full"],
                                          "E": x, "N": N, "ci95": cp(x, N),
                                          "U95_if_zero": (1 - 0.05 ** (1 / N)) if x == 0 else None,
                                          "consistency_violations": int(sum(r["consistency_violations"] for r in rs))}
    out["MC-D"] = md
    json.dump(out, open(RUN / "summary.json", "w"), indent=1)
    print(json.dumps({"MC-A": out["MC-A"], "MC-C failures": fails, "G1 failures": len(g1),
                      "MC-D": {k: (v["E"], v["N"]) for k, v in md.items()}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("verify", "selftest", "pilot", "run", "summarize"))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    {"verify": cmd_verify, "selftest": cmd_selftest, "pilot": cmd_pilot, "run": cmd_run,
     "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
