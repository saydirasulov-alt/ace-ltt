"""ACE-LTT MC pre-registration, §12: generator grid levels, constructed from the CLOSED-FORM risk only.

Never imports or runs the engine. No randomness. Construction (per truth config, d, L):
  1. b-levels: b_i = Rmax^{-1}(t_i) at v = 0, for the registered base risks t_i (bisection).
  2. v-levels (every veto coordinate): equally spaced on [0, v_max], with v_max found by bisection so that
     Rmax at the registered reference b-level, with every veto coordinate at v_max, equals the floor.
  3. Levels rounded to 4 decimals; risks recomputed from the rounded levels; acceptance checked.
R_k(pi) = 0.7 F_k0(b - v(S0)) + 0.3 F_k1(b - v(S1)),  F = Beta CDF, F(x) = 0 for x <= 0.
"""
import json, sys
import numpy as np
from scipy.special import betainc

P1 = 0.3
BETA = {1: ((2.0, 5.0), (1.5, 4.0)), 2: ((1.3, 2.0), (1.3, 2.0))}
SHAPE = {1: (8,), 2: (5, 5), 3: (4, 4, 4)}             # coordinates: b | b, v(S1) | b, v(S0), v(S1)

# (base risks at v = 0 for the b-levels, index of the reference b-level, floor for step 2)
DESIGN = {
    ("spread", 1):           (np.linspace(0.010, 0.120, 8).tolist(), None, None),
    ("spread", 2):           ([0.030, 0.052, 0.075, 0.097, 0.120], 0, 0.010),
    ("spread", 3):           ([0.040, 0.067, 0.093, 0.120], 0, 0.010),
    ("least-favourable", 1): ([0.020, 0.0515, 0.0535, 0.0555, 0.0575, 0.070, 0.090, 0.110], None, None),
    ("least-favourable", 2): ([0.020, 0.0530, 0.0560, 0.0590, 0.100], 1, 0.0505),
    ("least-favourable", 3): ([0.020, 0.0560, 0.0595, 0.100], 1, 0.0505),
    ("global-null", 1):      (np.linspace(0.053, 0.090, 8).tolist(), None, None),
    ("global-null", 2):      ([0.060, 0.0675, 0.075, 0.0825, 0.090], 0, 0.053),
    ("global-null", 3):      ([0.063, 0.072, 0.081, 0.090], 0, 0.053),
}

def F(ab, x):
    return betainc(ab[0], ab[1], np.clip(np.asarray(x, float), 0.0, 1.0))

def rmax(L, b, v0, v1, arg=False):
    R = np.array([(1 - P1) * F(BETA[k][0], b - v0) + P1 * F(BETA[k][1], b - v1) for k in range(1, L + 1)])
    return (R.max(0), R.argmax(0) + 1) if arg else R.max(0)

def bisect(f, lo, hi, target, it=200):
    for _ in range(it):                                  # f increasing on [lo, hi]
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)

def levels(cfg, d, L):
    base, ref, floor = DESIGN[(cfg, d)]
    b = [bisect(lambda x: float(rmax(L, x, 0.0, 0.0)), 0.0, 1.0, t) for t in base]
    lv = [np.round(b, 4)]
    if d >= 2:
        bref = lv[0][ref]
        both = (d == 3)
        vmax = bisect(lambda v: -float(rmax(L, bref, v if both else 0.0, v)), 0.0, bref, -floor)
        vl = np.round(np.linspace(0.0, vmax, SHAPE[d][1]), 4)
        lv += [vl] * (d - 1)
    return lv

def family_risks(d, lv, L):
    g = np.meshgrid(*lv, indexing="ij")
    b = g[0].ravel()
    v0 = g[1].ravel() if d == 3 else np.zeros_like(b)
    v1 = g[-1].ravel() if d >= 2 else np.zeros_like(b)
    return rmax(L, b, v0, v1, arg=True)

def accept(cfg, d, L, lv):
    R, arg = family_risks(d, lv, L); N = len(R)
    inc = all(np.all(np.diff(l) > 0) for l in lv)
    band = int(np.sum((R > 0.050) & (R <= 0.060)))
    ok = {"spread": R.min() >= 0.008 and R.max() <= 0.125,
          "least-favourable": band >= N / 2,
          "global-null": R.min() > 0.050}[cfg]
    bind = [int(np.sum(arg == 1)), int(np.sum(arg == 2))] if L == 2 else [N, 0]
    bind_ok = (L == 1) or cfg == "global-null" or min(bind) >= N / 4
    return dict(ok=bool(ok and inc and bind_ok), increasing=bool(inc), bind=bind, min=float(R.min()),
                max=float(R.max()), band=band, nulls=int(np.sum(R > 0.05)), N=N)

if __name__ == "__main__":
    res = {}
    for cfg in ("spread", "least-favourable", "global-null"):
        for d in (1, 2, 3):
            for L in (1, 2):
                lv = levels(cfg, d, L); a = accept(cfg, d, L, lv)
                res[f"{cfg}|d={d}|L={L}"] = {"levels": [l.tolist() for l in lv], **a}
                print(f"{cfg:17s} d={d} L={L} {'PASS' if a['ok'] else 'FAIL'}  min={a['min']:.4f} "
                      f"max={a['max']:.4f} band={a['band']}/{a['N']} nulls={a['nulls']} bind={a['bind']}  "
                      f"b={lv[0].tolist()}" + (f" v={lv[1].tolist()}" if d >= 2 else ""))
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "mc_generator_params.json", "w"), indent=1)
