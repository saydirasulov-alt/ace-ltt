# ACE-LTT on a generic synthetic instance (2026-09-24T07:24:55Z)

**Illustration, not evidence.** The generator is declared and seeded; its constants were chosen so that the instance exhibits a truncating chain, a non-truncating chain and the refusal paths. No number here is a measurement of any real system.

generator seed 20260925, 900 units, 5392 frames; band (0.2, 0.36), A = 0.55, b ladder [0.0, 0.15, 0.3, 0.45], lambda_S = 0

alpha = 0.05, beta = 0.05, delta = 0.1; M = 2, delta_m = 0.05, n_min = 59; cover `c5eae7965d30c72977afbb74eaf1c21c433d224ed4b5b8ee8fc778b8492e3465`

audit population: 415 miss units, 630 budget units, membership `388adb8c929aa5c8...` (identical for all 8 candidates)

C4, discharged statically: every escalated frame of mode `a` transmits exactly 22 KiB (`a1`), 41 KiB (`a2`) by construction, against the declared transport cap B_max = 48 KiB. For contrast only, the design-fold empirical maximum of C_u is 17.6 (`a1`), 32.8 (`a2`) KiB - a sample statement, never the cap. Note that it lies strictly BELOW the construction bound: the design fold never observed the worst case, which is exactly why a sample maximum cannot discharge a forall-x cap.

## 1 Family design, select fold only

| candidate | R_sel(miss) | Z_sel(budget) | joint slack | FA | calls | hand-off |
|---|---|---|---|---|---|---|
| `a1|b=0` | 0.0117 | 0.0000 | +0.00335 | 0.0823 | 0.2663 | 0.1690 |
| `a1|b=0.15` | 0.0117 | 0.0000 | +0.00335 | 0.0823 | 0.2663 | 0.1471 |
| `a1|b=0.3` | 0.0117 | 0.0000 | +0.00335 | 0.0823 | 0.2663 | 0.1064 |
| `a1|b=0.45` | 0.0166 | 0.0000 | -0.00152 | 0.0823 | 0.2663 | 0.0432 |
| `a2|b=0` | 0.0117 | 0.0222 | +0.00335 | 0.0814 | 0.2663 | 0.1854 |
| `a2|b=0.15` | 0.0146 | 0.0222 | +0.00043 | 0.0814 | 0.2663 | 0.1617 |
| `a2|b=0.3` | 0.0224 | 0.0222 | -0.00737 | 0.0814 | 0.2663 | 0.1167 |
| `a2|b=0.45` | 0.0400 | 0.0222 | -0.02491 | 0.0814 | 0.2663 | 0.0450 |

| chain | Gamma(S) design | select bar | required h* | status | spanning | max/min joint slack |
|---|---|---|---|---|---|---|
| `a1` | 0.09357 | 0.01505 | 0.00000 | MARGIN_CAPABLE | BOUNDARY_SPANNING | +0.00335 / -0.00152 |
| `a2` | 0.09357 | 0.01505 | 0.00000 | MARGIN_CAPABLE | BOUNDARY_SPANNING | +0.00335 / -0.02491 |

pre-freeze checklist: **8/8**

| item | ok |
|---|---|
| loss_by_loss_monotonicity | yes |
| deterministic_chain_cover | yes |
| predeclared_delta_plan | yes |
| policy_independent_audit_membership | yes |
| fixed_payload_semantics | yes |
| weights_hash_populated | yes |
| chain_margin_capacity | yes |
| chain_boundary_spanning | yes |

## 2 Certification, certify fold

r* = a1 0.02651, a2 0.02651; g* = a1 0.02349, a2 0.02349

| candidate | R_cert(miss) | Z_cert(budget) | p_joint | certified | utility |
|---|---|---|---|---|---|
| `a1|b=0` | 0.0213 | 0.0000 | 0.00728 | yes | 1.7317 |
| `a1|b=0.15` | 0.0213 | 0.0000 | 0.00728 | yes | 1.6441 |
| `a1|b=0.3` | 0.0237 | 0.0000 | 0.0167 | yes | 1.4968 |
| `a1|b=0.45` | 0.0269 | 0.0000 | 0.062 | no | 1.2588 |
| `a2|b=0` | 0.0213 | 0.0238 | 0.00728 | yes | 1.7818 |
| `a2|b=0.15` | 0.0233 | 0.0238 | 0.0167 | yes | 1.6953 |
| `a2|b=0.3` | 0.0285 | 0.0238 | 0.0675 | no | 1.5288 |
| `a2|b=0.45` | 0.0466 | 0.0238 | - | no | 1.2395 |

reasons: none; operating point **`a1|b=0.3`** (utility 1.4968 over 5 certified candidates)

## 3 Ordering: the strict-front counterexample

b grid 0 .. 0.55 step 0.01 (56 candidates), cost = hand-off rate on the select fold (operator load). Select-fold tie groups (equal R_sel, unequal cost):

- R_sel = 0.011696: `a2|b=0.00`, `a2|b=0.01`, `a2|b=0.02`, `a2|b=0.03`, `a2|b=0.04`, `a2|b=0.05`, `a2|b=0.06`, `a2|b=0.07`, `a2|b=0.08`, `a2|b=0.09`, `a2|b=0.10`
- R_sel = 0.014620: `a2|b=0.11`, `a2|b=0.12`, `a2|b=0.13`, `a2|b=0.14`, `a2|b=0.15`, `a2|b=0.16`, `a2|b=0.17`, `a2|b=0.18`, `a2|b=0.19`, `a2|b=0.20`
- R_sel = 0.016569: `a2|b=0.21`, `a2|b=0.22`, `a2|b=0.23`, `a2|b=0.24`, `a2|b=0.25`
- R_sel = 0.019493: `a2|b=0.26`, `a2|b=0.27`, `a2|b=0.28`, `a2|b=0.29`
- R_sel = 0.022417: `a2|b=0.30`, `a2|b=0.31`, `a2|b=0.32`
- R_sel = 0.038012: `a2|b=0.35`, `a2|b=0.36`, `a2|b=0.37`, `a2|b=0.38`, `a2|b=0.39`, `a2|b=0.40`, `a2|b=0.41`, `a2|b=0.42`
- R_sel = 0.039961: `a2|b=0.43`, `a2|b=0.44`, `a2|b=0.45`, `a2|b=0.46`, `a2|b=0.47`, `a2|b=0.48`, `a2|b=0.49`
- R_sel = 0.045809: `a2|b=0.50`, `a2|b=0.51`
- R_sel = 0.050682: `a2|b=0.53`, `a2|b=0.54`, `a2|b=0.55`

Whole grid: ACE (weak front + inclusion order) certifies **32** of 56 kept candidates; the strict front keeps 12 and certifies **4**.

**The witness.** Tie group at R_sel = 0.022417, tested as its own family (M = 1, delta_m = 0.1, r* = 0.02892):

| member | cost_sel | R_cert | p_joint | kept by strict front |
|---|---|---|---|---|
| `a2|b=0.30` | 0.1167 | 0.0285 | 0.0675 | no |
| `a2|b=0.31` | 0.1094 | 0.0285 | 0.0675 | no |
| `a2|b=0.32` | 0.1082 | 0.0309 | 0.12 | yes (cheapest) |

- ACE family ['a2|b=0.30', 'a2|b=0.31', 'a2|b=0.32'] -> certified **['a2|b=0.30', 'a2|b=0.31']**
- strict front family ['a2|b=0.32'] -> certified **[]**, reasons ['NO_REJECTION']

The strict front is not merely less efficient here: it returns the empty certified set on an instance where the ACE order certifies. The candidates it discarded are risk-equivalent to the one it kept on the fold that chose them, and strictly safer on the fold that tests them.

## 4 Fail-closed refusals

| clause | outcome |
|---|---|
| C3 audit support | declared 12 units < n_min 59 -> ['C3_CERTIFIABILITY_AUDIT', 'NO_REJECTION'], 0 certified |
| C6 no monotonicity lemma | chain sizes [4, 4] -> [1, 1, 1, 1, 1, 1, 1, 1] |
| C7 payload semantics | ['C7_FROZEN_PAYLOAD_SEMANTICS', 'NO_REJECTION'], 0 certified |

without a written lemma every candidate is its own chain: M rises from 2 to 8, delta_m falls from 0.05 to 0.0125, and no fail-safe path exists

## 5 The two diagnostics are declarations, not properties of the data

| chain | select bar / S | Gamma | required h* | status |
|---|---|---|---|---|
| `a1` | bar=r*, S full | 0.09357 | 0.00000 | MARGIN_CAPABLE |
| `a1` | bar=alpha, S full | 0.09357 | 0.03495 | MARGIN_CAPABLE |
| `a1` | bar=alpha, S narrow | 0.01871 | 0.03495 | MARGIN_INCAPABLE |
| `a2` | bar=r*, S full | 0.09357 | 0.00000 | MARGIN_CAPABLE |
| `a2` | bar=alpha, S full | 0.09357 | 0.03495 | MARGIN_CAPABLE |
| `a2` | bar=alpha, S narrow | 0.01871 | 0.03495 | MARGIN_INCAPABLE |

safe-only subfamily ['a1|b=0', 'a2|b=0']: **NOT_SPANNING** (max +0.00335, min +0.00335) - every member on the certifiable side: the sequence has nothing to reject
