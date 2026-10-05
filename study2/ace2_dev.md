# ACE-LTT Study 2 (b ladder) — 2026-09-24T04:29:21Z

**exploratory / method-development; NOT a confirmatory guarantee on fresh data**

registration 2026-09-24T04:29:18Z · spec v2.3 · code `a73364122806` · cover `80cc3c33aeff`
split: 1797 select / 4192 certify units

## Frozen family (registered from the select fold)

| acquisition | t_low | t_high | A | b ladder (safest first) | select-fold joint slack |
|---|---|---|---|---|---|
| crop | 0.2012 | 0.3601 | 0.4500 | 10 members, b 0.000..0.450 | BOUNDARY_SPANNING (+0.0092 .. -0.0049) |
| overlay | 0.2012 | 0.3601 | 0.2500 | 6 members, b 0.000..0.250 | BOUNDARY_SPANNING (+0.0092 .. -0.0005) |

## Certification

delta = 0.1, M = 2, delta_m = 0.05, 0.05

A member is tested only if every safer member of its chain was rejected (fail-safe path).

| policy | fire risk | smoke risk | p_joint | delta_m | certified |
|---|---|---|---|---|---|
| crop|b=0 | 0.0166 | 0.0130 | 0.00000 | 0.0500 | YES |
| crop|b=0.05 | 0.0166 | 0.0130 | 0.00000 | 0.0500 | YES |
| crop|b=0.1 | 0.0166 | 0.0130 | 0.00000 | 0.0500 | YES |
| crop|b=0.15 | 0.0166 | 0.0130 | 0.00000 | 0.0500 | YES |
| crop|b=0.2 | 0.0166 | 0.0130 | 0.00000 | 0.0500 | YES |
| crop|b=0.25 | 0.0167 | 0.0130 | 0.00000 | 0.0500 | YES |
| crop|b=0.3 | 0.0190 | 0.0159 | 0.00000 | 0.0500 | YES |
| crop|b=0.35 | 0.0200 | 0.0169 | 0.00000 | 0.0500 | YES |
| crop|b=0.4 | 0.0227 | 0.0198 | 0.00001 | 0.0500 | YES |
| crop|b=0.45 | 0.0256 | 0.0217 | 0.00012 | 0.0500 | YES |
| overlay|b=0 | 0.0166 | 0.0130 | 0.00000 | 0.0500 | YES |
| overlay|b=0.05 | 0.0176 | 0.0130 | 0.00000 | 0.0500 | YES |
| overlay|b=0.1 | 0.0179 | 0.0141 | 0.00000 | 0.0500 | YES |
| overlay|b=0.15 | 0.0204 | 0.0151 | 0.00000 | 0.0500 | YES |
| overlay|b=0.2 | 0.0213 | 0.0151 | 0.00000 | 0.0500 | YES |
| overlay|b=0.25 | 0.0222 | 0.0180 | 0.00001 | 0.0500 | YES |

## Operating point

chosen: **overlay|b=0.25** (utility 0.8091 over 16 certified candidates)
selection-side on the certify fold: FA 0.0738, calls 0.0715, hand-off 0.0000

*the operating point is chosen inside the certified set, which does not weaken the guarantee; but FA, calls and hand-off measured on the certify fold are selection-side quantities, not an unbiased performance estimate. With b < A the operator tier is non-empty, so hand-off load is priced by the utility.*

## Sec. 4.5 capacity

| loss | g* | design Γ (registered) | realized Γ (diagnostic) | design status |
|---|---|---|---|---|
| fire | 0.0141 | 0.0216 | 0.0284 | MARGIN_CAPABLE |
| smoke | 0.0145 | 0.0294 | 0.0286 | MARGIN_CAPABLE |

*a capacity recomputed on the certify fold is DIAGNOSTIC ONLY and never accepts a family, changes a chain or changes an order*

## Diagnostics (never constrained)

- `fire|S`: 4 certify units, n_min = 59 → NOT certifiable (clause C3 would refuse)
- `smoke|S`: 13 certify units, n_min = 59 → NOT certifiable (clause C3 would refuse)
