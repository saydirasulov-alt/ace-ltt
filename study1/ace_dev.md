# ACE-LTT dev run — 2026-09-23T10:32:00Z

registration 2026-09-23T10:31:20Z · code `db3c5a7600c7` · cover `030f725011a3`
split: 1797 select / 4192 certify units

## Thresholds chosen on the select fold (lambda = 0)

| acquisition | t_low | t_high | b = A | select risk | objective |
|---|---|---|---|---|---|
| crop | 0.1170 | 0.3816 | 0.6500 | 0.0493 | 0.1511 |
| overlay | 0.1170 | 0.3816 | 0.5000 | 0.0434 | 0.1230 |

## Certification

delta = 0.1, M = 2, delta_m = 0.05, 0.05

A member is tested only if every safer member of its chain was rejected (fail-safe path).

| policy | fire risk | smoke risk | p_joint | delta_m | certified |
|---|---|---|---|---|---|
| crop|lambda=1 | 0.0380 | 0.0370 | 0.11383 | 0.0500 | no |
| crop|lambda=0.8 | — | — | — | — | not tested (sequence stopped earlier) |
| crop|lambda=0.6 | — | — | — | — | not tested (sequence stopped earlier) |
| crop|lambda=0.4 | — | — | — | — | not tested (sequence stopped earlier) |
| crop|lambda=0.2 | — | — | — | — | not tested (sequence stopped earlier) |
| crop|lambda=0 | — | — | — | — | not tested (sequence stopped earlier) |
| overlay|lambda=1 | 0.0312 | 0.0327 | 0.01421 | 0.0500 | YES |
| overlay|lambda=0.8 | 0.0312 | 0.0327 | 0.01421 | 0.0500 | YES |
| overlay|lambda=0.6 | 0.0312 | 0.0327 | 0.01421 | 0.0500 | YES |
| overlay|lambda=0.4 | 0.0312 | 0.0332 | 0.01421 | 0.0500 | YES |
| overlay|lambda=0.2 | 0.0314 | 0.0343 | 0.03345 | 0.0500 | YES |
| overlay|lambda=0 | 0.0323 | 0.0362 | 0.07125 | 0.0500 | no |

## Operating point

chosen: **overlay|lambda=0.2** (utility 0.5263 over 5 certified candidates)
selection-side on the certify fold: FA 0.0399, calls 0.1269

*the operating point is chosen by minimising this utility INSIDE the certified set, which does not weaken the risk guarantee - the policy is already in the simultaneously certified set. But the FA and calls values used for that choice are measured on the certify fold and are therefore NOT an unbiased final performance estimate; they are reported as selection-side quantities. Final FA/calls performance is measured once on the sealed test.*

## Diagnostics (never constrained)

stratum-restricted risks at delta_m = 0.05 need n_min = 59 units:
- `fire|S`: 4 certify units → NOT certifiable (clause C3 would refuse)
- `smoke|S`: 12 certify units → NOT certifiable (clause C3 would refuse)

not measured on these agent CSVs; the measured payload lives in the acquisition-ladder outputs. No payload claim is made in this study.
