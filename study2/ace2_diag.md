# Study 2 — paired post-registration diagnostic (2026-09-24T04:39:55Z)

**paired post-registration diagnostic; descriptive only, no p-value, no decision rule**

policy `overlay|b=0.25` from registration 2026-09-24T04:29:18Z (v2.3), run 2026-09-24T04:29:21Z
same certify units (4192), same t_high = 0.3601, same detector. Edge-only = same (t_high); t_low collapsed to t_high so nothing escalates.

## Event-miss risk (unit averaged) with raw event counts

| risk | cascade | edge-only | difference | cascade missed / total | edge missed / total |
|---|---|---|---|---|---|
| fire | 0.0222 | 0.0450 | -0.0228 | 109 / 2626 | 168 / 2626 |
| smoke | 0.0180 | 0.0416 | -0.0237 | 56 / 3330 | 127 / 3330 |

## Frame-level rates with raw numerators

| quantity | cascade | edge-only | difference |
|---|---|---|---|
| false alarm | 0.0738 (287/3891) | 0.0439 (171/3891) | +0.0298 |
| cloud calls | 0.0715 (789/11037) | 0.0000 (0/11037) | +0.0715 |
| hand-off | 0.0000 (0) | 0.0000 (0) | +0.0000 |

## Paired per-unit differences (cascade − edge-only)

| risk | mean | 90% interval | units worse | units better | units |
|---|---|---|---|---|---|
| fire | -0.02282 | [-0.03010, -0.01584] | 0 | 31 | 1141 |
| smoke | -0.02365 | [-0.03178, -0.01631] | 0 | 31 | 1043 |

*Intervals are a descriptive cluster (unit) bootstrap on a policy chosen after the certify fold was seen. They are POST-SELECTION: not a confidence statement about its risk, and not a test. No p-value and no decision rule appear in this diagnostic.*

*the registered dev probe measured edge_LTT at FA 0.0505 with zero calls, on a different fold and different thresholds; the same-threshold edge-only column here is the comparison*
