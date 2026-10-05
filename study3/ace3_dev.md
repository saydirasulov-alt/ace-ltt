# ACE-LTT Study 3 — prospective stress test of the Sec. 4.5 extension — 2026-09-28T02:49:23Z

**prospective / pre-registered on units never read by Studies 1-2; the P-4.5 predictions are finite-sample predictions evaluated once, carrying no p-value and no population claim**

R1 2026-09-28T02:09:58Z · spec v2.4.3 · code `aa461f6b8f1f` · branch `full`
n_sel {'fire': 488, 'smoke': 454} → n_ref {'fire': 1138, 'smoke': 1059} → n_cert {'fire': 1130, 'smoke': 1114}
r*_ref {'fire': 0.03427, 'smoke': 0.03399} · r*_cert {'fire': 0.03451, 'smoke': 0.03411}

## Registered cover (per arm; arms are never mixed)

| arm | M | δ_arm | δ_m | chains |
|---|---|---|---|---|
| B | 2 | 0.05 | 0.025, 0.025 | 15 members; 12 members |
| L | 2 | 0.05 | 0.025, 0.025 | 6 members; 6 members |

## Certification

### arm B (δ = 0.05)

| policy | fire | smoke | p_joint | δ_m | certified |
|---|---|---|---|---|---|
| B|crop|b=0 | 0.0076 | 0.0048 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.05 | 0.0076 | 0.0048 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.1 | 0.0076 | 0.0048 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.15 | 0.0076 | 0.0048 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.2 | 0.0076 | 0.0048 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.25 | 0.0105 | 0.0070 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.3 | 0.0117 | 0.0101 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.35 | 0.0138 | 0.0128 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.4 | 0.0160 | 0.0178 | 0.00000 | 0.0250 | YES |
| B|crop|b=0.45 | 0.0219 | 0.0277 | 0.00045 | 0.0250 | YES |
| B|crop|b=0.5 | 0.0255 | 0.0326 | 0.01158 | 0.0250 | YES |
| B|crop|b=0.55 | 0.0308 | 0.0380 | 0.11620 | 0.0250 | no |
| B|crop|b=0.6 | — | — | — | — | not tested (sequence stopped earlier) |
| B|crop|b=0.65 | — | — | — | — | not tested (sequence stopped earlier) |
| B|crop|b=0.7 | — | — | — | — | not tested (sequence stopped earlier) |
| B|overlay|b=0 | 0.0076 | 0.0048 | 0.00000 | 0.0250 | YES |
| B|overlay|b=0.05 | 0.0095 | 0.0048 | 0.00000 | 0.0250 | YES |
| B|overlay|b=0.1 | 0.0138 | 0.0066 | 0.00000 | 0.0250 | YES |
| B|overlay|b=0.15 | 0.0145 | 0.0115 | 0.00000 | 0.0250 | YES |
| B|overlay|b=0.2 | 0.0175 | 0.0172 | 0.00000 | 0.0250 | YES |
| B|overlay|b=0.25 | 0.0198 | 0.0239 | 0.00003 | 0.0250 | YES |
| B|overlay|b=0.3 | 0.0207 | 0.0284 | 0.00084 | 0.0250 | YES |
| B|overlay|b=0.35 | 0.0237 | 0.0308 | 0.00440 | 0.0250 | YES |
| B|overlay|b=0.4 | 0.0255 | 0.0318 | 0.00723 | 0.0250 | YES |
| B|overlay|b=0.45 | 0.0294 | 0.0368 | 0.08382 | 0.0250 | no |
| B|overlay|b=0.5 | — | — | — | — | not tested (sequence stopped earlier) |
| B|overlay|b=0.55 | — | — | — | — | not tested (sequence stopped earlier) |

### arm L (δ = 0.05)

| policy | fire | smoke | p_joint | δ_m | certified |
|---|---|---|---|---|---|
| L|crop|lam=1 | 0.0105 | 0.0070 | 0.00000 | 0.0250 | YES |
| L|crop|lam=0.8 | 0.0105 | 0.0070 | 0.00000 | 0.0250 | YES |
| L|crop|lam=0.6 | 0.0105 | 0.0070 | 0.00000 | 0.0250 | YES |
| L|crop|lam=0.4 | 0.0105 | 0.0070 | 0.00000 | 0.0250 | YES |
| L|crop|lam=0.2 | 0.0105 | 0.0070 | 0.00000 | 0.0250 | YES |
| L|crop|lam=0 | 0.0105 | 0.0071 | 0.00000 | 0.0250 | YES |
| L|overlay|lam=1 | 0.0198 | 0.0239 | 0.00003 | 0.0250 | YES |
| L|overlay|lam=0.8 | 0.0198 | 0.0239 | 0.00003 | 0.0250 | YES |
| L|overlay|lam=0.6 | 0.0198 | 0.0239 | 0.00003 | 0.0250 | YES |
| L|overlay|lam=0.4 | 0.0198 | 0.0239 | 0.00003 | 0.0250 | YES |
| L|overlay|lam=0.2 | 0.0198 | 0.0240 | 0.00003 | 0.0250 | YES |
| L|overlay|lam=0 | 0.0198 | 0.0273 | 0.00045 | 0.0250 | YES |

## P-4.5

**INFORMATIVE** — as registered at R1; never recomputed

T: 8/8 pairs · S: 4/6 capable pairs

| (C,k) | J | Γ_sel | h*_ref | CAPABLE | sp_cert | T | S | N | X | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| B|crop|fire | 15 | 0.0554 | 0.0157 | yes | 0.0352 | ✓ | ✓ | — | ✓ | — |
| B|crop|smoke | 15 | 0.0684 | 0.0160 | yes | 0.0493 | ✓ | ✓ | — | ✓ | — |
| B|overlay|fire | 12 | 0.0554 | 0.0157 | yes | 0.0278 | ✓ | ✓ | — | ✓ | — |
| B|overlay|smoke | 12 | 0.0684 | 0.0160 | yes | 0.0412 | ✓ | ✓ | — | ✓ | — |
| L|crop|fire | 6 | 0.0021 | 0.0157 | no | 0.0000 | ✓ | — | ✓ | ✓ | — |
| L|crop|smoke | 6 | 0.0177 | 0.0160 | yes | 0.0001 | ✓ | ✗ | — | ✓ | — |
| L|overlay|fire | 6 | 0.0021 | 0.0157 | no | 0.0000 | ✓ | — | ✓ | ✓ | — |
| L|overlay|smoke | 6 | 0.0177 | 0.0160 | yes | 0.0033 | ✓ | ✗ | — | ✓ | — |

*N is derived and never counted as evidence; X is secondary and confounded. T, S and N read the registered h\*_ref and CAPABLE bits; only X reads r\*_cert.*

## Operating point (reported per arm; never pooled)

- arm B: **B|crop|b=0.5** (utility 2.4465 over 20 certified) — FA 0.0464, calls 0.1422, hand-off 0.0920
- arm L: **L|overlay|lam=0** (utility 0.7121 over 12 certified) — FA 0.0330, calls 0.1422, hand-off 0.0120

*FA, calls and hand-off measured on the certify fold are selection-side quantities, not an unbiased performance estimate*

## Scope

- the prospective fold is fresh with respect to the registered unit split, but it is not a new-domain evaluation: 88.6% of calibration units come from the dominant source, and the smallest source contributes only two units
- allowed: an independent fresh-fold test of the Sec. 4.5 extension; evaluated on units not used to design the family
- never written: independent of the ACE-LTT certification result — they are the same units
