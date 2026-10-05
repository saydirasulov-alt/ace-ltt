# ACE-LTT — provenance record (Studies 1 and 2)

Reference document for the reproducibility section. Every hash below is a full SHA-256 as written by the
tooling; nothing here is abbreviated except where a short prefix is marked as such.

## 0. Shared inputs (identical for both studies)

| object | sha256 |
|---|---|
| `protocol.yaml` | `a7a654d672a0a0d251e8da39082d4d7c6b41743912a3b782028c8baa686fe3ea` |
| `splits.csv` | `362d8689b193d61b3f97ff08ea1acd1a0956824cbbea3544f96921940994d27f` |
| detector CSV `det_yolo26-base_s0_nms-none.csv` | `82997578e96f7ebe556c2e9f29bfecac27faa9db5761bce79636d3529af30d94` |
| detector weights `runs/fsc_v02/yolo26-base_s0/weights/best.pt` | `b4dd9d7edf515401524af23d40c392bfbc600bf005c1188c5453909e4d640230` |

`protocol.yaml` reached this state on 2026-09-23 when `make_protocol fill-weights` copied the weights digest
from the detector meta (`a21d853b2699f1ad…` → `a7a654d672a0a0d2…`, snapshot
`protocol.yaml.a21d853b.snapshot` kept). Every registration hash below was taken after that change.

Dev split (5989 units, 13937 frames) is shared; the select/certify partition differs by registered seed:
Study 1 seed 20260923 (1797 / 4192 units), Study 2 seed 20260924 (1797 / 4192 units).

## 1. Packages, and which one executed what

| package | zip sha256 | analysis_code sha256 | role |
|---|---|---|---|
| `cascade_v2_19.zip` | `335753548a8b4b7915b2420ef14180276040daaabe884547b3574cf272f076e2` | `db3c5a7600c7d9c82a13a7f5733909bd760136a614fbdf4f380725ac7fe582d1` | **executed Study 1** (spec v2.2) |
| `cascade_v2_3_0.zip` | `288a982e14d93b282bd1cd4b8e9729b6a9f12beafd92fc487bbdc83acd8aa07c` | `62beabf7d75d1b35634af7e085efc34e49ec62a10a04b161d6861f976f887901` | first v2.3; **refused Study-2 design audit 2A** |
| `cascade_v2_3_1.zip` | `a62c91821d40eb3360a5406904d5dd9130ef90c5f736657410f236d9db32ffd5` | `a73364122806a9e4bfa230e2677b829d34c8385c444e58808ede20092b86a4a6` | §4.5 generalised to the select bar, §4.6 added |
| `cascade_v2_3_2.zip` | `8b76f0125476b958857502fe64bb8592189b1d931779dde13fee24317d07221c` | `a73364122806a9e4bfa230e2677b829d34c8385c444e58808ede20092b86a4a6` | **registered and executed Study 2** |
| `cascade_v2_3_3.zip` | `3072e13e9cbdda2551ee28d45361900770990bbbc368afa4a8792f81f613a7a9` | `a73364122806a9e4bfa230e2677b829d34c8385c444e58808ede20092b86a4a6` | **ran the paired diagnostic** |

v2.3.2 and v2.3.3 carry the *same* `analysis_code_sha256` as v2.3.1: the OpenCV selftest skip touches
`selftest.py` and the diagnostic lives in `ace2_diag.py`, neither of which is in `ANALYSIS_CODE`. That is
why both registrations remain verifiable against the installed tree today.

**Reproduction:** Study 1 from `cascade_v2_19.zip`, Study 2 from `cascade_v2_3_2.zip`. Installation is a
clean swap (extract to a staging directory, `code-hash --verify-tree` there, then two `mv`s); the previous
tree is moved to a timestamped backup, never deleted. Every install in this chain verified the ZIP digest
first and the runtime tree against `MANIFEST.sha256` after.

## 2. Study 1 — `ace_ltt_dev`, spec v2.2

| | |
|---|---|
| registered | 2026-09-23T10:31:20Z |
| cover_id | `030f725011a36785cdca2b69b755d6b25248a024e0f45d6bbc631038bdc5c0e6` |
| family | λ ladder {1.0 … 0.0} per acquisition {crop, overlay}, M = 2, δ_m = 0.05 |
| run | 2026-09-23T10:32:00Z |
| outcome | `overlay\|λ ∈ {1.0 … 0.2}` certified (5 of 16); the crop chain stopped at its safest member (p_joint 0.1138) |
| operating point | `overlay\|λ=0.2` |

Preconditions at registration: no event spanned two units; audit 1138 fire / 1079 smoke certify units with
full membership digests; checklist 6/6 (the v2.2 checklist had six items).

## 3. Design audit 2A — Study 2's first proposed rule, REFUSED

Run under `cascade_v2_3_0` (code `62beabf7…`) on 2026-09-24. The proposed rule minimised utility subject to
`R̂_select ≤ r*` and checked capacity afterwards. It selected a narrow, low-`t_high` band in which almost
every positive event is edge-alarmed, leaving `b` nearly inert:

```
fire   Γ = 0.0072   g* = 0.0141   MARGIN_INCAPABLE
smoke  Γ = 0.0127   g* = 0.0145   MARGIN_INCAPABLE
```

Nothing was registered. **Provenance gap, stated explicitly:** `ace2_register` did not yet write a
`design_audit_<UTC>.json` on refusal — that behaviour was added in v2.3.1, after this audit. The record of
2A is therefore this document and the session transcript, not a machine-written file. From v2.3.1 onward a
refusal writes its own audit record.

Two corrections followed, both to the *criterion*, neither to the data:
- §4.5's required margin generalised to the declared select bar, `h*(bar) = max(0, bar − r*)`; with
  `bar = r*` it is 0, so the capacity condition is vacuous for a rule of this shape — correctly so.
- §4.6 added: `BOUNDARY_SPANNING` on the select fold, `max_π s_joint ≥ 0` and `min_π s_joint < 0` with
  `s_k(π) = r*_k − R̂^sel_k(π)`. The earlier claim that `Γ < g*` implies the chain cannot straddle `r*` was
  **withdrawn**: with `r* = 0.036` and `Γ = 0.007`, members at 0.034 and 0.040 differ by less than Γ and
  still fall on opposite sides.

## 4. Study 2 — `ace_ltt_study2_bladder`, spec v2.3

| | |
|---|---|
| registered | 2026-09-24T04:29:18Z |
| cover_id | `80cc3c33aeff4a15616f303cdf5c63e27262956ee36e41562ab2ab64b6924f2e` |
| declaration_sha256 | `57b224c13ddeef8ae4fed8bb38be316053b749fb313eff841884d4a04e7e2290` |
| status | exploratory / method-development; **not** a confirmatory guarantee on fresh data |
| family | b ladder per acquisition, λ frozen at 0.4, M = 2, δ_m = 0.05; crop t=(0.2012, 0.3601) A=0.45, 10 members; overlay t=(0.2012, 0.3601) A=0.25, 6 members |
| pre-registration spanning | crop +0.0092 … −0.0049, overlay +0.0092 … −0.0005, both `BOUNDARY_SPANNING` |
| checklist | 8/8 |
| run | 2026-09-24T04:29:21Z — **16/16 certified**, worst p_joint 0.00012 |
| operating point | `overlay\|b=0.25`, hand-off 0.0000 |
| paired diagnostic | 2026-09-24T04:39:55Z |

The declaration digest is not a field of the record; it is recomputed from the stored declaration with

```bash
python -c "import json,hashlib,sys; d=json.load(open(sys.argv[1]))['declaration']; \
print(hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest())" \
  ~/Norqobil/runs/cascade/ace2.registration.json
```

Thresholds, b ladders, chains, cover id and design-time capacity were all frozen **at registration** from
select-fold data; `ace2_dev` re-derived each of them and would have refused on any mismatch.

## 5. The paired post-registration diagnostic

Descriptive only: no p-value, no decision rule, no re-selection. Same 4192 certify units, same
`t_high = 0.3601`, same detector. Edge-only is the same policy object with `t_low` collapsed to `t_high`,
so nothing escalates.

| quantity | cascade `overlay\|b=0.25` | edge-only, same t_high | difference |
|---|---|---|---|
| fire event-miss risk | 0.0222 (109/2626) | 0.0450 (168/2626) | −0.0228 |
| smoke event-miss risk | 0.0180 (56/3330) | 0.0416 (127/3330) | −0.0237 |
| false alarm | 0.0738 (287/3891) | 0.0439 (171/3891) | +0.0298 |
| cloud calls | 0.0715 (789/11037) | 0 | +0.0715 |
| hand-off | 0 | 0 | 0 |

Paired per-unit: fire −0.02282 [−0.03010, −0.01584], smoke −0.02365 [−0.03178, −0.01631]; **0 units worse**,
31 better, of 1141 / 1043. Intervals are a descriptive cluster bootstrap on a policy chosen after the
certify fold was seen — post-selection, not inferential.

**Do not confuse two numbers.** The same-threshold comparator is **0.0439**. The figure **0.0505** is
`edge_LTT` from the earlier registered dev probe, on a different fold at different thresholds; it is
background context only and must not be used as the baseline in this comparison.

Descriptive, and the sharpest observation in the run: edge-only at these thresholds sits at 0.0450 / 0.0416
— under α = 0.05 but above r* = 0.0359 / 0.0355 — so its HB p-values would be 0.733 and 0.374 against
δ_m = 0.05, while the certified cascade's are 6·10⁻⁶ and < 10⁻⁶. At these thresholds on this fold the
cascade is certifiable and the edge-only baseline is not. Edge-only was **not** a registered candidate, so
this is stated descriptively; a confirmatory version requires it declared in a future registration.

## 6. The three layers, as the paper should separate them

1. **Certification result.** 16 of 16 policies certified under the registered pooled-risk procedure,
   FWER ≤ δ = 0.10 across M = 2 chains at δ_m = 0.05.
2. **Operating-point result.** The registered utility (`calls` 1.0, `fa` 10.0, `handoff` 20.0) selected
   `overlay|b=0.25`, with zero hand-off: the certified set contained operator-assisted policies and the
   deployment utility chose the operator-free endpoint.
3. **Post-registration paired diagnostic.** Against a same-threshold edge-only comparator the selected
   cascade reduced fire and smoke event-miss risk by ≈ 2.3 percentage points absolute (roughly halving it),
   with no unit worse off, at the cost of +0.0298 false alarms and 7.15 % cloud calls.

The empirical lesson the two studies carry together:

> certification validity ≠ family informativeness ≠ deployment desirability

Study 1: the chain coordinate could not move the constrained risk enough (λ acts only inside a stratum too
small to span the margin) — the sequence stopped at its first member. Study 2: the coordinate did move the
risk and the chain spanned the boundary *by select design*, yet a fold-to-fold shift the size of g* left
every member inside the boundary, so the sequence was *non-discriminating on the certify fold* — a different
statement from "not informative". And the utility, a third and separate pre-registered object, selected a
policy that used none of the mechanism the chain was built around.

## 7. Open, carried forward

- **v2.4 `ROBUST_BOUNDARY_SPANNING`** beside `BOUNDARY_SPANNING`: `U_safe < r*` and `L_aggressive > r*` with
  simultaneous one-sided bounds over the full select candidate family, or a separate design/select split.
  Not applied to either registered study.
- **`handoff = 20.0` is not an ACE default** — it belongs to a study's pre-registered utility. An
  operator-layer study should constrain it (`min (FA, calls) s.t. handoff ≤ H₀`) or report the
  FA/calls/hand-off Pareto front over the certified set.
- **Edge-only as a declared candidate**, so the §5 observation becomes confirmatory rather than descriptive.
- **Sealed test** untouched. `fire|S` (4 units) and `smoke|S` (13) remain below n_min = 59 and stay
  diagnostic; the guarantee is silent about the stratum, on the record.
- No Study 3 is opened.
