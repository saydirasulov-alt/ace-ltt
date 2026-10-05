# ACE-LTT — controlled validation: Monte-Carlo pre-registration

**Status: R0 — frozen 2026-09-30.** This file is the registration. Its SHA-256 is **not** written inside it
(a file cannot carry its own digest); it is recorded in the detached freeze record
`ACE_MC_preregistration_R0.sha256`, which also carries the engine, reference and generator digests. After R0
nothing below may change; any deviation is a dated addendum in a separate file, never an edit of this one.
Same discipline as the Study 3 registration.

**Changes from draft 4 to R0**, from the authors' pre-freeze review; none relaxes a rule or changes a grid:
(1) detached freeze record instead of a self-hash (§11); (2) timing pilot made fully determinate — exact
pilot levels and the exact projection equation (§7); (3) MC-D zero-event bound written as the exact
Clopper–Pearson value (§6); (4) MC-B made explicit for pairs in which $C_{\min}$ is refused (§6); (5) Python,
SciPy and platform versions added to the run manifest (§10). The authors independently recomputed all 18
§12 combinations and the draft-4 digest before approving.

**Changes from draft 3 to draft 4**: the pre-freeze claim→evidence audit of §13 — a claim map (§0b),
eleven decision rules made fully determinate, one generator check added (G1), the provenance decision that
the harness is written only after R0, and the timing pilot moved after R0. No rule was made weaker.

**Changes from draft 1 to draft 2**, forced by the Q1–Q8 answers: engine lineage and digest fixed;
replications instead of "families"; semantic check A0; MC-D split into D1/D2; §12 as a procedure.

**Changes from draft 2 to draft 3**, forced by the Q9–Q14 answers and by the §12 computation. No MC code has
run, and no engine call has touched synthetic data; the only engine calls made so far are data-free
(declaration and `build_chains` on the registered grids).

1. *Lemma removed was mis-bound in draft 2.* `LossSpec(..., antitone_under_alarm_inclusion=True, lemma="")`
   is **refused at declaration** (`ValueError`, checked). A loss without a lemma can only be declared with
   `antitone_under_alarm_inclusion=False`, which is also how `ace_toy` builds its C6 refusal scene. §5 is
   re-bound to that construction; the relation now refuses at line 625, not 635–638.
2. *Path L is required* (Q9): `certify_family` always builds its own cover; `plan` carries only shares.
3. *Two refusal paths, not one* (Q13): step 2 audits every candidate at a provisional uniform share
   $\delta/M$ of the engine's own cover of the **whole** family, before any cover is registered. When that
   fails, nothing survives and the reason is C3 + `NO_REJECTION`, **without** `CHAIN_TERMINAL`. The
   definition of a successful execution (§5) and the §4 tables are corrected accordingly; under draft 2,
   such runs would have been mislabelled successful and would have produced false A2 violations.
4. *The §3 decision rule is the engine's* (Q11), with the policy fields that make it so now fixed.
5. *§12 procedure replaced.* Draft 2's numerical fit with 3-decimal rounding was run once on 2026-09-30 and
   failed acceptance in 15 of 18 combinations, mostly through tied levels after rounding. It was replaced by
   the constructive procedure of §12, which passes all 18. Targets, acceptance criteria and the Beta table
   are unchanged. This replacement is flagged for the authors' explicit approval.

---

## 0. What this study is, and what it is not

Proposition 3 and Corollary 1 are **proved** (supplement M5.5). Theorem 1 is proved. This study proves
nothing. It has four jobs, and each block is labelled by which one it does:

| block | job | kind of result |
|---|---|---|
| **MC-A** | the frozen engine behaves as Proposition 3 says it must | deterministic conformance; any violation is a defect |
| **MC-B** | how much a minimum admissible cover returns relative to larger covers | effect size; no pass/fail |
| **MC-C** | family-wise error of the full procedure against known truth | validity check with a pre-declared failure rule |
| **MC-D** | how the greedy fallback behaves, including under deletion | algorithmic stress test; existence of an event |

A reader should be able to delete MC-B and MC-D and still have the paper's validity story intact; they
report *yield* and *algorithm behaviour*, not the guarantee.

## 0b. Claim → evidence map

Each paper claim that the study touches, the check that bears on it, and what outcome would count against
it. A claim not in this table is not supported by this study.

| paper claim | where | evidence here | counts against the claim |
|---|---|---|---|
| Equality (1): under a successful execution the returned set is the threshold set $\{p_{\mathrm{joint}} \le \delta/M\}$, strict-`>` stop included | M5.5 Prop. 3; main Cor. 1 | A2 (via A6) | any A2 mismatch |
| Inclusion (2): fewer chains never return less | M5.5 Prop. 3; main Cor. 1 | A1 | any A1 violation |
| Remark (a): a smaller $M$ never fails C3 where a larger one passes | M5.5 | A4, A5 | any A4 violation; an A5 disagreement |
| Lemma 2 + A2-O along admissible edges | Lemma 2; A2-O | A3 | any A3 violation |
| The admissibility restriction is not vacuous ($E_{\mathrm{adm}} \subsetneq E_{\mathrm{str}}$ can force $M_{\min} = \lvert\Pi\rvert$) | M5.5 witness | lemma-removed factor, A5 | $M \ne \lvert\Pi\rvert$ under lemma removed |
| Theorem 1: FWER $\le \delta$ for the full procedure | Theorem 1 | MC-C | the MC-C failure rule in any cell |
| Remark (b): the gain from fewer chains is weak and instance-dependent | M5.5 | MC-B (effect size only) | none: MC-B has no pass/fail |
| Remark (c): the greedy cover carries no minimality or deletion-monotonicity guarantee | M5.5 | MC-D | none: a non-occurrence of $E$ is reported, not claimed as a property |
| The frozen engine implements the above | engine v2.4.7 | A0, A2, A6, the MC-D consistency checks | any of them failing |

**Not supported by this study:** validity on real data, any "power" statement, any property of
non-uniform allocation, any statement about the Study 1–3 artifacts, and any statement about the greedy
routine beyond the grids of MC-D.

## 1. Hard constraints inherited from the project

- The harness **calls the frozen engine as a library**: `cascade/ace_ltt.py` from the frozen
  `cascade_v2_4_7` release, recorded by the release record as byte-identical through v2.4.8 and v2.4.9
  (SHA-256 in §11). No engine file is edited, patched or monkey-patched, and no module constant is
  reassigned — in particular `GREEDY_ABOVE = 400` is used as frozen.
- On the server the engine is imported from `~/Norqobil/v249/cascade/`. The copies in `v245b/`, `v247/`
  and `v248/` carry the same SHA-256; `v245/` is not readable by the user account and is not used.
- The harness may call engine functions whose names begin with an underscore (`_min_chain_cover`,
  `_greedy_chain_cover`). Calling a function is library use; it changes no engine behaviour.
- Engine source comments are not evidence. The comment at `ace_ltt.py:1217–1218` (a greedy deletion
  instance, 3 → 4 chains) is not cited, reproduced from, or used to choose any setting here.
- **Provenance order: frozen engine → frozen R0 → harness.** The harness is written only **after** R0 is
  frozen, as a new file under `~/Norqobil/mc_prereg/`, and carries the R0 SHA-256 in its header. No MC code
  is added to `ace_toy.py`, which stays frozen (SHA-256 in §11) as the reference implementation of the
  synthetic-input pattern. The harness imports from `cascade.ace_ltt` only.
- The harness is a **new module outside `ANALYSIS_CODE`**, on the pattern of `cascade.ace_toy`: it reads
  no dataset, no protocol, no detector output and no registration of Studies 1–3, so running it changes no
  registered digest. `ace_toy` is the existence proof that the engine runs on synthetic inputs this way.
- It does **not** touch the sealed test set, does **not** re-run `STAGE=protocol`, does **not** re-run
  `ace3_run.sh`, and trains nothing.
- All inputs and outputs live under `~/Norqobil/mc_prereg/`.

## 2. What the engine supplies and what the harness supplies

The line matters: MC-A is only an implementation check if the parts being checked are the engine's.

| component | supplied by | checked in |
|---|---|---|
| HB $p$-value `hb_pvalue(rhat, n, alpha)` via `_pvalue_hb_unweighted_unit_mean` | **engine** | A, C |
| joint $p$-value `joint_pvalue(obs, losses)` | **engine** | A, C |
| stopping rule `fixed_sequence(pvalues, delta_m)`, strict `>`, returns prefix length | **engine** | A, C |
| chain-relation certificate `certify_chain_relation` (per ordered pair) | **engine** | A, B, D |
| C3 support rule `n_min(alpha, delta)` | **engine** | A, C |
| exact minimum chain cover `_min_chain_cover` (Kuhn matching), via `build_chains` when $\le 400$ | **engine** | A, B, C, D |
| greedy chain cover `_greedy_chain_cover`, via `build_chains` when $> 400$ | **engine** | D |
| full procedure `certify_family` | **engine** | A, C |
| synthetic units, policies, losses, contracts, evaluator | harness, on the `ace_toy` pattern | all |
| alternative covers for MC-A (refinements, random admissible covers) | harness, **validated by the engine's certificate** | A, B |
| true risk $R(\pi)$ | harness, closed form (§3) | C |

## 3. Data-generating process (frozen)

**Units.** $n$ i.i.d. units per replication. Each unit carries exactly one event of each constrained risk
type, so under `exclude_unit` the audit population is $n_k = n$ for every $k$ — common to the family, as A4
requires.

**Strata.** Two declared strata, $\sigma \in \{0, 1\}$ (engine names `S0`, `S1`), with $\Pr(\sigma = 1) = 0.3$;
one stratum per unit, shared by all its events.

**Sampling order (frozen, for reproducibility).** Per replication, with that replication's generator:
$\sigma$ for all $n$ units as `rng.random(n) < 0.3`; then $g_1$ for all units by `rng.beta` with per-unit
parameters; then $g_2$ likewise if $L = 2$. Unit ids are `u0 … u{n-1}`; the audit-membership digest is
SHA-256 of the sorted ids joined by `|`, as `ace_toy.toy_digest` computes it, used on both the contract and
the evaluation side.

**Verifier scores.** For risk $k$ and stratum $\sigma$, the event's verifier score is
$g \sim \mathrm{Beta}(a_{k\sigma}, b_{k\sigma})$, independent across units and across risks.

**Policies.** Fixed acquisition mode, escalation band, $B_0$ and stratification, so every candidate lies
in one comparability group: `Policy(acq="a", t_low=0.0, t_high=1.0, b, A=1.0, veto=(("S0", v0), ("S1", v1)), B0=inf)`.
Both strata are declared on every policy (the relation requires equal `strata`); $v(S0) = 0$ for
$d \le 2$ and $v(S1) = 0$ for $d = 1$. Coordinates vary by the dimension factor $d$:

| $d$ | varying coordinates | grid | $\lvert\Pi\rvert$ | $M_{\min}(G_{\mathrm{str}})$ |
|---|---|---|---|---|
| 1 | response authority $b$ | 8 levels | 8 | 1 |
| 2 | $b$, veto on stratum 1 | $5 \times 5$ | 25 | 5 |
| 3 | $b$, veto on stratum 0, veto on stratum 1 | $4 \times 4 \times 4$ | 64 | 12 |

$M_{\min}$ values are exact (Dilworth via bipartite matching, computed at draft time). $d = 2, 3$ are
genuine multi-coordinate posets of width 5 and 12.

**MC-D production grid.** For arm D2 only: $d = 3$ on an $8 \times 8 \times 8$ grid, $\lvert\Pi\rvert = 512$,
above `GREEDY_ABOVE = 400`, with $M_{\min}(G_{\mathrm{str}}) = 48$ (the largest rank level, which for a
product of chains equals the width). MC-D reads no data, so this grid needs coordinate levels (§12) but no truth
configuration.

**Decision and loss (bound by Q11).** Each event is one frame with edge score $s \equiv 0.5$, so with the
band $[0, 1)$ every event is escalated and none is edge-alarmed. The engine's reference rule
(`decision_sets`, `ace_ltt.py:583–589`) then reads: $g_v = \min(1, g + v(\sigma))$, and the event is in
`response` iff $g_v \ge b$. $A = 1$ affects only `alarm`, which no loss uses. The unit loss is the miss
indicator. Each loss is `miss_loss("m<k>", 0.05, "L-mc-<k>", catch_set="response",
aggregation="unit_any_positive_frame")` (weighting uniform, `exclude_unit`, built-in inference). With one
event and one frame per unit and risk, the three monotone aggregations coincide. No payload loss is
declared, so `payload_semantics=None` is admissible (step 1 requires it only for a payload loss).

**True risk (closed form).** With $F_{k\sigma}$ the Beta CDF,

$$R_k(\pi) = \sum_{\sigma} \Pr(\sigma)\, F_{k\sigma}\!\bigl(b_\pi - v_\pi(\sigma)\bigr),
\qquad F(x) = 0 \text{ for } x \le 0.$$

$R_k$ is non-decreasing in $b$ and non-increasing in each $v(\sigma)$, so it is antitone along
$\succeq_{\mathrm{cert}}$ exactly as Lemma 2 requires, and A2-V holds by construction. The direction
matches the engine's structural test (`ace_ltt.py:623`: $\pi \succeq \pi'$ needs $b_\pi \le b_{\pi'}$,
$A_\pi \le A_{\pi'}$ and `veto_at` of $\pi$ $\ge$ that of $\pi'$ at every index).

**Binding (Q11), closed.** The rule above is the engine's own for this configuration. A0 (§6) checks it
unit by unit against `decision_sets` on every replication.

**Truth configurations.** The Beta parameters are common to all configurations; only the grid levels
differ, so that true risks land as follows (exact parameter values in Appendix §12, fixed before R0):

| config | true risks $R_k(\pi)$ across the family | purpose |
|---|---|---|
| `spread` | roughly uniform over $[0.01, 0.12]$ | typical mix of nulls and non-nulls |
| `least-favourable` | at least half of the family in $(0.050, 0.060]$ | nulls pressed against $\alpha$: the hardest case for FWER |
| `global-null` | every policy above $\alpha$ | FWER $= \Pr(\text{anything certified})$ |

**Number of constrained losses.** $L \in \{1, 2\}$. With $L = 2$ the two risk types have independent scores
and a policy is null if either risk exceeds $\alpha$; this exercises $p_{\mathrm{joint}} = \max_k p_k$. For
$L = 2$ the targets of the table above apply to $\max_k R_k(\pi)$, and §12 additionally requires, under
`spread` and `least-favourable`, each risk type to be the larger one for at least a quarter of the
family, so that the maximum is not decided by one loss throughout.

## 4. Fixed constants

| symbol | value |
|---|---|
| $\delta$ | 0.10 |
| $\alpha$ | 0.05 for every constrained loss |
| $n$-grid | {50, 80, 150, 400, 1000} |
| $d$-grid | {1, 2, 3} |
| $L$-grid | {1, 2} |
| truth configs | {`spread`, `least-favourable`, `global-null`} |
| allocation | uniform, $\delta_m = \delta / M$ |
| admissibility factor | {lemma present, lemma removed} — see §5 |

**Pre-computed support thresholds.** $n_{\min}(\alpha, \delta/M) = \lceil \log(\delta/M) / \log(1-\alpha) \rceil$:

| $M$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $n_{\min}$ | 45 | 59 | 67 | 72 | 77 | 80 | 83 | 86 | 88 | 90 | 92 | 94 |

**Cells refused by construction** (registered now, so they are predictions, not findings). Two
engine paths exist (Q13): **R2** = every candidate refused at step 2, at the provisional share
$\delta/M_{\mathrm{prov}}$ with $M_{\mathrm{prov}}$ the engine's own cover of the whole family (reason C3,
then `NO_REJECTION`, no `CHAIN_TERMINAL`); **T4** = step 2 passes, but the re-check at the executed share
$\delta/M(C)$ fails and every chain is `CHAIN_TERMINAL`.

*Lemma present* ($M_{\mathrm{prov}} = M_{\min}$):

| cover | $n=50$ | 80 | 150 | 400 | 1000 |
|---|---|---|---|---|---|
| $d=1$, $C_{\min}$ ($M=1$, needs 45) | pass | pass | pass | pass | pass |
| $d=2$, $C_{\min}$ ($M=5$, needs 77) | **R2** | pass | pass | pass | pass |
| $d=3$, $C_{\min}$ ($M=12$, needs 94) | **R2** | **R2** | pass | pass | pass |
| $d=1$, singletons ($M=8$, needs 86) | **T4** | **T4** | pass | pass | pass |
| $d=2$, singletons ($M=25$, needs 108) | **R2** | **T4** | pass | pass | pass |
| $d=3$, singletons ($M=64$, needs 126) | **R2** | **R2** | pass | pass | pass |

*Lemma removed* ($M_{\mathrm{prov}} = M_{\min} = \lvert\Pi\rvert$, the exact cover is the all-singleton
cover): $d = 1, 2, 3$ are **R2** at $n = 50$ and $80$ and pass at $n \ge 150$.

The cell $d = 1, n = 80$ (lemma present) is a direct instance of Remark (a) of M5.5: the minimum cover
passes C3 and the Bonferroni cover does not. Intermediate covers in MC-A are predicted by the same rule
from $n_{\min}(\alpha, \delta/M(C))$. Any disagreement between these predictions and the engine's outcome
is itself a finding, reported under A5.

## 5. Definitions frozen for all blocks

**Admissible chain.** A chain is admissible iff the engine's `verify_chain(chain, losses)` returns `ok`
(`ace_ltt.py:755–764`: every consecutive pair, in walk order, has a non-`None` `certify_chain_relation`).
For a fixed loss family this is equivalent to every ordered pair passing, because the structural part is a
product order and the loss and inference parts do not depend on the pair. Walk order is the engine's:
most inclusive first, by `_inclusion_key` $= (A, b, (-v(S0), -v(S1)), \text{name})$. Singletons are always
admissible. A cover is admissible iff every chain is.

**Admissibility factor.** The engine's relation has a structural part (lines 617–624), a loss part for every
loss of kind `"miss"` (line 625: `antitone_under_alarm_inclusion`; lines 635–638: aggregation, `lemma`,
`empty_denominator`), and an inference part (line 632: `chain_compatible`). A loss that claims
antitonicity must name its lemma at declaration (`LossSpec.__post_init__`), so the two levels are:

- *Lemma present*: every constrained loss is `miss_loss(...)` as in §3, with its lemma id; so
  $E_{\mathrm{adm}} = E_{\mathrm{str}}$.
- *Lemma removed*: the **first** constrained loss is declared without a lemma, which the engine allows only
  as `LossSpec("m1", 0.05, False, kind="miss", catch_set="response", aggregation="unit_any_positive_frame",
  empty_denominator="exclude_unit")` — the construction `ace_toy` uses for its C6 refusal scene. Every other
  field, and every other loss, is unchanged. The relation refuses at line 625 for every pair, so
  $E_{\mathrm{adm}} = \varnothing$ and every chain is a singleton (checked data-free: $M$ = 8, 25, 64 for
  $d$ = 1, 2, 3). This is the generalisation of the measured M9 refusal path 4. *Note for the M5 rewrite:*
  M5.5 says "remove the lemma and change nothing else"; in the engine, removing the lemma necessarily also
  withdraws the antitonicity claim.

No further level is added.

**Survivors and the minimum cover.** `certify_family` builds its registered cover over the candidates
that survive step 2 (Q13). With common $n$ and common membership, step 2 either keeps every candidate or
refuses every candidate, so $C_{\min}$ is the engine's exact cover of the whole grid whenever the run gets
past step 2.

**Successful execution.** Under cover $C$, the execution is successful iff the engine's `inadmissible`
map is empty: no candidate refused at step 2 (**R2**), and no chain made terminal afterwards (**T4**, a
failed chain certificate, or a certify-fold membership mismatch — `ace_ltt.py:1293–1345`). An empty
certified set with `NO_REJECTION` alone *is* a successful execution. With common $n$, R2 and T4 are
family-wide and give $\widehat\Pi(C) = \varnothing$. The flag is read from the engine's output (Path L:
from the same engine calls), not inferred by the harness.

**Refusal treatment.** Refused executions (R2 or T4) are never discarded. They are counted, tagged by
path and refusal code, and enter the checks of §6 that are defined for them.

**Null policy.** $\pi$ is null iff $R_k(\pi) > \alpha$ for at least one constrained $k$, evaluated from the
closed form of §3. An **invalid certificate** is a returned null policy.

## 6. Blocks, metrics and decision rules

### MC-A — deterministic implementation check

*Per replication* (a fresh sample of $n$ units on the cell's fixed grid): the engine's exact minimum
admissible cover $C_{\min}$ (`build_chains`); 10 refinements; 10 random admissible covers; and the
all-singleton cover. The covers depend only on the grid and the replication's generator, never on data.

- *Refinement $j = 1, \dots, 10$.* Target $M_j = M_{\min} + \mathrm{round}\bigl(j (\lvert\Pi\rvert - M_{\min})/11\bigr)$.
  Starting from $C_{\min}$, repeat until $M = M_j$: choose a chain of length $\ge 2$ uniformly, a cut
  position uniformly among its $\text{length} - 1$ gaps, and split it there. Sub-chains of admissible chains
  are admissible.
- *Random admissible cover.* Visit the items in a uniformly random order; put each into a uniformly chosen
  existing chain all of whose members are comparable with it under `certify_chain_relation` (in either
  direction), or open a new chain if there is none; finally sort each chain by `_inclusion_key`.
- Every cover is validated with `verify_chain` on each chain before use; a cover that fails is recorded as a
  harness defect and not executed. For every cover, $p$-values from the engine, stopping by the
engine's `fixed_sequence` at $\delta/M(C)$.

*How a non-minimum cover is executed — Path L (bound by Q9).* `certify_family` always builds its own cover
with `build_chains` and uses `plan` only to allocate shares (`ace_ltt.py:1289–1290`), so a non-minimum
cover cannot be passed to it. Every cover, $C_{\min}$ included, is therefore executed by the harness with
the engine's own parts and in the engine's own order, transcribing `ace_ltt.py:1268–1356` with no added
logic: step 2 exactly as the engine does it (`build_chains` over the whole family for the provisional
share, `check_admissibility` per candidate); then per chain of $C$ at $\delta_m = \delta/M(C)$:
`check_admissibility` at $\delta_m$ (T4 on failure), `verify_chain`, and the test loop — `as_observation`,
the membership check, `joint_pvalue`, stop at the first `pj > dm`, certified prefix
`chain[:fixed_sequence(ps, dm)]`.

*Check A6 (primary).* On $C_{\min}$, in every replication, Path L and `certify_family` (with `plan=None`)
return the same certified set, the same `inadmissible` keys and the same reason codes. **Zero mismatches
required.** An A6 mismatch makes the A1, A2 and MC-B results of that cell non-interpretable; they are
reported as such, not repaired.

Under *lemma removed* every admissible cover is the all-singleton cover, so only A0, A2, A5 and A6 are
evaluated there.

*Primary check (A1).* For every pair of **successful** executions with $M(C_1) < M(C_2)$:
$\widehat\Pi(C_2) \subseteq \widehat\Pi(C_1)$. **Decision rule: zero violations required.**

*Primary check (A2).* For every successful execution: the returned set equals the threshold set
$\{\pi : p_{\mathrm{joint}}(\pi) \le \delta/M(C)\}$, with $p_{\mathrm{joint}}$ obtained by calling the
engine's `joint_pvalue` on **every** policy (the engine's own loop stops early and does not compute them all).
**Zero mismatches required.** This is equality (1) of M5.5, including the strict-`>` boundary.

*Primary check (A0), run first.* For every unit of every replication and every policy, the harness's
caught indicator (§3) equals the engine's `decision_sets(p, s, g, stratum)["response"]`. **Zero mismatches
required.** A0 is what licenses the closed-form truth used by MC-C: if A0 fails in a cell, that cell's MC-C
result is reported as non-interpretable, and nothing is re-parameterised.

*Primary check (G1), generator.* A0 checks the decision rule, not the sampler. For every cell, policy and
loss, the empirical miss rate pooled over all MC-C replications is compared with the closed-form $R_k$:
$z = (\bar r - R_k)/\sqrt{R_k(1-R_k)/(n \cdot \text{reps})}$. **Failure iff $\lvert z\rvert > 5$** for any
triple (about $4.4 \times 10^3$ triples over the 90 cells; expected false triggers $< 0.01$). A G1 failure makes that
cell's MC-C result non-interpretable.

*Secondary checks.*
- (A3) Chain compatibility directly: for every $E_{\mathrm{adm}}$ edge $(\pi, \pi')$ with
  $\pi \succeq_{\mathrm{cert}} \pi'$ (π earlier in walk order) and every $k$,
  $p_k(\pi) \le p_k(\pi')$ at equal audit metadata. Zero violations required.
- (A4) Remark (a): no pair in which the smaller-$M$ execution is refused by C3 (R2 or T4) while the
  larger-$M$ execution is successful. Zero required.
- (A5) The refusal path observed (pass, R2, T4) agrees with the prediction of §4 for every cover. Zero
  disagreements required; a disagreement is a non-conformance, treated as an A-check failure.

*If any A-check fails:* the run is not stopped, not repeated with a new seed, and the grid is not
changed. The failing cases are archived in full, and the result is reported as a non-conformance of the
frozen engine with Proposition 3 — a finding the paper must disclose, not a nuisance to remove.

*Scale:* 200 replications per cell of $d \times n \times L \times$ config $\times$ admissibility
(180 cells). Walk order and group key are the engine's (`build_chains`, Q12).

### MC-B — yield of the minimum admissible cover

*Same runs as MC-A.* For every admissible cover $C$ such that **both** $C$ and $C_{\min}$ execute
successfully in the same replication:

$$\Delta(C) = \lvert\widehat\Pi(C_{\min})\rvert - \lvert\widehat\Pi(C)\rvert \;\ge\; 0 .$$

*Primary quantity:* the distribution of $\Delta$ as a function of $M(C)/M_{\min}$ — mean, median, and
$\Pr(\Delta > 0)$ — per $d$ and $n$. *Secondary:* **yield fraction**, the share of non-null policies
returned, for $C_{\min}$ and for the all-singleton cover; defined only when the family has at least one
non-null, so it is not reported for `global-null` ($\Delta$ still is). The word "power" is not used for
either. All quantities are reported for every cell; the main-paper figure uses the slice fixed in §8.

*No pass/fail.* $\Delta < 0$ cannot occur without an A1 violation and is reported there. If $C$ executes
successfully while $C_{\min}$ is refused in the same replication, that is an A4 non-conformance; the pair
is recorded under A4 and marked `non-interpretable` for MC-B, and it enters no MC-B quantity. By
Proposition 3 and Remark (a) this cannot occur; the rule fixes the treatment in advance.

### MC-C — validity against known truth

*Per replication:* a fresh sample of $n$ units; the engine's full `certify_family` with its own exact cover;
the returned set compared with the closed-form truth. Refused executions (R2, T4) return the empty set and count
as replications without an invalid certificate; their number is reported per cell.

*Primary quantity:* $\widehat{\mathrm{FWER}} = $ share of replications with at least one invalid
certificate, with a two-sided 95% Clopper–Pearson interval, per cell.

*Cells:* $d \times n \times L \times$ config, lemma present only (90 cells). Under lemma removed the exact
cover *is* the all-singleton cover, which the secondary quantity below already covers.

*Decision rule:* a validity failure is declared for a cell iff the **one-sided 99% Clopper–Pearson lower
bound** of its FWER exceeds $\delta = 0.10$. A cell whose true FWER is at most $\delta$ triggers this rule
with probability below 1%. The rule is **not** multiplicity-corrected across the 90 cells: if the true FWER
sat exactly at $\delta$ in every cell, about 0.9 cells would be expected to trigger. The uncorrected rule is
chosen deliberately because it is the stricter one for the method under test, and every triggered cell is
reported as a validity failure of that cell. This choice is registered here, not made after the results.
(Draft 1 said "60 cells"; the grid has $3 \times 5 \times 2 \times 3 = 90$.)

*Vacuous cells.* Cells refused by construction (R2 in §4: $d = 2, n = 50$; $d = 3, n \in \{50, 80\}$)
certify nothing and have $\widehat{\mathrm{FWER}} = 0$ trivially. They are reported, flagged as vacuous, and
excluded from the pooled FWER; they are not dropped.

*Secondary:* the same FWER for the all-singleton cover (the Bonferroni baseline, executed by Path L), which is the $M =
\lvert\Pi\rvert$ special case and needs no separate procedure.

*Scale:* 5 000 replications per cell. Fallback 2 000, chosen **only** by the timing rule of §7.

### MC-D — greedy stress test

The cover depends only on the family, the losses and the relation, never on the unit sample, so MC-D has no
sampling replications: its randomness is the deletion sets. Lemma present only; under lemma removed both
routines return the all-singleton cover by construction, and that is recorded as one deterministic check.

*Arm D1 — direct call, registered grids.* For $d = 1, 2, 3$: the engine's `_min_chain_cover` and
`_greedy_chain_cover`, each called with the same `items` (sorted by `_inclusion_key`, as `build_chains` passes them, Q12)
and the same `comparable` predicate as `ace_ltt.py:722`. Then 1 000 deletion sets $S$ per grid, $\lvert S\rvert$
uniform on $\{1, \dots, \lfloor \lvert\Pi\rvert/4 \rfloor\}$, members uniform without replacement, and both
routines recomputed on $\Pi \setminus S$. D1 characterises the greedy *routine*; it is not the production
regime, since these grids are below `GREEDY_ABOVE`.

*Arm D2 — production regime.* The registered $8 \times 8 \times 8$ grid of §3 ($d = 3$,
$\lvert\Pi\rvert = 512$), run through `build_chains`, which selects the greedy routine because
$512 > 400$. Exact comparison by a direct call of `_min_chain_cover` on the same items. Then 1 000 deletion
sets with $\lvert S\rvert$ uniform on $\{1, \dots, 111\}$, so that $\lvert\Pi \setminus S\rvert \ge 401$ and the
production path stays in the greedy regime throughout.

*Primary quantities (per arm and grid):* $M_{\mathrm{greedy}}(\Pi) - M_{\mathrm{exact}}(\Pi)$; and the frequency of

$$E = \{\, M_{\mathrm{greedy}}(\Pi \setminus S) > M_{\mathrm{greedy}}(\Pi) \,\}$$

with a 95% Clopper–Pearson interval. *Consistency checks, zero required:* the exact routine never shows
$M_{\mathrm{exact}}(\Pi \setminus S) > M_{\mathrm{exact}}(\Pi)$; the greedy cover never has fewer chains
than the exact one; every multi-member chain of either cover passes the relation pair by pair. Any
instance is an A-type non-conformance.

*Witness rule.* If $E$ occurs, the smallest instance (arm D1 before D2, then least $\lvert\Pi\rvert$, then
least $\lvert S\rvert$, then lowest replication index) is archived with its seed, grid, $S$, and both greedy
covers. **Only an archived witness may enter the paper**, in the supplement. If $E$ is not observed, the
paper states that it was not observed in $N_D$ trials with the exact one-sided 95% Clopper–Pearson upper
bound $U_{0.95} = 1 - 0.05^{1/N_D}$ ($\approx 3/N_D$; at $N_D = 1000$, $U_{0.95} = 0.0029912$), and quotes no
example.

*Scale:* 1 000 deletion sets per grid; 4 grids (D1: three, D2: one).

## 7. Seeds, scale and the timing rule

**Seeds.** Master seed `20261001`. `numpy.random.SeedSequence(20261001).spawn(4)` gives the four block
seeds in the order A, B, C, D; each block spawns one child per replication (MC-D: per deletion set) in
cell order. Seed B is spawned and recorded but unused, because MC-B reuses the MC-A runs; it is kept so
that the A, C and D streams are the ones registered in draft 1. The `numpy`
version is recorded in the run manifest. **Cell order** is lexicographic in the order the factors are
listed in §4: config (`spread`, `least-favourable`, `global-null`), then $d$, then $L$, then $n$, then
admissibility (present, removed); MC-D in the order D1 $d = 1, 2, 3$, then D2.

**Timing rule.** After R0 and before the registered run, a timing pilot runs on **off-grid** families only,
on the server and with the worker count $P$ that the registered run will use ($P$ is written to the pilot
record before the pilot starts). Pilot: lemma present, $L = 1$, $n = 300$ (not on the $n$-grid), Beta table
of §12, seed `99`, 20 replications per pilot family, each replication doing exactly the MC-C per-replication
work (sampling, `certify_family`, the Path-L all-singleton baseline, and the G1 accumulators). Pilot
families (no pilot $b$-level equals any $b$-level of §12, so no pilot family coincides with a registered one):

| pilot family | $b$ levels | veto levels |
|---|---|---|
| $d = 1$ (8) | 0.03, 0.04, …, 0.10 | $v(S0) = v(S1) = 0$ |
| $d = 2$ ($5 \times 5$) | 0.030, 0.045, 0.060, 0.075, 0.090 | $v(S1) \in$ {0, 0.01, 0.02, 0.03, 0.04}, $v(S0) = 0$ |
| $d = 3$ ($4 \times 4 \times 4$) | 0.04, 0.06, 0.08, 0.10 | $v(S0), v(S1) \in$ {0, 0.01, 0.02, 0.03} |

Let $\bar t_d$ be the mean wall-clock seconds per replication of pilot family $d$. The projected MC-C
runtime at 5 000 replications is

$$\widehat T = \frac{5000}{P} \sum_{\text{cells } c} \bar t_{d(c)} \cdot \frac{n_c}{300} \cdot L_c
= \frac{5000}{P} \cdot 50.4 \cdot (\bar t_1 + \bar t_2 + \bar t_3),$$

since each $d$ appears in 3 configurations $\times$ ($L = 1, 2$) $\times$ 5 values of $n$, with
$\sum_L L = 3$ and $\sum_n n / 300 = 5.6$. **If $\widehat T > 259\,200$ s (72 h), MC-C uses 2 000
replications; otherwise 5 000.** The pilot record keeps $P$, the three $\bar t_d$, $\widehat T$ and the
decision; every other pilot output is deleted unread. This is the only scale choice that depends on anything
observed, and it depends only on time.

## 8. What the main paper will show, fixed in advance

Irrespective of outcome, the main text carries **at most one figure and one compact table**.

- **Figure.** Yield fraction against $M(C)/M_{\min}$ (MC-B), mean over replications and covers at each
  observed $M$; one panel per $d$, one curve per $n$; slice fixed now: `spread`, $L = 1$, lemma present.
  Every other slice goes to the supplement in the same format.
- **Table.** One row per block: MC-A violations over checks made (A0, A1, A2); MC-C FWER with 95% interval,
  worst cell and pooled; MC-D frequency of $E$ per arm, with its interval.

Seeds, generator parameters, all confidence intervals, cover-by-cover traces, terminal-cell tables and any
greedy witness go to the supplement. Every registered cell is reported, including cells whose result is
unflattering; no cell is dropped for being uninformative.

## 9. API binding

Questions about the frozen engine, not design choices. Answers are recorded in §11 with the line they were
read from. No answer may change the targets, constants, decision rules or reporting of §3–§8; an answer may
only bind a registered item to the engine's actual interface, or mark it infeasible in a dated note before
R0. Nothing is substituted silently.

**All answered (read-only, 2026-09-30):** Q1–Q14, see §11.

## 10. Outputs

Under `~/Norqobil/mc_prereg/run_R0/`: one JSONL file per block, one row per replication or per cover
pair; `manifest.json` with engine SHA-256, harness SHA-256 (first recorded here, since the harness is
written after R0), this document's SHA-256 (from the detached record), Python, `numpy` and `scipy` versions, platform string,
the worker count $P$, the MC-C replication count with its pilot record, seeds
and start/end times; and `witness/` for MC-D, if any.

## 11. Freeze record

| item | value |
|---|---|
| this document, SHA-256 | see the detached freeze record `ACE_MC_preregistration_R0.sha256` |
| harness module, SHA-256 | not part of R0: written after R0, recorded in the run manifest (§10) |
| `cascade/ace_ltt.py`, SHA-256 | `ca93d2f3e4eafa85255f512318c0e7a350d259640518696ddbfed23e93960241` — from the frozen `cascade_v2_4_7.zip`; the release record states byte-identity through v2.4.8 and v2.4.9 |
| `cascade/ace_toy.py`, SHA-256 | `adc4b284a1ddaf6c39d9a7cc8584abd0508307a0b09979796edcc7069de9fd1a` (server, `v249/`) |
| engine location on the server | `~/Norqobil/v249/cascade/`; identical copies in `v245b/`, `v247/`, `v248/` |
| §12 script `mc_generator_params.py`, SHA-256 | `0426b2f14ec0e7ca6febca588b236f389c37ef89c7d5b7edab7d70524e5e7646` |
| MC-A execution path | **L** (Q9) |
| date of freeze | 2026-09-30 |

**API answers, read-only, 2026-09-30** (line numbers in `ace_ltt.py`). Q1–Q8 and the Q9–Q14 excerpts were
read on the server by the author; the remaining lines of Q9–Q14 were read by the assistant from a local
copy whose SHA-256 matches both digests above.

| Q | binding | lines |
|---|---|---|
| Q1 | Built-in inference `_pvalue_hb_unweighted_unit_mean(obs, loss)` returns `hb_pvalue(obs.risk, obs.n, loss.alpha)`. Scalar `hb_pvalue(rhat, n, alpha)` tests $H_0: R > \alpha$; raises on non-finite or out-of-range `rhat` or `n < 0`; returns 1.0 at `n == 0`. The built-in pair is registered with weighting `uniform` and `chain_compatible: True`. | 184–187, 198–204, 820–832 |
| Q2 | `fixed_sequence(pvalues, delta_m)` returns the rejected **prefix length** `k`, stopping at the first `p > delta_m`. `certify_family` uses the same strict test (`if pj > dm: break`) and certifies `chain[:fixed_sequence(ps, dm)]`. So $p_{\mathrm{joint}} = \delta_m$ is certified, as equality (1) of M5.5 states. | 1094–1101, 1346–1352 |
| Q3 | `GREEDY_ABOVE = 400` is a module constant. `build_chains` uses `comparable(i, j) = certify_chain_relation(...) is not None` and chooses `_greedy_chain_cover` if `len(items) > GREEDY_ABOVE`, else `_min_chain_cover` (Dilworth, maximum bipartite matching, Kuhn). Group key, item order and within-chain order: Q12. | 100, 661–662, 693–694, 708–724 |
| Q4 | `certify_chain_relation` is a per-pair predicate returning `None` on refusal. Structural: equal `acq`, band `(t_low, t_high)`, `B0`, `strata`; `p.b <= q.b`, `p.A <= q.A`, `veto_at` of `p` $\ge$ of `q` at every index. Loss side, for every loss of kind `"miss"`: `antitone_under_alarm_inclusion`; aggregation in `MONOTONE_AGGREGATIONS`; non-empty `lemma`; registered `empty_denominator`. Inference side: every loss's inference `chain_compatible`. This is $E_{\mathrm{adm}}$ of M5.5. | 608–638 |
| Q5 | `_greedy_chain_cover(items, comparable)`: deterministic, no minimality guarantee, callable directly. The comment at 1217–1218 is not evidence (§1). | 693–694 |
| Q6 | `certify_family(policies, losses, *, delta, contracts, certify_eval, plan=None, payload_semantics=None, inference_registration=None)`. Order: (1) family-wide preconditions; (2) candidate-level checks before the cover; (3) registered cover over survivors; (4) re-check at the actual `delta_m`; (5) post-cover failure terminal for the whole chain. `CHAIN_TERMINAL` on three paths: re-check (1293–1308), chain certificate `C6_CHAIN_CERTIFICATE` (1309–1314), membership mismatch `C3_CERTIFIABILITY_AUDIT` (1338–1345). `ace_toy.py:319` calls the real `certify_family` on synthetic inputs. | 1199–1221, 1293–1352; `ace_toy.py:319` |
| Q7 | `n_min(alpha, delta) = ceil(log(delta) / log(1 - alpha))`; C3 requires `contract.units_for(L) >= n_min(L.alpha, delta_chain)` for every loss. Identical to the formula of §4. | 504–506, 562–566 |
| Q8 | SHA-256 above. | — |
| Q9 | `certify_family` always builds its cover with `build_chains` (provisional over all candidates at step 2, registered over survivors at step 3); `plan` (a `DeltaPlan`) only allocates shares via `allocate_delta`. A non-minimum cover cannot be executed through it → Path L. | 1269, 1289–1290 |
| Q10 | `ace_toy` builds contracts as `Contract(True, True, {loss: n}, n_budget, {loss: digest}, static_bytes, cap, True)` with one digest for the whole family; `certify_eval(p)` returns `{loss: (risk, n, digest_of_evaluated_units)}`; `IREG = declare_inference(REGISTRATION, LOSSES)`; `plan` built from chain signatures. The harness copies this, with registration id `ace_ltt_mc_prereg/R0`, no payload loss, `plan=None`. | `ace_toy.py` 221–312 |
| Q11 | `decision_sets`: $g_v = \min(1, g + v(\sigma))$; `esc` = $t_{\mathrm{low}} \le s < t_{\mathrm{high}}$; `edge` = $s \ge t_{\mathrm{high}}$; `response` = edge ∪ (esc ∧ $g_v \ge b$); `alarm` = edge ∪ (esc ∧ $g_v \ge A$). `veto_at(stratum)` takes a stratum name; veto ∈ [0, 1]; `Policy` requires $b \le A$. A loss claiming antitonicity must name a lemma at declaration. | 117–146, 390–432, 583–589 |
| Q12 | Group key `(acq, t_low, t_high, B0, strata)`; items sorted by `_inclusion_key = (A, b, (-veto...), name)`, most inclusive first; chains returned most-inclusive-first and sorted by their first element. | 656–658, 708–724 |
| Q13 | Step 2 audits each candidate by `check_admissibility` at the provisional uniform share of the engine's own cover of the **whole** family; if none survive, the result is C3 + `NO_REJECTION` with no `CHAIN_TERMINAL`. | 1268–1287 |
| Q14 | `joint_pvalue(observed, losses)`: `observed` maps loss name → `ObservedLoss` or `(risk, n, membership)`; a missing loss gives $p = 1$; returns `(p_joint, parts)`. `verify_chain(chain, losses)` checks consecutive pairs and returns `(ok, certificates)`. | 755–764, 890–… |

## 12. Appendix — generator parameters: registered procedure

Values are computed by the procedure below from the closed-form $R_k$ **only**, never by running the
engine. They were computed after Q11 closed.

**Fixed now.** $\Pr(\sigma = 1) = 0.3$. Beta parameters, common to all configurations:

| risk type $k$ | $\sigma = 0$ | $\sigma = 1$ |
|---|---|---|
| 1 | Beta(2.0, 5.0) | Beta(1.5, 4.0) |
| 2 | Beta(1.3, 2.0) | Beta(1.3, 2.0) |

Truth configurations differ **only** in the grid levels of $b$ and $v(\sigma)$. Risk type 2 has the heavier lower tail, so at $v = 0$ the two risk curves cross once, near $R \approx 0.055$ (checked in closed form at draft time): type 2 is the larger risk below the crossing and type 1 above it. This places the crossing inside the `least-favourable` band and inside the `spread` range.

**Procedure (constructive, no randomness).** Script `mc_prereg/mc_generator_params.py` (SHA-256 in §11).
It never imports the engine. For each (config, $d$, $L$):

1. *$b$-levels.* $b_i = R_{\max}^{-1}(t_i)$ at $v \equiv 0$, by bisection, for the registered base risks
   $t_i$ below.
2. *Veto levels.* Each veto coordinate takes the same equally spaced levels on $[0, v_{\max}]$, where
   $v_{\max}$ is found by bisection so that $R_{\max}$ at the reference $b$-level, with every veto
   coordinate at $v_{\max}$, equals the floor.
3. Round every level to four decimals; recompute all risks from the rounded levels; check acceptance.

| config | $d$ | base risks $t_i$ at $v = 0$ | reference $b$-level, floor |
|---|---|---|---|
| `spread` | 1 | 8 equally spaced on $[0.010, 0.120]$ | — |
| `spread` | 2 | 0.030, 0.052, 0.075, 0.097, 0.120 | 1st, 0.010 |
| `spread` | 3 | 0.040, 0.067, 0.093, 0.120 | 1st, 0.010 |
| `least-favourable` | 1 | 0.020, 0.0515, 0.0535, 0.0555, 0.0575, 0.070, 0.090, 0.110 | — |
| `least-favourable` | 2 | 0.020, 0.0530, 0.0560, 0.0590, 0.100 | 2nd, 0.0505 |
| `least-favourable` | 3 | 0.020, 0.0560, 0.0595, 0.100 | 2nd, 0.0505 |
| `global-null` | 1 | 8 equally spaced on $[0.053, 0.090]$ | — |
| `global-null` | 2 | 0.060, 0.0675, 0.075, 0.0825, 0.090 | 1st, 0.053 |
| `global-null` | 3 | 0.063, 0.072, 0.081, 0.090 | 1st, 0.053 |

In `least-favourable` the veto range is kept small on purpose: it must not push the in-band $b$-levels out
of $(0.050, 0.060]$. The poset is unchanged by the size of the levels; only the risks are.

**Acceptance** (unchanged from draft 2), on the rounded levels: `spread` min $\ge 0.008$ and max
$\le 0.125$; `least-favourable` at least $N/2$ risks in $(0.050, 0.060]$; `global-null` min $> 0.050$;
always strictly increasing levels; for $L = 2$ under `spread` and `least-favourable`, each risk type is the
larger one for at least $N/4$ policies.

**Values** (computed 2026-09-30; all 18 pass). For $d = 2$ the $v$ levels are $v(S1)$ with $v(S0) = 0$; for
$d = 3$ the same levels serve $v(S0)$ and $v(S1)$. "bind" is the number of policies at which risk type 1 /
type 2 is the larger.

| config | $d$ | $L$ | levels | min / max $R_{\max}$ | in $(0.05, 0.06]$ | nulls | bind | acceptance |
|---|---|---|---|---|---|---|---|---|
| `spread` | 1 | 1 | $b$: 0.0208, 0.0367, 0.0489, 0.0594, 0.0689, 0.0778, 0.0862, 0.0942 | 0.0100 / 0.1200 | 1/8 | 5 | — | pass |
| `spread` | 1 | 2 | $b$: 0.0154, 0.0320, 0.0465, 0.0594, 0.0689, 0.0778, 0.0862, 0.0942 | 0.0100 / 0.1200 | 1/8 | 5 | 5/3 | pass |
| `spread` | 2 | 1 | $b$: 0.0402, 0.0561, 0.0702, 0.0824, 0.0942; $v$: 0.0000, 0.0100, 0.0201, 0.0302, 0.0402 | 0.0152 / 0.1200 | 3/25 | 16 | — | pass |
| `spread` | 2 | 2 | $b$: 0.0361, 0.0556, 0.0702, 0.0824, 0.0942; $v$: 0.0000, 0.0090, 0.0180, 0.0271, 0.0361 | 0.0210 / 0.1200 | 2/25 | 16 | 13/12 | pass |
| `spread` | 3 | 1 | $b$: 0.0478, 0.0655, 0.0802, 0.0942; $v$: 0.0000, 0.0090, 0.0180, 0.0270 | 0.0100 / 0.1200 | 8/64 | 38 | — | pass |
| `spread` | 3 | 2 | $b$: 0.0452, 0.0655, 0.0802, 0.0942; $v$: 0.0000, 0.0099, 0.0199, 0.0298 | 0.0100 / 0.1200 | 7/64 | 36 | 35/29 | pass |
| `least-favourable` | 1 | 1 | $b$: 0.0315, 0.0558, 0.0571, 0.0584, 0.0596, 0.0673, 0.0786, 0.0891 | 0.0200 / 0.1099 | 4/8 | 7 | — | pass |
| `least-favourable` | 1 | 2 | $b$: 0.0263, 0.0551, 0.0568, 0.0584, 0.0596, 0.0673, 0.0786, 0.0891 | 0.0200 / 0.1099 | 4/8 | 7 | 5/3 | pass |
| `least-favourable` | 2 | 1 | $b$: 0.0315, 0.0567, 0.0587, 0.0606, 0.0839; $v$: 0.0000, 0.0010, 0.0021, 0.0031, 0.0042 | 0.0180 / 0.0999 | 15/25 | 20 | — | pass |
| `least-favourable` | 2 | 2 | $b$: 0.0263, 0.0564, 0.0587, 0.0606, 0.0839; $v$: 0.0000, 0.0018, 0.0036, 0.0053, 0.0071 | 0.0180 / 0.0999 | 15/25 | 20 | 9/16 | pass |
| `least-favourable` | 3 | 1 | $b$: 0.0315, 0.0587, 0.0609, 0.0839; $v$: 0.0000, 0.0012, 0.0024, 0.0036 | 0.0163 / 0.0999 | 32/64 | 48 | — | pass |
| `least-favourable` | 3 | 2 | $b$: 0.0263, 0.0587, 0.0609, 0.0839; $v$: 0.0000, 0.0015, 0.0029, 0.0044 | 0.0158 / 0.0999 | 32/64 | 48 | 30/34 | pass |
| `global-null` | 1 | 1 | $b$: 0.0567, 0.0601, 0.0634, 0.0666, 0.0697, 0.0727, 0.0757, 0.0786 | 0.0529 / 0.0900 | 2/8 | 8 | — | pass |
| `global-null` | 1 | 2 | $b$: 0.0564, 0.0601, 0.0634, 0.0666, 0.0697, 0.0727, 0.0757, 0.0786 | 0.0530 / 0.0900 | 2/8 | 8 | 7/1 | pass |
| `global-null` | 2 | 1 | $b$: 0.0612, 0.0658, 0.0702, 0.0745, 0.0786; $v$: 0.0000, 0.0030, 0.0060, 0.0090, 0.0120 | 0.0530 / 0.0900 | 4/25 | 25 | — | pass |
| `global-null` | 2 | 2 | $b$: 0.0612, 0.0658, 0.0702, 0.0745, 0.0786; $v$: 0.0000, 0.0041, 0.0082, 0.0123, 0.0164 | 0.0530 / 0.0900 | 5/25 | 25 | 21/4 | pass |
| `global-null` | 3 | 1 | $b$: 0.0631, 0.0684, 0.0736, 0.0786; $v$: 0.0000, 0.0021, 0.0042, 0.0064 | 0.0529 / 0.0900 | 12/64 | 64 | — | pass |
| `global-null` | 3 | 2 | $b$: 0.0631, 0.0684, 0.0736, 0.0786; $v$: 0.0000, 0.0022, 0.0045, 0.0067 | 0.0530 / 0.0900 | 12/64 | 64 | 61/3 | pass |

**MC-D production grid** ($8 \times 8 \times 8$): $b \in \{0.10, 0.15, \dots, 0.45\}$,
$v(S0), v(S1) \in \{0.00, 0.02, \dots, 0.14\}$, other fields as in §3. Structure only; no risk is evaluated
on this grid.

## 13. Pre-freeze audit of the decision rules (2026-09-30)

Every primary and secondary rule was checked for four things: it is tied to a claim in §0b; it is fully
determined before data (no choice left to the harness author); it cannot pass trivially; and it is not
weaker than in the previous draft. Findings and what was done:

| rule | finding | action |
|---|---|---|
| A0 | checks the decision rule, not the sampler | added **G1** (generator check, $\lvert z\rvert > 5$) |
| A1 | implied by A2 (if every returned set is its threshold set, inclusion follows) | kept as the direct check of (2), which the paper states separately |
| A2 | the engine stops early, so not every $p_{\mathrm{joint}}$ is computed by it | already bound: `joint_pvalue` on every policy (draft 3) |
| A3 | edge direction was implicit | stated: $\pi \succeq_{\mathrm{cert}} \pi'$, π earlier in walk order |
| A4 | determinate under Path L (step 2 common to all covers) | none |
| A5 | "reported" had no status | made a zero-tolerance non-conformance |
| A6 | determinate | none |
| MC-A covers | "random cut points" and "random-order decomposition" were not algorithms | both specified; every cover validated by `verify_chain` |
| MC-B | yield undefined without non-nulls; figure slice left open | `global-null` excluded from yield only; figure slice fixed in §8 |
| MC-C | R2 cells give FWER = 0 trivially and would dilute the pooled FWER | flagged vacuous, reported, excluded from pooling only |
| MC-C rule | uncorrected over 90 cells | kept (stricter for the method); arithmetic already stated |
| MC-D | determinate; rule of three per grid | none |
| seeds | cell order undefined | fixed lexicographically (§7) |
| timing pilot | placed before R0, but the harness is now written after R0 | moved to after R0, before the registered run |
| harness SHA in R0 | impossible under the provenance decision | recorded in the run manifest; the harness carries the R0 SHA instead |
| §8 | no rule depends on the result | none |

No rule was relaxed. Two rules were made stricter (A5; G1 added), and the rest were made determinate.

*Authors' review of draft 4 (2026-09-30)* added four items, all applied: the self-hash was impossible and is
replaced by a detached record; the timing pilot's truth levels and projection equation were not fixed; the
MC-D zero-event bound was the rule-of-three approximation; and MC-B did not say what happens when
$C_{\min}$ is refused while a larger cover succeeds.

