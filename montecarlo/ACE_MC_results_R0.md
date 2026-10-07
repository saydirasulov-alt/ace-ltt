# ACE-LTT — controlled validation: results of the registered run R0

Results of the single registered run of `ACE_MC_preregistration_R0.md` (SHA-256 `a8e0b6fc…1cedd`, detached
record `ACE_MC_preregistration_R0.sha256`), read against the decision rules fixed there. Nothing in this
report changes R0. Paper text has not been edited.

## 1. Provenance

| item | value |
|---|---|
| R0 document | `a8e0b6fca234f043d7ead1e4a6f7d00b605a455888c1101cf504274c6271cedd` |
| engine `v249/cascade/ace_ltt.py` | `ca93d2f3e4eafa85255f512318c0e7a350d259640518696ddbfed23e93960241` |
| generator values `mc_generator_params.json` | `0f0ed400138b15a28b3408b323e7964f420d698dddb1fc3d5b5e5284cc902ab8` |
| harness `mc_harness.py` (written after R0) | `3c091eb22370232a50d85353dc06989685c6d9370edc719889372c530f0399df` |
| environment | Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1, Linux 5.15.0-139; $P = 24$ workers |
| pilot | written 2026-10-01 08:57:48 KST; $\bar t_d$ = 0.0019 / 0.0068 / 0.0227 s; $\widehat T = 330$ s $\le$ 259 200 s → MC-C 5 000 replications |
| registered run | 2026-10-01 08:58:38 → 09:08:17 KST, single execution, `run complete` |
| outputs | `manifest.json` `2cfd7bec…`, `mcA.jsonl` `c499a0f5…`, `mcC.jsonl` `ffd728cf…`, `mcD.jsonl` `f2f8d6bb…`, `summary.json` `ae7c5d88…`, `pilot_record.json` `d77d058b…` |

## 2. Outcome against the R0 decision rules

| rule | registered requirement | observed | status |
|---|---|---|---|
| A0 decision rule = engine `decision_sets` | zero mismatches | 0 / 586 656 000 unit checks | met |
| A1 inclusion (2) | zero violations | 0 / 2 306 485 successful cover pairs | met |
| A2 equality (1), strict `>` | zero mismatches | 0 / 296 934 successful executions | met |
| A3 chain compatibility along $E_{\mathrm{adm}}$ | zero violations | 0 | met |
| A4 Remark (a) | zero | 0 | met |
| A5 refusal path (pass / R2 / T4) as predicted | zero disagreements | 0 | met |
| A6 Path L = `certify_family` on $C_{\min}$ | zero mismatches | 0 / 36 000 | met |
| cover validation by `verify_chain` | defects reported | 0 | — |
| G1 generator, $\lvert z\rvert > 5$ | no trigger | 0 | met |
| MC-C failure rule (99% one-sided CP lower bound > 0.10) | no cell | 0 / 90 | met |
| MC-B non-interpretable pairs ($C$ succeeds, $C_{\min}$ refused) | recorded under A4 | 0 | — |
| MC-D consistency checks | zero | 0 in every arm | met |

Every primary and secondary check of MC-A is met in all 36 000 replications, so the frozen engine conforms to
Proposition 3 and Remark (a) on every registered cell. As an additional implementation-conformance diagnostic
of equality (1) — not a further proof of Proposition 3 — all covers with the same $M$ returned identical sets
within every replication (measured spread 0).

## 3. MC-C — family-wise error against known truth

**Headline: 0 of 90 cells triggered the pre-registered validity-failure rule.** Worst cell: FWER **0.070**
[0.063, 0.077] (least-favourable, $d = 1$, $L = 1$, $n = 50$). Secondary, descriptive only: the pooled rate over
the 72 non-vacuous cells is 0.0064 [0.0061, 0.0067] (2 306 / 360 000); it averages different registered
cells and is not an estimate of one population FWER. The 18 vacuous cells (R2 by construction: $d = 2, n = 50$; $d = 3, n \in \{50, 80\}$, all configs and $L$) certified
nothing and are reported with FWER 0.

| config | worst cell | FWER [95% CI] | one-sided 99% lower bound |
|---|---|---|---|
| `spread` | $d=1, L=1, n=50$ | 0.048 [0.042, 0.054] | 0.041 |
| `least-favourable` | $d=1, L=1, n=50$ | **0.070** [0.063, 0.077] | 0.062 |
| `global-null` | $d=1, L=1, n=50$ | 0.067 [0.060, 0.074] | 0.059 |

The largest observed FWER occurred in the pre-registered least-favourable setting with $M = 1$, the smallest
admissible $n$, and risks concentrated near $\alpha$. Its whole interval lies below $\delta = 0.10$. Theorem 1
gives an upper bound; it does not predict which cell attains the maximum.
The Bonferroni (all-singleton) baseline has FWER at most 0.0034 in every cell.

## 4. MC-B — yield of the minimum admissible cover (effect size, no pass/fail)

Registered figure slice (`spread`, $L = 1$, lemma present): `fig_mcB_yield.tex`. Selected values, all
singletons versus $C_{\min}$, 200 replications each:

| $d$ | $n$ | yield $C_{\min}$ | yield singletons | mean $\Delta$ | $\Pr(\Delta > 0)$ |
|---|---|---|---|---|---|
| 1 | 400 | 0.613 | 0.457 | 0.48 | 0.47 |
| 2 | 400 | 0.369 | 0.226 | 1.29 | 0.77 |
| 3 | 400 | 0.300 | 0.173 | 3.28 | 0.95 |
| 3 | 1000 | 0.548 | 0.461 | 2.29 | 0.94 |
| 1 | 80 | 0.218 | 0 (T4) | — | — |

In the pre-registered figure slice, the observed yield gap was larger at higher $d$, which in this design
also corresponds to a larger gap between $M_{\min}$ and $\lvert\Pi\rvert$. Strict improvement is not
guaranteed; $\Delta = 0$ occurs in registered runs, as Remark (b) allows. $\Delta$ is computed only over pairs
in which both executions succeed. At $d = 1, n = 80$ the minimum cover certifies and
the singleton cover is refused (T4), the registered Remark (a) cell; the refused singleton execution is
reported as yield 0 (T4) and enters no $\Delta$. Yield is not monotone in $n$ at fixed
$M$ (e.g. $d = 1$, $M = 1$: 0.325 at $n = 50$, 0.218 at $n = 80$), as the step-function boundary $r^*$ of
Remark (b) allows.

## 5. MC-D — greedy cover under deletion

| arm | grid | $M_{\mathrm{greedy}} - M_{\mathrm{exact}}$ on $\Pi$ | $E$ | 95% CI | consistency |
|---|---|---|---|---|---|
| D1 | $d = 1$ (8) | 0 | 0 / 1000 | $U_{0.95} = 0.0029912$ | 0 |
| D1 | $d = 2$ (25) | 0 | 0 / 1000 | $U_{0.95} = 0.0029912$ | 0 |
| D1 | $d = 3$ (64) | 0 | 0 / 1000 | $U_{0.95} = 0.0029912$ | 0 |
| D2 | $8 \times 8 \times 8$ (512, production greedy) | 0 | **1 / 1000** | [0.000025, 0.0056] | 0 |

**Archived witness (eligible for the supplement under the R0 witness rule).** Arm D2, deletion set 210,
seed `SeedSequence(20261001, spawn_key=(3, 3210))`. Full grid: production greedy cover 48 chains, exact cover
48. Deleting three candidates — $(b, v(S0), v(S1))$ = (0.10, 0.02, 0.08), (0.10, 0.00, 0.08),
(0.30, 0.06, 0.02) — raises the greedy cover to **49** chains while the exact cover stays at 48. Under
uniform allocation that lowers $\delta_m$ from $0.10/48$ to $0.10/49$. Reproduced independently from the seed
and the frozen engine: same $S$, greedy 48 → 49, exact 48, every chain valid, both stored covers identical to
the recomputed ones.

This is the paper's Remark (c) observed, in two ways: the greedy chain count is not monotone under deletion,
and on the reduced family the greedy cover is not minimum (49 against an exact 48). It is our
own registered witness; the engine comment (3 → 4 on a 402-candidate group) remains uncited.

## 6. Disclosures

Accepted by the authors on 2026-10-01: the run (no rerun), the figure reading (item 2), and refused
execution = yield 0 (item 2).


1. **Thread environment.** Pilot and run were both started with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
   MKL_NUM_THREADS=1`; these variables are not recorded in the manifest. They affect timing only.
2. **Reading of the registered figure.** R0 says "mean over replications and covers at each observed $M$".
   It is computed as the mean over replications of the per-replication mean over covers of that $M$, so that
   replications with more random covers at a given $M$ do not get more weight; because covers of equal $M$
   return identical sets within a replication, this changes weighting only. Refused executions count as
   yield 0 (they return the empty set). Points for $M$ values that occur in all 200 replications are joined
   by lines; $M$ values produced only by some random covers (as few as 1 replication) are open markers and are
   not joined. With this reading the joined curves are non-increasing in $M$ everywhere, as A1 implies.
3. **Witness storage.** The full-grid greedy cover of D2 is stored once (deletion set 0), not inside the
   witness file; it is deterministic and was recomputed and matched.
4. **Harness implementation choices** (do not change any value): per-replication memo of the engine's
   `joint_pvalue` output in Path L; the exact cover in MC-D uses a precomputed table of
   `certify_chain_relation` values; D1 posets use the `spread` levels (the poset is identical across configs).
5. A premature `summarize` call during the run failed on the missing `mcD.jsonl` and wrote nothing.

## 7. What may enter the paper (R0 section 8)

Main text: `fig_mcB_yield.tex` and `tab_mc_summary.tex` (one figure, one compact table). Supplement: this
report's tables, all 90 MC-C cells, MC-B per cell, the bucket counts behind the open markers, and the D2
witness. Claims are limited to §0b of R0.
