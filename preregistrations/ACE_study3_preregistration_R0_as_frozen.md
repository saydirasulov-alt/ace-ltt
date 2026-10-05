# Study 3 — $R_0$, v5 (FROZEN TEXT). Nothing opened, nothing frozen on disk, nothing run.

Supersedes v1–v4 and the amendment note. This is what `ace3_register.py` will emit as the $R_0$ receipt.
Every field is a literal value or a rule with no freedom left. Artifact-level document; the main paper
carries Study 3 as one table row and three or four sentences.

**The governing invariant**, and the reason the rest of this document looks the way it does:

$$\boxed{\text{design chooses; } \mathcal{A} \text{ may only refuse.}}$$

Audit metadata $\mathcal{A}$ — the indicators $Z_k$, the counts $n_k$, the membership digests on
calibration-certify — may turn a pre-fixed procedure **off**. It may never change what procedure will be
run: not the family, not the cover, not the order, not the shares, not the branch.

Protocol constants, frozen and quoted by value: `calibration_fraction_of_test_units = 0.5`,
`select_fraction = 0.3` (`sf`), `calibrate_seed = 1`, `min_test_units_per_source = 5`. Roles: calibration
6045 units (select $\approx 1814$, certify $\approx 4232$), dev 5989, `sealed_test` 6045,
`external_shift` 874. Calibration sources: fasdd_cv 5353, dfire 690, pyro_sdis 2. Neither `protocol.lock`
nor `final_receipt.json` exists at the time of writing.

---

## 1 Stages

$$P_0 \;\to\; R_0 \;\to\; \text{design on calibration-select} \;\to\; R_1 \;\to\; \mathcal{A}
\;\to\; \text{C3 verify/refuse} \;\to\; \text{one execution},$$

each receipt recording the digest of its predecessor. $R_1$ freezes the family, cover, order and shares
**before** $\mathcal{A}$ is opened.

The design stage therefore does not test clause C3. It uses a **design-time support proxy**: the C3
threshold $n_{\min}(\alpha_k, \delta_m)$ evaluated at the design reference counts of §2. Those counts are
select-fold quantities, so the proxy may shape the family and the branch. It is **not** C3 and it discharges
nothing: $n^{\mathrm{ref}}_k$ is not the audit population, and no theorem premise is satisfied by it. Formal
**C3** is tested exactly once, after $\mathcal{A}$ is opened, as
$n^{\mathrm{cert}}_k \ge n_{\min}(\alpha_k, \delta_m)$ together with the membership digest, and it can
then only PASS or REFUSE. Nowhere in $R_0$, $R_1$ or the paper is the design proxy called C3.

## 2 The design reference count, and the branch

$$n^{\mathrm{ref}}_k \;=\; \Big\lfloor n^{\mathrm{sel}}_k \cdot \frac{1-\mathrm{sf}}{\mathrm{sf}} \Big\rfloor
\;=\; \Big\lfloor \tfrac{7}{3}\, n^{\mathrm{sel}}_k \Big\rfloor,$$

where $n^{\mathrm{sel}}_k$ is the positive-unit count for loss $k$ on calibration-**select**. `floor` is
deliberate: `certification_boundary` takes an integer $n$, and a fractional future unit earns the design no
credit. $n^{\mathrm{ref}}_k$ is a **design reference only** — never an estimator, never a validity
quantity — and it is the only count the design stage may see.

> **Rule R1 (branch, from $n^{\mathrm{ref}}$ alone).**
> If $n^{\mathrm{ref}}_k \ge 72$ for both $k$: the **full** design, $M = 4$, $\delta_m = 0.025$.
> Else if $n^{\mathrm{ref}}_k \ge 59$ for both: the **overlay fallback**, $M = 2$, $\delta_m = 0.05$.
> Else: design-stage **REFUSE**, `C3_CERTIFIABILITY_AUDIT`.
>
> $r^{*,\mathrm{ref}}_k = r^*(n^{\mathrm{ref}}_k, \alpha_k, \delta_m)$ for that branch is the **only**
> certification boundary the design stage and every $R_1$ status may use. $72 = n_{\min}(0.05, 0.025)$ and
> $59 = n_{\min}(0.05, 0.05)$.
>
> **Verification at $R_1$.** After the cover is built, the realized $M$ must equal the declared $M$ and
> $\delta/M$ must equal the declared $\delta_m$ to within $10^{-12}$; otherwise **REFUSE**. No re-planning,
> no re-seeding. (`select_stage` consumes $\delta_m$ before it can choose a threshold, so $\delta_m$ is
> declared and then verified rather than derived — this is the only acyclic arrangement that does not
> require rewriting the released selection rule.)
>
> **Formal C3, after $\mathcal{A}$ is opened.** The realized certify count must satisfy
> $n^{\mathrm{cert}}_k \ge n_{\min}(\alpha_k, \delta_m)$ for the branch already chosen — $72$ for the full
> design, $59$ for the fallback — and the membership digest must equal the registered one. If either fails:
> **REFUSE**. There is no full $\to$ fallback switch, and $\mathcal{A}$ changes nothing else.
>
> **The two boundaries, and their namespaces.**
>
> $$r^{*,\mathrm{ref}}_k = r^*(n^{\mathrm{ref}}_k, \alpha_k, \delta_m)
> \;\longrightarrow\; \text{design, } h^{*,\mathrm{ref}}, \texttt{CAPABLE}, \text{T, S, N, }
> \texttt{KNIFE\_EDGE}, \text{R2};$$
>
> $$r^{*,\mathrm{cert}}_k = r^*(n^{\mathrm{cert}}_k, \alpha_k, \delta_m)
> \;\longrightarrow\; \text{the certificate itself and } X, \text{ post-}R_1 \text{ diagnostics only}.$$
>
> Every quantity in the first line is known at $R_1$; nothing in the first line may be recomputed from
> $n^{\mathrm{cert}}_k$. A bare $r^*_k$ appears in this document only where Study 2's historical boundary is
> being described; no Study-3 quantity is ever written with an unsubscripted $r^*$.

## 3 The two arms, and the arm-wise cover

| | arm B | arm L |
|---|---|---|
| varying coordinate | response authority $b$ | veto authority $\lambda$ |
| ladder | every $g$-grid edge $b \le A$ forming a valid candidate at the chosen $(t_{\mathrm{low}}, t_{\mathrm{high}}, A)$, **ascending**; the $g$ grid is fixed in `cascade.policy.Grid`, so the ladder is a deterministic function of the chosen thresholds and of nothing observed | $\lambda \in \{0.0,\ 0.2,\ 0.4,\ 0.6,\ 0.8,\ 1.0\}$, safest first |
| fixed other coordinate | $\lambda = 0.4$ | $b = 0.25$ |
| envelope | $S_B$ (§4) | $S_L$ (§4) |
| `min_members` | 3 | 3 |
| intra-chain order | $\succeq_{\mathrm{cert}}$, safest first | $\succeq_{\mathrm{cert}}$, safest first |

$\lambda = 0.4$ is the pre-Study-1 working point Study 2 registered as `lambda_fixed`, not selected from any
study's result.

> **Fixed response authority for arm L.** $b = 0.25$, carried forward from the operating point selected by
> Study 2's pre-registered utility in its historical development run. It is used here **only as
> pre-existing design information; no validity claim from Study 2 is imported into Study 3.** The
> substantive reading: *prior selected operating point at fixed response authority; Study 3 asks what veto
> authority can do around that previously selected operating regime.* Importing arm B's safest endpoint was
> considered and rejected: as $b \to 0$ the response set absorbs the whole escalated population, so
> $\lambda$ becomes inert with respect to the response-miss risk it is supposed to move.

> **Arm-wise cover, and why.** `build_chains` groups by `(acq, t_low, t_high, B0, strata)`, and the group
> key holds the stratification *keys*, not the $\lambda$ values. On one acquisition mode at the same
> thresholds, arms B and L therefore share a group, and cross-arm pairs are $\succeq_{\mathrm{cert}}$-
> comparable: the minimum cover interleaves them (measured: one chain came back as
> `['B|b=0','B|b=0.1','B|b=0.25','L|lam=0.4','B|b=0.4']`). Such a chain has two varying coordinates, so
> neither $S_B$ nor $S_L$ is its envelope and the $(C,k)$ primitives of §5 have nothing to attach to. So the
> cover is computed **within an arm**, never over the union, and $\delta$ is split across arms:
> $\delta_B = \delta_L = 0.05$, one chain per (arm, mode), $\delta_m = 0.025$ — arithmetically the same
> $4 \times 0.025 = \delta$ the full branch declares. Validity is untouched: any partition into
> $\succeq_{\mathrm{cert}}$-chains fixed before the test fold gives the guarantee, and minimality is a power
> property, not a validity property (§3.4). The price is a little power; the return is chains whose varying
> coordinate, and therefore whose envelope, is well defined.

**Thresholds.** $(t_{\mathrm{low}}, t_{\mathrm{high}}, A)$ per mode are re-derived on calibration-select by
the contract in §6, not carried over from dev. Calibration-select is Study 3's design fold, so this is
design-fold work under A1; re-deriving keeps the family a function of the fresh fold rather than of the fold
Studies 1 and 2 used.

**Utility**, for the reported operating point only, never for T/S/N/X:
$(c_{\mathrm{calls}}, c_{\mathrm{FA}}, c_{\mathrm{handoff}}) = (1.0,\ 10.0,\ 20.0)$. Its value is a
selection-side quantity measured on the certify fold, not an unbiased performance estimate.

## 4 The influence envelopes, frozen as predicates

$S$ is a pre-registered **influence envelope**: any event set containing every event whose catch status the
coordinate can change. Frozen here as a *predicate*; realized membership follows deterministically on
calibration-select. One envelope per arm, and they are different objects.

**$S_B$ (arm B).** For mode $a$ at its frozen $(t^a_{\mathrm{low}}, t^a_{\mathrm{high}})$, a positive
$k$-event $e \in S_B$ iff

$$\big(\exists f \in e,\ y_k(f):\ t^a_{\mathrm{low}} \le s_f < t^a_{\mathrm{high}}\big)\ \wedge\
\big(\nexists f \in e,\ y_k(f):\ s_f \ge t^a_{\mathrm{high}}\big).$$

This is the minimal envelope for $b$: an event outside it is caught or missed regardless of $b$.

**$S_L$ (arm L).** A positive $k$-event $e \in S_L$ iff at least one positive frame of $e$ falls in
`StratumSpec(kind='small_smoke', area_max=0.02)` — *top smoke box area $\le$ `area_max` and
$s_{\mathrm{smoke}} \ge s_{\mathrm{fire}}$; label-free, from the detector boxes only* — the predicate the
released `lambda_capacity` implements, carried over unchanged.

> **Scope note on $S_L$.** It is a **conservative influence envelope, not the minimal acting domain**: it
> includes events already caught by the edge stage, whose catch status $\lambda$ cannot change. So
> $\widehat{\Gamma}_L$ is a valid reach upper bound but may be loose, and the direction of that looseness is
> stated rather than glossed: a larger $\widehat{\Gamma}_L$ makes **T easier to satisfy** (its right-hand
> side grows) and makes `CAPABLE_L` easier to trigger, which makes **S harder**. No stronger interpretation
> is attached to $\widehat{\Gamma}_L$, and no tightened envelope is substituted after the fact.

$q_k(u,S) = |E_{k,u} \cap S| / |E_{k,u}|$ and $\widehat{\Gamma}_{k,D}(S) = |D|^{-1}\sum_u q_k(u,S)$ over
units carrying at least one positive $k$-event, computed **per chain at its own mode**. Study 2's minimum
over modes is kept only for the pre-freeze checklist and never for a $(C,k)$ status.

## 5 The predictions — $(C,k)$-level primitives, chain label derived

$$h^{*,\mathrm{ref}}_k = \max\!\big(0,\ \mathrm{bar}_k - r^{*,\mathrm{ref}}_k\big),
\qquad \mathrm{bar}_k = \alpha_k,
\qquad \mathrm{CAPABLE}_k(C) := \mathbf{1}\{\widehat{\Gamma}^{\mathrm{sel}}_k(S) \ge h^{*,\mathrm{ref}}_k\}.$$

The margin is built from $r^{*,\mathrm{ref}}_k$ and from nothing else, so `CAPABLE`, T, S, N, `KNIFE_EDGE`
and R2 are all functions of the design fold, the declared constants and $n^{\mathrm{ref}}$ — every one of
them known at $R_1$, none of them touched by $\mathcal{A}$. This is what makes the refuse-only invariant
hold literally rather than approximately.

> Study 2 registered the tight bound $\widehat{R}^{\mathrm{sel}}_k \le r^*_k$, which makes the margin $0$
> and the capacity condition vacuous. Since $r^{*,\mathrm{ref}}_k \le \alpha_k$ the selection rule also
> guarantees $\widehat{R}^{\mathrm{sel}}_k \le \alpha_k$, so Study 3 pre-registers the weaker but still
> selection-guaranteed **reference bar** $\mathrm{bar}_k = \alpha_k$, specifically in order to evaluate a
> non-vacuous reach margin. The effect is not one-directional: a larger $h^{*,\mathrm{ref}}_k$ makes
> `CAPABLE` harder to reach, which can reduce the pairs eligible for S and can move the R2 verdict either
> way.

On a fold $D$, $\mathrm{sp}_k^{D}(C) = \max_{\pi} \widehat{R}_{k,D}(\pi) - \min_{\pi} \widehat{R}_{k,D}(\pi)
= \widehat{R}_{k,D}(\pi^{(J)}) - \widehat{R}_{k,D}(\pi^{(1)})$ by Lemma 2, and
$X_k(C) = \mathbf{1}\{\exists \pi: \widehat{R}_{k,\mathrm{cert}}(\pi) \le r^{*,\mathrm{cert}}_k\}
= \mathbf{1}\{\widehat{R}_{k,\mathrm{cert}}(\pi^{(1)}) \le r^{*,\mathrm{cert}}_k\}$, with
$r^{*,\mathrm{cert}}_k = r^*(n^{\mathrm{cert}}_k, \alpha_k, \delta_m)$ — the same boundary the engine
uses to certify, so $X$ reads the actual certificate and not a design surrogate. $X$ is the **only**
primitive that touches $r^{*,\mathrm{cert}}$, and it is secondary and confounded.

> **T$_{C,k}$ — reach transfer (primary).** $\mathrm{sp}^{\mathrm{cert}}_k(C) \le \widehat{\Gamma}^{\mathrm{sel}}_k(S)$.
>
> **S$_{C,k}$ — sufficiency (primary).** $\mathrm{CAPABLE}_k(C) = 1 \Rightarrow \mathrm{sp}^{\mathrm{cert}}_k(C) \ge h^{*,\mathrm{ref}}_k$.
>
> **N$_{C,k}$ — necessity (derived).** $\mathrm{CAPABLE}_k(C) = 0 \wedge \mathrm{T}_{C,k} \Rightarrow \mathrm{sp}^{\mathrm{cert}}_k(C) < h^{*,\mathrm{ref}}_k$.
>
> **X$_{C,k}$ — crossing (secondary, confounded).** $\mathrm{CAPABLE}_k(C) = 1$ and
> $\widehat{R}^{\mathrm{sel}}_k(\pi^{(1)}) > r^{*,\mathrm{ref}}_k \Rightarrow X_k(C) = 1$. (The
> antecedent is an $R_1$ quantity; only the consequent reads $r^{*,\mathrm{cert}}$.)

**T's scope.** T is a **cross-fold reach bound**: whether the design-fold capacity upper-bounds the reach
the coordinate *realizes* on the evaluation fold. It asserts nothing about the ordering of
$\widehat{\Gamma}^{\mathrm{sel}}_k$ and $\widehat{\Gamma}^{\mathrm{cert}}_k$; a chain may pass T while its
certify-fold capacity exceeds its design-fold capacity and goes unused.

The chain-level label $\bigwedge_k \mathrm{CAPABLE}_k(C)$ is **derived**, reported for continuity with
§3.6's vocabulary, used in no prediction. A chain capable for one risk and not the other contributes one
pair to each side.

**N is never evidence.** The study reports two primary predictions (T, S), one derived implication (N,
marked `derived` in its own column) and one secondary diagnostic (X). N never appears as a third supported
hypothesis and never enters a tally.

**What these statements are.** T, S, N and X are **pre-registered finite-sample predictions**, evaluated
once on the fresh certify fold. No $p$-value, confidence statement, sampling distribution or
population-level generalisation attaches to any of them. A pass is one deterministic observation on one fold
pair and supports no claim that capacity transfers in general; a failure is likewise one observation. The
only statements in this work carrying a probability are the certificates of Theorem 1, which do not depend
on T, S, N or X (Theorem 1(iv)).

**Comparison semantics.** Float64, absolute tolerance $\tau = 10^{-12}$: status uses plain $\ge$;
T holds iff $\mathrm{sp}^{\mathrm{cert}}_k \le \widehat{\Gamma}^{\mathrm{sel}}_k + \tau$; S holds iff
$\mathrm{sp}^{\mathrm{cert}}_k \ge h^{*,\mathrm{ref}}_k - \tau$; $X$ uses the plain test, which must match
the engine bit for bit. If $|\widehat{\Gamma}^{\mathrm{sel}}_k - h^{*,\mathrm{ref}}_k| \le 10^{-6}$ the
pair is `KNIFE_EDGE`: T and S together would force
$\mathrm{sp}^{\mathrm{cert}}_k = h^{*,\mathrm{ref}}_k$ to that precision, which tests nothing, so it
is excluded from the S verdict and from R2's count, kept in T, and reported with its numbers. Both
quantities are known at $R_1$, so the exclusion is decided blind to the outcome. $10^{-6}$ is frozen.

## 6 The `select_stage` contract, literally

Inputs: calibration-**select** rows for the mode; the frozen $s_{\min}$ from `protocol.yaml`;
$n^{\mathrm{ref}}_k$ from §2; $\alpha = 0.05$; the branch's declared $\delta_m$; the declared stratum;
$\lambda_{\mathrm{fixed}} = 0.4$; the declared cost weights.

1. $r^{*,\mathrm{ref}}_k = \texttt{certification\_boundary}(n^{\mathrm{ref}}_k,\ \alpha,\ \delta_m)$.
2. $g_v = \texttt{discount}(g_a,\ \mathbf{1}[\,\text{frame} \in S_{\mathrm{stratum}}\,],\ \lambda_{\mathrm{fixed}})$ with $S_{\mathrm{stratum}} = \texttt{StratumSpec('small\_smoke', 0.02)}$.
3. Candidate grid $\texttt{Grid.default}(s,\ s_{\min})$: $s$ edges for $(t_{\mathrm{low}}, t_{\mathrm{high}})$, $g$ edges for $(b, A)$, fixed in `cascade.policy.Grid`, a function of the select-fold score distribution and $s_{\min}$ only.
4. Per-configuration select-fold risks from `_grid_stats`, unit level over events; validity mask $\texttt{candidate\_mask}(\text{grid},\ \texttt{cloud=True},\ \texttt{handoff=True})$ and finite FA.
5. Slacks $s_k(\pi) = r^{*,\mathrm{ref}}_k - \widehat{R}^{\mathrm{sel}}_k(\pi)$, $s_{\mathrm{joint}}(\pi) = \min_k s_k(\pi)$; a NaN risk counts as $+\infty$ risk, hence $-\infty$ slack.
6. For each $A = g_{\mathrm{edges}}[k_2]$ the ladder is $\{b = g_{\mathrm{edges}}[k]: k \le k_2,\ \text{valid}\}$, ascending. A configuration is **admissible** iff it has $\ge \texttt{min\_members} = 3$ valid members, $\max_\pi s_{\mathrm{joint}} \ge 0$ and $\min_\pi s_{\mathrm{joint}} < 0$ (boundary spanning).
7. Objective $\texttt{\_objective}(E, \text{costs})$ with $(1.0, 10.0, 20.0)$, minimised over the ladder members on the certifiable side ($s_{\mathrm{joint}} \ge 0$).
8. Tie-break: the configuration minimising that objective; ties resolved by `np.argmin` over the flattened $(t_{\mathrm{low}}, t_{\mathrm{high}})$ index at the smallest $A$ index attaining the minimum — lowest $A$ first, then row-major order of the threshold grid.
9. No admissible configuration for a mode: **REFUSE**, `no threshold configuration gives a boundary-spanning ladder whose safe endpoint is select-feasible`.

Arm L then fixes $b = 0.25$ and sweeps the declared $\lambda$ grid at the $(t_{\mathrm{low}},
t_{\mathrm{high}}, A)$ step 8 selected for that mode. Two properties of this contract are registered
explicitly because both are visible in it: the rule guarantees only that the **safe endpoint** is
select-feasible (hence Study 2's $\mathrm{bar} = r^*$, hence §5's disclosure), and it *requires* boundary
spanning at design time, so a mode whose ladder cannot straddle $r^{*,\mathrm{ref}}$ refuses rather than
being tested.

## 7 $P_0$ — set operations first, digests as attestation

The code reconstructs the unit-id sets and asserts, in order:

1. $U_{\mathrm{sel}} \cap U_{\mathrm{cert}} = \varnothing$, $U_{\mathrm{sel}} \cap U_{\mathrm{sealed}} = \varnothing$, $U_{\mathrm{cert}} \cap U_{\mathrm{sealed}} = \varnothing$;
2. $U_{\mathrm{sel}} \cup U_{\mathrm{cert}} \cup U_{\mathrm{sealed}} \cup U_{\mathrm{dev}} \cup U_{\mathrm{train}} \cup U_{\mathrm{ext}} = U_{\mathrm{frozen}}$, every unit in exactly one role;
3. $U_{\mathrm{sel}} \sqcup U_{\mathrm{cert}}$ reproduces by re-running the split function from `select_fraction = 0.3` and `calibrate_seed = 1` alone;
4. **only then** the canonical sorted id list of each set is digested into $R_0$ as attestation of what was asserted.

Plus `protocol.lock` present and matching, `analysis_code_sha256` equal to the executing tree's, and
`sealed_test` with no unseal receipt. Any failure refuses before the design stage, so no label is read.

## 8 Exclusions and the outcome vocabulary

$$\texttt{P-4.5} \in \{\texttt{INFORMATIVE},\ \texttt{UNINFORMATIVE},\ \texttt{REFUSED}\}.$$

**`NO_REACH`** — decided at $R_1$, before any outcome. A pair whose chain has $|C| = 1$ has
$\mathrm{sp}^D_k \equiv 0$, so T holds trivially and S fails trivially for any $h^{*,\mathrm{ref}}_k > 0$: no reach exists
to measure. Flagged, excluded from the T, S and N verdicts and from R2's count, reported with its numbers.
Legitimate because it is decided before the certify fold is opened.

**`CHAIN_TERMINAL`** — no denominator shrinking after execution.

> **Any post-registration `CHAIN_TERMINAL` affecting a registered primary pair $\Rightarrow$
> `P-4.5 = REFUSED`.** The terminal chain's pair-level numbers are reported as `CHAIN_TERMINAL`; other
> chains' results may be reported descriptively; no study-level T or S confirmatory verdict is given; R2 is
> **not** recomputed; the denominator is **not** shrunk after execution.

Fail-closed: if part of the registered primary prediction turns out not to be executable, the study does not
carry on with what is left. It differs from `NO_REACH` precisely because `NO_REACH` is known at $R_1$, blind
to the outcome, while `CHAIN_TERMINAL` is observed during execution.

> **Rule R2.** `INFORMATIVE` iff, among the pairs of the registered cover that are neither `KNIFE_EDGE` nor
> `NO_REACH`, there exists $(C,k)$ with $\mathrm{CAPABLE}_k(C) = 1$ **and** $(C',k')$ with
> $\mathrm{CAPABLE}_{k'}(C') = 0$. Decided in $R_1$, before any certify outcome, never recomputed.

If `UNINFORMATIVE`: no confirmatory interpretation of T, S or N; the run still executes and is reported as
an independent registered ACE-LTT certification on fresh units; no certify-fold outcome is used to redesign
the `P-4.5` family, in this paper or a later one; and no recovery of the form "after execution we happened
to see both types" is permitted. In every non-`INFORMATIVE` case the statuses are reported as measured. The
large calibration fold makes `UNINFORMATIVE` a live possibility — a bigger $n^{\mathrm{ref}}_k$ raises
$r^{*,\mathrm{ref}}_k$ and shrinks $h^{*,\mathrm{ref}}_k$, so a coordinate that was incapable on a smaller
design reference count can come back capable — and that outcome is accepted rather than designed around.
The realized $n^{\mathrm{cert}}_k$ cannot move `CAPABLE`, the R2 verdict or the `KNIFE_EDGE` set at all;
its only powers are to certify, to feed $X$, and to refuse.

**Reading of outcomes.** T and S both hold $\Rightarrow$ `P-4.5` survives its first independent test, as one
instance. T fails $\Rightarrow$ the design-fold capacity does not bound the realized reach across folds, and
§3.6 must then state $\widehat{\Gamma}$ as a same-fold bound with no cross-fold content. T holds and S fails
$\Rightarrow$ capacity transfers as a bound but is not sufficient; §4.5 stays a necessary-condition
diagnostic. X fails while T and S hold $\Rightarrow$ attributed to the fold shift by §9, and reported as
further evidence for Observation 1 rather than against §4.5.

**Refusal list** (`REFUSED`): any $P_0$ item; R1's design-time support-proxy branch or formal C3 after
$\mathcal{A}$; protocol
lock or `analysis_code_sha256` mismatch; a realized $M$ or $\delta_m$ that does not match the declaration;
audit membership differing from the registered one; a failed chain certificate; an inference-registration
mismatch; the pre-freeze checklist not closing; any `CHAIN_TERMINAL` on a registered primary pair.

## 9 Registered diagnostics — reported, never tested

Per $(C,k)$: $\widehat{\Gamma}^{\mathrm{sel}}_k$, $\widehat{\Gamma}^{\mathrm{cert}}_k$ (provenance
`realized`), $h^{*,\mathrm{ref}}_k$, $r^{*,\mathrm{ref}}_k$ and $r^{*,\mathrm{cert}}_k$, $\mathrm{sp}^{\mathrm{sel}}_k$,
$\mathrm{sp}^{\mathrm{cert}}_k$, the anchor shift
$\Delta_k = \widehat{R}^{\mathrm{cert}}_k(\pi^{(1)}) - \widehat{R}^{\mathrm{sel}}_k(\pi^{(1)})$ as a
multiple of $g^*_k$, $n^{\mathrm{sel}}_k$, $n^{\mathrm{ref}}_k$ and $n^{\mathrm{cert}}_k$, the select-fold
standard error, the stop index, and the operating point the utility selects. The pair
$(\Delta_k, \mathrm{sp}^{\mathrm{cert}}_k)$ is what makes X interpretable: crossing mixes *where the chain
lands* with *how far the coordinate moves it*.

## 10 One fold, two uses — the allowed wording

The fresh certify fold both issues the ACE-LTT certificate and evaluates T, S and X.

> **Allowed:** "an independent fresh-fold test of the §4.5 extension"; "evaluated on units not used to
> design the family". **Not allowed, and written nowhere:** "independent of the ACE-LTT certification
> result". They are the same units.

## 11 What Study 3 will not claim

One instance, one dataset, one fold pair, finite-sample throughout. A pass makes `P-4.5` a measured
regularity, not a theorem; a failure refutes the extension and leaves §4.5, Theorem 1 and every certificate
in the paper standing. The certified statement, if any, concerns calibration units carrying an event of the
corresponding type (`exclude_unit`) — half of the test units — not a deployed system, and not `sealed_test`,
which stays closed with no unseal receipt. For the main text:

> The prospective fold is fresh with respect to the registered unit split, but it is not a new-domain
> evaluation: 88.6% of calibration units come from the dominant source, and the smallest source contributes
> only two units.

---

## Next: the single code change-set

```
cascade/ace_ltt.py      NOT_CERTIFIED["C5_SPLIT_INDEPENDENCE"] =
    "family, chains, order or delta shares depend on a candidate-dependent certify-fold quantity"
cascade/ace_ltt.py      chain_capacity docstring: "acting domain" -> "influence envelope"
cascade/ace3_register.py   P0, R0, design on calibration-select, R1, A-stage verify, R2
cascade/ace3_dev.py        the single registered execution
cascade/test_ace_ltt.py    + ace3 tests
```

One new `analysis_code_sha256`, one clean-package re-verification, `SPEC_VERSION` unchanged at `v2.4.3`.
Order: patch + code $\to$ tests $\to$ `write_manifest` $\to$ zip $\to$ `freeze` $\to$ design $\to$ $R_1$
$\to$ $\mathcal{A}$ $\to$ execution. Nothing in `ANALYSIS_CODE` changes after `freeze`.
