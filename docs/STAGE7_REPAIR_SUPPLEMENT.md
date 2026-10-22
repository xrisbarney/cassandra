# Bounded Stage 7 convergence supplement

Decision date: 2026-10-22. Stage 8 remains closed. This supplement follows the
user's instruction to repair convergence within a bounded scope. It does not
replace, rerun or remove any of the original 55 cases. Implementation and
preflight verification must precede activation of the fits listed below.

## Findings that change the proposed work order

The [premise audit](checks/stage7-repair-premises-20261022/README.md) shows that
the existing representative anchors already maximize generating loading
magnitudes by column and the two-row absolute determinant. The proposed
stronger-anchor rule is therefore a no-op on this fixture; no duplicate refit
is authorized for that rule. Arbitrary row reordering with a new triangular
constraint need not preserve the diagonal-AR model or its loading prior.
No density-ordering amendment follows from these results.

The pass pattern is not a perfect zero/one-factor versus two-factor split:
latent-AR and oracle-effort cases include failures, and two proxy passes remain
excluded for unexplored regions. The fallback fixture is synthetic. In its
seed-7101 final attempt the fallback level mixes substantially better than
factor-two persistence and several known-category levels. The supplement tests
reduction and coupling; it does not assume the failure mechanism is proven.

## Fixed experiment set

Use the primary anchored-level prior, R=1, count channel only, no effort, iid
residual or kernel. Preserve category order and all observations unless the
row explicitly defines a mask or a generated-data change. Use new directories.
The canonical representative and fallback fixtures both use seed 7101, ten
categories and 108 training months. No other seed or prior candidate may be
added to this round.

| Case | Observations | Fitted factors | Purpose |
|---|---|---:|---|
| Masked fallback | Original fallback fixture; only category-10 count mask set false | 2 | Test whether removing its count contribution changes joint mixing. |
| Reduced representative | Original representative fixture, unchanged | 1 | Assess a reduced approximation to the original two-factor generated data. |
| Reduced fallback | Original fallback fixture, unchanged | 1 | Assess that approximation with the independent fallback level and all 108 zero observations retained. |
| Matched one-factor fallback | One-factor counterpart defined below | 1 | Check a correctly specified one-factor model at the intended dimensions and with an independent fallback level. |

For the masked case, preserve the original prior anchors explicitly; missing
observations must not trigger a changed empirical anchor. Keep category 10 in
the parameterization. Its masked posterior is not evidence that the original
zero-count target is validated, and no primary score may silently drop it.

For the matched case, retain the fallback fixture's stationary first-factor
training path, first loading column, log levels and NB dispersion; remove the
second factor contribution. Redraw the 108 training counts from this specified
NB law using NumPy RNG seed 78101. Draw 12 future factor innovations with RNG
seed 78102, from the retained terminal first-factor state, using the original
first-factor persistence and stationary innovation variance. Generate future
counts after the training counts using RNG 78101. Record both generators and
all resulting array hashes before fitting. Derive prior anchors from these new
training counts under the same declared rule. This is one fixed-truth case,
not an independent replication of the entire original experiment or SBC.

The reduced cases are deliberately fitted to two-factor generated data; they
test approximation, not recovery of nonexistent one-factor parameter truths.
Compare operational intensity, count means and implied covariance with the
original generating quantities. Do not score their loading entries against a
two-factor matrix or reinterpret the original truth.

## Budget and unchanged gates

Exactly one sampling attempt per case: four sequential chains, 2,000 warmup
and 2,000 retained draws per chain, target acceptance 0.99 and maximum tree
depth 15. Sampler seed 7103 and initialization-jitter RNG seed 7301 match the
original seed-7101 final-budget attempt. Maximum round budget is four fits,
32,000 warmup and 32,000 retained draws. Existing irreversible retained-failure
stopping rules apply. No extra warmup, retry, seed search or prior adjustment
is permitted after seeing these outcomes.

Use the previously verified conditional count coordinates and diagonal mass
adaptation for one-factor fits; use the original exact joint coordinates for
the two-factor masked diagnostic. Pin one numerical thread and NUMBA_OPT=3.
Before sampling, verify complete native/compiled targets and gradients, masks,
canonical data provenance, and the one-factor prior-predictive plausibility
check. Preserve warnings and failed preflights. A failed preflight blocks that
case rather than authorizing a statistical-law change.

Every admitted fit still needs four complete chains, finite diagnostics,
R-hat <1.01, bulk and tail ESS >400, BFMI >0.3, zero retained divergences and
depth saturations, and no unresolved numerical warnings. Previously documented
unexplored regions require separate exploration review. Save all attempts and
independently recompute diagnostics. Recovery summaries do not override these
gates; no new recovery threshold is chosen from observed results.

## Stop and adoption rule

Stop after these four terminal outcomes, including failures. Kernel and
misspecified extensions remain unvalidated and excluded from the proposed
initial Stage 8 fit. The masked experiment cannot decide the separate KEV
mask-versus-intercept question; all KEV additions and provenance remain intact.

A one-factor pilot is a candidate amendment, not an automatic fallback that
waives validation. Require all three one-factor cases to pass numerical and
exploration gates, review their recovery and approximation limitations, and
record a dated pilot amendment before any Stage 8 fit. If they fail, Stage 8
stays closed and the bounded round ends with that finding. Do not launch another
repair round implicitly. No held-out forecast or score is authorized here.
