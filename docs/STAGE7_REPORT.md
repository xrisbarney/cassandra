# Stage 7 synthetic validation record

Status: execution in progress, 2026-10-21. The
[recorded design](STAGE7_DESIGN.md) fixes seeds, component checks, scenarios,
budgets and interpretation before fitting. Forty generated datasets are
scheduled; none uses a real-data panel or held-out outcomes.

## Implementation and verification

The generator uses NumPy/SciPy independently of the PyMC likelihood. Shared
components include an emission-only HMM with exact first derivatives, forward
filtering/backward sampling, stationary factors, constrained loadings, a causal
kernel and gathered-cell observation likelihoods. Censored mark endpoints are
classified on the original mark scale to avoid floating-point misclassification
after division by ten.

Numerical checks compare HMM likelihoods with exhaustive enumeration through
six months and three regimes, compare gradients with finite differences and
native/NUMBA evaluation, check backward samples against exact marginals, and
check observation densities, masking, kernel covariance and conditional forecasts.
The factor and kernel forecast tests compare Monte Carlo means with independent
analytic lognormal-mixture calculations. Finite kernel-truncation and
known-parameter count-calibration outputs are in
[the numerical audit](checks/stage7-numerics-20261021/numerics.json).

Hierarchical Gaussian group means are integrated out exactly: the category-level
prior covariance is `s^2 I + hyper_sd^2 11'`. An independent quadrature check
confirms equality with the original hierarchical prior. This improves geometry
without changing that prior. Two earlier uncompleted engineering runs retain
their source/data records and explicit abort records under `artifacts/stage7/`;
they are not recovery evidence. The campaign uses `artifacts/stage7/recovery-v1`.

Known missing-BLAS, noncontiguous-dot and object-mode indexing warnings are
recorded as performance messages after native/compiled value and gradient
checks. Numerical warnings and all frozen convergence thresholds remain gates.

## Findings already established

The [prior-tail finding](STAGE7_PRIOR_FINDING.md) records 16 integer-range failures
among 80,000 generated category-count draws and derives the infinite prior
rate-scale first moment under the frozen level and factor priors. The original
priors remain unchanged in this campaign. A documented prior amendment and
revalidation are required before real-data proposed-model fitting.

The completed independent-count fixtures passed their first four-chain budgets.
Latent AR fixtures have required all retry budgets; unsuccessful fits are kept
and will not contribute recovery claims. Remaining scenario results will be
reported with accepted/attempted denominators after all scheduled runs finish.

## Limits and next gates

These are fixed-truth stress cases, not prior-generated SBC. Coverage over three
datasets and correlated category-month cells cannot establish general Bayesian
calibration. Within-fit regime probabilities are retrospective. Auxiliary-channel
coefficients and synthetic exposure are known in the missing-channel fixture;
it does not establish real-channel eligibility or coefficient identification.

Stage 8 must integrate separately tested additions into production model variants
and revalidate their combinations. Stage 9 must assess full-model inference cost.
The six remaining named comparison systems, incident/economic eligibility and
economic calibration remain later work. January 2024-December 2025 remains
unscored, and the Stage 6 score tables and original Stage 2 freeze are unchanged.
