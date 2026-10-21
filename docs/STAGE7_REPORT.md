# Stage 7 synthetic validation record

Current status, 2026-10-21: **v2 campaign running; not complete**. The
[second amendment](amendments/STATE_SPACE_LEVELS_20261021.md) now governs new
fits. Its primary prior passed all declared pilot plausibility gates. The wider
level candidate failed the tail gate; the wider dynamic candidate remains
unresolved after its declared 8,000-draw repeat. Both remain stress comparisons.
The [prior audit](checks/stage7-levels-20261021/audit.json) uses only the initial
2010--2018 count panel; posterior fixtures use synthetic data only. The new
campaign contains 43 primary and 12 sensitivity cases. All 161 tests passed
before the runtime-only native-thread policy change. The first sparse-factor
fits have passed; full recovery and component decisions await the campaign.
The record below is historical evidence and will be replaced by the completed
audit, not reinterpreted as v2 validation.

## Historical campaigns

Status: revised-prior validation resumed, 2026-10-21; Stage 7 incomplete. The
[recorded design](STAGE7_DESIGN.md) fixes seeds, component checks, scenarios,
budgets and interpretation before fitting. Forty generated datasets
were planned; none uses a real-data panel or held-out outcomes. The
[interruption audit](simulations/stage7-20261021-interrupted/audit.json) records
13 completed cases (6 accepted, 7 failed diagnostic gates), 5 user-stopped
cases and 22 not started. Completed files were hash-verified, and partial
directories were preserved. No old-prior sampling jobs remain. The
[prior amendment](amendments/STATE_SPACE_PRIORS_20261021.md) is now
[activated](amendments/STATE_SPACE_PRIORS_ACTIVATION_20261021.md) for a separate
synthetic campaign. Its original sealed draft remains unchanged. The fitting,
forecast, covariance and truth-coordinate code now use the stationary-amplitude
law together. New directories preserve both original and translated truth.

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
Across 72 widths from 0.25 to 18 and integer lags 0 to 120, the largest observed
kernel-covariance discrepancy is about 0.0002384 (width 18, lag 34). This is a
finite-grid result, not an all-width error bound. The known-parameter count
experiment compares interval coverage with exact discrete coverage, which need
not equal the nominal level when counts have large atoms.

The full suite passed 137 tests; two additional runner tests then verified early
rejection and checkpoint handling. Ruff and the lock check passed. The original
five-file protocol freeze and the panel-manifest hash still match.

Hierarchical Gaussian group means are integrated out exactly: the category-level
prior covariance is `s^2 I + hyper_sd^2 11'`. An independent quadrature check
confirms equality with the original hierarchical prior. This improves geometry
without changing that prior. Two earlier uncompleted engineering runs retain
their source/data records and explicit abort records under `artifacts/stage7/`;
they are not recovery evidence. The interrupted campaign used
`artifacts/stage7/recovery-v3`, retaining the six independent/latent-AR controls
from `recovery-v1`. The [geometry decision](decisions/STAGE7_GEOMETRY_20261021.md)
records the centered factor coordinates and dense metric selected from
computational diagnostics. Earlier factor attempts remain available as geometry
comparisons; they are not additional independent simulated datasets.

Known missing-BLAS, noncontiguous-dot and object-mode indexing warnings are
recorded as performance messages after native/compiled value and gradient
checks. Numerical warnings and all frozen convergence thresholds remain gates.
The [fail-fast execution decision](decisions/STAGE7_FAIL_FAST_20261021.md) stops
later attempts once a retained divergence/depth event makes admission impossible.
Partial failed attempts are explicit and provide no recovery evidence. Accepted
fits still require their entire four-chain budget. Processes started before
that decision retained their original complete-attempt behavior until stopped.

Before its fits began, the deliberately zero-inflated fixture's reference mean
was corrected to include its 30% independent zero inflation. It retains the
unselected NB-component intensity as a separate generating target. Consequently,
observable-mean recovery and underlying-intensity recovery answer different
questions in that misspecified scenario; the generated observations were unchanged.

## Findings already established

The [prior-tail finding](STAGE7_PRIOR_FINDING.md) records 16 integer-range failures
among 80,000 generated category-count draws and derives the infinite prior
rate-scale first moment under the frozen level and factor priors. The original
priors remain unchanged in this campaign. A documented prior amendment and
revalidation are required before real-data proposed-model fitting.

The three independent-count fixtures passed their first four-chain budgets.
Two of three sparse-factor cases and one completed regime case passed. The
three latent-AR cases, remaining sparse-factor case, one severity/missingness
case and two dispersion cases failed their diagnostic gates. The other cases
were interrupted or unstarted; their status must not be classified as a
diagnostic outcome. Accepted fits are conditional old-prior evidence, not
validation of the proposed revision. No old-prior recovery claim resolves
the prior-tail finding.

The prior revision changes the model. Fresh diagnostic/recovery runs are
required and are recorded separately. The optional iid-noise
identification and kernel identification gates remain unresolved.

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
