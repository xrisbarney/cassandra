# Design amendments

## 2026-10-21: activation for revised Stage 7 validation

The user instructed continuation of Stage 7 with issues patched as found.
The [activation record](amendments/STATE_SPACE_PRIORS_ACTIVATION_20261021.md)
activates the sealed prior proposal as `stationary_bounded_v1` for synthetic
fits, forecasts and recovery. Its companion JSON binds the specification;
new runs verify and record it. Old priors/results and the original freeze stay
unchanged. Real-data fitting and held-out scoring remain disabled.

## 2026-10-21: proposed state-space prior revision (not active)

At the user's request, unfinished old-prior Stage 7 jobs were stopped and their
artifacts preserved. The [draft prior amendment](amendments/STATE_SPACE_PRIORS_20261021.md)
addresses stationary factor amplitude, hierarchical scales, optional residuals,
kernel loadings and the version-G KEV slope. Its analytic moment proof and
prior-only simulation audit are ready for review. This is a proposal, not an
activated model or a completed Stage 7 validation. The original Stage 2 freeze,
Stage 6 controls/results and inference code remain unchanged. Revised posterior
fits have not started.

## 2026-10-21: primary independent control replacement

The [conditional observation-driven NB amendment](amendments/OBSERVATION_CONTROL_20261021.md)
replaces the unsuccessful latent independent control as the Stage 6 primary
control, under the user's instruction to resolve it using the literature.
Its distinct ID is `independent_nb_glar`. The stationary-scale revision below
remains a recorded diagnostic variant; it did not pass every required initial
category. Its partial rolling run was aborted before scoring. The replacement
keeps the four-chain budgets, convergence gates, held-out period and eight named
comparators unchanged. Its companion JSON freezes the new specification.

## 2026-10-21: independent-control stationary-scale prior

Under the user's instruction to resolve the independent Bayesian control using
best practices or the literature, the scale prior is moved to the stationary
log-rate SD. This is a substantive model change following a prior-predictive
failure, separate from the exact computational reparameterizations. It is
recorded before any independent-control development scoring or held-out scoring.
See the [decision and mathematical rationale](amendments/INDEPENDENT_CONTROL_20261021.md)
and [supplemental freeze](amendments/INDEPENDENT_CONTROL_20261021.json).
The original Stage 2 freeze, other controls and all diagnostic thresholds remain
unchanged. The prior scale is an explicit design choice, not a published result.

## 2026-10-21: all-zero ARIMA/ETS boundary convention

User-approved substantive supplement to design v1.0.1, activated before ARIMA
screening and before held-out scoring. ARIMA and ETS use a declared joint zero
point mass when the complete observed training series is all zero. Other series
retain the frozen fitting/retry rules. Category masks, candidate grids and the
comparison family are unchanged; infinite log-score penalties remain infinite.

The [approved text](amendments/ZERO_SERIES_20261021.md) and
[supplemental freeze](amendments/ZERO_SERIES_20261021.json) are recorded separately.
The original Stage 2 freeze and Stage 5 panel hashes remain unchanged. New run
provenance includes both records. The earlier diagnostic ARIMA failures remain
historical evidence and are not rewritten.

## 1.0.1 — 2026-10-20

Editorial clarifications requested before Stage 3. The Bakdash adaptation explicitly follows the time-only fit rather than the alternative including report length. KEV launch handling depends on the presence of a backlog, not an exact count. No target, model equation, selection grid, score, or evaluation date changed; no held-out scores have been computed.

The preceding hash record is retained in [freezes/STAGE2_FREEZE_v1.0.json](freezes/STAGE2_FREEZE_v1.0.json). [STAGE2_FREEZE.json](STAGE2_FREEZE.json) records the current clarified files; the statistical analysis remains design version 1.0.
