# Design amendments

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
