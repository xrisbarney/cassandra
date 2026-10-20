# Historical draft: Stage 6 zero-series amendment

Superseded on 2026-10-21 by the user-approved
[active amendment](amendments/ZERO_SERIES_20261021.md). The original draft follows
as a record of the proposal; its pending language no longer describes the active design.

Drafted 2026-10-20, before ARIMA screening and before any 2024–2025 scoring.
This draft is **not active** and does not change the frozen baseline behavior.

The Gaussian fitting likelihood for an all-zero series has an unbounded limit
as innovation variance approaches zero. It cannot supply a regular positive-
variance MLE at that limit. Treating this only as an optimizer retry problem
would make the intended ARIMA comparison depend on numerical boundary behavior.

Proposed rule: for ARIMA and ETS only, if every observed count in the complete
training prefix is zero, issue an explicitly declared boundary model with joint
probability one on twelve future zeros. Apply the rule to any category at any
origin, using training observations only. Do not inspect the future target to
decide whether the rule applies. Keep the category in all target masks.

The issued law has mean, median and quantiles zero. Its CRPS equals the realized
count; its log mass is zero at zero and negative infinity at a positive count.
Thus a positive realization receives infinite negative log score. Do not apply
a probability floor. Once a positive count enters the training prefix, use the
ordinary fitted method and its frozen optimizer sequence. This rule does not
repair any other failed fit, add a grid candidate, or select a different model
for an ordinary nonzero series.

The change would be an explicit extension of the ARIMA/ETS adapters, not a claim
that an optimizer found a finite Gaussian MLE. Report use of the boundary rule
by category and origin. Preserve the previous failed run and report the unknown
stratum separately so its easy target does not imply substantive threat skill.

Alternative: retain the original rule and report ARIMA/ETS as not estimable when
the complete panel forecast is unavailable. The user decision will determine
which rule, if any, is frozen in a dated supplement. The existing Stage 2 and
Stage 5 manifests will remain immutable; subsequent run provenance must identify
the supplement explicitly.
