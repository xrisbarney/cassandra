# Active amendment: all-zero ARIMA and ETS training series

Approved by the user on 2026-10-21, before ARIMA screening and before any
2024-2025 scoring. Identifier: `2026-10-21-zero-series`. Supplements the frozen
Stage 2 design v1.0.1; the original freeze and panel manifests remain immutable.

## Decision and rationale

For ARIMA and ETS only, if every observed count in the complete training prefix
is zero, issue the declared boundary model with joint probability one on twelve
future zeros. Apply this to any category at any origin, using training data only.
Retain the category in the primary target mask and all scheduled comparisons.
Missing data do not qualify as observed zeros.

This handles a structural zero-variance boundary where the Gaussian likelihood
does not have a regular positive-variance maximum. It is not a converged Gaussian
MLE and is not an optimizer retry or an alternative candidate selected by score.
Label these comparators ARIMA/ETS with the all-zero boundary rule.

The issued mean, median and quantiles are zero. CRPS equals the realized count.
Log mass is zero at zero and negative infinity at a positive count. Accordingly,
a positive target receives infinite negative log score, which must propagate
into averages; no clipping, dropping or probability floor is permitted. The
certainty assumption is a limitation, not evidence that new activity is
impossible. Once a positive count enters the training prefix, use the original
frozen fitting and retry procedure. Do not repair any other fitting failure.

## Scope and reporting

The amendment changes these two baseline laws on all-zero training series only.
It leaves the proposed Bayesian model, its priors, convergence gates, forecast
periods, candidate grid, selection schedule and comparison family unchanged.
Seasonal naive and the other count models retain their existing probability laws.
For ETS this is binding when its adapter is implemented; approval does not claim
that the remaining ETS adapter or other comparators are already implemented.

Record the rule identifier, affected category and training cutoff for every use;
report the number of boundary uses. Preserve previous failed runs. Continue
reporting all-category, active-category, unknown-category and aggregate results,
and describe unknown as a fallback category, not a forecasting success.

The companion JSON freezes this amendment's text and identifies its parent
freeze. New run provenance records both the original protocol and this
supplement. The previously discussed draft remains historical, superseded by
this approved decision. This is a dated development amendment, not an assertion
that the original protocol already contained the boundary convention.
