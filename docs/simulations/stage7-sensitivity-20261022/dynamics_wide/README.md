# Wider-dynamics sensitivity: complete

All three declared seed-7101 cases have terminal results, and all three fail
their diagnostic schedules: the sparse one-factor fixture, the representative
two-factor fixture and the independent-fallback fixture. The
[summary audit](summary/audit.json) preserves the three-case denominator and
binds the routed results. There are no recovery-eligible posterior summaries;
the empty [analysis](analysis/details.json) reflects that exclusion, not zero
estimation error.

The fallback final retry completes four chains but fails mixing; the
representative final retry stops on a retained divergence. Their earlier
completed chains document poorly explored level/factor regions. No failed
attempt supplies a covariance-recovery or fallback-intensity estimate.

The prior itself remains a sensitivity stress case: its declared
prior-predictive plausibility check was unresolved after the prescribed repeat.
These posterior failures do not retrospectively change that separate
classification or select a different prior. The other sensitivity candidates
remain separate experiments on the same generated datasets.
