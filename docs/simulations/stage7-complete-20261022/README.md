# Original Stage 7 campaign: complete

The [campaign audit](campaign/audit.json) verifies all 55 canonical cases,
their artifact and fixture hashes, fixed retry budgets, admission gates,
recovery exclusions and 39 preserved earlier execution records. It independently
recomputes diagnostics for admitted posteriors. Severity-coordinate verification
reuses three canonical cases and is not counted as three additional fits.

| Group | Cases | Numerical passes | Recovery eligible | Diagnostic failures | Exploration exclusions |
|---|---:|---:|---:|---:|---:|
| Primary | 43 | 28 | 25 | 15 | 3 |
| Sensitivity | 12 | 2 | 2 | 10 | 0 |
| Total | 55 | 30 | 27 | 25 | 3 |

The 43 primary cases contain 37 distinct observed training count panels because
effort variants reuse observations. Distinct hashes are not a claim of independent
replication. These are fixed-truth experiments, not simulation-based calibration.
Failed fits and exploration exclusions remain in the denominators.

Sensitivity outcomes are 0/3 passes for tighter levels, 1/3 for wider levels,
1/3 for tighter dynamics, and 0/3 for wider dynamics. Both passes are the small
one-factor sparse fixture. Wider levels fail their separate prior-predictive
plausibility gate; wider dynamics remain an unresolved prior stress case.
Neither is admitted as a replacement default. No representative or fallback
sensitivity fit passes.

The [independent diagnostic inventory](independent-diagnostics.json) verifies
all 77 complete saved canonical attempts, including unsuccessful ones. Early
stops have no complete saved posterior and are excluded from that attempt
count. Numerical admission does not override a recovery exclusion.

The [coordinator reconciliation](coordinator-reconciliation.json) resolves all
eight reserved hosted cases against their verified imports and terminal
results. Raw local scheduling labels are preserved; they are not extra model
failures. The separately executed helper is likewise reconciled.

The [timing audit](timing/audit.json) covers 55 cases and sums to 68.477 recorded
component job-hours. These are overlapping job costs, not elapsed campaign time
or total research compute. Earlier engineering trials, transfers and analysis
are outside that total. Sampling-call time includes warmup, retained draws and
some deferred compilation; chain-phase times are subdivisions, not additional
costs. Explicit compile fields and the case/chain tables preserve these limits.
Failed synthetic fits cannot establish a production rolling-refit budget.

The [222-test record](../../checks/stage7-final-tests-20261022/audit.json) and
[technical integrity checks](../../checks/stage7-integrity-20261022/README.md)
and [final hash/check revalidation](final-checks.json) remain separate from
posterior validation. Tested runtime source is unchanged. The ten-category, two-factor pilot
is not admitted. The [bounded repair supplement](../../STAGE7_REPAIR_SUPPLEMENT.md)
is a new, limited investigation; no supplement fit or Stage 8 fit is included
in this completed campaign. January 2024--December 2025 remains unscored.
