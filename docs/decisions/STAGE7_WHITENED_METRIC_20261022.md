# Metric comparison after exact conditional whitening

Declared 2026-10-22 before the additional fit. The rank-two representative
conditional-count fit is spending its time in NUTS integration, confirmed by
a py-spy stack inspection; it has not yet printed its first 500-draw marker.
Its repeated gradient preflight was approximately 0.00252 seconds under the
concurrent workload. This is a computational finding, not recovery evidence.

The earlier dense metric was selected for amplitude coordinates. The new
conditional transformation has identity state covariance in its working
Gaussian conditional. Estimating a dense metric over all transformed states
need not help that geometry, especially within the fixed warmup budget.
Test the same primary representative dataset, seed 7101, with the exact block
conditional coordinates and PyMC's diagonal adaptation. Keep every prior,
initialization, seed, chain count, budget and diagnostic threshold unchanged.
No likelihood, tolerance or prior-support change is introduced.

Run the complete declared retry schedule in a fresh directory. Compare only
diagnostic admission and computational cost, never recovery scores. If the
diagonal fit is admitted, use that metric and these coordinates for all
rank-two primary/fallback and sensitivity cases, retaining the existing
attempts. If both metrics are admitted, prefer the diagonal metric to avoid
estimating an unnecessary dense state metric; report both diagnostics and
their concurrent timing limitations. If diagonal adaptation fails all gates,
retain the original metric decision. Do not make a separate metric selection
for each sensitivity candidate.

Other scenarios retain their recorded metric. Any unfinished dense attempt
stopped after a successful alternative is an interrupted engineering attempt,
not a completed failed schedule. The replacement must itself be a complete
four-chain admitted fit or a fully exhausted schedule. The comparison adds
no independent generated dataset and reads no real-data evaluation outcomes.
