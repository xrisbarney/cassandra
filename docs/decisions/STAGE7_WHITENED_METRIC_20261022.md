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

Numerical optimization follow-up: the verified scalar block implementation
removes the tiny-array allocation bottleneck. Repeat the diagonal representative
trial in a new directory under the full same schedule. Its preflight measures
the complete model gradient. If that is at least twice as fast as the previous
0.00252-second preflight under this concurrent workload, supersede the two
unfinished allocation-heavy representative trials and preserve their partial
records. This is a computational replacement; neither interrupted trial is
called a fully failed schedule. If the replacement is not admitted, the dense
metric still requires its own complete schedule with the verified faster
implementation before the metric comparison can be closed. No posterior
recovery score is consulted for this computational decision.

Scheduling follow-up: the verified scalar dense-metric alternative may start
concurrently when the early `dispersion_dense/7101` case releases its fit slot.
This avoids waiting for a fully failed diagonal schedule before beginning the
already required fallback. The preference remains diagonal upon admission,
otherwise dense only after its own complete admission/exhaustion schedule.
If diagonal adaptation is admitted while the dense alternative is unfinished,
stop that alternative with an explicit interruption record. Do not classify
the interrupted alternative as a diagnostic failure or use its recovery scores.
