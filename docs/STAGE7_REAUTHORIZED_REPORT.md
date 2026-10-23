# Completed reauthorization of the two Stage 7 diagnostics

Status, 2026-10-23: **both attempts completed; the free-persistence matched
fixture passes numerical admission, while the fixed-persistence diagnostic
fails R-hat and bulk ESS**. Recovery remains mixed. This round is closed after
exactly two fresh attempts. Stage 8 and held-out scoring remain closed.

The user reauthorized the same two previously unfinished diagnostics with
realistic limits. The [declaration](STAGE7_DIAGNOSTIC_REAUTHORIZATION_20261023.md)
and [activation](decisions/STAGE7_DIAGNOSTIC_REAUTHORIZATION_20261023.json)
preserved the model, priors, fixture, seeds, draw budgets and diagnostic gates.
The earlier [20-minute timeouts](STAGE7_TERMINAL_REPORT.md) remain historical
resource failures, not statistical verdicts. This report supersedes their
current-status conclusion only to the extent established by the new evidence.

## Execution and numerical results

Both runs started from clean commit `d67871e9c1f1eb4c08c474d3015b68e231babfa7`.
Each used the matched one-factor seed-7101 fixture, ten categories and 108 months,
including all category-10 zero counts; four sequential chains; 2,000 warmup and
2,000 retained draws per chain; target acceptance 0.99; depth 15; sampler seed
7103; initialization seed 7301; NUMBA_OPT=3; and one numerical thread. The two
fits ran concurrently, each with a 90-minute supervisor limit. Neither timed
out or needed the 120-minute global deadline. There were no retries.

| Diagnostic | Maximum R-hat | Minimum bulk ESS | Minimum tail ESS | Minimum BFMI | Retained divergences / depth saturations | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Free persistence, unchanged Uniform(-1,1) prior | 1.005285 | 581.834 | 1064.547 | 0.901503 | 0 / 0 | Numerical pass |
| Persistence fixed at generating value 0.65 | 1.011221 | 354.219 | 618.871 | 0.978947 | 0 / 0 | Failed R-hat and bulk ESS |

All diagnostics are finite and neither fit has unresolved numerical warnings.
The unchanged thresholds are R-hat <1.01, bulk and tail ESS >400, BFMI >0.3,
zero retained divergences and depth saturations, and no unresolved numerical
warnings. The fixed fit's proximity to the limits does not change its failure.
Its largest R-hat is `joint_level_whitened[4]`; its smallest bulk ESS is
`f_free[8]`, the fallback category's loading. In the free fit, `m_sd` has both
the largest R-hat and the smallest bulk ESS.

The [independent audit](simulations/stage7-reauthorized-20261023/diagnostics.json)
recomputed diagnostics from both full saved posteriors, including the failed
fixed fit. It verified the complete artifact inventories, hashes, chain labels,
budgets, source revision and all eight completed-chain checkpoints. Every
posterior variable and every sampler statistic present in the final posterior
is exactly equal to its checkpoint counterpart. Callback conversion additionally
retains a `warning` statistic that `pm.sample` removes by default; this column
is entirely missing-valued in all eight checkpoints. The audit explicitly checks
that exception rather than dropping or ignoring nonempty warnings. No sampler
or fitting source was changed to accommodate this serialization distinction.

## Exploration and recovery

The [chain review](simulations/stage7-reauthorized-20261023/exploration.json),
[plots](simulations/stage7-reauthorized-20261023/README.md) and
[review decision](simulations/stage7-reauthorized-20261023/review-decision.json)
show no additional within-run exploration exclusion for the free fit. Its chain
distributions overlap for persistence, level scale, fallback level, anchor and
fallback loadings, factor time mean and the inspected joint coordinate. Loading
orientation summaries do not reveal chains trapped in opposing orientations.
This is not proof of global exploration. Earlier divergent callback points
are not mode-weight estimates, and conditioning persistence changes the target.

Existing recovery methods were then applied to the admitted free posterior;
no recovery summary is used to rehabilitate the failed fixed fit. The
[recovery record](simulations/stage7-reauthorized-20261023/recovery.json) retains
all results, including the following limitations:

- For the nine known categories, 90% log-intensity intervals cover 90.64% of
  category-month truths; 50% intervals cover 46.50%. Log-intensity RMSE is 0.1090.
- Across all ten categories, 90% log-intensity coverage is 81.57%. The fallback
  category has zero coverage of its generating log-intensities at both reported
  levels. Its expected-count RMSE is 0.01554, while the fixture's reference rate
  is 0.0001. Observed zeros constrain the rate without identifying that exact
  very small generating value.
- Persistence has posterior mean 0.96716 and central 90% interval
  [0.95053, 0.97934], excluding the generating value 0.65. Category-level 90%
  parameter coverage is only 10%. The factor time mean is positive and anchor
  loadings concentrate toward their upper bound. Numerically agreeing chains
  therefore do not establish recovery of the generating level/factor split.
- The 90% intervals for the stationary factor covariance cover only 18-19% of
  the generating matrix entries at lags 0, 1, 6 and 12. Entries are dependent;
  these are descriptive fixed-fixture fractions, not calibration tests.
- Synthetic future paths use the existing forecast recursion, all 8,000
  parameter/state draws and the pre-existing review RNG seed 17101. The fallback
  zero fraction is 95.71% across generated paths and horizons. Their horizon
  summaries concern one generated future realization; no real 2024-2025 target
  was scored and these are not general forecast-calibration evidence.

The practical result is narrower than either blanket success or failure:
joint coordinates permit a complete numerically admitted posterior on the
matched free-persistence fixture, with every fallback count included and no
new persistence restriction. They do not solve the observed structural-recovery
limitations. Fixing persistence did not pass the numerical gates in this single
attempt; this does not prove that estimating persistence always improves mixing.

## Timing and preserved evidence

| Component, seconds | Free persistence | Fixed persistence |
|---|---:|---:|
| Graph construction | 0.65 | 0.65 |
| NUTS step construction/compilation call | 12.16 | 10.77 |
| Sampling call | 3003.90 | 2868.52 |
| Checkpoint serialization within sampling call | 10.65 | 11.16 |
| Total child elapsed through result preparation | 3048.26 | 2917.97 |

The checkpoint row is a subset of the sampling-call row, not an additional cost.
Sampling calls can include initialization and deferred compilation; these fields
do not fully isolate warmup, retained sampling, cache state or process startup.
Supervisor timestamps are retained separately. The two totals correspond to
50.80 and 48.63 minutes and overlap in wall time. They are evidence for this
fixture, not an 84-origin production cost estimate.

Full posteriors, eight chain checkpoints, last unconstrained points, warnings,
heartbeats, logs, supervisor records and synthetic future paths remain under
`artifacts/stage7/reauthorized-20261023/`. Last points are not complete resumable
NUTS states. Committed summaries bind artifact bytes by hash; large artifacts
remain local and ignored by Git. The original interrupted attempts are intact.

The preflight records seven distinct passing checkpoint/guard tests. They verify
seeded draw preservation, checkpoint equality, interruption survival and refusal
of duplicate, expired, closed or altered-budget attempts. Final lint, lock,
configuration, panel and seal checks are in the
[check record](simulations/stage7-reauthorized-20261023/final-checks.json).
No repeated test invocation is counted as an additional test.

## Boundary of this result

The [outcome record](decisions/STAGE7_REAUTHORIZED_OUTCOME_20261023.json) closes
this authorization after its two outcomes. No further fit, seed, draw, prior
amendment, production-pilot amendment or evaluation-roster amendment is made.
The unused replication fixtures remain unused. The original 55-case campaign
and four-case supplement keep their separate totals and negative findings.

The two-factor production pilot remains unvalidated. A one-factor matched
numerical pass is new positive evidence, but poor structural recovery and lack
of replication prevent treating it as production validation. The proposed model
still has no admitted real-data forecast row. Any future model-validation work
needs a separately bounded declaration addressing these findings; this round
does not start another repair loop. Stage 8, incident/economic calibration and
January 2024-December 2025 scoring remain closed. The six outstanding named
comparators and final comparison roster are still required, and a control must
not be substituted into the proposed-model slot.
