# Stage 6: first development benchmark

Status: **in progress**. The rolling-origin engine, count-law scoring and initial
fixed controls are implemented. This is not completion of the comparator suite,
ARIMA grid selection, Bayesian posterior validation or proposed-model selection.
The held-out January 2024–December 2025 outcomes have not been scored.

## Recorded development run

Run `stage6-20261020` records clean source revision
`ecf9cbab43d25ceb509daf630d5c06da68304980`. Code and lock hashes still match the
run's provenance. See the [audit record](backtests/stage6-20261020/audit.json),
[full score table](backtests/stage6-20261020/summary.csv), and
[forecast manifest](backtests/stage6-20261020/forecast_manifest.json).
The manifest's artifact paths resolve under local
`artifacts/backtests/stage6-20261020/`; compressed forecasts and per-cell scores
are retained there and excluded from Git. The committed audit copies retain
the original bytes and hashes.

| Control | Accepted origins | CRPS h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|
| Historical mean | 60/60 | 104.9187 | 108.2726 | 112.1794 | 117.5687 |
| Seasonal naive | 60/60 | 40.5185 | 41.0589 | 41.8348 | 40.5182 |
| ARIMA(0,1,0), diagnostic candidate | 0/60 | — | — | — | — |

These are all-category, available-origin **development** averages, not held-out
results or evidence of a proposed-model advantage. The two fixed controls have
4,884 scored rows including aggregate rows. Another 2,442 scheduled ARIMA rows
retain failed-fit status and missing scores; altogether the score artifact has
7,326 rows. Across horizons there are 222 eligible origin/horizon pairs per method.

Every ARIMA failure occurs at category 10. In the
[first failure record](backtests/stage6-20261020/first_arima_failure.json), the
initial L-BFGS fit fails, Powell reports convergence, and the required final
L-BFGS refinement fails. All 60 failures are listed in the
[failure register](backtests/stage6-20261020/fit_failures.csv). No zero-series
fallback or alternate optimizer sequence has been introduced.

Unknown contributes one-month CRPS of about 0.000013 for historical mean and
0.025002 for seasonal naive. Their active-topic CRPS values are 116.5763 and
45.0177 respectively. Thus the primary equal-category average includes an easy
fallback target; it must not be presented as successful threat forecasting.
The seasonal scale floor leaves positive unknown-category predictive draws,
which also enter the aggregate forecast.

The [ensemble checks](backtests/stage6-20261020/ensemble_check.json) pass for both
fixed controls at all four horizons, using each horizon's last six eligible
development origins and 1,000/3,000/6,000 paths. Resimulation at 3,000 exactly
reproduces the saved random streams. Differences satisfy the frozen 1% tolerance.
This does not certify other methods' Monte Carlo stability.

Fitting plus forecast generation totals approximately 18.53 seconds for historical
mean, 19.07 seconds for seasonal naive and 12.54 seconds for the failed ARIMA
candidate; these exclude upstream verification and separate scoring. They provide
no timing evidence for the full Bayesian model.

## Preserved panel decisions

The [carry-forward note](STAGE6_8_CARRY_FORWARD.md) records the accepted loading
order, observed-zero fallback category, topic-wording limitation and two allowed
unknown-KEV treatments for Stage 8. The Stage 5 panel and its dictionary remain
unchanged. Neither KEV treatment has been selected or fitted.

## Implemented behavior

- Train afresh on January 2010 through each development origin. Sixty origins
  run from December 2018 through November 2023. The scorer admits only targets
  in 2019–2023: 60/58/55/49 origins at horizons 1/3/6/12.
- Save 3,000 matched twelve-month paths, distribution parameters, fit records,
  warnings, seeds and hashes before a separate scoring invocation. Future paths
  can extend beyond December 2023, but those targets are not scored. The engine
  excludes 2024–2025 outcome values before constructing any training matrix.
- Historical mean uses the frozen smoothed rate and Poisson law. Seasonal naive
  uses last year's count, training-only seasonal-difference RMS and the frozen
  0.5 scale floor, with the censored/rounded Gaussian count law.
- The ARIMA adapter implements the 65 frozen candidates and joint state-space
  future simulation. This first run exercises only ARIMA(0,1,0), no trend. It is
  explicitly a diagnostic candidate, not the selected ARIMA comparator.
- Independent NB AR(1) and pooled NB regression adapters implement the frozen
  priors, stationary initialization where required, four-chain sampling budgets,
  retries and diagnostic rejection. Their likelihoods and gradients compile and
  are finite on a small test that includes an all-zero series. Their substantive
  posterior validation and rolling fits remain outstanding. The effort-adjusted
  paired control also remains to be added with the Stage 8 effort input.
- Scores include empirical CRPS, actual predictive count log mass, median MAE,
  predictive-mean RMSE, 50%/90% coverage, widths and interval scores, and ten seeded
  randomized PIT values per cell. No clipping, NB moment matching or probability
  floor repairs a tail. Aggregate scores sum within matched paths; aggregate log
  mass is deliberately unreported until a validated evaluator exists.
- Report all-category, active, sparse, unknown and aggregate summaries, with both
  available origins and the common 49-origin horizon window. An incomplete fit
  schedule produces an incomplete primary row, not an average over successes.
  This first runner treats a multi-category origin atomically; partial-category
  ARIMA fits are preserved in failure records but are not scored as a completed
  active-stratum comparison.

The scoring pass checks source/panel and forecast hashes, calendar completeness,
observed counts, category order and the scheduled origin set. It opens a
development-count execution gate for that run only; the historical Stage 5
`scoring_ready: false` manifest is not rewritten. Auxiliary channels and held-out
execution remain locked. Dates on current-vintage CVE records do not establish
historical availability: this remains a **revised-history** experiment.

## Verification and limits

The tests exercise exact score calculations, normalized count probabilities,
extreme tails, discrete quantiles, serial dependence in ARIMA paths, trend
extension, deterministic random streams, aggregation, category permutations,
future-data mutation, missing-month rejection, failed-fit retention, immutable
artifacts and held-out refusal. The end-to-end test saves forecasts before
scoring and checks that a deliberate failed origin prevents a primary score.
The suite reports **77 passed**. Ruff passes, `uv lock --check` is current and
configuration validation reproduces development counts 60/58/55/49 and held-out
counts 24/22/19/13. The five frozen Stage 2 document hashes and the Stage 5 panel
manifest hash remain unchanged.

The Bayesian graph checks establish numerical construction, not convergence,
recovery or useful full-model runtime. PyTensor reports no linked BLAS library;
native C++ compilation works, but matrix-heavy performance must be measured in
Stage 7. Preserve that warning in the performance record. Stage 9 must separate
graph/compile/cache time, process startup, warmup, sampling and serialization;
the initial adapter currently reports step construction separately from the
combined sampling call, not a complete timing decomposition.

## Outstanding gates

1. Validate the Bayesian controls' posterior behavior and computational budget;
   run their development fits under the frozen convergence requirements.
2. Execute frozen ARIMA screening and finalist selection. Investigate the
   zero-series optimizer behavior without replacing failed forecasts silently.
3. Implement and validate the remaining requested comparators (RF, ETS, BSTS-U,
   Vuln4Cast, Almahmoud B-LSTM and Bakdash adaptation). These initial controls do
   not replace them.
4. Complete ensemble/Monte Carlo checks for every admitted method, proposed-model
   development selection and its selection record before held-out execution.

Commands and paths are in the [README](../README.md). The distribution definition
and selection rules remain the unmodified [frozen baseline specification](BASELINE_SPECIFICATIONS.md).
