# Stage 6 completion record

Status, 2026-10-21: **Stage 6 complete for the five initial controls and
rolling development benchmark specified in the plan**. All 60 origins per
control, posterior gates, forecast-ensemble checks and development score tables
are complete. Both final Bayesian scoring jobs finished before the computer
restart; the resumed session verified their saved artifacts rather than refitting.
January 2024 through December 2025 remains unscored.

## Scope and implementation

Stage 6 supplies the development rolling-origin engine and five initial controls:
historical mean, seasonal naive, independent Bayesian count autoregression,
selected ARIMA, and pooled Bayesian NB regression. This matches the initial
controls deliverable in the implementation plan. It does not complete the full
eight-system comparison or proposed-model selection.

The engine refits on each expanding training prefix and saves forecasts before
scoring. It records source revision, configuration and data hashes, seeds,
attempts, posterior draws, diagnostics, predictive checks and runtime components.
Each completed control has 60 scheduled monthly development origins. Eligible
target counts at horizons 1/3/6/12 are 60/58/55/49 origins, with 10 categories.
Forecast paths extending beyond December 2023 are excluded from development
scores. Held-out scoring remains disabled.

The count-law scorer evaluates empirical CRPS, the issued law's negative log
PMF, median MAE, mean RMSE, coverage, widths, interval scores and randomized PIT.
It preserves infinite log scores and sums matched paths for aggregate scores.
Aggregate log scores remain unreported because no aggregate PMF evaluator has
been validated. Tables include all categories, active/sparse strata, aggregates,
and both available-origin and common-origin windows.

## Independent Bayesian control: resolved by a model amendment

The latent AR(1) implementation was not admitted. Its stationary-scale-prior
revision also failed the frozen initial-window gate for category 9: final
bulk/tail ESS 399.15/318.89, despite zero divergences and R-hat 1.00889. Its partial
rolling run was stopped before scoring. The original and amended latent variants,
failed attempts, and the aborted-run record remain available.

The replacement `independent_nb_glar` is a Bayesian conditional log-linear NB
autoregression:

`log(mu[t]) = (1-phi)*m + phi*log(1+N[t-1])`.

It conditions on the first observed count and retains proper priors on m, phi
and NB dispersion. Its future paths recursively feed simulated counts into the
next conditional mean; parameter uncertainty is retained throughout each path.
The independent posterior is also retained for the all-zero fallback category.
There is no Bayesian zero point mass or artificial intensity floor.

The [dated replacement amendment](amendments/OBSERVATION_CONTROL_20261021.md)
specifies the complete law and its empirical initial-window prior anchor.
The observation-driven model family follows
[Liboschik, Fokianos and Fried (2017)](https://www.jstatsoft.org/article/view/v082i05);
Bayesian count-autoregression context is supplied by
[Andrade, Ehlers and Andrade](https://arxiv.org/abs/1509.08666).
The numerical priors are project choices, not purported published recommendations.
This is a substantive replacement, not an exact transformation of the failed model.
It changes none of the proposed model's factors, regimes or channel specifications.

The [validation record](STAGE6_BAYESIAN_VALIDATION.md) gives the likelihood,
gradient, probability and posterior checks. Initial fits passed with both PyMC
and nutpie; production retains PyMC NUTS with NUMBA density/gradient evaluation.
The original four-chain budgets, retry sequence and all convergence thresholds
remain unchanged. The independent control is a forecasting comparator, not a
matched ablation of the proposed latent model.

## ARIMA and zero-series resolution

The [approved zero-series rule](amendments/ZERO_SERIES_20261021.md) is applied
before Gaussian fitting: an entirely-zero observed training series receives a
declared joint zero forecast. Other fitting failures retain their failure status.
The rule also binds the future ETS adapter; ETS itself is not implemented here.

All 65 ARIMA candidates were screened at the 20 quarterly development origins.
Sixty-three completed all origins; two completed 19 and were ineligible.
The two finalists were both refit at all 60 monthly origins. ARIMA(2,1,1) without
trend was selected at the common 6,000-path budget with one-month CRPS
27.28006, versus 27.47851 for ARIMA(3,1,1)
with trend. Both finalists passed the forecast-ensemble check at all four horizons.
The selection is development selection, not a held-out superiority claim.

The declared all-zero boundary was used for category 10 at **60 of 60 origins**
for the selected ARIMA(2,1,1), and at 60 of 60 for the other monthly finalist.
Quarterly screening records **1,298 boundary uses** across accepted candidate-origin
fits (63 candidates with 20 completed origins and two with 19). The two failed
candidate-origin fits have no completed fit record and are not added to that
count. Reissuing forecasts at 6,000 paths reuses those fits; it does not count as
another boundary application.


## Development results and audit

The original 3,000-path check exceeded the frozen 1% CRPS tolerance at horizons
1 and 6 for the independent control. The
[budget decision](decisions/ENSEMBLE_BUDGET_20261021.json) applies the existing
fallback in BASELINE_SPECIFICATIONS: 6,000 reporting paths for every method,
checked against 12,000 on the last six eligible origins at each horizon.
Posterior samples, fitted parameters, NUTS budgets and convergence gates are
unchanged. More predictive paths do not create additional posterior information.
The original 3,000-path artifacts remain historical; both monthly ARIMA finalists
are rescored at 6,000, retaining the original quarterly screening.

Reissuing ARIMA uses its exact joint conditional Gaussian covariance, calculated
from the fitted state-space matrices. It preserves temporal dependence and the
analytic marginal forecast laws; its random-number realization differs from
statsmodels' per-path simulator. Closed-form AR(1)/random-walk tests and an
independent native-simulation comparison validate the covariance calculation.
Bayesian and fixed-control source forecasts reproduce exactly before reissuing.

All 24 method/horizon ensemble checks passed at 6,000 versus 12,000 paths,
including both ARIMA finalists. The final five-control table includes only the
selected ARIMA. The [completion audit](backtests/stage6-complete-20261021/completion.json)
binds verified forecast, score and ensemble artifacts. The
[complete score table](backtests/stage6-complete-20261021/summary.csv) contains
200 rows across horizons, strata and evaluation windows; every row is complete.

One-month development scores, all ten categories and all 60 eligible origins
(600 cells; lower CRPS, negative log mass, MAE and RMSE are better):

| Control | CRPS | Negative log mass | MAE | RMSE | 50% coverage | 90% coverage |
|---|---:|---:|---:|---:|---:|---:|
| Historical mean | 104.92 | 70.579 | 109.34 | 190.52 | 0.137 | 0.235 |
| Seasonal naive | 40.50 | 5.330 | 53.54 | 88.88 | 0.652 | 0.883 |
| Pooled Bayesian NB | 65.58 | 5.227 | 98.36 | 144.02 | 0.410 | 0.933 |
| ARIMA(2,1,1), no trend | 27.28 | 5.018 | 35.27 | 59.08 | 0.645 | 0.872 |
| Independent Bayesian NB autoregression | 36.03 | 4.805 | 48.67 | 78.71 | 0.665 | 0.960 |

Category 10 contributes exactly zero CRPS and negative log mass for the selected
ARIMA on all **222 eligible origin-horizon cells** (60+58+55+49), because its
observed targets are all zero. Other controls assign small positive uncertainty
to that category. The all-category average therefore reflects both performance
on the nine active categories and this difference in boundary treatment. It
remains the protocol's primary score; the active-stratum comparison below
excludes the fallback category and isolates performance on the nonempty series.
The zero-series result is not evidence of successful prediction of threat activity.

One-month development scores on the nine active categories (540 cells):

| Control | CRPS | Negative log mass |
|---|---:|---:|
| Historical mean | 116.57 | 78.420 |
| Seasonal naive | 45.00 | 5.903 |
| Pooled Bayesian NB | 72.82 | 5.803 |
| ARIMA(2,1,1), no trend | 30.31 | 5.576 |
| Independent Bayesian NB autoregression | 40.04 | 5.338 |

CRPS by horizon, using the available-origin window at each horizon:

| Control | 1 month | 3 months | 6 months | 12 months |
|---|---:|---:|---:|---:|
| Historical mean | 104.92 | 108.27 | 112.18 | 117.57 |
| Seasonal naive | 40.50 | 41.05 | 41.83 | 40.50 |
| Pooled Bayesian NB | 65.58 | 67.44 | 69.53 | 71.86 |
| ARIMA(2,1,1), no trend | 27.28 | 29.36 | 32.03 | 38.51 |
| Independent Bayesian NB autoregression | 36.03 | 50.06 | 66.74 | 83.98 |

These are development comparisons, not untouched-test results or significance
claims. Selected ARIMA has the lowest development CRPS here; the independent
Bayesian control has the lowest one-month negative log mass. Passing posterior
diagnostics does not guarantee predictive calibration: the independent control
covers 96% at the nominal 90% level at one month, but only 77.1% at twelve months.
Different horizons have different eligible origins; the saved common-origin
tables support comparisons using the same origin window.

Log-mass Monte Carlo sensitivity is reported separately from the frozen CRPS
admission gate. On the last six eligible origins at each horizon, the maximum
absolute per-cell differences are:

| Control | 6,000 versus 12,000 | Independent 6,000-path seed |
|---|---:|---:|
| Pooled Bayesian NB | 0.0081 | 0.0259 |
| Independent Bayesian NB | 0.0546 | 0.1689 |

These maxima are numerical sensitivity diagnostics, not new admission thresholds.
The independent-seed evidence retains its original source hash.

Both Bayesian controls passed all 60 four-chain fits at the first frozen budget.
The [posterior diagnostic table](backtests/stage6-complete-20261021/posterior_diagnostics.csv)
records all 120 fits; none has divergences or tree-depth saturation.

| Control | Maximum R-hat | Minimum bulk ESS | Minimum tail ESS | Minimum BFMI |
|---|---:|---:|---:|---:|
| Independent Bayesian NB | 1.00661 | 811.16 | 404.28 | 0.7636 |
| Pooled Bayesian NB | 1.00645 | 4262.36 | 2030.79 | 0.8433 |

The [runtime table](backtests/stage6-complete-20261021/bayesian_runtime.csv)
separates step construction/compilation from the sampling call. Median times
were 4.15/176.36 seconds for the independent control and 3.76/184.95 seconds
for the pooled control. The sampling call includes startup and warmup; these
measurements were collected under concurrent load. They are not full-model
Stage 9 cost estimates or isolated post-warmup sampling times.

## Verification and remaining stages

The full implementation test run passed **113 tests**, with one documented
missing-BLAS performance warning. Its saved log and source hashes were checked
after restart; no source changes followed that test run. Ruff, the dependency-lock
check and configuration validation passed again after restart. The
[verification record](backtests/stage6-verification-20261021.json) binds the tested
implementation and configuration. The original Stage 2 freeze remains unchanged;
substantive decisions are separately hashed supplements.

The independent rolling fits record clean source commit `45bd87f`; the pooled
fits record `cd7752c`. All four final forecast-reissue runs record clean source
commit `8095f2a`, and preserve the original fits and diagnostics. The final audit
assembler verified forecast and score artifact hashes after restart. Source
audits are revised-history audits; these results concern monthly CVE-publication
counts and do not establish historical availability of every source field.

The all-zero unknown category remains in the category-equal primary score and
can compress differences. Active-category and aggregate results are also reported;
the fallback category should not be presented as a substantive forecasting success.
Several NMF categories reflect vendor/template wording, limiting their description
as distinct threats. The 21 unlinked KEV additions are preserved for the Stage 8
mask/intercept-only decision; no KEV likelihood is fitted in these controls.

Next is Stage 7's synthetic recovery and calibration work, followed by construction
and selection of the proposed model. The remaining named comparators are RF, ETS,
BSTS-U, Vuln4Cast, Almahmoud B-LSTM and Bakdash BSSM. They remain required before
the full Stage 13 comparison. Incident-channel eligibility and economic target
compatibility remain separate gates. No previous failed evaluation values or
conclusions are reused.
