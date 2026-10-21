# Stage 6 completion record

Status: **execution in progress**. The independent control's rolling run and its
ensemble/scoring checks must finish before this record can declare completion.
The implementation and initial posterior validation are complete. January
2024 through December 2025 remains unscored.

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
trend was selected with one-month CRPS 27.26839, versus 27.48251 for ARIMA(3,1,1)
with trend. Both finalists passed the forecast-ensemble check at all four horizons.
The selection is development selection, not a held-out superiority claim.

## Development results and audit

Pending completion of the independent control's full schedule, ensemble checks
and scoring. Final tables will be assembled only from verified admitted artifacts.

## Verification and remaining stages

The full source test run passed **109 tests**; Ruff and the dependency-lock check
passed. The original Stage 2 freeze remains unchanged, with substantive decisions
in separately hashed supplements. The model run records clean source commit
`45bd87f815beb79d2c270130e2b255a4b73e080a`. Source audits are revised-history audits;
these results concern monthly CVE-publication counts.

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
