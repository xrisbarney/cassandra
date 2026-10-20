# Stage 2: frozen statistical analysis and eligibility rules

Design version 1.0, 2026-10-20. Complements [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) and [BASELINE_SPECIFICATIONS.md](BASELINE_SPECIFICATIONS.md). All formulas and thresholds below are prospective design choices for the rerun; no reported statistic is inherited from the earlier evaluation.

## 1 Targets and eligibility

The primary target is the monthly publication count in each of K=10 frozen categories, including unknown. The primary score is one-month CRPS; all four horizons and all requested metrics are reported without selecting favourable subsets.

For each horizon, use the protocol's fixed origin calendar. A count cell is eligible only if its source-month is audited available and its CVE cohort is assignable under the declared vintage track. A CVE with missing text goes to unknown. Missing severity or CPE does not remove an otherwise valid count.

Counts require at least 60 audited training months at an origin. ARIMA/ETS/BSTS additionally require a complete monthly count grid in their primary implementation; supervised methods require all feature/target entries in a training row. Do not close calendar gaps by compressing time. If the initial count panel cannot support a common complete grid, amend the modelling design before fitting rather than silently filling gaps.

All model comparisons use the same target mask. Training-data requirements do not redefine the target mask: a method unable to fit is a recorded method failure. No test topic is excluded because it is sparse, emerging, or poorly predicted.

Severity requires a nonempty observed-score cohort. EPSS additionally requires an acceptable dated snapshot and a supported measurement version under the origin rule. Require at least 12 observed training marks per category for a fitted-channel mark comparison; retain prior-dominated categories in the joint model but label their mark forecasts separately. Version changes unsupported at the forecast origin have explicit eligibility flags; do not select them by their realised scores.

SEC Item 1.05 has no full-month training observations before the December 2023 initial test origin under the current masking rule. It therefore cannot be a fitted auxiliary channel at that origin. Incident-channel research is a separate later-start task, with at least 12 audited full training months and a protocol frozen before its scores are viewed. It is not silently presented as part of the full-window primary model.

Primary totals use equal weighting over eligible categories within each origin and then over origins. Publish category and origin counts. Aggregate-count scoring requires all categories present; a partial category sum is a different target. Reference-loss validation remains gated by compatible economic evidence and disjoint calibration/evaluation observations.

## 2 Fixed selection and result tables

Compare one development-selected proposed count model against seven monthly comparators: RF, ARIMA, ETS, seasonal naive, BSTS-U, Almahmoud B-LSTM adaptation, and Bakdash-inspired NB BSSM. Vuln4Cast is the eighth comparator and has its own eight-origin quarterly table.

The result set must contain:

1. Monthly metrics by method and horizon: CRPS, negative log mass, median MAE, mean RMSE, 50%/90% coverage, interval width, and interval score.
2. All-origin and 13-common-origin horizon comparisons.
3. Active, sparse, and unknown category summaries, using the initial-window stratification.
4. PIT histograms, per-origin paired differences, and computational success rates.
5. Quarterly aggregate comparison with Vuln4Cast.
6. Development ablations, with any subsequently reported held-out ablations explicitly predeclared and not used for further selection.

A run may refit parameters on elapsed test observations according to the fixed schedule. It may not retune hyperparameters, select channels, change transforms or increase search based on test scores. A model-design change after test inspection requires a new exploratory label; it does not overwrite the frozen run.

## 3 Exact paired DM/HAC specification

For each baseline b, selected model a, and horizon h, define

`d[t] = mean_k(CRPS_b[t,h,k] - CRPS_a[t,h,k])`.

Compute this on chronological monthly origins using the same eligible categories. Positive differences favour a. Do not flatten topics into the time axis. If the required origin series has gaps or model failures, report descriptive differences only; do not treat nonadjacent months as adjacent for HAC.

Let n be the number of complete contiguous origins, dbar their mean, and

`gamma[j] = sum_{t=j+1..n} (d[t]-dbar)*(d[t-j]-dbar) / n`.

The frozen lag rule is

`L = min(n-1, max(h-1, floor(4*(n/100)**(2/9))))`.

Use the Bartlett HAC long-run variance

`V = gamma[0] + 2*sum_{j=1..L} (1-j/(L+1))*gamma[j]`.

The reported statistic is

`DM_HLN = sqrt((n+1-2*h+h*(h-1)/n)/n) * dbar / sqrt(V/n)`.

Use the two-sided Student-t reference with n−1 degrees of freedom as an **approximation**, not an exact finite-sample distribution. Report raw dbar and the variance estimate beside the statistic. This combines a predeclared HAC bandwidth and the HLN correction; it does not guarantee valid inference under arbitrary nonstationarity or heavy-tailed loss differences.

Sources: [Harvey, Leybourne and Newbold (1997)](https://doi.org/10.1016/S0169-2070(96)00719-4) for the small-sample comparison approach, and the [Bank of England's implementation](https://github.com/bank-of-england/forecast_evaluation/blob/main/src/forecast_evaluation/tests/diebold_mariano.py) as an independent implementation reference. Our bandwidth, eligibility thresholds and multiplicity family are explicit project choices.

Report a p-value only when n>=max(20,4*(L+1)), V is positive and finite, the correction is real and positive, and every scheduled comparison cell is available. Otherwise report `not estimable under protocol`, not p=1 or a favourable significance label. This threshold is a conservative reporting rule, not a demonstrated size-control guarantee.

Consequences for the fixed calendar:

| h | n | L | Required n | Decision |
|---|---|---|---|---|
| 1 | 24 | 2 | 20 | Exploratory statistic and p-value permitted |
| 3 | 22 | 2 | 20 | Exploratory statistic and p-value permitted |
| 6 | 19 | 5 | 24 | Descriptive difference only |
| 12 | 13 | 11 | 48 | Descriptive difference only |

The fixed multiplicity family is the 14 comparisons from seven monthly baselines at h=1 and h=3, using CRPS only. Apply Holm adjustment across all 14 hypotheses. Reserve unavailable comparisons as p=1 **inside the adjustment calculation only**, and display their raw and adjusted p-values as unavailable. This prevents missing methods from silently reducing the family. Report raw and adjusted values, the sign of dbar, and a 0.05 adjusted threshold. No confirmatory interpretation is claimed from this short series.

Sensitivity: repeat using L=h−1 and L=min(n−1,max(h−1,3)), subject to the same eligibility rule. These are labelled sensitivities; they cannot replace the primary lag rule because they give a smaller p-value. Joint scores, mark scores, diagnostic controls, and quarterly comparisons are descriptive secondary results outside the 14-test family.

## 4 Bootstrap, calibration, and Monte Carlo uncertainty

For exploratory uncertainty on mean differences, use the protocol's moving-block bootstrap: 2,000 replicates; block length max(3,h); sample uniformly from all overlapping full blocks, concatenate and truncate to n. Keep model pairing and the full topic vector. Report percentile 2.5%/97.5% bounds. Require at least four block lengths in n; otherwise omit the interval. Report block-length sensitivities at 2 and 4 for h=1, and 2 and 4 for h=3 where permitted. These do not establish robustness to structural breaks.

Coverage is the fraction inside each model's central interval. Also report width and the interval score `(u-l) + (2/alpha)*(l-y)*I(y<l) + (2/alpha)*(y-u)*I(y>u)`. Quantiles of count laws use the smallest integer with CDF at least the requested probability. For empirical forecasts use the corresponding inverse empirical CDF convention.

PIT uses `F(y-1)+v*(F(y)-F(y-1))` for counts. Fix randomization streams, draw ten randomizations for sensitivity, and display ten equal-width bins for the first stream and bin-frequency ranges across streams. No iid KS p-value is reported. An uncalibrated forecast does not become acceptable because a short-sample test fails to reject.

Report predictive-simulation and posterior Monte Carlo uncertainty separately from between-origin sampling uncertainty. Use the common ensemble-size check in the baseline specification. For NB-mixture log scores, repeat the forecast calculation with independent future-innovation seeds on development origins; numerical underflow is investigated with log-domain arithmetic, never hidden by a probability floor.

## 5 Manifest contract and execution gates

Stage 2 freezes the schema and eligibility rules. Stage 4 populates the source/cell manifests; it cannot fabricate coverage before ingestion. Each forecast/score record must contain:

`run_id, method_id, configuration_hash, source_manifest_hash, vintage_track, origin_month, target_month, horizon, target_kind, topic_or_sector, training_cutoff, observed_flag, eligibility_reason, fit_status, retry_number, seed, forecast_artifact_hash, scoring_version`.

Source records additionally contain source identifier, publication date, retrieval timestamp, available-at timestamp or `unknown`, source/model version, file hash, revision policy, and exclusions. Training, development and evaluation partitions must be recoverable without examining outcomes.

Before held-out scoring: populate and hash manifests; pin runtime packages; pass scorer and forecasting validation; complete development selection; save selected configurations and a protocol hash. These are execution checks for the frozen design. A substantive change to a probability law, candidate grid, target, or statistical rule requires a dated design amendment before test-score access.
