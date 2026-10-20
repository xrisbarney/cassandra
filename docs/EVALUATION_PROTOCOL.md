# Evaluation protocol: decisions before implementation

Version 1.0.1, 2026-10-20. Stage 2 design is frozen for the requested 2024–2025 forecast window, together with [BASELINE_SPECIFICATIONS.md](BASELINE_SPECIFICATIONS.md) and [STATISTICAL_ANALYSIS_PLAN.md](STATISTICAL_ANALYSIS_PLAN.md). The patch revision records the editorial clarifications in [AMENDMENTS.md](AMENDMENTS.md); scientific choices remain version 1.0. This is a design record, not an executed evaluation or a claim of external preregistration. It resolves the evaluation choices in [the implementation plan](../IMPLEMENTATION_PLAN.md) and uses the model definitions in [SPECIFICATION.md](SPECIFICATION.md). Substantive changes require a dated amendment before held-out score access.

Use the supplied failed evaluation only to identify the requested baselines, measurement families, and comparison tasks. Do not use its reported values, rankings, procedural narrative, or conclusions as evidence or selection criteria. The carry-forward scope is recorded in [PRIOR_EVALUATION_REVIEW.md](PRIOR_EVALUATION_REVIEW.md). Every result will be computed afresh.

The relevant original user-supplied passages are saved in [USER_EVALUATION_EXTRACTS.md](sources/USER_EVALUATION_EXTRACTS.md). This closes the comparator/measurement provenance gap without importing previous results.

## 1 What the first implementation must establish

The primary task is forecasting monthly, integer CVE publication counts by a fixed topic map. One-month count CRPS is the primary measure, with the requested comparator suite in Section 5. An independent negative-binomial latent AR(1) model using the same data and effort treatment additionally isolates the value of shared structure.

Latent intensity is not observed malicious activity. Count accuracy cannot validate that interpretation. Severity prediction, regime recovery, additional observation channels, and economic translation are separate questions with separate evidence.

| Decision | Initial choice |
|---|---|
| Time unit and horizons | Calendar months; simulate a coherent 12-month path and report horizons 1, 3, 6, 12 |
| Initial topic map | TF–IDF plus NMF on 2010–2018 descriptions only; hard assignments; freeze vocabulary and topics |
| Topic count | Primary pilot: 9 learned topics plus the reserved unknown category, total K = 10 |
| Category assignment | Largest NMF coefficient; deterministic tie break by topic ID; unknown if no usable text representation or all coefficients are zero |
| Initial Bayesian factor model | Two intensity factors, one regime, kernel off, count channel only, effort off |
| Severity extension | Two severity factors; the bounded, coarsened observation model in SPECIFICATION.md |
| Effort extension | Origin-fitted filtered trend from SPECIFICATION.md; hold-last forecast; compare against no effort |
| Regime challengers | R = 2 and R = 3, assessed on development data; neither is required to win or retained automatically |
| Kernel challenger | Off initially; the specified finite kernel is tested after stable factor/regime fits |
| Auxiliary channels | EPSS, KEV, and incidents excluded from the first benchmark; introduced separately after their audit |
| Inference | NUTS for the initial small panel; VI is a separately labelled approximation experiment |

The old eight-topic panel is a possible reconstruction exercise only if its topic definitions and provenance are recovered. Do not call the new ten-category panel a replication of it. K = 20 is a later resolution sensitivity check, not a competitor selected by lower raw per-topic CRPS: changing category granularity changes the targets.

Use all ten categories, including unknown, in the primary count score. For descriptive strata, adopt the old active-topic rule: at least 50 assignments and 24 nonzero months **in 2010–2018**, frozen thereafter. Report active and sparse results without dropping either from the main result. Do not enforce the old claim that all categories are active.

## 2 Calendar, development, and held-out forecasts

| Period | Role |
|---|---|
| January 2010–December 2018 | Initial estimation window and topic-map training |
| January 2019–December 2023 | Development outcomes for model and implementation choices |
| January 2024–December 2025 | Held-out forecast evaluation for this rerun |

Freeze preprocessing rules, model choices, hyperparameters, metrics, and comparison procedures using data through December 2023. Do not inspect new 2024–2025 evaluation scores until those choices are frozen. The previous run's numbers and conclusions do not guide selection. “Held out” describes separation from development in this rerun; do not claim that this calendar period has never been examined in earlier work. A separate prospective test is optional, not a prerequisite for the requested historical comparison.

This is a rolling held-out evaluation, not a single fixed-origin forecast of all 24 months. Once a month has elapsed, its observations may enter the next scheduled training fit. Its evaluation score may not be used to change the model design. At every origin, all target months remain future to that origin.

An origin T is the end of a month; an h-month forecast targets month T+h, not a cumulative h-month count. Each fit uses observations dated no later than T. A separate cumulative forecast must sum a coherent simulated path and carry its own target label.

For the 2024–2025 historical evaluation, use all monthly origins from December 2023 for which the target remains inside the period:

| Horizon | Origins, inclusive | Targets, inclusive | Number of origins |
|---|---|---|---|
| 1 | December 2023–November 2025 | January 2024–December 2025 | 24 |
| 3 | December 2023–September 2025 | March 2024–December 2025 | 22 |
| 6 | December 2023–June 2025 | June 2024–December 2025 | 19 |
| 12 | December 2023–December 2024 | December 2024–December 2025 | 13 |

Also report a common-origin comparison using December 2023–December 2024, 13 origins at every horizon. This separates horizon effects from changes in the origin set. Missing source coverage can reduce eligible cells, but never just because a forecast performed badly.

Development follows the same construction, beginning at December 2018 and ending at December 2023 minus h months: 60, 58, 55, and 49 origins respectively. Screening uses the 20 quarterly development origins defined in the baseline specification, then the best two valid configurations per grid use all monthly development origins. The held-out schedule remains monthly.

### Availability check for the requested window

Checked on 2026-10-20 without downloading or scoring the target panel:

- NVD's official annual-feed metadata endpoints for 2024 and 2025 both returned HTTP 200. Both feeds reported modification dates of 2026-10-16. This establishes feed availability, not complete CVSS/CPE coverage or historical metadata vintages. Select target records by publication timestamp, not the year in the CVE identifier. [NVD feeds](https://nvd.nist.gov/vuln/data-feeds).
- The official EPSS repository's 2024 and 2025 directory listings contain all 12 expected month-end files in each year: 24 of 24 listed, none missing. File contents and CVE-level score availability remain to be audited. [Official EPSS archive](https://github.com/empiricalsec/epss_scores).
- EPSS v4 began publishing on 2025-03-17, so the extended test window crosses a measurement-version change. Preserve snapshot versions and apply the specification's origin-aware handling. [FIRST data documentation](https://www.first.org/epss/data.html).
- SEC provides historical public filing indexes and documents. This establishes an access route, not a verified incident panel. [SEC data access](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).

These checks support proceeding with a 2024–2025 technical backtest. They do not establish that compatible realised economic-loss observations exist for every test month. Economic validation retains its separate evidence requirements.

## 3 Information available at an origin

Use an expanding-window refit at **every monthly origin for every model** in the main comparison. Warm starts may use earlier fits only. A once-fitted full-panel posterior, smoothed state, HP trend, or parameter estimate cannot be used to forecast earlier months. Filtering regime beliefs afterward does not remove that leakage.

Within a fit ending at T, it is legitimate to estimate historical states using observations through T. The forecast must start from that fit's terminal distribution, not from a later fit's estimate of the state at T. Hyperparameters chosen on development are fixed for the historical evaluation.

This follows the rolling-origin principle that training precedes each forecast target. [Hyndman and Athanasopoulos, time-series cross-validation](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation).

Keep two explicit data tracks:

- **Revised historical data:** a pinned retrieval snapshot, with records restricted by publication month. Later-revised text, severity, and product metadata remain a disclosed limitation. This is the initial feasible track.
- **Point-in-time reconstruction:** only fields with verified availability by the applicable cutoff. Run it separately if the audit supports it. No historical real-time claim is permitted merely because rows were filtered by publication date.

Fit imputation rules, scaling, trend parameters, learned text features, calibration wrappers, and forecast residual models inside the permitted training window. The frozen topic map is the stated exception: it is fitted once on the initial window. Source availability masks are common to all models forecasting a given target.

Future realised EPSS, CVSS, counts, event identities, and disclosure outcomes are never covariates for an unconditional forecast. Announced future reporting changes may enter only if documented as known at the origin. Preserve the actual source version and its date.

## 4 Data decisions and audit gates

| Item | Decision now | Evidence still required |
|---|---|---|
| NVD counts | One publication-month assignment per retained CVE ID; explicit rejected/withdrawn-record policy in the data dictionary | Reconcile duplicates, statuses, monthly totals, and retrieval coverage; compute dataset sizes afresh |
| Empty versus unavailable | An audited zero count is observed; an unavailable source-month is masked | Coverage report explaining each missing interval |
| CVSS | Preserve raw scores and source/version; one deterministic selected score per CVE before computing the monthly mean | Freeze version/source precedence in the ingestion data dictionary before forming severity targets; report missing-score fractions |
| EPSS cohort | Newly published CVEs, using the latest snapshot no later than month-end and no more than 7 calendar days old | Verify archive dates, availability, gaps, and version changes; seven days is an operational tolerance, not an empirical finding |
| EPSS absent CVE | Missing score, never zero | Snapshot-level join and score-count audit |
| KEV | Separate count channel based on catalogue addition date; exclude launch backlog month as specified | Reconstruct dated additions; do not adopt the old catalogue count |
| SEC | First qualifying disclosure per incident–registrant pair; link amendments and repeats | Manually review the small candidate set; separate pre-mandate text matches from Item 1.05 records |
| Incident coverage | Apply the specification's partial-month masks and reporting-population rule | Audit actual collection coverage; absence of a filing is zero only within an audited observation window |
| BEA and event losses | No monetary calibration in the first technical benchmark | Compatible geography, accounting units, period, vintage, and loss definition |

If the seven-day EPSS rule leaves insufficient coverage, report that limitation. A changed tolerance defines a dated development amendment or sensitivity analysis; it is not selected to improve historical evaluation scores. No usable observations means the channel is unavailable, not identified by its prior.

Freeze a machine-readable source manifest and cell-eligibility manifest after the audit and before model fitting. Their schemas must include retrieval hash, source date, available-at date or an explicit unknown flag, origin, target, vintage track, channel, category, and exclusion reason.

## 5 Baselines, selection, and ablations

The required comparison suite carried forward from the supplied evaluation is:

| Comparator | Comparison task |
|---|---|
| RF, following the Kia-style forecasting approach | Per-topic monthly forecasts |
| ARIMA | Per-topic monthly forecasts |
| ETS | Per-topic monthly forecasts |
| Seasonal naive | Per-topic monthly forecasts |
| BSTS-U | Univariate Bayesian structural time-series forecasts |
| Vuln4Cast | Matched quarterly aggregate forecasts; topic disaggregation only as an explicitly defined adaptation |
| Almahmoud B-LSTM | Matched monthly forecasts with verified preprocessing and predictive distribution |
| Bakdash BSSM | Matched monthly rolling forecasts using an audited implementation or documented reimplementation |

Historical-mean counts, independent negative-binomial latent AR(1) processes (Version A), and a pooled count regression are additional diagnostic controls, not replacements for the requested suite. Probability laws, priors, tuning grids, compute rules and adaptation records are now fixed in [BASELINE_SPECIFICATIONS.md](BASELINE_SPECIFICATIONS.md). Runtime package builds are pinned during Stage 3; the external source revisions are already recorded. The old performance rankings and claims of exact replication are not carried forward.

A point method needs an explicit distributional wrapper. The baseline specification fixes a log-scale residual mixture for RF/B-LSTM, native transformed Gaussian forecasts for ARIMA/ETS/seasonal naive, and posterior predictive laws for Bayesian models. Forecast-error calibration respects the training cutoff; no substitute distribution is created only at scoring time.

For a continuous forecast X converted to a nonnegative integer forecast Y = max(0, round(X)), the induced probability law must be scored: P(Y=0)=F_X(0.5) and P(Y=k)=F_X(k+0.5)-F_X(k-0.5) for k >= 1 when X is continuous. Generate draws using the same conversion. This is an explicitly transformed forecast, not its original Gaussian density.

Use development one-month mean count CRPS as the primary selection criterion. Report all other horizons and log scores for each shortlisted model. Apply the fixed grids, simulation stability check and tie rules in the baseline specification. No model qualifies with unresolved inference failures. Historical evaluation results cannot change the selected configuration without relabelling the exercise exploratory.

Separate these comparisons:

| Comparison | Controlled change |
|---|---|
| A versus B′ | Add factors while retaining AR residuals |
| B′ versus B | Replace AR residuals with iid residuals |
| Effort off versus on | Same architecture, with or without the origin-fitted proxy |
| R = 1 versus 2 versus 3 | Add switching levels; retain only with an explicit development rationale |
| Kernel off versus on | Test on the selected count model, not only a discarded four-channel model |
| N versus N+B, N+E, N+V, N+D | Add each eligible source separately; then test justified combinations |
| Hierarchical versus fixed independent priors | Change pooling explicitly while matching reasonable marginal scales |

With independent intensity and severity processes and R = 1, adding B need not change the count posterior at all. Count improvement is not an acceptance condition for that severity extension; assess severity forecasts and synthetic recovery. If a channel changes counts through shared regimes or other coupling, state the coupling.

Setting factor loadings to zero does not make topics independent while shared regimes or other random common components remain. Widening all priors tenfold is prior sensitivity, not removal of hierarchy. If D is absent, its disclosure parameters are not estimated from data; they are not required by the stated Mλσ economic shock.

Run population-scale experiments only after the small-panel experiment is stable. Hold channels, target definition, and refit schedule fixed when attributing differences to scale. If VI is necessary, compare VI with NUTS on the same small panel first. Record all other changes.

## 6 Scores must evaluate the issued distribution

Use lower-is-better scores throughout. Proper scores assess the probability forecast actually issued; interval coverage should be considered together with width. [Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

1. **Primary count CRPS:** compute the empirical-ensemble CRPS using the pairwise M² formula in the supplied excerpt. It is the exact CRPS of the empirical forecast, not an unbiased estimate of the underlying model's CRPS. Use equal ensemble sizes across methods and assess Monte Carlo sensitivity; dependent posterior draws require additional care.
2. **Count log score:** use the predictive probability mass. For the NB mixture, compute `-(logsumexp(log_NB_pmf(y | mu[m], psi[m])) - log(M))`, averaging over posterior and future latent draws. Retain the corresponding component means and concentrations in forecast artifacts. This is a Monte Carlo approximation to the predictive log score; examine its stability.
3. **No post-hoc tail repair:** no winsorization, moment-matched substitute law, density floor, or removal of extreme draws from the primary forecast. Diagnose numerical failures separately from valid heavy-tailed forecasts. A deliberately robust forecast is a separate model whose complete distribution is issued before scoring.
4. **Point errors:** MAE uses the predictive median; RMSE uses the predictive mean. Label any median-based RMSE separately. Report unstable Monte Carlo mean estimates rather than substituting the median unnoticed. The mean/median distinction matches squared/absolute error objectives. [Forecast accuracy definitions](https://otexts.com/fpp3/toolbox.html#scale-dependent-errors).
5. **Intervals:** central 50% and 90% coverage, mean width, and interval score. Do not bold the greatest coverage as best calibration. Discreteness, sample size, and dependence affect interpretation.
6. **PIT:** randomized count PIT from the same predictive CDF used by the model, with reproducible randomization. Display histograms and sensitivity to randomization. A pooled iid KS p-value is not the default because topics and forecast errors can be dependent; the KS statistic alone does not identify the cause of rejection.
7. **Marks:** apply the specification's conditional mark forecast, actual availability masks, and censored endpoint masses. Raw severity observations are not observations of latent severity.
8. **Aggregate counts:** sum categories within each joint draw; sum months within each path for quarterly totals. Use aggregate CRPS. Report aggregate log score only after validating its probability calculation; summing marginal log scores is not an aggregate log score.

Primary summaries weight every eligible category equally, then every origin equally. Report the aggregate-count score and active/sparse strata separately. Changes in K require common aggregate targets or another explicitly comparable task. Ratios to a baseline must use the same target cells and origin set.

## 7 Dependence and uncertainty in comparisons

For baseline b, model a, and fixed horizon h, form one paired difference per origin:

`d[T,h] = mean_k(CRPS[b,T,h,k] - CRPS[a,T,h,k])`.

Positive values favour a. Topic-month cells are not independent replicates. A 24-origin design yields 24 origin-level differences regardless of the number of topics. Retain the full topic vector when resampling an origin.

Report mean differences, origin-level plots, and dependence-aware uncertainty. DM comparison remains a requested measurement, subject to a defensible origin-level implementation and the sample-size limits below. No significance values or conclusions are inherited from the previous run.

For an exploratory moving-block bootstrap, use contiguous origin blocks of length max(3,h), 2,000 replicates, and preserve model pairing and the complete topic vector. Report sensitivity to feasible shorter/longer blocks. This is a chosen diagnostic, not a guarantee of valid coverage under structural change. Suppress the interval when fewer than four such blocks fit in the origin series; report the descriptive difference instead. The primary historical design consequently has no such interval at h = 6 or 12.

The [statistical analysis plan](STATISTICAL_ANALYSIS_PLAN.md) now fixes the DM/HAC calculation: Bartlett long-run variance with an explicit bandwidth formula, HLN correction, an approximate t reference, a 14-comparison Holm family, and minimum-length rules. Only h=1 and h=3 qualify for exploratory p-values under the current complete calendar. Horizons 6 and 12 and the eight-quarter comparison remain descriptive. No statistic is inherited from the old run.

## 8 Computation, failures, and reproducibility

The baseline specification fixes the initial NUTS budget and two retry budgets, and defines MLE/neural failure handling. An initial budget is not evidence of convergence. Apply the specification's R-hat, bulk/tail ESS, divergence, and Monte Carlo error requirements. Estimate rolling-fit cost before expanding the experiment.

Use the frozen retry policy. A feasibility-driven budget change must be documented before held-out scoring. Structural reparameterization requires a new version and revalidation; it is not a hidden origin-specific repair.

Persist every failed fit, warning, retry, and score-computation failure. No difficult topic, emerging category, or extreme but valid outcome may be deleted to improve a table. The primary comparison requires valid forecasts on all scheduled eligible cells. If that cannot be achieved, report the incomplete result and failure rate; a common-success subset is secondary and potentially selected.

A seasonal fallback can be evaluated as a separately named operational hybrid, including every fallback use. It cannot be reported as the unmodified model. A successful process exit does not establish successful inference, and equal MCMC budgets do not imply equal posterior accuracy.

Save the training cutoff, feature fit cutoff, source manifest, model configuration, posterior diagnostics, forecast draws/component parameters, eligible target IDs, seeds, code revision, and scoring version. Never use realised outcomes to select an ensemble subset. Check score stability against simulation size on development forecasts before freezing the reporting budget.

Published benchmarks require an audited source/version and a written adaptation record. Distinguish verbatim reproduction, reimplementation, and adaptation. Match rolling versus open-loop forecasts, target frequency, available information, and update schedule. An old software version does not intrinsically prevent rolling-origin evaluation.

## 9 Economic evidence is a separate gate

The candidate reference events require independent source verification before they can enter a common loss dataset. Do not copy dollar ranges from the failed evaluation into a calibration likelihood. For each source establish geography, time window, currency year, direct versus indirect loss, accounting quantity, and whether it is a realised estimate or a modelled scenario.

The old lognormal-range rule is not adopted. A published range need not be a probabilistic 90% interval. If a source genuinely supplies positive 5th/95th percentiles a,b of a lognormal quantity, its log-location is `(log(a)+log(b))/2` and log-standard deviation `(log(b)-log(a))/(2*z_0.95)`; its median is sqrt(a*b), not the arithmetic midpoint. This conversion alone does not define the likelihood relating a documented estimate to model-predicted loss.

Before calibration, settle the event likelihood and whether the annual anchor measures the same quantity as the model. If no compatible anchor exists, omit it and disclose the remaining scale uncertainty. Do not use insurance claims, ransom payments, global estimates, and US gross-output disruption as interchangeable targets.

Keep these tasks distinct:

- **Calibration check:** reproduce training events; useful for fit diagnostics only.
- **Retrospective leave-one-event-out check:** exclude the held-out event from the likelihood, tuning, and any anchor that incorporates its loss; disclose any later information in the technical and economic inputs.
- **Genuine pre-event forecast:** every technical fit, economic calibration input, exposure estimate, and model-selection decision must predate the event.
- **Event-conditioned scenario:** may condition on the event's identity or exposure footprint; it is not an unconditional pre-event prediction.

A monthly total-loss forecast is not automatically a prediction of the eventual loss from a selected incident. Reconcile the horizon, background losses, event attribution, and target definition before comparing them. Range overlap and width ratios are descriptive; four events cannot establish calibrated 95% VaR or ES.

## 10 Stage 2 completion and subsequent execution

Resolved now: the initial model scope, primary target, requested comparator suite, 2024–2025 held-out forecast period and origin counts, rolling-refit rule, topic treatment, EPSS snapshot tolerance, scoring definitions, dependence treatment, failure policy principles, ablation interpretations, and boundaries on economic claims.

The Stage 2 design gates are closed: baseline probability laws and finite selection grids, source/adaptation records, exact comparison rules, eligibility schemas, and compute/failure rules are documented. Initial and challenger regime settings agree with the plan.

[STAGE2_FREEZE.json](STAGE2_FREEZE.json) records the SHA-256 hashes of the frozen design files and the source-extract record. It is a local change-control artifact, not an external preregistration. Amendments must update the version and explain the change rather than silently replacing the record.

Stage 3 is the project skeleton. Stage 4 populates the already specified source and eligibility manifests; Stage 5 finalizes source-dependent score-selection and record-status implementation; Stage 9 verifies computational feasibility. The prescribed development experiments choose among the frozen candidates. No data manifest or successful model fit is claimed to exist yet. Economic likelihoods still require compatible evidence at Stage 12.

No model, ingestion pipeline, or evaluation has been implemented by this document.
