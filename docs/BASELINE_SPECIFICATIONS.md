# Stage 2: frozen comparator specifications

Design version 1.0.1, 2026-10-20 (editorial clarification only). This file specifies what will be implemented; no comparator has been fitted. Read with [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). Numerical values here are design settings, not results from the failed evaluation. The [user-source extracts](sources/USER_EVALUATION_EXTRACTS.md) establish the requested comparator list.

## 1 Shared rules and selection budget

All monthly methods forecast the same integer topic counts, including unknown, through horizons 1–12. They receive their own series only unless explicitly labelled pooled or multivariate. No future marks, event identities, or outcomes enter a count baseline. Additional same-input negative-binomial controls isolate architecture from information differences.

Candidate grids below are finite and fixed. Choose one configuration per method for the entire topic panel, not a winning method for each test topic. The selection objective is mean one-month CRPS over categories and development origins, subject to valid fits; the same choice serves every horizon.

Screen on quarterly origins December 2018, March 2019, ..., September 2023 (20 origins), targeting the following month. Evaluate the two lowest-scoring valid configurations at all 60 monthly development origins December 2018–November 2023; select using those 60 scores. Exact ties after numerical precision are resolved by fewer parameters, then the listed grid order. Fixed-configuration methods require no grid search. Do not expand a grid after inspecting held-out results.

All supervised training pairs satisfy feature date < target date <= current origin. Forecast-error libraries also obey that cutoff. Estimate coefficients and states afresh each origin under the fixed configuration. No carried-over later fit is permitted.

The proposed-model selection grid is also fixed: K=10, intensity rank 2, severity rank 2 when enabled; channels {N, N+B}, R in {1,2,3}, effort {off,on}, kernel {off,on}: 24 configurations with the specification's priors. The kernel uses the specified centre 30 and maximum lag 60. Apply the same quarterly screening and monthly two-finalist rule. Versions A and B′ are diagnostic controls, not silently additional candidates. Rank changes, K=20, full factor dynamics, and extra-channel configurations are separately labelled sensitivity/development experiments, not additions to the primary search after test results.

EPSS/KEV/incident extensions retain separate coverage-appropriate designs; they do not change the frozen full-window count-model selection grid. This scope is a preimplementation choice, not a claim that these sources hurt accuracy.

Default ensemble size is 3,000 full paths per origin for every method. Use 1,000 and 6,000 as development-only Monte Carlo sensitivity checks on the last six eligible development origins. If differences in mean CRPS exceed 1% of max(1, the 6,000-draw mean CRPS), increase the common reporting budget to 6,000; if still unstable against 12,000, amend the simulation budget before held-out scoring. This threshold is an engineering tolerance, not a statistical theorem.

Use experiment seed 20261020 and derive distinct reproducible stream seeds from method, topic, origin, chain, and purpose. For neural fitting, use the fixed three seeds 11, 23, 47. Save generated candidate IDs, selected settings, and all failures.

## 2 Count probability laws

Native count models retain their own PMFs. Native continuous models must declare a transformation and score its induced PMF:

- For Gaussian forecasts on the count scale, Y = max(0, floor(X+0.5)). P(Y=0)=F_X(0.5); P(Y=k)=F_X(k+0.5)-F_X(k-0.5), k>=1.
- For a continuous forecast Z on the log1p scale, Y = max(0, floor(exp(Z)-1+0.5)). P(Y=0)=F_Z(log(1.5)); P(Y=k)=F_Z(log(k+1.5))-F_Z(log(k+0.5)), k>=1.

Use stable log-CDF or survival-function differences. The distributions used to draw, compute intervals, and score must agree. Do not round the realised target differently for different methods.

Gaussian state-space forecasts use conditional maximum-likelihood parameter estimates; they include observation and state innovations but omit parameter uncertainty. Bayesian forecasts integrate sampled parameters and states. Report this distinction rather than pretending the uncertainty treatments are identical.

### Residual distribution for RF and B-LSTM adaptations

Define an explicitly issued distribution rather than attaching a density after scoring. Work on z=log(1+y). At origin T, construct errors from the last 36 eligible inner monthly origins s satisfying s+12<=T, with at least 60 training months at s. For every s, refit the same fixed candidate using data through s and save its 12-horizon error vector on z. Inner forecasts never use intermediate realised future values.

For horizon h, error r[s,h] is realised z[s+h] minus the inner forecast's mean z. For RF the latter is its direct point output. For B-LSTM it is the average of 300 dropout trajectories, 100 per training seed. Preserve the residual vector without centering, so the distribution includes the estimated forecast bias.

Let n be the number of complete residual vectors. Use bandwidth

`b[h] = max(0.02, 1.06 * max(sample_sd(r[:,h]), 0.05) * n**(-1/5))`.

For each future path, sample one residual-vector index uniformly and add its entries plus independent N(0,b[h]^2) innovations to the base forecast path. For B-LSTM also draw a seed uniformly and a dropout path; the residual adjustment is a separately declared calibration layer, not a claim of an exact Bayesian posterior. This may overdisperse and is assessed on development.

The issued marginal is the Gaussian mixture over residual indices and, for B-LSTM, over the saved base dropout paths. Compute its CDF/PMF analytically conditional on these components; save the components. Do not substitute an NB moment match. Generate topic-level paths independently for these per-series baselines; their aggregate thus encodes an explicit independence assumption.

If fewer than 12 complete inner-origin vectors exist, use a fixed N(0,1) log-scale error law and label it `prior_residual` in the artifact. No variance or residual is estimated from the target window. Missing counts invalidate the affected supervised rows and complete residual vectors, never create zeros.

## 3 Standard monthly comparators

| ID | Model and fixed grid | Forecast distribution |
|---|---|---|
| RF | One direct random forest for each h=1,...,12 and topic. Features are the last L log1p counts at the feature origin; L in {6,12,24}. 500 trees, squared-error splits, bootstrap enabled, max_features=1.0; max_depth in {8,None}; min_samples_leaf in {1,5}. Twelve configurations. | Log-scale residual mixture above around each horizon's output; coherent error vectors for path aggregates. |
| ARIMA | Raw counts; p,q in {0,1,2,3} with p+q<=4; d in {0,1,2}. For d=0 allow no trend or constant; d=1 no trend or linear drift; d=2 no deterministic trend. No seasonal terms in this named baseline. Enforce stationary AR and invertible MA terms. Select the grid using the common development rule, not test AIC or test scores. | Conditional Gaussian state-space forecasts transformed to counts; simulate the full joint future path. |
| ETS | Additive errors on raw counts. Trend none or additive; additive trend may be damped or undamped. Seasonality none or additive with period 12. Six configurations. Estimated initial states; no multiplicative terms because zeros are allowed. | Native additive-error state-space simulation and Gaussian marginal law, transformed to counts. |
| Seasonal naive | Conditional mean y[T+h-12], h<=12. No tuning. Add iid Gaussian innovations with sd equal to the root mean square of all audited training differences y[t]-y[t-12], minimum sd 0.5 counts. This estimates variation about the model's zero innovation mean. Each horizon through 12 has one future seasonal innovation. | Transformed Gaussian on counts; the sd floor is part of the model, not a scoring floor. |
| BSTS-U | Independent Bayesian Gaussian local-linear-trend model per topic, defined below; no factors, regimes, marks, or effort. No tuning grid. | Posterior Gaussian mixture transformed to counts, including observation and state noise. |

The RF row is an adaptation inspired by the paper's forecasting strand, not a replication of WikiTopic/CyRiPred: it uses our frozen NMF count panel and declared lag features. Kia et al.'s work is identified by [DOI 10.1016/j.eswa.2023.121599](https://doi.org/10.1016/j.eswa.2023.121599).

ARIMA and ETS terminology and implementation interfaces are documented by [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) and [ETSModel](https://www.statsmodels.org/stable/generated/statsmodels.tsa.exponential_smoothing.ets.ETSModel.html). The grids and count transformations above are our choices.

### BSTS-U equations and priors

For each topic let c=max(1, sample sd of its 2010–2018 counts), fixed for every subsequent origin; let a be its initial-window mean divided by c. On the scaled raw-count variable x=y/c:

`x[t] ~ Normal(level[t], obs_sd^2)`

`level[t] = level[t-1] + slope[t-1] + level_sd * epsilon[t]`

`slope[t] = slope[t-1] + slope_sd * xi[t]`.

All innovations are independent standard Normal. Priors: level[0]~Normal(a,2^2), slope[0]~Normal(0,0.1^2); level_sd~HalfNormal(0.5), slope_sd~HalfNormal(0.1), obs_sd~HalfNormal(1). Multiply future continuous draws by c before applying the count transformation. Priors are independent across topics. These are declared benchmark choices, not priors attributed to Bakdash.

## 4 Published-system adaptations and source record

### Vuln4Cast

Pinned repository: [FIRSTdotorg/Vuln4Cast](https://github.com/FIRSTdotorg/Vuln4Cast/tree/c93910d268aa5b5d2260adb9b5d92094b2d06daf), commit `c93910d268aa5b5d2260adb9b5d92094b2d06daf`, inspected 2026-10-20. Chosen source is `QuarterlyVuln4Cast-V1.ipynb`, whose model call uses order=(1,2,4), exog=None, trend='ct'. It does not specify a seasonal order. This is the chosen V1 variant, not a statement about every notebook in that repository.

Freeze that model configuration, default nonseasonal SARIMAX state-space structure, and refit it on aggregate quarterly publication counts. Use eight quarter-end origins December 2023–September 2025, targeting Q1 2024–Q4 2025. Compare with the proposed model's sum of the next three months and all categories **within each draw**. Use the same quarterly target definitions and transformed Gaussian count law for Vuln4Cast. Report CRPS, log mass where supported, point errors and intervals; no significance test on eight quarterly observations.

No topic-share disaggregation is in the primary comparison. Any later disaggregation is a separately specified adaptation. The primary experiment is a monthly-to-quarterly, CVE-publication-date adaptation of the pinned notebook; do not execute its data-fetching or exploratory full-series cells as part of backtesting.

### Almahmoud B-LSTM

Pinned repository: [zaidalmahmoud/Cyber-threat-forecast](https://github.com/zaidalmahmoud/Cyber-threat-forecast/tree/4fa160c885a22359d700d521969865d3ea47649d), commit `4fa160c885a22359d700d521969865d3ea47649d`, inspected 2026-10-20. Read `univariate/hp_optimisation_u.ipynb` and `univariate/forecast_u.ipynb`. The authors' [2023 article](https://www.nature.com/articles/s41598-023-35198-1) links this repository.

Our monthly adaptation uses one encoder and one decoder LSTM with the same U units, ReLU activations, recurrent dropout active at inference, RepeatVector(1), a time-distributed Dense(100, ReLU), then Dense(1). Grid: lookback {6,12}, U {25,50}, recurrent dropout {0.2,0.4}: eight configurations. Use Adam learning rate 0.001, batch size 8, MSE training loss, shuffle=False, maximum 200 epochs. Fit all three seeds and average their predictive distributions; do not select the best seed.

Use log1p counts, min–max scaled using the training subwindow alone; if its range is zero set the scaling denominator to 1. No clipping of later scaled inputs. No pre-smoothing in this adaptation, so the target stays the raw monthly count. Withhold the last 24 training months for early stopping, patience 30, restore best weights; choose the best epoch then retrain from the same seed on all origin-available training data for that many epochs. This inner validation is contained within each origin, including residual-library fits.

Generate 12-step recursive dropout trajectories without injecting realised future observations. Transform back to log1p units, then apply the issued residual-mixture distribution in Section 2. The changed target, bounded grid, preprocessing, rolling evaluation and residual calibration make this an explicitly labelled adaptation, not a verbatim reproduction or an exact Bayesian posterior.

### Bakdash BSSM

Source: the authors' [OSF analysis component](https://osf.io/rq82a/), linked from their [2018 paper](https://doi.org/10.1093/cybsec/tyy007). Inspected file `06.05.2018.Forecasting_CyberEvents_Bayesian_model.Rmd`, OSF file ID `5b1a9c6cb796ba000d3f2bf9`, [download](https://osf.io/download/b8f62/), SHA-256 `7e47e2d62609bc6bc151f0293fa3a6071655bc9cc72f9b7ae0b89377d61d160b`.

The inspected source specifies `Counts ~ Time` with `cor_bsts(formula = ~1)` and a negative-binomial family. Thus its deterministic time predictor must not disappear unnoticed in a count-model adaptation. No claim is made here that the original software defaults equal the priors below.

The Rmd also fits `Counts ~ Time + Length`; this adaptation follows the first, time-only fit and does not use report length.

Implement the following transparent Python adaptation per topic:

`y[t] ~ NB2(exp(alpha + beta*t/12 + u[t]), psi)`

`u[0]=0; u[t]=u[t-1]+tau*epsilon[t]`.

Time t starts at zero in January 2010 and increases by one per calendar month; t/12 is years. Independent priors: alpha~Normal(log(initial-window mean count+0.5),2^2), beta~Normal(0,0.5^2), tau~HalfNormal(0.3), log(psi)~Normal(0,1). Refit monthly; simulate latent innovations and count observations. No seasonality, pooled factors, or extra channels. There is one fixed configuration.

Label the result **Bakdash-inspired NB BSSM (monthly adaptation)**. It changes weekly malware counts to monthly CVE counts and makes priors and initialization explicit. Do not label it an exact port of obsolete `brms` internals. An exact legacy reproduction is optional additional work, not a prerequisite or a replacement for the declared comparison.

## 5 Additional controls

These supplement the eight requested comparators:

- Historical mean: fixed mean `(sum(training counts)+0.5)/(number of observed training months+0.5)`, with a Poisson forecast. Zero training exposure is a failure, not a zero forecast.
- Independent NB AR(1): specification Version A, with independent m~Normal(log(initial-window mean+0.5),2^2), tau~HalfNormal(0.3), log(psi)~Normal(0,1), and the specification's stationary AR prior. Fit a paired effort-adjusted version with the same effort input as the factor model.
- Pooled NB regression: log mean b0+b[k]+beta*t/12, b~Normal(0,1), b0~Normal(log(initial aggregate mean/K+0.5),2^2), beta~Normal(0,0.5^2), log(psi[k])~Normal(0,1). One fixed configuration; sampled shared coefficients induce dependence.

A no-factor ablation with shared random parameters remains distinct from the independent control.

## 6 Frozen compute and failure rules

Bayesian fits: four chains; initial target_accept=0.95, max_treedepth=12, 1,000 warmup and 1,000 retained draws each. Retry 1: target_accept=0.99, depth 15, 2,000 warmup and 1,000 retained. Retry 2: same acceptance/depth, 2,000 warmup and 2,000 retained. Use the specification's convergence requirements; unresolved warnings or material Monte Carlo instability prevent an accepted fit.

MLE fits: first L-BFGS with 1,000 iterations; if convergence fails, use Powell with 2,000 iterations then L-BFGS refinement with 1,000. Keep the declared model order; do not search another order based on test performance. A finite objective alone is not successful convergence.

RF/neural fits receive the fixed budgets above. A neural seed with nonfinite loss is a failed seed; all three are required. No retries that choose a lucky seed. Missing forecasts are recorded under the protocol's failure policy. Runtime and memory are measured in Stage 9; any budget amendment is recorded and validated on development before held-out scoring.

Library versions, device information, and code hashes are populated when the environment is built in Stage 3. They must not change the specified probability laws. Pinning package builds is an execution artifact, not a remaining choice of scientific comparator.
