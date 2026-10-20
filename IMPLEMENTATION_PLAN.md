# Python Implementation and Evaluation Plan

This plan develops the framework in `main.tex` from an empty Python project through to the experimental sections of the paper. Work proceeds through small, verifiable stages, each producing an artifact to inspect before adding the next layer.

The aim is to test the paper's claims and establish which ones the evidence supports. The numerical results currently in the introduction are unsupported by the available artifacts and must be replaced by new, valid evaluation results rather than treated as targets to reproduce.

Status, 2026-10-20: Stage 1 design is complete; Stage 2 design is frozen with editorial clarifications as protocol v1.0.1; Stage 3 project structure is implemented and checked. See [specification](docs/SPECIFICATION.md), [evaluation protocol](docs/EVALUATION_PROTOCOL.md), [baseline specifications](docs/BASELINE_SPECIFICATIONS.md), [statistical analysis plan](docs/STATISTICAL_ANALYSIS_PLAN.md), [manuscript-change record](docs/MANUSCRIPT_CHANGES.md), and [Stage 3 setup and verification](docs/STAGE3_SETUP.md). [Original comparator/metric extracts](docs/sources/USER_EVALUATION_EXTRACTS.md) document the user-supplied source. The package provides configuration validation and run preparation; no cyber model or data pipeline has been implemented. Next is Stage 4, the data audit and ingestion. Full-model inference feasibility and economic evidence remain later-stage work.

## 1 Resolve the mathematical specification

Corresponds to: Problem Formulation and Proposed Solution.

Before coding, turn the manuscript into an explicit specification of every random variable, likelihood, prior, transformation, and forecast target.

The specification resolves the initial defaults for the following details; source-dependent and empirical choices retain explicit gates:

- **Latent intensity:** define its operational meaning and scale. Without additional evidence, a disclosure-adjusted vulnerability signal cannot automatically be interpreted as the true rate of malicious activity.
- **Reporting effort:** specify how the CVE-volume trend is estimated using only information available at each forecast origin. Treat separation of reporting effort from threat growth as an assumption to investigate.
- **Factor identification:** supplement triangular loading constraints with sufficient scale constraints; positive diagonal entries alone do not remove all ambiguity when factor innovation scales are also free.
- **Temporal kernel:** specify its finite support, normalization, and forecast boundary treatment. The sigmoid weights in the draft do not themselves have compact support. A causal construction would simplify forecasting.
- **Observation distributions:** resolve bounded CVSS scores, exploitation probabilities at zero or one, and the distinction between EPSS probabilities and KEV counts.
- **Loss marks:** either specify a likelihood for loss marks or explicitly reserve them for economic calibration.
- **Economic inference:** decide whether this is one joint posterior or a modular model with a separately calibrated economic layer. Start with the modular formulation and describe it accurately.
- **Regime alerts:** use filtered probabilities based on information available at the time; historical smoothed probabilities can be shown separately.

**Deliverable:** an equation-to-implementation specification and a record of changes required in the manuscript.

## 2 Define the evaluation protocol before fitting anything

Corresponds to: a new Experimental Design section.

Separate three questions:

| Question | Evidence required |
|---|---|
| Can the model predict observable technical signals? | Historical forecasts scored against subsequently observed data |
| Can inference recover hidden structure? | Synthetic experiments with known latent states |
| Can technical forecasts support economic loss estimates? | Independent loss evidence, held-out events, and sensitivity analysis |

Use monthly observations and evaluate horizons of 1, 3, 6, and 12 months.

The historical design is now specified in [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md):

- Initial training: 2010–2018.
- Development and model selection: 2019–2023.
- Held-out forecast evaluation for this rerun: January 2024–December 2025.
- Freeze design and tuning using data through December 2023. Earlier elapsed test observations may enter later rolling fits; test scores cannot guide redesign. A separate prospective evaluation is optional.

The coverage audit must document any necessary amendment to these boundaries. The full multi-channel model will need a separate, shorter evaluation window: historical EPSS scores begin on April 14, 2021, and model changes introduce score shifts. SEC Item 1.05 reporting began much later than the CVE history. [FIRST historical data](https://www.first.org/epss/data.html), [SEC disclosure timeline](https://www.sec.gov/newsroom/press-releases/2023-139).

The protocol fixes 24, 22, 19, and 13 historical origins for horizons 1, 3, 6, and 12, plus a 13-origin common comparison. Audit masks then determine eligible cells. Refit every model through each origin; a full-panel posterior or smoother is forbidden in historical forecasts. Keep point-in-time and revised-data experiments separate. Version changes or reporting dates learned after an origin must not enter its forecast; announced future changes can enter. Late-arriving channels need their own development/evaluation design.

**Deliverable:** frozen protocol v1.0.1 (editorial amendment to design v1.0), baseline probability laws and selection grids, comparison/inference rules, source/adaptation records, compute budgets, and source/eligibility manifest schemas. Stage 4 populates those manifests before fitting; Stage 3 pins package builds. Design completion does not claim completed data ingestion or external preregistration.

## 3 Create the Python project structure

Corresponds to: reproducibility and implementation details.

Use one installable package, with notebooks for exploration and presentation.

Implemented package namespaces and their planned responsibilities:

| Area | Responsibility |
|---|---|
| `data` | Source downloads, parsing, provenance, and historical availability |
| `features` | Topic assignments, monthly panels, exposure maps |
| `models` | Baselines, measurement models, latent dynamics |
| `inference` | Model fitting, diagnostics, regime reconstruction |
| `forecasting` | Posterior simulation and observable predictions |
| `economics` | Damage calibration and input–output propagation |
| `evaluation` | Backtests, scoring, ablations, sensitivity analysis |
| `reporting` | Tables and figures generated from saved results |

The project uses NumPy/SciPy, pandas, Parquet/PyArrow, PyMC/PyTensor, ArviZ/xarray, scikit-learn, statsmodels, and pytest. Exact versions are recorded in `uv.lock`; a synthetic inference and library compatibility probe passed on Windows/Python 3.12. See [Stage 3 verification](docs/STAGE3_SETUP.md) for results and the missing C++ compiler limitation. Neural-comparator dependencies will be added when that adapter is implemented.

PyMC is a suitable starting point because its documented workflows support marginalizing hidden Markov chains before NUTS sampling and recovering states afterward. Verify compatibility with the particular model graph before relying on automatic marginalization. [PyMC HMM example](https://www.pymc.io/projects/examples/en/latest/mixture_models/discrete_markov_chain_marginalization.html).

Every run should record configuration, random seeds, data hashes, software versions, and source revision.

**Deliverable (complete):** installable package, locked environment, `configs/pilot.toml`, configuration/protocol validation, run provenance, CLI, and 17 passing scaffold tests. Run records explicitly leave data provenance pending until ingestion. The synthetic probe checks the environment, not convergence or correctness of the proposed cyber model.

## 4 Audit and ingest the data

Corresponds to: a new Data section.

Build independent ingestion components for:

| Source | Intended use | Main audit question |
|---|---|---|
| NVD CVE records | Counts, descriptions, severity | Which fields were available at each historical cutoff? |
| NVD CPE metadata | Affected products | How complete and historically reconstructable are product mappings? |
| EPSS archives | Exploitation-probability signal | How do model versions and scoring dates affect comparability? |
| CISA KEV | Known-exploitation signal | When was an entry publicly added? |
| SEC filings | Sectoral incident disclosures | Can amendments and repeated disclosures be linked to one incident? |
| BEA accounts | Output and inter-sectoral relationships | Which classifications and publication vintages are compatible? |
| Documented incidents | Economic calibration and evaluation | What does each reported loss measure actually include? |

Store raw responses unchanged. Record retrieval time, publication time, revision time where available, and source identifiers.

Downloading an old CVE today does not establish what its severity, description, or product metadata looked like years ago. Audit NVD change histories and archived records before deciding which fields can be reconstructed. Where reconstruction fails, label the experiment as using revised historical data; this limitation also applies to topic assignments from revised descriptions. NVD's change-history API exists, but its existence does not guarantee complete reconstruction. [NVD API announcement](https://groups.google.com/a/list.nist.gov/g/nvd-news/c/8FGMfqB2PQ8).

**Deliverable:** source adapters, a coverage report, and a documented assessment of historical-data limitations.

## 5 Construct the monthly modelling panels

Corresponds to: Inputs and Outputs and Measurement.

Create separate topic-month and sector-month datasets, with explicit availability masks.

For topic construction, use TF–IDF plus NMF and a hard assignment per CVE. This keeps category counts integer-valued for the count likelihood. The primary pilot has nine learned topics plus unknown (K = 10); K = 20 is a later resolution sensitivity experiment. Inspect topic coherence. Do not select topic count by comparing raw per-topic CRPS across different targets.

Fit the vocabulary and topic model on the initial training period and freeze them for the primary benchmark. This keeps topic identities consistent across forecast origins. Any later retraining strategy becomes a separate experiment.

Define precisely:

- Whether exploitation signals summarize newly disclosed CVEs or an existing vulnerability population.
- Which CVSS version takes precedence.
- How missing severity differs from an empty category-month.
- How companies map into economic sectors.
- How filing amendments and repeat reports are deduplicated.
- How changing reporting-population size enters sectoral incident rates.

**Deliverable:** validated modelling panels, a data dictionary, and descriptive coverage plots.

## 6 Build the backtesting engine and simple baselines

Corresponds to: Experimental Design and Benchmark Models.

Implement the evaluation machinery before the full Bayesian model.

At each forecast origin, the engine should construct the available training data, fit preprocessing, train the model, generate forecasts, and save predictions before scoring.

Start with:

- Historical-mean count forecasts.
- Seasonal naïve forecasts with a training-only uncertainty model.
- Independent negative-binomial autoregressions.
- Per-series state-space or SARIMA benchmarks with predictive distributions.
- A pooled count model without latent factors or regimes.

All models must predict the same observable targets and receive the same information. Also distinguish richer-input comparisons from comparisons of model architecture.

These initial controls do not replace the requested comparison suite: RF, ARIMA, ETS, seasonal naive, BSTS-U, Vuln4Cast, Almahmoud B-LSTM, and Bakdash BSSM. The protocol defines their comparison tasks. Carry forward baseline identities and measurement families from the supplied failed run, not its results, rankings, or conclusions.

**Deliverable:** a working rolling-origin benchmark and initial score tables.

## 7 Develop a synthetic-data test suite

Corresponds to: a new Simulation Study section.

**Entry gate:** resolve the missing PyTensor C++ compiler before Stage 7. Verify compiler detection, compiled log-density/gradient execution, and a timed synthetic NUTS fit in the chosen execution environment. Record the environment and timings before starting the simulation suite; do not defer this to Stage 9. Data ingestion in Stage 4 can proceed independently.

Generate datasets with known factors, regimes, intensities, severities, observation biases, and missingness.

Introduce difficulty gradually:

- Independent counts with no regime changes.
- Shared factors and sparse categories.
- Known regime transitions.
- Reporting changes without threat changes.
- Threat changes without reporting changes.
- Temporal-kernel dependence.
- Missing channels and deliberately misspecified observations.

Check predictive recovery, interval coverage, and identifiable parameter combinations. Do not demand recovery of arbitrary factor rotations or regime labels.

For the hidden Markov calculation, compare the forward-recursion likelihood against explicit enumeration on tiny examples. Check its gradients before using NUTS.

**Deliverable:** evidence that the implementation behaves correctly under controlled conditions.

## 8 Implement the Bayesian model incrementally

Corresponds to: Latent Threat Dynamics and Measurement.

Use successive model versions:

| Version | Addition | Purpose |
|---|---|---|
| A | Single-category negative-binomial latent process | Establish stable inference and forecasting |
| B | Shared dynamic factors and iid residuals | Compare the shared-factor model; also fit B′ retaining A's AR residuals to isolate the factor contribution |
| C | Severity dynamics | Estimate the intensity–severity inputs |
| D | Reporting-effort adjustment | Measure its effect on forecasts and interpretation |
| E | Markov-switching levels | Capture structural changes |
| F | Temporal kernel | Test additional local dependence |
| G | Exploitation and incident channels | Assess the value of additional observations |

Start with K = 10 including unknown, two intensity factors, one regime, count observations only, no kernel, and no effort adjustment. Add two severity factors, the effort proxy, regime challengers R = 2 and 3, and the kernel in the protocol's development sequence. These are starting configurations, not empirically established winners.

Each addition must pass synthetic checks and produce acceptable inference diagnostics before expanding the model.

For the observation layer, use the corrected distributions in the specification. The bounded, coarsened severity model is the default; the manuscript's lognormal approximation is a separately evaluated alternative on the same target.

**Deliverable:** separately runnable model variants that also support the ablation study.

## 9 Establish inference and computational feasibility

Corresponds to: Inference.

Implement the regime marginalization with a numerically stable forward recursion. Specify exactly which regime-dependent density enters the recursion so latent-state densities and observation likelihoods are not counted twice.

Use non-centred parameterizations where helpful, identifiable factor scales, constrained stable dynamics where assumed, and explicit regime-label conventions.

For final fits, assess:

- Multiple-chain agreement, with a target of R-hat below 1.01.
- Bulk and tail effective sample sizes.
- Monte Carlo error for reported quantities.
- Divergences and other sampler warnings.
- Prior and posterior predictive behavior.
- Runtime and peak memory.

Benchmark a representative fit early, then estimate the cost of all backtest origins and ablations. Use a reduced development schedule if necessary, and freeze the final schedule before evaluating the test set.

**Deliverable:** a documented inference configuration and an affordable experiment budget.

## 10 Implement posterior forecasting

Corresponds to: Forecast Simulation and Outputs.

For each posterior draw, simulate future regimes, factors, kernel contributions, idiosyncratic innovations, intensity, and severity.

Produce both:

- **Latent forecasts**, used for interpretation and economic translation.
- **Observable forecasts**, generated through the measurement distributions and scored against future counts, scores, and disclosures.

Preserve dependence across categories and months within each draw. Do not independently resample marginal forecasts before aggregating them.

Evaluate the manuscript's constant-effort assumption alongside plausible alternatives. Forecast future EPSS and disclosure observations under explicit assumptions about their measurement processes.

**Deliverable:** a common forecast format indexed by origin, horizon, category or sector, and posterior draw.

## 11 Build the sector exposure and economic structure

Corresponds to: Economic Translation.

Construct the exposure map in two documented steps:

1. Map affected products into technology groups.
2. Map technology groups into sectoral usage or dependence.

CPE metadata identifies affected products; it does not by itself measure sector adoption. Where adoption evidence is weak, use transparent assumptions and sensitivity ranges.

For the input–output layer, choose a consistent industry classification and derive or obtain compatible requirements matrices. BEA distinguishes use tables from the requirements tables derived from make/use information, so simply normalizing an arbitrary Use table is insufficient. [BEA requirements-table explanation](https://www.bea.gov/help/faq/32).

Check dimensions, units, invertibility, aggregation, and annual-to-monthly conversion.

Also establish what the loss quantity means. Applying a Leontief inverse to disrupted output needs an explicit economic interpretation because standard requirements tables describe output required to meet final demand. Compare propagation with no amplification and at least one alternative disruption assumption.

**Deliverable:** an auditable exposure map and independently checked economic transformation.

## 12 Calibrate the damage functions

Corresponds to: Economic Translation and a new Calibration subsection.

Write the exact zero-anchored sigmoid and specify the three shared parameters and sector operating points.

Create an event ledger recording geography, affected sectors, time window, currency year, direct versus indirect loss, and uncertainty. Distinguish output losses from ransom payments, remediation costs, and insurance claims.

Then:

- Fit the damage model using designated calibration events.
- Introduce the aggregate-loss anchor only after checking that its definition matches the modelled quantity.
- Test parameter identification and sensitivity to priors.
- Compare alternative damage functions.
- Propagate calibration and exposure uncertainty into loss forecasts.

Four events do not automatically identify three parameters. This must be checked.

Reserve independent events where possible. Leave-one-event-out analysis is useful with a small dataset, but must be labelled retrospective unless all information and calibration inputs predate the forecast.

**Deliverable:** a calibrated economic module with documented uncertainty and held-out checks.

## 13 Run the main evaluation

Corresponds to: a new Results section.

Evaluate technical predictions using:

| Aspect | Measures |
|---|---|
| Distributional accuracy | CRPS and negative log predictive density |
| Calibration | Empirical coverage and randomized PIT for counts |
| Sharpness | Interval width and interval score |
| Point accuracy | MAE and RMSE as secondary measures |
| Joint dependence | Aggregate forecast scores, energy or variogram scores |
| Robustness | Performance by horizon, category frequency, and historical period |

Compute log scores from predictive probability masses or densities, using stable likelihood averaging where possible. The protocol prohibits winsorizing draws and then scoring a moment-matched substitute as the original forecast. MAE uses the median and RMSE the mean; labels must make any alternative explicit.

For clipped score targets, endpoint observations require tail probability masses and interior observations require densities with the appropriate transformation Jacobian. Use the same coarsened target for every model. Mark-score eligibility must depend on actual observation availability, never on whether finite simulated count draws happen to be positive; explicitly define the conditional mark forecast. For aggregate counts, do not add marginal log scores or use empirical sample frequencies as an exact aggregate probability mass. Validate a convolution or another justified probability calculation before reporting aggregate log scores.

Report both category-averaged and aggregate-count summaries. Compare models using paired results at the same forecast origins. Average topic loss differences within origin before estimating temporal uncertainty; do not treat all topic–month cells as independent. Follow the protocol's exploratory block-resampling and short-series limits.

For economic evaluation, distinguish unconditional forecasts from scenarios conditioned on a known event. Event-range overlap alone cannot establish calibrated aggregate-loss predictions or reliable 95% tail risk.

An observed 97% coverage rate for nominal 90% intervals is empirical overcoverage. Whether it establishes systematic miscalibration depends on sample size, dependence, and discrete-outcome interval conservatism; report uncertainty, interval width, and proper scores.

**Deliverable:** the protocol-governed historical benchmark, any separately justified independent validation, and an appropriately qualified economic evaluation.

## 14 Run ablations and sensitivity analyses

Corresponds to: new Ablation and Robustness sections.

Remove one component at a time: regimes, kernel, shared factors, effort adjustment, exploitation data, incident data, and economic propagation.

Also vary topic count, factor rank, regime count, priors, exposure assumptions, damage functions, and loss anchors.

Use the same evaluation origins and comparable fitting effort. Include tests of residual dependence between observation channels: conditional independence is an assumption, especially when channels share underlying vulnerability information.

The purpose is to establish which components contribute useful predictive information and which mainly add uncertainty or computation.

**Deliverable:** an ablation table and a sensitivity analysis for the principal conclusions.

## 15 Generate the paper's evidence and revise its claims

Corresponds to: Results, Discussion, Limitations, and Reproducibility.

Generate tables and figures directly from saved experiment artifacts:

- Data coverage and source-version timeline.
- Benchmark scores by forecast horizon.
- Calibration and interval-width plots.
- Synthetic recovery results.
- Filtered regime probabilities.
- Sector exposure and loss distributions.
- Held-out event comparisons.
- Ablation, sensitivity, and computational-cost tables.

Maintain a claim-to-evidence table linking each statement in the paper to an experiment. Revise unsupported claims, including claims about cross-excitation, reporting-bias separation, economic calibration, and tail-risk accuracy.

The first practical milestone is a historically disciplined CVE forecasting benchmark with simple baselines and one small Bayesian factor model. Once that works, add regimes, other observation channels, and economic translation in the order above.
