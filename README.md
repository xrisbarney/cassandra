# CASSANDRA — Cyber Threat Modelling

Bayesian hierarchical state-space model of cyber threats and systemic economic risk.

## Standalone paper implementation

The `standalone` branch implements the numbered procedure in
`ESWA_Journal_Article_Template.pdf` in its stated order. The default inference
command is the paper's alternating NUTS/FFBS algorithm; it does not use the
older marginalized-regime approximation. Run the complete sequence with:

```bash
pip install -e ".[dev]"
python scripts/run_all.py --start 2010-01 --end 2024-12
```

A real paper run is computationally expensive because every rolling origin is
re-estimated. Optional curated loss marks belong at
`data/cache/incident_losses.csv` with columns `date,sector_idx,loss_usd`.

### Exact paper-step mapping

| Paper step | Implementation | Run |
|---|---|---|
| Step 1 | Observation history is assembled in `data/pipeline.py`, `data/incident_losses.py`, and `scripts/ingest_data.py`. | `python scripts/ingest_data.py --start 2010-01 --end 2024-12 --cache-dir data/cache` |
| Step 2 | CPE-to-sector exposure tensor `M_skt` is built in `features/exposure_map.py`. | `python scripts/build_features.py` |
| Step 3 | BEA coefficients and `(I-A)^-1` are implemented in `data/bea_io.py`. | `python scripts/build_features.py` |
| Step 4 | Reference events and loss ranges are in `configs/reference_events.yaml`. | `python scripts/calibrate_damage.py` |
| Step 5 | Markov transitions and mean-ordered labels are in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 6 | Regime-specific `Phi_z` and `Q_z` factor recursion is in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 7 | `eta_t`, `Gamma`, and `lambda_t=exp(eta_t)` are in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 8 | The lower-dimensional severity recursion is in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 9 | Inverse-gamma variances and hierarchical loading priors are in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 10 | Optional multivariate Hawkes-type excitation is in `model/paper_exact.py`. | Set `hawkes.enabled: true`; run `python scripts/train.py --method blocked` |
| Step 11 | Latent reporting effort is in `features/effort.py` and `model/paper_exact.py`. | `python scripts/build_features.py`; `python scripts/train.py --method blocked` |
| Step 12 | `N_kt ~ NegBin(e_t lambda_kt, psi_k)` is in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 13 | The exploitation channel on the logit scale is in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 14 | Time-varying sector reporting propensity and the December-2023 indicator are in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 15 | Multiplicative Poisson incident rate is in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 16 | Independent N, E, and D likelihood sites encode conditional independence in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 17 | Sector shock loads and `phi_s` are in `model/economic.py` and `model/paper_exact.py`. | `python scripts/forecast.py --horizon 12` |
| Step 18 | Weak-prior Bayesian event calibration and retained posterior draws are in `scripts/calibrate_damage.py`. | `python scripts/calibrate_damage.py` |
| Step 19 | Direct losses `d_s=g_s*x_s` are formed in `model/paper_exact.py`. | `python scripts/forecast.py --horizon 12` |
| Step 20 | Leontief and aggregate losses are in `model/economic.py` and `model/paper_exact.py`. | `python scripts/forecast.py --horizon 12` |
| Step 21 | The joint conditional posterior is `paper_model` in `model/paper_exact.py`. | `python scripts/train.py --method blocked` |
| Step 22 | NUTS smooth-block sampling is in `inference/blocked.py`. | `python scripts/train.py --method blocked` |
| Step 23 | Alternating FFBS is in `inference/blocked.py` and `inference/ffbs.py`. | `python scripts/train.py --method blocked` |
| Step 24 | The scalable variational/FFBS alternative is in `inference/blocked.py` and `inference/vi.py`. | `python scripts/train.py --method vi` |
| Step 25 | Regime/factor/intensity/severity forward simulation is `paper_exact.predict`. | `python scripts/forecast.py --horizon 12` |
| Step 26 | Draws pass through uncertain `M_t` and `phi_s` in `paper_exact.predict`. | `python scripts/forecast.py --horizon 12` |
| Step 27 | Direct and Leontief loss draws are generated in `paper_exact.predict`. | `python scripts/forecast.py --horizon 12` |
| Step 28 | Joint intensity, exposure, severity, and damage uncertainty is retained by `paper_exact.predict`. | `python scripts/calibrate_damage.py`; `python scripts/forecast.py --horizon 12` |
| Step 29 | Threat distributions and bands are written by `scripts/forecast.py`. | `python scripts/forecast.py --horizon 12` |
| Step 30 | Sector indices are written to `results/forecasts/sector_exposure.csv`. | `python scripts/forecast.py --horizon 12` |
| Step 31 | Loss, VaR95, ES95, and tail probabilities are written by `scripts/forecast.py`. | `python scripts/forecast.py --horizon 12` |
| Step 32 | Regime probabilities are written to `results/forecasts/regime_probs.csv`. | `python scripts/forecast.py --horizon 12` |
| Step 33 | Per-origin expanding-window refits are the default in `scripts/evaluate.py`. | `python scripts/evaluate.py` |
| Step 34 | CRPS and count LogS are written to `results/evaluation/channel_scores.csv`. | `python scripts/evaluate.py` |
| Step 35 | PIT, KS tests, and interval coverage are in `evaluation/calibration.py`. | `python scripts/evaluate.py` |
| Step 36 | RF, ARIMA, persistence, ETS, and BSTS-U are in `evaluation/baselines.py`. | `python scripts/evaluate.py` |
| Step 37 | Observation-channel checks are in `evaluation/posterior_predictive.py`. | `python scripts/evaluate.py` |

The end-to-end command executes the dependency order: ingest, features,
blocked inference, damage calibration, rolling validation, then forecasting.

Models K=8 threat topics across S=11 BEA economic sectors using latent intensity and severity factor processes with shared Markov regime switching. Four observation channels are treated as biased views of the latent state: CVE counts and severity marks (NVD/CVSS), exploitation probability (EPSS/CISA KEV), and SEC 8-K incident disclosures. Posterior draws are propagated through a BEA input–output Leontief inverse to produce predictive distributions of systemic economic loss.

## 🔮 Easiest way to use it: the dashboard

If you'd rather not use the command line, run the guided visual dashboard:

```bash
pip install -e ".[dashboard]"    # one-time: installs the dashboard
streamlit run app.py
```

A page opens in your browser with guided controls for the main pipeline stages.
The command-line runner is authoritative for the paper-exact six-stage order.
each shows a green light when finished, a live progress log while running, and
charts of your data and the final forecast. No commands, no jargon. Everything
below is the manual/command-line equivalent.

**Run everything, unattended:** the dashboard also has a **🌙 Run the whole
pipeline now** button that runs the stages in sequence in the background.
Tick *Start completely fresh* to wipe all caches and re-download from scratch.
It keeps running even if you close the browser — great for leaving overnight.
The command-line equivalent is:

```bash
python scripts/run_all.py --fresh          # wipe generated artifacts and run all stages
python scripts/run_all.py --quick          # keep caches, quick-preview training
```

## Legacy scalable-model notes

The older scalable/VI implementation made changes to the paper specification that
**require manuscript changes** — the model as written in Section 3 is not
identified and will not sample (NUTS step size collapses to ~1e-10):

1. The factor loadings **Γ** (Eq. 3) and severity loadings **Ψ** are not
   identified (rotation/scale/sign freedom) — the code imposes the standard
   positive-lower-triangular constraint.
2. The reporting propensity **π** (Eq. 8) is over-parameterized as free per
   sector-*month* on a near-empty channel — the code uses one baseline per
   sector.

3. The discrete regime path **z_t** is marginalized analytically (a standard
   HMM forward algorithm inside the scan, injected via `numpyro.factor`)
   instead of being Gibbs-sampled with `DiscreteHMCGibbs`, which NumPyro's own
   docs flag as an `[EXPERIMENTAL INTERFACE]` and which was empirically
   unreliable on the real data (frequent warmup step-size collapse). A
   consequence: the factor/severity AR *transition* dynamics are now
   regime-constant — only the emission *means* switch by regime
   ("Markov-switching mean" rather than the paper's full
   Markov-switching-VAR). Plain `NUTS` now samples the whole model.

These fix the sampling geometry (step size ~1e-10 → ~1e-2, with no collapse
observed at real scale after fix 3). **See
[docs/PAPER_NOTES.md](docs/PAPER_NOTES.md) for the exact equations to
update.**

**Also see docs/PAPER_NOTES.md §5-§6:** backtesting the model's 2025 forecast
(trained on 2010-2024 data) against the real 2025 data that has since come in
shows severe under-prediction (actual CVE volume ran ~3.7x the forecast
total) and poor uncertainty calibration (90%-interval coverage of only 4.2%
vs. a ~90% target). §5 traces the *dominant* cause to a concrete,
easily-fixed defect — `predict()` drops the reporting-effort offset `e_t`
(implicitly assuming it snaps from its 2024 peak back to the 15-year average
in the forecast), which alone accounts for essentially the whole ~3.7x gap;
AR mean-reversion and interval overconfidence are smaller, separate issues.
§6 discusses the CVE-volume surge, assesses the "AI-accelerated discovery"
hypothesis (plausible as one driver among several; our data can't causally
isolate it), and catalogues model-tuning options — including why "just train
on the last year" won't work and what to do instead.

**§7 documents the two fixes now implemented:** (i) `predict()` carries the
fitted likelihood's covariates into the forecast — the effort offset `e_t`
(hold-last) and the baseline disclosure rate `π_s` — closing the §5 gap;
(ii) `scripts/backtest.py` evaluates by **sequential one-step-ahead filtered
prediction**: starting from the first panel month, the model predicts each
month before seeing it, is scored, then updates its regime beliefs with that
month's actual data. Under this (correct) evaluation the 90%-interval
coverage is ~98% on the 2023–2024 test window, versus 4.2% under the old
open-loop design. Run `python scripts/backtest.py` (or dashboard **Step 6**)
to reproduce; interactive charts land in `results/backtest/figures/`.

**§8 documents the paper-conformance build** — everything the manuscript's
tables promise now has numbers behind it: the BSTS-U baseline and log score
in the evaluation (Tables 2/3/6); Bayesian calibration of the damage
functions against the reference event set plus an aggregate annual anchor
(Tables 7/8 — this also uncovered and fixed a saturated damage sigmoid and
an x_s units error that together made all earlier loss figures invalid);
systemic-event probabilities and regime-probability early-warning outputs;
the six-variant ablation study (Table 4); and the K=1024 sparse-topic
population run with the active-topic set and Full-vs-BSTS-U appendix
comparison (§4.1). Headline results: the full model beats every baseline at
every horizon (CRPS 41.2 vs best-baseline 49.6 at h=1, all DM tests
p < 1e-14), and the calibrated forecast puts aggregate 12-month US losses at
~$137B (90%: $128–148B).

**§9 documents a test theory (quarantined from the main model):** a learned
moving-window covariance kernel that quantifies how strongly nearby months
co-move beyond the AR factor dynamics. The data chose a ±20-month window
(90% CI 18.7–21.6) with decisively nonzero amplitude, improving sequential
CRPS by ~11% in a settings-matched comparison. Opt-in via the `kernel:`
config block (off by default — the paper-native model is untouched); results
live in the dashboard's **🧪 Kernel lab** tab.

## Requirements

- Python 3.11+
- JAX-compatible hardware (CPU works; GPU/TPU accelerates MCMC)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install the package and dev dependencies
pip install -e ".[dev]"

# 3. Configure API keys
cp .env.sample .env
# Edit .env and fill in your keys (see API Keys section below)
```

## API Keys

Copy `.env.sample` to `.env` and fill in the values. The scripts load it automatically.

| Variable | Required | Where to get it |
|----------|----------|-----------------|
| `NVD_API_KEY` | Recommended | [nvd.nist.gov/developers/request-an-api-key](https://nvd.nist.gov/developers/request-an-api-key) — free, raises rate limit from 5 req/30s to 50 req/30s |
| `BEA_API_KEY` | Required | [apps.bea.gov/api/signup](https://apps.bea.gov/api/signup/) — free, used to fetch I-O tables automatically |
| `SEC_USER_AGENT` | Required | Your name and email, e.g. `Jane Smith jane@example.com` — SEC fair-access policy |

## Running the pipeline

The five scripts run in order: ingest → features → train → evaluate → forecast.

### 1. Ingest data

Pulls CVE records from NVD, EPSS scores, CISA KEV catalog, and SEC EDGAR 8-K filings.

```bash
python scripts/ingest_data.py --start 2010-01 --end 2024-12 --cache-dir data/cache/
```

> **Note:** Without an NVD API key, requests are rate-limited to 5/30s — a 15-year fetch will take several hours. With a key it drops to ~30–60 minutes. The date range is automatically chunked into 119-day windows as required by the NVD API, and each window is cached so an interrupted run resumes where it stopped.

> **EPSS coverage:** EPSS scores exist only from **2021-04-14** onward (the EPSS v1 launch). Months earlier than that are skipped automatically without issuing requests. One mid-month snapshot is fetched per month — each daily EPSS file already contains the full CVE catalog — so this channel is fast. Pre-2022 (EPSS v1) files carry scores but no percentile column; that is handled transparently.

### 2. Build features

Fits the topic model, constructs the sector exposure tensor M_skt, estimates attacker effort e_t, and fetches the BEA I-O tables automatically via the BEA API.

```bash
python scripts/build_features.py
```

> BEA Use table (Summary level, 2022) is fetched automatically using `BEA_API_KEY` and cached to `data/processed/bea_use_table_2022.json`.

### 3. Train the model

> **"Training" means Bayesian posterior inference, not weight fitting.** The default alternates NUTS draws for the smooth blocks with FFBS draws for the complete discrete regime path, exactly as Steps 22–23 prescribe. `--method vi` replaces the smooth NUTS block with variational inference while retaining FFBS, as allowed by Step 24.

The command saves the posterior to `results/idata.pkl` and, when supported by the installed ArviZ stack, `results/idata.nc`.

```bash
python scripts/train.py
```

Runtime depends strongly on the panel and Gibbs settings. To use the scalable VI/FFBS alternative:

```bash
python scripts/train.py --method vi
```

### 4. Calibrate economic damage

```bash
python scripts/calibrate_damage.py
```

### 5. Evaluate

Rolling-origin expanding-window evaluation against five baselines (RF, ARIMA, ETS, seasonal Naive, BSTS-U). Outputs CRPS/LogS/MAE/RMSE tables and Diebold-Mariano test results.

```bash
python scripts/evaluate.py
```

Results are written to `results/evaluation/scores.csv`, `results/evaluation/dm_test.csv`, `results/evaluation/calibration.csv`, and rendered as an interactive baseline-comparison chart in the dashboard's forecast tab.

### 6. Forecast

Generates 12-month-ahead predictive draws, quantile CSV, interactive fan charts, the loss distribution with systemic-event probabilities `P(loss > c)`, the ranked sector-exposure table, and the regime-probability early-warning series.

```bash
python scripts/forecast.py --horizon 12
```

Interactive figures (`.html` to open in a browser, `.json` for the dashboard) land in `results/forecasts/figures/`. Uses the event-calibrated damage functions automatically when `results/calibration/damage_params.json` exists.

### 6. Backtest (check the model against reality)

Sequential one-step-ahead filtered evaluation over the full panel: starting
from the first month, the model predicts each month **before** seeing it, is
scored, then updates its regime beliefs with that month's actual data across
all four observation channels. Needs only the trained model — no network.

```bash
python scripts/backtest.py                       # test window 2023-01..2024-12
python scripts/backtest.py --test-start 2022-01  # or pick your own window
```

Outputs `results/backtest/backtest_scores.csv`, per-topic and per-sector
predicted-vs-actual comparisons, and interactive charts in
`results/backtest/figures/` (also rendered in the dashboard's forecast tab,
including the per-sector incident charts with a sector filter).

### Paper analyses (run after training)

```bash
# Damage-function calibration against the reference event set + Table 7
# economic backtest (edit documented loss ranges in configs/reference_events.yaml)
python scripts/calibrate_damage.py

# Table 4 ablation study: -regimes, -factors, -hierarchy, -EPSS channel,
# -incident channel, -effort, and the optional "+kernel" variant.
# Resumable; merges results across invocations.
python scripts/ablation.py --num-warmup 300 --num-samples 300

# §4.1/Appendix: K=1024 sparse-topic population (MiniBatchNMF), active-topic
# set, Full-via-VI vs per-series BSTS-U. Long run (~20h) -- leave overnight.
python scripts/full_population.py --k-full 1024 --vi-steps 6000

# Kernel experiment (test theory, see §9): train the variant, then distill
# its posterior into the dashboard's Kernel lab tab.
python scripts/ablation.py --variants kernel
python scripts/kernel_report.py
```

## Tests

```bash
pytest
# or with coverage
pytest --cov=cassandra_threatcast --cov-report=term-missing
```

50 tests cover Leontief correctness, panel aggregation, scoring, calibration, the legacy model, and the paper-exact model. Four live BEA tests are skipped unless `BEA_API_KEY` is set.

## Project structure

```
paper-cyber-threatmodelling/
├── app.py                         # 🔮 guided Streamlit dashboard (streamlit run app.py)
├── docs/PAPER_NOTES.md            # ⚠️ model corrections vs. the paper (identifiability, π, regime marginalization)
├── docs/MODEL_VARIANTS.md         # paper-native vs. --enhanced-mode comparison
├── configs/default.yaml          # K=8 topics, S=11 sectors, r=3 intensity + r_sigma=2 severity factors, R=3 regimes
├── configs/reference_events.yaml # documented loss ranges for damage calibration (edit as sources improve)
├── scripts/                      # CLI entry points
│   ├── ingest_data.py             # pipeline steps, run in order
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── forecast.py
│   ├── backtest.py                # sequential 1-step-ahead filtered backtest
│   ├── run_all.py                 # run all steps end-to-end (--fresh to wipe & restart)
│   ├── calibrate_damage.py        # paper analyses:
│   ├── ablation.py                #   Tables 7, 4
│   ├── full_population.py         #   §4.1 / Appendix (K=1024 topics)
│   └── kernel_report.py           #   Kernel lab artifacts (test theory, §9)
├── src/cassandra_threatcast/
│   ├── data/       # NVD, EPSS, CISA KEV, SEC 8-K, BEA I-O ingestion
│   ├── features/   # Topic mapper (TF-IDF+NMF; MiniBatchNMF at large K), exposure map, effort estimators
│   ├── model/      # NumPyro model (factor AR + optional moving-window kernel, Markov regimes, 4 obs channels, Leontief)
│   ├── inference/  # paper-exact blocked NUTS/FFBS, scalable VI/FFBS, legacy inference
│   ├── evaluation/ # CRPS, log score, DM test, PIT calibration, 5 baselines, sequential filter machinery
│   └── viz/        # interactive plotly charts (fan, loss, regimes, baselines, kernel) + legacy matplotlib
└── tests/
```

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.K` | 8 | Number of threat topics |
| `model.S` | 11 | Number of BEA sectors |
| `model.r` | 3 | Number of latent intensity factors |
| `model.r_sigma` | 2 | Number of latent severity factors |
| `model.R` | 3 | Number of regimes |
| `mcmc.num_chains` | 1 | Blocked MCMC chains (run additional seeds independently) |
| `mcmc.num_samples` | 2000 | Posterior samples per chain |
| `evaluation.horizons` | [1,3,6,12] | Forecast horizons in months |
| `bea.year` | 2022 | BEA I-O table year fetched via API |
| `enhanced.enabled` | false | Enhanced model variant (also set by `--enhanced-mode`) — see [docs/MODEL_VARIANTS.md](docs/MODEL_VARIANTS.md) |
| `enhanced.student_t_df` | 4.0 | Student-t d.o.f. for heavy-tailed innovations (enhanced only) |
| `kernel.enabled` | false | Learned moving-window covariance kernel (test theory — PAPER_NOTES §9, dashboard 🧪 Kernel lab) |
| `kernel.halfwidth_prior_months` | 24 | Prior centre for the kernel window half-width |
