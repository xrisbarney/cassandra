# CASSANDRA — Cyber Threat Modelling

Bayesian hierarchical state-space model of cyber threats and systemic economic risk.

Models K=8 threat topics across S=11 BEA economic sectors using latent intensity and severity factor processes with shared Markov regime switching. Four observation channels are treated as biased views of the latent state: CVE counts and severity marks (NVD/CVSS), exploitation probability (EPSS/CISA KEV), and SEC 8-K incident disclosures. Posterior draws are propagated through a BEA input–output Leontief inverse to produce predictive distributions of systemic economic loss.

## 🔮 Easiest way to use it: the dashboard

If you'd rather not use the command line, run the guided visual dashboard:

```bash
pip install -e ".[dashboard]"    # one-time: installs the dashboard
streamlit run app.py
```

A page opens in your browser with **five buttons**, one per step (Collect data →
Prepare inputs → Train → Check accuracy → Forecast). Click them top to bottom;
each shows a green light when finished, a live progress log while running, and
charts of your data and the final forecast. No commands, no jargon. Everything
below is the manual/command-line equivalent.

**Run everything, unattended:** the dashboard also has a **🌙 Run the whole
pipeline now** button that runs all five steps in sequence in the background.
Tick *Start completely fresh* to wipe all caches and re-download from scratch.
It keeps running even if you close the browser — great for leaving overnight.
The command-line equivalent is:

```bash
python scripts/run_all.py --fresh          # wipe everything and run all 5 steps
python scripts/run_all.py --quick          # keep caches, quick-preview training
```

## ⚠️ Model corrections vs. the paper

Making the model sample revealed issues in the paper's stated specification that
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

> **"Training" here means Bayesian posterior inference, not machine-learning weight-fitting.** The model defines a posterior `p(parameters, latent states | data) ∝ likelihood × prior` that has no closed form for a state-space model of this complexity (regime switching, factor dynamics, hierarchical priors, non-conjugate likelihoods). `train.py` therefore *approximates* that posterior by drawing samples via MCMC (NUTS, with the discrete regime path marginalized analytically inside the model) — or, optionally, variational inference. This is exactly the inference procedure the model's mathematics prescribes; every downstream output (predictive distributions, loss VaR/ES, regime probabilities, CRPS scores) is a functional of this posterior. Nothing here is trained by gradient descent on a loss.

Runs plain NUTS — the discrete Markov regime path is marginalized analytically via an HMM forward algorithm inside the model rather than Gibbs-sampled (4 chains × 2000 samples after 1000 warmup) — and saves the posterior to `results/idata.nc`.

```bash
python scripts/train.py
```

Training takes ~30–90 minutes on CPU depending on dataset length. To use VI instead (faster, less accurate):

```bash
python scripts/train.py --method vi
```

#### Model variant: paper-native vs. enhanced

By default the code runs the model **exactly as specified in the paper**. An opt-in `--enhanced-mode` flag switches on a set of Claude-proposed statistical enhancements (heavy-tailed Student-t latent innovations + a Negative-Binomial incident channel) aimed at better tail-risk calibration:

```bash
python scripts/train.py --enhanced-mode          # fit the enhanced model
python scripts/forecast.py --enhanced-mode ...    # forecast must match how it was fit
```

The two variants share all priors, latent structure, and the economic layer, so they form a clean ablation. See **[docs/MODEL_VARIANTS.md](docs/MODEL_VARIANTS.md)** for the full paper-vs-enhanced comparison table and the rationale for each change.

### 4. Evaluate

Rolling-origin expanding-window evaluation against five baselines (RF, ARIMA, ETS, Naive, BSTS). Outputs CRPS/MAE/RMSE tables and Diebold-Mariano test results.

```bash
python scripts/evaluate.py
```

Results are written to `results/scores.csv`, `results/dm_test.csv`, `results/calibration.csv`.

### 5. Forecast

Generates 12-month-ahead predictive draws, quantile CSV, and fan-chart figures.

```bash
python scripts/forecast.py --horizon 12
```

Figures are saved to `results/figures/`.

## Tests

```bash
pytest
# or with coverage
pytest --cov=cassandra_threatcast --cov-report=term-missing
```

46 tests covering: Leontief correctness (live BEA API smoke test), panel aggregation, CRPS (including the analytical value (√2−1)/√π ≈ 0.2337 for N(0,1) at 0), DM test sign/symmetry, the NumPyro model forward pass, the latent severity process (with NaN-mark masking), and both model variants (paper-native and `--enhanced-mode`). The four BEA live tests are skipped automatically unless `BEA_API_KEY` is set.

## Project structure

```
paper-cyber-threatmodelling/
├── app.py                         # 🔮 guided Streamlit dashboard (streamlit run app.py)
├── docs/PAPER_NOTES.md            # ⚠️ model corrections vs. the paper (identifiability, π, regime marginalization)
├── docs/MODEL_VARIANTS.md         # paper-native vs. --enhanced-mode comparison
├── configs/default.yaml          # K=8 topics, S=11 sectors, r=3 intensity + r_sigma=2 severity factors, R=3 regimes
├── scripts/                      # CLI entry points (run in order)
│   ├── ingest_data.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── forecast.py
│   └── run_all.py                 # run all steps end-to-end (--fresh to wipe & restart)
├── src/cassandra_threatcast/
│   ├── data/       # NVD, EPSS, CISA KEV, SEC 8-K, BEA I-O ingestion
│   ├── features/   # Topic mapper (TF-IDF+NMF), exposure map, HP-filter effort
│   ├── model/      # NumPyro model (intensity + severity factor AR, Markov regimes, 4 obs channels, Leontief)
│   ├── inference/  # plain NUTS (regime path marginalized in-model), FFBS terminal-regime recovery, VI fallback
│   ├── evaluation/ # CRPS, log score, DM test, PIT calibration, 5 baselines
│   └── viz/        # Fan charts, PIT histograms, regime-prob plots, sector exposure
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
| `mcmc.num_chains` | 4 | MCMC chains (reduce to 1 for quick tests) |
| `mcmc.num_samples` | 2000 | Posterior samples per chain |
| `evaluation.horizons` | [1,3,6,12] | Forecast horizons in months |
| `bea.year` | 2022 | BEA I-O table year fetched via API |
| `enhanced.enabled` | false | Enhanced model variant (also set by `--enhanced-mode`) — see [docs/MODEL_VARIANTS.md](docs/MODEL_VARIANTS.md) |
| `enhanced.student_t_df` | 4.0 | Student-t d.o.f. for heavy-tailed innovations (enhanced only) |
