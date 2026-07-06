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

> **"Training" here means Bayesian posterior inference, not machine-learning weight-fitting.** The model defines a posterior `p(parameters, latent states | data) ∝ likelihood × prior` that has no closed form for a state-space model of this complexity (regime switching, factor dynamics, hierarchical priors, non-conjugate likelihoods). `train.py` therefore *approximates* that posterior by drawing samples via MCMC (NUTS, with Gibbs updates for the discrete regime path) — or, optionally, variational inference. This is exactly the inference procedure the model's mathematics prescribes; every downstream output (predictive distributions, loss VaR/ES, regime probabilities, CRPS scores) is a functional of this posterior. Nothing here is trained by gradient descent on a loss.

Runs NUTS wrapped in `DiscreteHMCGibbs` — Gibbs updates for the discrete Markov regime path, NUTS for all continuous parameters (4 chains × 2000 samples after 1000 warmup) — and saves the posterior to `results/idata.nc`.

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
├── docs/MODEL_VARIANTS.md         # paper-native vs. --enhanced-mode comparison
├── configs/default.yaml          # K=8 topics, S=11 sectors, r=3 intensity + r_sigma=2 severity factors, R=3 regimes
├── scripts/                      # CLI entry points (run in order)
│   ├── ingest_data.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── forecast.py
├── src/cassandra_threatcast/
│   ├── data/       # NVD, EPSS, CISA KEV, SEC 8-K, BEA I-O ingestion
│   ├── features/   # Topic mapper (TF-IDF+NMF), exposure map, HP-filter effort
│   ├── model/      # NumPyro model (intensity + severity factor AR, Markov regimes, 4 obs channels, Leontief)
│   ├── inference/  # NUTS + DiscreteHMCGibbs, FFBS regime path sampler, VI fallback
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
