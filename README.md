# Cyber-threat modelling

Repository: [xrisbarney/cassandra](https://github.com/xrisbarney/cassandra).

Python implementation of the staged research plan. Stage 3 supplies packaging,
configuration validation and run provenance. Stage 4 adds raw source adapters,
coverage audits and hash-verified data manifests. Stage 5 adds frozen initial-window
topics, monthly panels and cell eligibility. Models, backtesting, economic
calibration and paper results remain later-stage work.

The held-out forecast period is January 2024–December 2025. The initial pilot
has ten categories including unknown, two factors, one regime, counts only,
and no effort adjustment or kernel. The configuration does not launch training.

## Setup on Windows

Python 3.12 and uv 0.13.4 are used for this environment. With uv already installed:

    uv sync --locked --extra science
    uv run --locked --extra science ctm validate configs/pilot.toml
    uv run --locked --extra science pytest
    uv run --locked --extra science ruff check .
    uv run --locked --extra science python scripts/check_environment.py

The default development group includes pytest and Ruff. The science extra installs
NumPy, SciPy, pandas, PyArrow, PyMC/PyTensor, ArviZ/xarray, scikit-learn and statsmodels.
The base package's configuration/provenance commands use only the Python standard library.
The neural runtime will be added and locked when that comparator is implemented.

If uv is absent, bootstrap it in a separate local tools environment:

    python -m venv .tools
    .\.tools\Scripts\python.exe -m pip install uv==0.13.4
    .\.tools\Scripts\uv.exe sync --locked --extra science

Use that local uv executable in place of uv in the other commands. No activation,
global package installation, API key or credential-file loading is required.

## Prepare a run

    uv run --locked --extra science ctm prepare-run configs/pilot.toml --output artifacts/runs/pilot-001

This verifies the design hashes and writes a new run.json containing settings,
the seed, dependency versions, source-file hashes and Git revision when available.
It refuses an existing output directory. Its status is prepared_not_executed;
no data hashes are invented before ingestion. It never reads environment variables
or .env files. Configure credentials separately when the source adapters exist.

An explicit project location can be used from another directory:

    ctm --project-root C:\path\to\cyber-threat-modelling validate configs/pilot.toml

## Layout

| Directory | Responsibility |
|---|---|
| configs | Versioned experiment TOML |
| src/cyber_threat_modelling/data | Source adapters and provenance (Stage 4) |
| src/cyber_threat_modelling/features | Topic assignments and panels (Stage 5) |
| src/cyber_threat_modelling/models | Baselines and proposed models |
| src/cyber_threat_modelling/inference | Fitting and diagnostics |
| src/cyber_threat_modelling/forecasting | Joint posterior forecast paths |
| src/cyber_threat_modelling/economics | Exposure, damage and propagation |
| src/cyber_threat_modelling/evaluation | Rolling backtests and scoring |
| src/cyber_threat_modelling/reporting | Tables and figures from saved artifacts |
| tests | Configuration, provenance and command-line checks |
| notebooks | Later exploration; reusable logic belongs in the package |
| data/raw, data/interim, data/processed | Local data, excluded from version control |
| artifacts | Generated run records and results, excluded from version control |

## Audit source data

Use the science environment for the data adapters. A full audit retrieves NVD
feeds, EPSS month-end archives, KEV, SEC candidates and filing documents, BEA
tables, and the event evidence register. The existing audited snapshot is
`stage4-20261020`; its report is [Stage 4 data audit](docs/STAGE4_DATA_AUDIT.md).

    uv run --locked --extra science ctm audit-data --snapshot new-audit-id --credentials-file C:\path\to\.env --finalize
    uv run --locked --extra science ctm verify-data docs/audits/stage4-20261020/data_manifest.json
    uv run --locked --extra science ctm prepare-run configs/pilot.toml --output artifacts/runs/pilot-with-data --data-manifest docs/audits/stage4-20261020/data_manifest.json

Credentials are optional for public feeds; BEA needs BEA_API_KEY, SEC needs a
declared SEC_USER_AGENT, and NVD_API_KEY changes the API request interval. They
are not stored in the experiment config. BEA raw responses echo the API key, so
keep raw data local. The audit exports contain redacted request URLs and hashes.
`--sources nvd epss` limits an unfinished audit to named components. Successful
raw downloads are reused after checksum verification. `--finalize-only` freezes
completed component outputs without rerunning them; a frozen snapshot cannot be
overwritten. A fresh checkout has the manifests, but needs the corresponding
local data to pass `verify-data`. Current-source downloads may have changed since
the snapshot; use a new ID rather than claiming a newer download has its hashes.

See the [data dictionary](docs/DATA_DICTIONARY.md) for record exclusions, missing
values, vintage limitations, and the distinction between source coverage and
eligibility for modelling/scoring. No forecasts are produced by these commands.

## Design and scope

Stage 5's primary count panel is ready; see the
[panel report](docs/STAGE5_PANELS.md) and [panel dictionary](docs/PANEL_DICTIONARY.md).
The SEC sector panel remains masked because collection/population coverage is
uncertified. Development count scores are available; held-out scores remain locked.

    uv run --locked --extra science ctm verify-panels docs/panels/stage5-20261020/panel_manifest.json
    uv run --locked --extra science python scripts/check_panels.py docs/panels/stage5-20261020/panel_manifest.json

To rebuild from the pinned local source data, choose a fresh panel ID:

    uv run --locked --extra science ctm build-panels --data-manifest docs/audits/stage4-20261020/data_manifest.json --panel-id new-panel-id

The builder fits text features only on 2010–2018, freezes them for all later
assignments, and refuses to overwrite an existing panel directory.

See [implementation plan](IMPLEMENTATION_PLAN.md),
[evaluation protocol](docs/EVALUATION_PROTOCOL.md),
[baseline specifications](docs/BASELINE_SPECIFICATIONS.md),
and [statistical analysis plan](docs/STATISTICAL_ANALYSIS_PLAN.md).
Clarifications to the frozen design are recorded in [amendments](docs/AMENDMENTS.md).

The dependency lock is a software-resolution artifact, not proof that a future
high-dimensional HMM model will converge. The small synthetic environment probe
passed on Windows/Python 3.12; see [Stage 3 verification](docs/STAGE3_SETUP.md).
The Windows GCC 14.2.0 compiler is installed and native PyTensor density/gradient
execution plus a four-chain synthetic NUTS probe passed. See
[compiler setup and verification](docs/COMPILER_SETUP.md). To reproduce the setup:

    .\.venv\Scripts\python.exe scripts/setup_compiler.py
    . .\scripts\Use-Compiler.ps1
    .\.venv\Scripts\python.exe scripts/check_compiler.py --output artifacts/compiler/new-check.json

The installer verifies the pinned archive checksum and adds the compiler to the
Windows user PATH. Dot-sourcing configures an already-open PowerShell session.
Full-model runtime and convergence also require assessment before substantial fitting.

Packaging follows the [Python packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

Stage 6 now provides separate development forecast and scoring commands:

    .\.venv\Scripts\python.exe -m cyber_threat_modelling forecast-development configs/stage6.json --panel-manifest docs/panels/stage5-20261020/panel_manifest.json --output artifacts/backtests/new-development-run
    .\.venv\Scripts\python.exe -m cyber_threat_modelling score-development artifacts/backtests/new-development-run
    .\.venv\Scripts\python.exe scripts/check_stage6_ensemble.py --run artifacts/backtests/new-development-run --output artifacts/backtests/new-development-run/ensemble_check.json

Commands refuse held-out execution and refuse to overwrite runs or scores.
The [Stage 6 completion record](docs/STAGE6_COMPLETION.md) reports all five initial
controls across 60 development origins, posterior validation, ARIMA selection
and complete development results. The 2024-2025 holdout remains unscored. The
original [first benchmark report](docs/STAGE6_BACKTEST.md) is historical. See the
[category and KEV carry-forward decisions](docs/STAGE6_8_CARRY_FORWARD.md).
`configs/stage6.json` reproduces that first fixed-control run with diagnostic
ARIMA(0,1,0); it is not the selected ARIMA specification.

The [independent-control amendment](docs/amendments/OBSERVATION_CONTROL_20261021.md)
replaces the unsuccessful latent control with a Bayesian conditional NB
autoregression (`independent_nb_glar`). Its priors, conditional likelihood and
recursive forecasts are explicit; its original convergence thresholds remain.
The [validation record](docs/STAGE6_BAYESIAN_VALIDATION.md) preserves both the
failed latent attempts and the successful replacement's initial checks.
`configs/stage6_bayesian.json` and `configs/stage6_independent.json` retain the
superseded latent control for diagnostics. Use `configs/stage6_observation_nb.json`
for the primary independent control and `configs/stage6_pooled.json` for the pooled one.
The common reporting budget is now 6,000 paths, following the protocol's existing
fallback after the 3,000-path Monte Carlo check failed. The
[budget decision](docs/decisions/ENSEMBLE_BUDGET_20261021.json) binds the evidence;
checks compare 6,000 with 12,000. `scripts/reissue_stage6.py` reuses accepted fits
at this budget, preserving the original runs and all posterior diagnostics.

    . .\scripts\Use-Compiler.ps1
    .\.venv\Scripts\ctm.exe forecast-development configs/stage6_observation_nb.json --panel-manifest docs/panels/stage5-20261020/panel_manifest.json --output artifacts/backtests/new-independent-run --workers 4
    .\.venv\Scripts\python.exe scripts/check_stage6_bayesian_ensemble.py --run artifacts/backtests/new-independent-run --output artifacts/backtests/new-independent-run/ensemble_check.json
    .\.venv\Scripts\ctm.exe score-development artifacts/backtests/new-independent-run

The [2026-10-21 amendment](docs/amendments/ZERO_SERIES_20261021.md) is active for
new ARIMA/ETS runs. It retains all categories and uses an explicit zero point
mass for an entirely-zero observed training series. Historical runs retain
their original definitions. New run provenance records the supplemental freeze.
The frozen ARIMA screen can now be run with a fresh output directory:

    .\.venv\Scripts\python.exe scripts/stage6_arima_screen.py --workers 2 --output artifacts/backtests/new-arima-screen

It evaluates all 65 candidates at the 20 quarterly development origins. The
recorded run and the two-finalist 60-origin comparison selected ARIMA(2,1,1)
without trend. `configs/stage6_arima_finalists.json` binds that particular local
screening record by hash. A new screening run requires its own verified record
path and hash in the finalist configuration.

Stage 7 posterior fixtures use generated data only. The current prior-plausibility
audit separately reads January 2010--December 2018 counts, never held-out counts.
The [design](docs/STAGE7_DESIGN.md),
[execution record](docs/STAGE7_REPORT.md), and
[prior-tail finding](docs/STAGE7_PRIOR_FINDING.md) distinguish numerical component
checks, fixed-truth recovery, and remaining model-validation gates. The user
stopped the old-prior campaign on 2026-10-21. The subsequent
[activation record](docs/amendments/STATE_SPACE_PRIORS_ACTIVATION_20261021.md)
records the first stationary-amplitude revision. The
[second amendment](docs/amendments/STATE_SPACE_LEVELS_20261021.md) now governs
`anchored_levels_v2`: anchored known-topic levels, a separate fallback level,
declared predictive gates and both directions of sensitivity. The primary prior
passed its [recorded checks](docs/checks/stage7-levels-20261021/audit.json).
Revised runs use new directories; old results remain historical evidence.
To reproduce the v2 prior audit with a new output path:

    .\.venv\Scripts\python.exe scripts/check_stage7_level_prior.py --output artifacts/stage7/new-level-prior-audit

Revised posterior runs verify and record the activation hashes:

    . .\scripts\Use-Compiler.ps1
    $env:NUMBA_OPT='1'
    .\.venv\Scripts\python.exe scripts/run_stage7.py --scenario factors_sparse --seed 7101 --anchor-factors --dense-mass --conditional-counts --block-coordinates --output artifacts/stage7/new-amended-factor-case

The campaign comprises 43 primary cases and 12 sensitivity cases, including
deliberate reuse of observations in the three effort variants. Coordinate and
metric repairs follow the decisions linked in the report; one generic batch
command is not the complete routed campaign. `scripts/stage7_batch.py` accepts
explicit scenario/seed/candidate subsets and preserves existing partial fits.
`scripts/stage7_rank_two.py` normally requires representative admission. Its
explicit `--synthetic-grid` mode follows the recorded completion decision:
verified joint coordinates and generator hashes permit the diagnostic grid
to run independently of representative success, while every posterior still
faces the original admission gates. The joint grid uses `NUMBA_OPT=0`; the
other recorded conditional-coordinate configurations use their declared profile. New fit processes record and limit native numerical pools
to one thread. Four chains are sampled sequentially within each fit.

The [original 55-case campaign is complete](docs/simulations/stage7-complete-20261022/README.md):
30 numerical passes and 27 recovery-eligible cases. The two-factor pilot remains
unvalidated. The [bounded convergence supplement](docs/STAGE7_SUPPLEMENT_REPORT.md)
is also complete: all four single-attempt cases failed the unchanged diagnostic
gates. No one-factor pilot amendment is admitted. Stage 8 and held-out scoring
remain closed; no further repair round has been launched.

The historical campaign's control cases used a diagonal metric, as recorded in
the geometry decision. `scripts/summarize_stage7.py --controls-from` preserves
that distinction in its audit. The current summary accepts explicit `--routing`
and `--candidate` arguments. `scripts/audit_stage7_campaign.py` checks completion,
preserved earlier attempts and source/data hashes; `--recompute-diagnostics`
independently checks admitted saved posteriors. `scripts/analyze_stage7.py`
describes admitted posteriors, and `scripts/plot_stage7.py --audit` follows the
same routed summary. Failed fits remain failures. The
[test audit](docs/checks/stage7-test-suite-20261022/audit.json) records 175 distinct
passing tests; posterior convergence is a separate gate.

Use locked commands to prevent unnoticed dependency resolution changes; see
[uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).
