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
uncertified. No forecast scores have been computed.

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

The initial run contains historical mean, seasonal naive, and a diagnostic
ARIMA(0,1,0) candidate. This is not the frozen ARIMA grid selection. Commands
refuse held-out execution and refuse to overwrite runs or scores. Bayesian
control adapters are implemented with diagnostic gates; their posterior
validation and rolling fits remain outstanding. See the
[category and KEV carry-forward decisions](docs/STAGE6_8_CARRY_FORWARD.md).
The [Stage 6 report](docs/STAGE6_BACKTEST.md) contains the initial development
tables, failed-fit register and remaining validation gates.

Use locked commands to prevent unnoticed dependency resolution changes; see
[uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).
