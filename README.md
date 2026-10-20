# Cyber-threat modelling

Repository: [xrisbarney/cassandra](https://github.com/xrisbarney/cassandra).

Python implementation of the staged research plan. Stage 3 supplies packaging,
configuration validation and run provenance. Models, ingestion, backtesting,
economic calibration and paper results are not implemented yet.

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

## Design and scope

See [implementation plan](IMPLEMENTATION_PLAN.md),
[evaluation protocol](docs/EVALUATION_PROTOCOL.md),
[baseline specifications](docs/BASELINE_SPECIFICATIONS.md),
and [statistical analysis plan](docs/STATISTICAL_ANALYSIS_PLAN.md).
Clarifications to the frozen design are recorded in [amendments](docs/AMENDMENTS.md).

The dependency lock is a software-resolution artifact, not proof that a future
high-dimensional HMM model will converge. The small synthetic environment probe
passed on Windows/Python 3.12; see [Stage 3 verification](docs/STAGE3_SETUP.md).
PyTensor currently uses its Python fallback because no C++ compiler was detected.
The compiler must be configured and verified before Stage 7's synthetic test suite.
Full-model runtime and convergence also require assessment before substantial fitting.

Packaging follows the [Python packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
Use locked commands to prevent unnoticed dependency resolution changes; see
[uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).
