# Stage 3: project setup and verification

Completed 2026-10-20 in `C:\Users\Chris\Documents\cyber-threat-modelling`.
The January 2024–December 2025 held-out forecast period is unchanged. No held-out
scores, real-data fits, or downloads were produced during this stage.

## Deliverables

- Installable `src/cyber_threat_modelling` package with eight namespaces matching
  the implementation plan. Domain modules are placeholders for later stages.
- Python 3.12 environment, `pyproject.toml`, and exact dependency resolution in
  `uv.lock`; uv itself is pinned to 0.13.4. `.venv` and `.tools` are local and ignored.
- `configs/pilot.toml`: the frozen initial count-only, one-regime pilot with K=10,
  two factor ranks, no effort adjustment or kernel, and the initial sampling budget.
- `ctm validate`: validates types, dates, primary grid, approved sampling budgets,
  and the current Stage 2 file hashes. The schema covers the primary experiment;
  later baseline adapters will need their own configuration sections.
- `ctm prepare-run`: writes settings, seed, platform and dependency versions,
  source hashes, design hashes, and Git revision when available. Existing run
  directories cannot be overwritten. Status is `prepared_not_executed` and data
  provenance remains explicitly pending. No credential files are loaded.
- Tests for period boundaries, frozen-design changes, invalid settings, canonical
  hashes, run overwrite protection, provenance, and CLI behavior.

At the Stage 3 verification, this folder had no Git revision, so the original
example record reports Git as unavailable and records source hashes. That record
is retained unchanged. The subsequent repository setup connects this folder to
[xrisbarney/cassandra](https://github.com/xrisbarney/cassandra), with a fresh history
on `main`; future run preparations record its source revision. This is separate
from the `paper-cyber-threatmodelling` repository.

## Commands and observed results

Commands below use the local uv installation; see the root README for bootstrap.

```powershell
.\.tools\Scripts\uv.exe sync --locked --extra science
.\.tools\Scripts\uv.exe lock --check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\ctm.exe validate configs/pilot.toml
.\.venv\Scripts\ctm.exe prepare-run configs/pilot.toml --output artifacts/runs/stage3-smoke
.\.tools\Scripts\uv.exe build
```

The preparation command succeeds once; a repeat with the same output directory
must fail. Choose a new name for subsequent runs.

| Check | Observed result |
|---|---|
| Scaffold test suite | 17 passed |
| Ruff | All checks passed |
| Dependency lock and installation | Resolved and installed successfully |
| Installed CLI and current freeze | Valid; evaluation origins 24/22/19/13 and development origins 60/58/55/49 |
| Preparation record | `artifacts/runs/stage3-smoke/run.json`, prepared only |
| Distribution build | Source archive and wheel built successfully |
| Parquet round trip | Passed |
| Toy random forest and ARIMA | Finite predictions |
| PyMC log density and gradient | Finite |
| Synthetic Normal posterior | Mean 0.944900; exact mean 1.0; zero divergences |

The synthetic probe uses only generated data and two short chains (300 warmup
and 300 retained draws each). Its R-hat was 1.019279, with a corresponding PyMC
warning. This is an import, differentiation, and sampler-execution check, **not**
a convergence-qualified research fit. The research protocol retains four chains
and its stronger diagnostics; this probe does not relax those requirements.

## Verified environment and compatibility correction

Tested on Windows with CPython 3.12.0. These are the installed versions; the lock
also records transitive dependencies and artifact hashes.

| Package | Version |
|---|---|
| NumPy | 2.3.5 |
| Numba | 0.63.1 |
| SciPy | 1.18.1 |
| pandas | 2.3.3 |
| PyArrow | 26.0.0 |
| PyMC | 5.28.5 |
| PyTensor | 2.38.3 |
| ArviZ | 0.23.4 |
| xarray | 2026.9.0 |
| scikit-learn | 1.9.1 |
| statsmodels | 0.15.0 |

The first resolution installed NumPy 2.5.4 with Numba 0.63.0b1; importing ArviZ
failed because that Numba code referenced the unavailable `numpy.row_stack`.
The science extra now requires NumPy below 2.4 and stable Numba 0.63.1 below 0.64.
The revised lock passed the probe. This bound agrees with the
[Numba 0.63.1 support table](https://numba.readthedocs.io/en/0.63.1/user/installing.html).

PyTensor detected no `g++` compiler (`pytensor.config.cxx` is empty). The probe ran
with its Python fallback. Stage 3 setup is complete, but compiled performance has
not been demonstrated. Before Stage 7's synthetic tests, establish a working C++
compiler environment, verify compiled density/gradient execution, and benchmark
a synthetic NUTS fit. Record the chosen execution environment and timings. Other
operating systems, GPU execution, and the future neural-comparator runtime were
not tested here.

## Next stage

Stage 4 audits and ingests the sources, populates availability/vintage and source
manifests, verifies coverage, and creates an auditable panel. Existing-file metadata
checks do not establish complete or historically reconstructable data. The frozen
protocol's eligibility gates apply before any historical forecast evaluation.
