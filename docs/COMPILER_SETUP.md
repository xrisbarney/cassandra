# Windows compiler setup and verification

Verified 2026-10-20 on Windows, Python 3.12.0, PyTensor 2.38.3 and PyMC 5.28.5.
The compiler prerequisite for Stage 7 is satisfied in this environment. Stage 5
remains paused; this check used only two synthetic observations.

## Installation

The versioned [compiler specification](../configs/windows_compiler.json) pins
WinLibs GCC 14.2.0 / MinGW-w64 12.0.0, x86_64 POSIX SEH UCRT, release
`14.2.0posix-12.0.0-ucrt-r3`. The archive SHA-256 was verified against the
[publisher's checksum](https://github.com/brechtsanders/winlibs_mingw/releases/download/14.2.0posix-12.0.0-ucrt-r3/winlibs-x86_64-posix-seh-gcc-14.2.0-mingw-w64ucrt-12.0.0-r3.zip.sha256):

```text
88868d745b807f083a117ff69348d8bc021ad7389aa503379dbed1866efcaeb9
```

The installed g++ binary SHA-256 is:

```text
f45f8fd915aecbf8a15692089dee8ebef0f210148f13ea5c1642e55376dd234c
```

Run from the project root after installing the locked science environment:

```powershell
.\.venv\Scripts\python.exe scripts/setup_compiler.py
. .\scripts\Use-Compiler.ps1
.\.venv\Scripts\python.exe scripts/check_compiler.py --output artifacts/compiler/new-check.json
```

The installer downloads into ignored `.tools/compiler`, checks the archive hash,
and installs below `%LOCALAPPDATA%\Programs\CassandraToolchain\<release>`.
It adds `mingw64\bin` to the user PATH, preserving the previous value in the local,
ignored `user-path-before.json`. System PATH and Python dependencies are unchanged.
An installation record is saved locally for activation. The verifier refuses to
overwrite an existing report; select a fresh output name for a later run.

New terminals should inherit the updated user PATH. Existing PowerShell sessions
can dot-source `Use-Compiler.ps1`, which updates that session's PATH and appends an
explicit `cxx` setting to `PYTENSOR_FLAGS`. A fresh Python process with PATH rebuilt
from the Windows machine/user registry values and without `PYTENSOR_FLAGS` also
detected the compiler automatically. These mechanisms follow PyTensor's
[compiler configuration](https://pytensor.readthedocs.io/en/latest/library/config.html).

## Measured verification

The [saved probe output](checks/compiler_check_20261020.json) contains the exact
measurements. The probe forces the C-only linker for an expression and derivative,
then for a PyMC model's log density and gradient, and checks native C thunks and
numerical results. Compilation therefore cannot silently fall back to Python in
these checks.

The synthetic model has prior `mu ~ Normal(0, 1)` and observations `[1, 2]` with
known standard deviation 1. Its exact posterior is Normal with mean 1 and variance
1/3. NUTS used four chains in four processes, 1,000 warmup and 1,000 retained draws
per chain, seed 20261020, target acceptance 0.95 and maximum tree depth 12.

| Check | Measured result |
|---|---:|
| Native expression and derivative compilation | 8.57 s |
| Model log density and gradient compilation | 4.74 s |
| Sampling wall time, including initialization | 44.29 s |
| Posterior mean / variance | 1.0234 / 0.3048 |
| R-hat | 1.0011 |
| Bulk / tail effective sample size | 1,604 / 1,574 |
| Mean Monte Carlo standard error | 0.01381 |
| Minimum chain BFMI | 0.9763 |
| Divergences / tree-depth saturations | 0 / 0 |

The scripted acceptance criteria are R-hat below 1.01, bulk/tail ESS at least 400,
no divergences or tree-depth saturation, BFMI above 0.3, posterior mean within
five Monte Carlo standard errors of the exact value, and posterior variance within
0.08 of the exact value. All passed. These are environment smoke-test criteria;
the frozen research protocol continues to govern substantive model diagnostics.

Timings describe this machine and run, including compilation/cache and process
startup effects. They do not establish a speedup against the earlier Python-only
probe, which used a different sampling budget. Full-model runtime, convergence,
identifiability and synthetic recovery still require the later planned checks.

Repository checks after the setup changes: 37 tests passed, Ruff lint and the new
Python scripts' formatting passed, the dependency lock is current, and
`ctm validate configs/pilot.toml` passed with the frozen design intact.
