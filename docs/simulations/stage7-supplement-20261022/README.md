# Bounded supplement: four failed attempts

The [report](../../STAGE7_SUPPLEMENT_REPORT.md) records the complete outcome.
[audit.json](audit.json) binds the four result inventories and timing fields to
the declared execution. Each file under `imports/` records provider provenance
and independently checked import diagnostics. Full posterior files and partial
states remain locally under `artifacts/stage7/supplement-v1/`; canonical inputs
and executable fitting code are committed. The original 55 cases are untouched.

Two complete posteriors fail the unchanged numerical gates. Their diagnostic
plots were visually inspected and remain labeled as failures:

- [Reduced representative chains](plots/reduced_representative/chains.png):
  the tail-ESS failure spans factor loadings, persistence and levels.
- [Reduced fallback chains](plots/reduced_fallback/chains.png): persistence
  shows long visits to high values and different occupancy across chains.
  This is diagnostic evidence, not a validated estimate of region weights.

The masked and matched cases stop at retained divergences. Neither has a
complete four-chain posterior, and neither receives a chain/recovery plot
pretending otherwise. No case is recovery eligible. These are four targeted,
fixed-truth experiments with explicit data reuse, not independent SBC replicates.

The retained audit helpers do not launch any sampler. To recompute the closure
summary from locally preserved, already-verified imports:

```powershell
.venv/Scripts/python.exe docs/simulations/stage7-supplement-20261022/summarize.py
```

To draw a complete failed posterior's diagnostics again, use `plot.py` with
`--case reduced_fallback` or `--case reduced_representative` and a fresh
`--output` directory. Existing output directories are refused. The plotting
script checks source-result artifact hashes and records its own hash.

The [preflight/test record](../../checks/stage7-supplement-20261022/README.md)
contains 39 distinct passing tests, four complete-target comparisons and the
one-factor prior plausibility audit. Software checks and prior plausibility
do not override these posterior failures. The [final checks](final-checks.json)
cover lint, lock, production configuration, panel integrity and unchanged
activation. Stage 8 and held-out scoring remain closed.
