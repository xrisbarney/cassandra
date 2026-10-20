# Stage 6 continuation and review disposition

Stage 6 remains **in progress**. No held-out outcomes have been scored, no ARIMA
screen has started, and no Bayesian rolling benchmark has been admitted by this
continuation. The completed fixed-control development results remain in
[the initial report](STAGE6_BACKTEST.md).

## Zero-series decision

The all-zero Gaussian likelihood issue is structural. A regular positive-variance
MLE does not exist at its zero-variance boundary; more optimizer iterations are
not a solution. With the current atomic-origin rule, failure on category 10
prevents a complete primary ARIMA row. ETS needs the same design decision before
it is used. This is not evidence that either method has poor predictive skill.

[The draft amendment](STAGE6_BOUNDARY_AMENDMENT_DRAFT.md) proposes a declared
zero-point-mass boundary model for any entirely-zero training series, retaining
all categories and assigning infinite negative log score to a subsequent
positive outcome. It is inactive pending the requested user decision. No silent
fallback is implemented. The screening script currently raises an explicit
error before opening a run directory. The frozen Stage 2 documents and Stage 5
panel remain unchanged.

## Bayesian implementation work

- Exact Gaussian reparameterization of the pooled regression removes its
  intercept ridge while preserving the complete original joint prior.
- The independent NB AR(1) now uses white innovations and a linear-time stationary
  AR transform, with an explicit reverse derivative. Value, derivative and
  stationary-covariance checks cover negative and near-boundary autocorrelation.
- Native and NumPy/Numba compiled log densities and gradients agree in tests.
  Four-chain diagnostics, explicit failure/retry records, prior/posterior
  predictive checks and posterior forecast simulation are implemented.
- The development runner can use persistent independent-origin workers. Every
  origin still constructs a fresh model and adapts fresh chains; no later fitted
  state is reused. Forecast persistence still precedes scoring.
- A separate optional nutpie execution probe is pinned in
  `configs/nuts-runtime.lock`. The base scientific lock is unchanged. Both PyMC
  and nutpie tree-depth diagnostic conventions are checked explicitly. This
  probe has not replaced the production sampler.
- A Bayesian ensemble-check script reproduces saved 3,000-path forecasts,
  compares 1,000/3,000/6,000 paths over the last six eligible development origins,
  and reports NB-mixture log-mass sensitivity to an independent future seed.
  It has not yet been run: it requires completed rolling artifacts.

The implementation details and mathematical checks are recorded in
[Bayesian validation](STAGE6_BAYESIAN_VALIDATION.md). Execution probes use the
initial development origin, December 2018. They are exploratory implementation
checks, not candidate-selection scores or evidence of forecasting superiority.
Intermediate probe parameterizations used an uncommitted working tree; do not
cite them as a clean production run. Original failures and aborted timing probes
are retained under `artifacts/stage6-completion/`. A small
[diagnostic snapshot](backtests/stage6-validation-20261021/exploratory_diagnostics.json)
preserves the completed attempt records. Timings were collected under concurrent
probe load and are not isolated performance estimates for the rolling schedule.

## Verified findings so far

The pooled control passed the PyMC/Numba four-chain check: maximum R-hat 1.00278,
minimum bulk/tail ESS 5358.9/2202.2, zero divergences and no depth saturations.
The compiled nutpie probe also passed (R-hat 1.00339, minimum bulk/tail ESS
4882.2/2533.7, zero divergences and no depth saturations). These are convergence
checks at one origin; neither constitutes a completed rolling backtest.

The independent white-innovation fit's first attempt had maximum R-hat 1.00708
and minimum bulk/tail ESS 480.0/585.9, but **47 divergences**. It is rejected.
Its fixed retry budget, not a relaxed convergence threshold, determines whether
it can be admitted. A separate compiled-sampler probe is still being evaluated.

The final full test run in this continuation reported **85 passed**, with the
documented missing-BLAS performance warning. Ruff, `uv lock --check`, installed
package compatibility and panel verification pass. The panel still has its
original SHA-256 `0448a3eb7f6d4a17d3808b98ad7a315105ed2cc72974cf7828afb4c3d5515840`
and `scoring_ready: false`. Configuration validation retains development counts
60/58/55/49 and held-out counts 24/22/19/13. Test output is saved locally at
`artifacts/stage6-completion/pytest-final.txt`.

For the lint review, the current working script was formatted and checked.
In this checkout `stage6_bayesian_probe.py` was untracked before this continuation
was committed; `git ls-tree` finds no such path in `10666b1`. An extracted copy
of the preceding HEAD `60d08a7` also passed Ruff. That establishes the scope of
the local verification without attributing another workspace's error to a
commit that does not contain the file.

## Remaining completion gates

For the five initial Stage 6 controls: resolve the zero-series amendment; complete
posterior validation and rolling Bayesian fits; screen/select the ARIMA control;
run ensemble checks and generate initial tables from admitted artifacts.

The six other requested comparator implementations, proposed-model development
selection and its saved selection record remain required for the eventual main
comparison. They are not supplied by these initial controls. Full-model synthetic
recovery is Stage 7 work; factor/regime model construction is Stage 8 work. None
of those gates is waived by completing the backtesting infrastructure.
