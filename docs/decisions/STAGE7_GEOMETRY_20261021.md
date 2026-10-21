# Stage 7 inference geometry decision

The recovery campaign exposed slow mixing under the initial diagonal metric.
This is an inference configuration decision, not a change to the frozen
likelihood, priors, target, four-chain budgets or diagnostic thresholds.

- `recovery-v1`: exact Gaussian group-mean marginalization, innovation factor
  coordinates and diagonal NUTS metric. All three independent-count fixtures
  passed; all three latent AR fixtures exhausted their retries. Sparse-factor
  attempts are retained, including their unsuccessful energy/ESS diagnostics.
- `recovery-v2`: centered factors with their normalized stationary AR density
  outside the HMM, diagonal metric; first sparse-factor seed only. Its attempts
  are retained. Centering alone did not pass the first two budgets.
- `recovery-v3`: centered factors and PyMC's `QuadPotentialFullAdapt` dense metric.
  The first sparse-factor seed passed the initial budget: maximum R-hat 1.00651,
  minimum bulk/tail ESS 626.56/616.39, minimum BFMI 0.30775, no divergences or
  depth saturation. This BFMI is close to the 0.3 threshold; it is not described
  as strong evidence merely because it passes.

Use v3 for the remaining fixtures, with identical settings for all three seeds
within each scenario. Preserve the completed v1 independent and latent-AR cases
as explicitly labelled controls; do not claim that the AR failure proves failure
under all possible metrics or priors. Their old diagonal configuration remains
visible in the audit. Do not choose a configuration separately by seed or based
on recovery/forecast scores. The v3 decision is made from computational diagnostics
before examining its recovery scores.

The dense potential starts with identity covariance and prior weight 10. Its
first covariance window is `max(101, 2*dimension)` and Cholesky updates occur every
10 draws during adaptation. The longer initial window avoids attempting an
unregularized covariance estimate from fewer observations than dimensions.
Adaptation stops for retained sampling. This configuration is explicitly marked
experimental by the installed PyMC implementation; that message is recorded,
not treated as a numerical warning. Other numerical warnings still reject a fit.

Validation includes the centered AR prior against its stationary Gaussian
covariance, native/NUMBA gradients and HMM enumeration in both coordinate systems,
and a four-chain known-Gaussian dense-metric check. The latter passed, with maximum
mean error 1.182 MC standard errors and maximum absolute covariance error 0.0312.
It is a sampler check, not full-model validation. The
[Stan reference manual](https://mc-stan.org/docs/reference-manual/mcmc.html)
describes dense-metric adaptation and its limits: it can address linear
correlations but does not generally resolve nonlinear posterior geometry.

The centered fixture uses a flat base measure plus the normalized AR potential;
prior simulation therefore uses the independent generator, not PyMC's generic
prior-predictive method for that flat base measure. This is research fixture code;
production interfaces and combinations still need Stage 8 validation.
