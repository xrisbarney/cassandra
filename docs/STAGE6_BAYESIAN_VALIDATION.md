# Stage 6 Bayesian control validation

The pooled control keeps its frozen law. The primary independent control is now
the [conditional NB autoregression](amendments/OBSERVATION_CONTROL_20261021.md),
identifier `independent_nb_glar`. It replaces the unsuccessful latent AR(1)
control; this is a model change, not a claim that the latent control converged.
The original Stage 2 freeze and failed exploratory fits remain preserved.
Held-out scoring remains locked.

## Primary independent control

The conditional law is `log(mu[t])=(1-phi)*m+phi*log(1+N[t-1])` with NB2 counts.
Priors retain `m~Normal(log(initial_mean+0.5),2)`, `phi~Uniform(-1,1)` and
`log(psi)~Normal(0,1)`. Condition on January 2010 and use subsequent counts in
the likelihood. The reference-level parameter m is not an unconditional mean.
The amendment gives the complete specification and distinguishes its priors
from the model-family support in the literature.
The initial-window mean is a frozen empirical prior anchor, as in the original
control; its estimation uncertainty is not an additional posterior component.
It is never recomputed from development or held-out outcomes.

Every category retains an independent posterior, including the all-zero category.
This is an independent forecasting control, not an ablation that changes only
the proposed model's factors or regimes; those effects require later matched
ablations. The named BSTS-U and Bakdash state-space comparisons remain required.
Future paths recursively use simulated counts and fixed posterior parameter tuples.
The forecast PMF is the mixture of conditional NB masses over parameters and
preceding simulated counts. It is not a moment-matched NB approximation.
Tests compare the likelihood with SciPy, native/NUMBA gradients, joint versus
separate-category density, and two-step forecast probabilities with an explicit
sum over the preceding count. They also check replay, independent category
draws, and the informative all-zero likelihood without a point-mass exception.

The initial PyMC four-chain fit passed at the first budget: maximum R-hat
1.00432, minimum bulk/tail ESS 1471.83/1043.10, BFMI 0.9105, no divergences
or depth saturations. Prior/posterior replications recursively generate the
training period conditional on its first count. These are posterior and
implementation checks, not forecast performance evidence.
An independent nutpie fit also passed at the first budget (maximum R-hat
1.00345, minimum bulk/tail ESS 1526.78/1026.09, BFMI 0.9007, no divergences or
depth saturations). The largest difference between sampler means for m, phi
and psi was 2.04 combined MC standard errors. This is a descriptive agreement
check, not a new admission threshold. Production retains PyMC NUTS with the
NUMBA density/gradient backend. The
[initial validation record](backtests/stage6-control-resolution-20261021/observation_initial_validation.json)
and [recursive predictive checks](backtests/stage6-control-resolution-20261021/observation_initial_predictive_checks.json)
preserve the evidence and source artifact hashes.

## Superseded latent independent control

The earlier [stationary-scale amendment](amendments/INDEPENDENT_CONTROL_20261021.md)
retains the NB2 observation law, AR(1) log-rate deviations,
initial-window mean anchor, and `phi~Uniform(-1,1)`. It puts
`s~HalfNormal(0.75)` on the stationary log-rate SD and derives
`tau=s*sqrt(1-phi²)`. Its initial state is `u0~Normal(0,s)`.
The unchanged priors are `m~Normal(log(initial_mean+0.5),2)` and
`log(psi)~Normal(0,1)`. Normal's second argument denotes SD.

The [prior-check script](../scripts/check_stage6_independent_prior.py) uses only
2010-2018 and saves [its results](checks/independent_prior_20261021.json).
The largest simulated log rate is 160.99 under the original prior and 14.02
under the amended prior. Ten category-draw combinations exceed the log of the
int64 count limit under the original prior and none under the amended prior.
All amended count replications complete. These finite simulations do not bound
the support or prove calibration. The amendment derives the finite amended
prior predictive count mean; it does not claim every moment is finite.

This decision follows prior-predictive criticism in
[Gelman et al., Bayesian Workflow](https://arxiv.org/abs/2011.01808) and the
[Stan predictive-check guidance](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html).
Neither source establishes 0.75 as a cyber-risk prior; it is our design choice.
Independent-control forecast scores were not consulted.

Each category is fitted separately because its posterior factorizes exactly
from the others. A test compares the joint log density with the sum of the
individual log densities. Every category receives four fresh chains and fresh
adaptation under the frozen three-budget sequence. An origin is admitted only
if every category passes. Compiled code is reused between categories of the
same shape within an origin; observations and the fixed initial-window prior
anchor are rebound explicitly. Tests compare the rebound density and gradient
with a fresh model. No fit or posterior is carried between categories or origins.

The independent implementation uses pinned nutpie 0.15.2 with the NUMBA backend
and ordinary diagonal gradient-based mass adaptation. Its wheel is pinned in
`configs/nuts-runtime.lock`, separately from the unchanged base `uv.lock`.
The pooled implementation retains PyMC NUTS. New runs record the installed
inventory, source hashes and design amendments.

## Exact computational transformations

Let `m0=log(initial_mean+0.5)` and `v=s²`. The initial-state construction is

`eta0~Normal(m0,sqrt(4+v))`;

`m|eta0,s~Normal(m0+4/(4+v)*(eta0-m0), sqrt(4*v/(4+v)))`.

It is exactly equivalent to independent `m~Normal(m0,2)`, `u0~Normal(0,s)`,
`eta0=m+u0`. The conditional residual is standard Normal. Subsequent states use
innovation coordinates except category 8, whose informative persistent series
benefited from centered states in initial-window probes. Outputs keep category
order 1 through 10. This computational split changes no probability law.

Centered states use a flat base measure plus the complete normalized Gaussian
transition density, yielding the same proper AR prior. Tests compare centered
and non-centered densities with the exact Jacobian and compare native/NUMBA
gradients. Basic slices avoid an object-mode fallback in indexed gradients.

The complementary nature of centered and non-centered coordinates is the basis
in [Papaspiliopoulos, Roberts and Sköld (2007)](https://doi.org/10.1214/088342307000000014)
and the [Stan efficiency guidance](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html).
Those references motivate the computational work, not guaranteed convergence.
The original linear-time stationary-innovation transform has an explicit reverse
derivative including the initial-state dependence on phi. Tests compare it with
a triangular solve, finite differences and stationary covariance simulations.
Its deterministic Numba kernels are cached without fast-math; NUTS uses first
derivatives only. See the
[PyTensor Op documentation](https://pytensor.readthedocs.io/en/latest/extending/creating_an_op.html)
and [Numba caching documentation](https://numba.readthedocs.io/en/stable/developer/caching.html).

For the pooled model, write `a=b0+b`, `c=(T-1)/24`, `q=a+beta*c`. Its exact prior
has `Cov(a)=I+4*11'`, `Var(beta)=0.25`, and independence between a and beta.
Sample `(q,beta)` jointly, evaluate `eta=q+beta*(t/12-c)`, and recover
`b0|a~Normal(m0+4/(1+4K)*sum(a-m0), sqrt(4/(1+4K)))` with an independent residual.
This removes a likelihood ridge without changing any prior. A 150,000-draw test
reconstructs the original joint Gaussian mean and covariance.

## Admission and limits

The gate remains R-hat below 1.01, bulk/tail ESS above 400, zero divergences,
no tree-depth saturation, finite diagnostics and BFMI above 0.3. All monitored
variables are checked. Unresolved warnings reject a fit. The documented missing
BLAS warning is retained as a performance warning. Dense-metric and low-rank
probes were not adopted on speed alone.

Separate accepted posteriors may have different retained draw counts. Forecasts
resample each category's posterior with an independent seeded stream, preserve
whole twelve-month paths, then concatenate categories. Tests cover replay,
category ordering, path confinement and independence of parameter draws.
Accepted fits save prior/posterior checks of training means, variances and zero
fractions. These are in-sample checks, not out-of-sample calibration evidence.
Ensemble and density Monte Carlo checks are separate gates.

Failed and incomplete exploratory runs remain under `artifacts/stage6-completion/`.
Zero divergences alone did not qualify several attempts that failed other
diagnostics. Incomplete experiments have abort records. Exploratory scripts
evolved; production evidence must come from separately source-pinned runs.

Timing separates compilation/data rebinding, the complete sampler call including
warmup, and serialization. This is not the full per-chain timing decomposition
required for Stage 9. Proposed-model recovery, simulation-based calibration and
factor/regime feasibility remain Stage 7 onward.
