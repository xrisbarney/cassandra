# Stage 6 Bayesian control validation record

This working record documents numerical implementation choices for the frozen
independent NB AR(1) and pooled NB controls. It does not amend their priors,
likelihoods, origin schedule, sampling budgets or diagnostic thresholds.

## Equivalent parameterizations

For the pooled model, let `a[k] = b0+b[k]`, `c = (T-1)/24` and
`q[k] = a[k]+beta*c`. Under the original independent Gaussian priors,
`Cov(a) = I + 4*11'`, `Var(beta)=0.25`, and `a` and `beta` are independent.
The code samples `(q,beta)` from its exact multivariate Gaussian prior, evaluates
`eta[k,t]=q[k]+beta*(t/12-c)`, and recovers `b0` using its exact conditional law:

`b0 | a ~ Normal(m0 + 4/(1+4K)*sum(a-m0), 4/(1+4K))`.

The conditional residual is a separate standard-Normal draw, and `b[k]=a[k]-b0`.
This removes a likelihood ridge without narrowing the prior. A test reconstructs
the complete original joint Gaussian mean/covariance from 150,000 prior draws.

For the independent control, sample independent standard-Normal innovations
`epsilon[k,t]` and use

`v[k,0]=epsilon[k,0]/sqrt(1-phi[k]^2)`;
`v[k,t]=phi[k]*v[k,t-1]+epsilon[k,t]`;
`eta[k,t]=m[k]+tau[k]*v[k,t]`.

This is the same stationary AR(1) law as the initial `pm.AR` representation.
The linear-time transform has an explicit reverse derivative, including the
dependence of the stationary initial state on `phi`. Tests compare values with
a triangular linear-system solution, compare every derivative with central
finite differences (including phi=0 and negative phi), and verify the stationary
covariance by simulation. No second derivative is provided or used by NUTS.

The deterministic kernels use the existing pinned Numba dependency with caching
and without fast-math. The PyTensor wrapper supplies the Op's value, shape and
first derivative interfaces; it does not replace the NB likelihood or NUTS.
See the [PyTensor Op documentation](https://pytensor.readthedocs.io/en/latest/extending/creating_an_op.html)
and [Numba caching documentation](https://numba.readthedocs.io/en/stable/developer/caching.html).
Implementation follows the installed PyTensor 2.38 API (`grad`); newer online
documentation also describes `pullback` APIs.

## Checks and execution

### Independent-control geometry investigation (2026-10-21)

The first non-centered four-chain probe exhausted the frozen retry budgets with
remaining divergences. Its final attempt had six divergences despite a maximum
R-hat near 1.006; those draws are rejected. A separate dense-metric probe on the
most persistent populated category also failed chain agreement. Neither a small
R-hat nor zero divergences alone establishes acceptance. Failed probe artifacts
are retained under `artifacts/stage6-completion/` and are not forecast results.

The methodological basis for the follow-up is that centered and non-centered
coordinates are complementary, with their performance depending on the
information in the observations. See
[Papaspiliopoulos, Roberts and Sköld (2007)](https://doi.org/10.1214/088342307000000014)
and the [Stan efficiency guidance](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html).
The latter also distinguishes constant posterior correlation, which a mass
matrix can address, from varying curvature, which may require different
coordinates. PyMC documents diagonal and dense adaptation in
[`init_nuts`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.init_nuts.html);
the installed implementation, rather than an unpinned online API, determines
the actual calls. These references motivate computational investigations; they
do not guarantee that this particular control will pass its diagnostic gates.

For exact initial-state conditioning, let `m0=log(initial_mean+0.5)` and
`s²=tau²/(1-phi²)`. The original law is `m~N(m0,4)`, independently
`u0~N(0,s²)`, with `eta0=m+u0`. Equivalently,

`eta0 ~ N(m0,4+s²)`;

`m | eta0,tau,phi ~ N(m0+4/(4+s²)*(eta0-m0), 4*s²/(4+s²))`.

Sample the conditional residual as a standard Normal and propagate subsequent
states with the original independent standard-Normal innovations. This centers
the initial observable log intensity while leaving subsequent innovations
non-centered. No stationary tail, coefficient prior, likelihood, or forecast
law changes. Analytic covariance tests reconstruct the independent original
`(m,u0)` prior for positive, negative and near-unit persistence; native and
Numba gradients are compared at perturbed coordinates. This representation is
being probed before production admission.

The independent control's joint posterior factorizes by category, because all
its priors and observations do. Separate category fits therefore target the
same posterior law, and permit independent step-size/mass adaptation. They do
not permit accepting a panel when any category fails. The implementation test
compares the joint log density with the sum of single-category log densities.
This factorization would not be valid for the pooled or proposed factor model.

Additional initial-window probes show why a single centered/non-centered choice
for the entire panel is insufficient. Initial-state conditioning with innovation
coordinates passes the all-zero category's gate (four chains, no divergences,
maximum R-hat 1.0067, minimum bulk/tail ESS 2038/1118). Centering subsequent states
passes the strongly persistent category 8's gate (R-hat 1.0070, ESS 644/1464,
zero divergences). But centering the first eight or nine categories jointly
fails: several innovation scales mix poorly. These joint attempts are rejected,
including the zero-divergence nutpie attempt whose BFMI and ESS fail.

The next exact representation centers only category 8 and keeps all other
categories in innovation coordinates, with initial-state conditioning for all.
The split uses initial-window posterior geometry only. It must still pass the
full-panel and rolling-origin gates. Centered transition states are represented
by a flat base measure plus the complete normalized Gaussian transition log
density; this is the original proper AR prior, not a replacement flat state
prior. Tests compare centered and non-centered log densities with the exact
`(T-1)*sum(log(tau))` Jacobian and compare native/Numba derivatives. Basic slices
and concatenation preserve the original category order without an object-mode
Numba fallback.

The backend investigation includes nutpie's diagonal and low-rank covariance
adaptations, at the same four-chain 1000/1000 initial budget. The low-rank probe
uses eigenvalue cutoff 3 and default regularization 1e-5; it is experimental in
the installed nutpie 0.15.2 API. The current
[official sampling guide](https://pymc-devs.github.io/nutpie/sampling-options.html)
describes its purpose and a cutoff-3 example, but uses a newer argument name.
Code follows the installed version's `low_rank_modified_mass_matrix` argument.
No backend or coordinate variant is accepted on speed alone, and no forecasting
score is consulted for these choices. Exploratory scripts changed during this
investigation; production evidence must come from a separately source-pinned run.

Every production fit retains four chains and the frozen warmup/draw/retry
budgets. Persistent outer workers run independent origins; each worker executes
the four chains sequentially to avoid repeatedly launching Windows sampling
workers. Each origin builds a fresh model, initializes fresh chains and adapts
from scratch. Compiled numerical kernels can be cached; fitted states and
posterior samples are never carried between origins.
Chains receive distinct seeded jittered initial points in unconstrained
coordinates. PyMC's NUTS uses the PyTensor NUMBA backend for the full log density
and gradient; tests compare both values with the native backend at perturbed
parameter points. No VI approximation is introduced.

A separate execution probe evaluates nutpie 0.15.2 with its compiled Rust NUTS
sampler and the same PyTensor/Numba density. It is not yet the production backend.
Its Windows CPython 3.12 wheel is pinned by version and SHA-256 in
`configs/nuts-runtime.lock`. Install this optional probe dependency after the
base environment with `uv pip install --python .venv/Scripts/python.exe
--no-deps --require-hashes -r configs/nuts-runtime.lock`. The base `uv.lock` and
its installed package versions remain unchanged. New run provenance includes
the supplemental lock hash and the installed-package inventory.

The probe uses four chains, the initial frozen 1,000/1,000 budget and target
acceptance 0.95, diagonal gradient-based mass adaptation and maximum depth 12.
It does not choose priors or compare forecasting scores. Its native
`maxdepth_reached` diagnostic is checked explicitly, alongside the PyMC
`tree_depth` convention. See the installed API and the
[official nutpie PyMC documentation](https://pymc-devs.github.io/nutpie/pymc-usage.html).

Acceptance checks include R-hat below 1.01, bulk and tail ESS above 400, zero
divergences, no tree-depth saturation, finite diagnostics and BFMI above 0.3.
All monitored variables are checked, including the latent states and innovations.
Unknown warnings remain a failure. The specific, already-diagnosed warning about
missing BLAS acceleration is retained as an environment/performance warning;
it is not a numerical-convergence failure. The missing library has not been
silently installed or the locked environment changed.

Accepted fits also save prior and posterior predictive distributions of the
training-panel mean, variance and zero fraction by category, with observed
statistics and explicitly defined posterior predictive tail fractions. These
are in-sample checks, not evidence of out-of-sample calibration. Predictive
ensembles and density evaluations receive separate Monte Carlo checks before
their development scores are admitted.

Full-model simulation-based recovery and calibration remain Stage 7 work. This
record concerns the Stage 6 controls; it does not claim that the factor/regime
model has been validated.
