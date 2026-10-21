# Proposed state-space prior amendment: stationary amplitude and finite moments

Prepared 2026-10-21 at the user's request: **prepare the prior amendment first
and stop unfinished old-prior runs**. Status: **draft for review, not active**.
No revised posterior fit has run. The inference and forecast implementations
still contain the old laws; this document must not be mistaken for an applied
change or a declaration that Stage 7 passed.

Parent: Stage 2 v1.0.1, SHA-256
`7597b76276b837f8db9bc2baff63cb035e14cdfa8656885c782795d38cb598e7`.
The [machine-readable proposal](STATE_SPACE_PRIORS_20261021.proposal.json)
specifies this candidate; the companion hash record seals the review artifacts
without activating them. This is a substantive prior revision, separate from
the exact centered-coordinate and mass-matrix changes already documented.

## 1. Evidence and stopped work

The [original audit and derivation](../STAGE7_PRIOR_FINDING.md) show that proper
frozen priors can nevertheless give an infinite prior mean publication rate.
The independent level-scale and factor-loading calculations each establish
this failure. A finite simulation also encountered count-generator range
failures. These facts do not establish posterior impropriety or infinite
posterior moments, nor prove that they caused every sampling failure.

All old-prior coordinators and their verified child fits were stopped. The
[interruption audit](../simulations/stage7-20261021-interrupted/audit.json)
records all 40 planned cases: 13 completed (6 accepted, 7 rejected by diagnostics),
5 stopped by user, and 22 not started. Completed result hashes were checked;
unfinished directories retain their data, truth, run records and completed
attempts, with a separate interruption record. A user-stopped case is not a
new convergence failure. Original failed fits and earlier geometry comparisons
remain on disk. None of these cases uses the real held-out panel.

## 2. Proposed probability laws

Normal arguments below are mean and **standard deviation**. `THN(s,U)` denotes
HalfNormal(s) conditioned on `0 < x < U`. `TN(0,s,-U,U)` is a normalized
truncated Normal. These are prior support restrictions, not clipping of latent
states, forecasts, or bad draws. Finite positive upper bounds express substantive
beliefs about plausible log-scale variation and require sensitivity analysis.

**Unit stationary factors.** For each intensity factor and severity factor:

```
phi ~ Uniform(-1,1)
f[0] ~ Normal(0,1)
f[t] | f[t-1],phi ~ Normal(phi*f[t-1], sqrt(1-phi**2))
```

Thus `Var(f[t] | phi)=1` and `Cov(f[t],f[t+d] | phi)=phi**abs(d)`.
Persistence no longer inflates marginal amplitude. The innovation covariance
becomes `diag(1-phi**2)`; it is not the old fixed identity. Loadings now measure
stationary log-state amplitude. The same equations apply to severity factors.
Stationary initialization, all transition normalizers and any transformation
Jacobians must be included in the implementation. Near-unit persistence can
still be computationally difficult; this change does not guarantee good NUTS
geometry or impose a new cutoff on phi.

| Component | Proposed law | Reason and change |
|---|---|---|
| Intensity levels | `m_bar ~ N(0,3)`; `s_m ~ THN(2,3)`; `m_k ~ N(m_bar,s_m)` | Retain the global location/scale; bound the random conditional SD. |
| Severity levels | `n_bar ~ N(1,1)`; `s_n ~ THN(1,2)`; `n_k ~ N(n_bar,s_n)` | Retain the bounded CVSS interpretation; regularize logit-scale heterogeneity. |
| Intensity loadings, rank r | Free entries `TN(0,1/sqrt(r),-2/sqrt(r),2/sqrt(r))`; positive leading diagonal uses the corresponding half law | Each row has squared norm at most 4; rank increases do not increase that upper variance budget. |
| Severity loadings, rank r_h | Same law with r_h | Same stationary-amplitude interpretation. |
| Optional iid count SD | `tau_k ~ THN(1,1)` | Remove the artificial .01 floor; bound iid log-rate variance by 1. NB dispersion remains separate. |
| Standalone latent-AR synthetic fixture | `s_k ~ THN(1,1)` on stationary SD; innovation SD `s_k*sqrt(1-phi_k**2)` | Covers the old Stage 7 latent-AR fixture; does not replace any Stage 6 control. |
| Kernel amplitude | `s_a ~ HalfNormal(.5)`; `a_k | s_a ~ TN(0,s_a,-1,1)`; first loading positive | Conditional truncation bounds each amplitude even though s_a remains unbounded. Include the s_a-dependent normalization. |
| KEV slope, version G only | `beta_V ~ TN(0,1,-1,1)` | An unbounded Normal slope multiplying a Gaussian log intensity reintroduces a product tail before exponentiation. |

Keep the lower-triangular zero pattern, positive diagonal and previously fixed
category ordering. Keep the causal kernel, unit squared-weight normalization,
width prior and support. Keep the regime transition prior, ordered increments
`Delta ~ HalfNormal(1)`, severity shifts and initial uniform regime law. Keep
NB dispersion `log(psi) ~ N(log(10),1)`. Channel intercepts, EPSS slopes/version
effects, observation SDs and incident coefficients retain their frozen laws.
No category gets a deterministic intensity floor or a Bayesian zero point mass.

The bounds are explicit design choices, **not numerical constants recommended
by a paper**. The factor row bound allows a stationary log SD as large as 2
(one-SD multiplier exp(2), approximately 7.39); optional iid and kernel each
allow a log SD up to 1. The level SD bound 3 allows substantial heterogeneity;
unbounded Gaussian category levels still allow arbitrarily high or low rates.
The loading SD divided by sqrt(rank) makes the prior allocation across ranks
more comparable, but the triangular anchor rows remain asymmetric.

The KEV bound is especially substantive: the signed elasticity of its mean
with respect to the reference publication rate lies between -1 and 1. This
does not resolve its cohort mismatch or authorize use of category 10 in that
likelihood. Masking that category or giving its 21 unmatched additions an
intercept-only component remains a separate Stage 8 decision. If the elasticity
restriction is not supported, widen the finite bound in a separately recorded
candidate before fitting G; do not silently revert to an unbounded slope.

## 3. What the amendment establishes mathematically

At a fixed month, conditional on scales and regime increments, a category's
Gaussian intensity terms have total variance bounded by

`V_max = 3**2 + 3**2 + 4 + 1 + 1 = 24`.

The terms are respectively global level, category-level variation, factor row,
iid residual, and kernel. Disabled terms disappear. Factor and kernel
innovations are independent in the model. For finite rank and finite R,
`c_z <= C = sum_{j=2}^R Delta_j`. HalfNormal increments have finite MGFs at
every finite argument. For every real p and any fixed finite effort e,

`E[exp(p*(e+eta))] <= exp(p*e + p**2*V_max/2) * E[exp(abs(p)*C)] < infinity`.

This holds uniformly over stationary phi and kernel widths: their marginal
variances are fixed at one. It proves finite positive moments of the reference
rate and the conditional observed rate. It also proves finite moments of a
finite sum across categories/months; independence of those cells is not needed.
Lognormal NB dispersion has every positive and inverse power moment, so the
NB2 factorial-moment identity gives finite integer predictive count moments.
This is an analytic statement; a successful finite Monte Carlo run alone would
not establish it.

The new KEV slope bound makes `exp(p*beta_V*eta)` dominated by
`exp(abs(p)*abs(eta))`. Combining positive/negative eta MGFs and the Normal
intercept proves finite KEV rate moments. Under the old slope, the retained
global level `m_bar ~ N(0,3)` alone yields
`E[exp(beta_V*m_bar) | beta_V] = exp(9*beta_V**2/2)`, whose integral under
`beta_V ~ N(0,1)` diverges. Fixing only the factor prior would miss this channel.

EPSS and default severity observations are bounded, and default severity
location is between zero and one. Thus CVSS-weighted publication rates inherit
the count-rate moment bound. Incident rates with bounded M, finite sectors,
and the frozen independent lognormal/Gaussian coefficients also have finite
positive moments. These arguments cover the specified default channels; they
do not validate their conditional-independence assumption or empirical fit.

The manuscript's alternative unbounded lognormal severity and unspecified
economic calibration priors are outside this guarantee. The count-derived
effort proxy is conditioned on, not jointly generated: the bound is conditional
on its finite fitted values, not a proof about a joint law for e(N) and N.
All-moment prior existence does not prove joint multichannel posterior propriety,
accurate tail Monte Carlo estimates, economic calibration, or convergence.

## 4. Draft checks and interpretation

`scripts/check_stage7_prior_proposal.py` reads only the proposal and generates
20,000 stationary parameter/state draws per case. Cases cover four categories
with one factor, the ten-category/two-factor pilot, and ten-category/three-factor
extensions with iid noise, kernel, three regimes and effort offsets 0 and 3.
For regime extensions it uses the highest ordered shift as a deliberate upper
envelope, not a sample from the Markov regime occupancy distribution. Unit
stationary marginals let this check integrate out persistence and width; it
does not test their temporal dependence. It separately draws the KEV law.

The [audit](../checks/stage7-prior-proposal-20261021/audit-final.json) reports
all numerical failures and untrimmed quantiles, including log means above 20.
No large draw is clipped or replaced. Quadrature checks the level moments and
an independently available closed-form loading moment. Prior spread can remain
very broad despite all moments existing; numerical feasibility is only one
check. See the audit note alongside that file for the measured values.

The proposal was specified before these simulations. No scale was selected
by held-out or development predictive scores. The frozen-prior diagnostic
failures motivated the revision, but the amendment cannot be presented as
preregistered before Stage 7 evidence existed.

## 5. Activation and validation sequence

1. Record activation and a new model ID/supplemental freeze before revised
   posterior sampling. The original five-file freeze remains unchanged.
2. Apply the same revised laws to fitting, prior simulation, forecasting and
   covariance/recovery calculations. Revise stationary initial/transition
   densities and forecast innovations together; compile once where possible.
   Audit bounded-prior normalizers and gradients against independent references.
3. Retain the existing synthetic observations and seeds for paired recovery.
   Translate old Q=I generating loadings to stationary loadings as
   `Gamma_new[:,j]=Gamma_old[:,j]/sqrt(1-phi_j**2)`; similarly for severity.
   Transform the latent factor truth inversely. Record a new truth-coordinate
   artifact and leave old truth/data files unchanged. Conditional noise/AR
   fixtures need their corresponding stationary-scale translation. If a fixed
   truth lies outside proposed support, report that fact; do not alter it to
   produce an easier recovery result.
4. Run the independent/AR/factor fixtures first, then the remaining predeclared
   fixed-truth scenarios. Keep all four-chain budgets and diagnostic gates.
   Failed or interrupted cases do not support recovery claims. Still assess
   tau/psi separation and kernel identification; finite moments resolve neither.
5. Add explicitly labelled prior-generated calibration work before claiming
   Bayesian calibration; the fixed-truth cases are not SBC. Freeze its replicate
   count, seed list, estimands and diagnostics before sampling. Stage 7 remains
   incomplete until the required validation and decisions are recorded.
6. Before production selection, predeclare sensitivity runs increasing finite
   amplitude/scale caps by 50%, and report posterior mass near support limits.
   Use development data only. Broad alternative priors must meet the same
   analytic moment and computational checks; no tuning against 2024-2025.

Severity measurement SDs near zero, optional iid/NB confounding and near-unit
AR persistence can still cause sampling problems. Diagnose them separately
instead of assuming this amendment cures every failed run. Cost estimates
must separate compilation/startup, warmup and retained sampling.

## 6. Literature basis and unchanged scope

[Gelman et al., Bayesian Workflow](https://arxiv.org/abs/2011.01808) supports
iterative construction, criticism and computational validation of Bayesian
models. [Stan's prior-predictive guidance](https://mc-stan.org/docs/2_36/stan-users-guide/posterior-predictive-checks.html)
and [PyMC's prior-predictive example](https://www.pymc.io/projects/docs/en/v5.15.1/learn/core_notebooks/posterior_predictive.html)
explain checking the observable implications of priors, including extreme
predictions under nonlinear links. These support the workflow; the specific
caps, rank scaling, moment proof and cyber model here are our design and
derivation, not borrowed empirical conclusions. Sources checked 2026-10-21.

This proposal leaves Stage 6 controls/results, all eight named comparators,
topic definitions, scoring rules, convergence thresholds, origin schedules,
and the January 2024-December 2025 holdout unchanged. No real holdout score
has been computed, and no inference from the earlier failed evaluation is
being reused as a result.
