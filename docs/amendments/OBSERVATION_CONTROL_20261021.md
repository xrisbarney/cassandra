# Replace the unsuccessful latent independent control with a conditional NB autoregression

Decision dated 2026-10-21 under the user's instruction to finish Stage 6 and
resolve its independent Bayesian control using best practices or the literature.
No independent-control forecast scores or held-out scores have been computed.
This changes a probability law and therefore supplements, rather than rewrites,
the Stage 2 freeze. The eight requested comparison systems are unchanged.

## Evidence and decision

The original latent AR(1) control and its stationary-scale-prior revision did
not pass the frozen diagnostic gates. With the revised prior, category 9 at
the December 2018 origin exhausted all three budgets. Its final attempt had
zero divergences and R-hat 1.00889, but bulk/tail ESS 399.15/318.89, below 400.
An earlier attempt also had poor energy mixing and differing exploration of
latent variation and NB dispersion. This is evidence of computational failure
under the allotted budgets, not proof that the model is mathematically invalid.
The partial rolling run is aborted and preserved, without scoring or admission.
The prior amendment and unsuccessful posteriors remain historical artifacts.

Use an observation-driven NB autoregression as Stage 6's simple independent
count control. It avoids a second stochastic source of variation at every month
while retaining autoregression, overdispersion and Bayesian parameter uncertainty.
It is not a reparameterization or a successful implementation of the latent model.
Use the distinct identifier `independent_nb_glar`; do not relabel the failed run.

## Complete probability law

For each category independently, let `c=log(initial-window mean count+0.5)`.
The initial window is fixed at January 2010 through December 2018.

- `m ~ Normal(c,2)`, where 2 is the standard deviation;
- `phi ~ Uniform(-1,1)`;
- `log(psi) ~ Normal(0,1)`;
- `log(mu[t]) = (1-phi)*m + phi*log(1+N[t-1])`;
- `N[t] | N[t-1],m,phi,psi ~ NB2(mu[t],psi)`, with variance `mu+mu^2/psi`.

Condition on the first observed count, January 2010, and use the product of
conditional masses from February 2010 to the origin. Do not impute a fictitious
pre-2010 count or fit a stationary initial distribution. `m` is a reference-level
parameter, not the unconditional mean or an exact deterministic equilibrium;
the addition of 1 matters at small counts. Priors are proper and independent
between categories. The first count is a conditioning value; all later observed
zeros retain their NB likelihood. Category 10 has no point-mass exception.

This is the lag-one, no-lagged-mean special case of the log-linear count model
in equation (3), with NB law (5), in
[Liboschik, Fokianos and Fried (2017), tscount](https://www.jstatsoft.org/article/view/v082i05).
Our intercept is parameterized as `(1-phi)*m`. That paper uses likelihood-based
estimation, not these priors. Bayesian inference for observation-driven NB
autoregressions also appears in
[Andrade, Ehlers and Andrade, Bayesian GARMA Models for Count Data](https://arxiv.org/abs/1509.08666).
Neither source supplies this exact prior, application or comparison result.
The reference-level parameterization and numerical prior scales are explicit
project choices, retained where possible from the earlier control.

## Inference, forecasts and checks

Use the same four-chain NUTS budgets, retries and diagnostic thresholds as the
original protocol. The joint posterior factorizes over categories; a vectorized
fit introduces no shared parameter or cross-category dependence. Every origin
uses a fresh fit and training prefix. Retain all attempt diagnostics and posterior
draws. No relaxation is allowed because the model has fewer parameters.

For each category draw a posterior parameter tuple independently, hold it fixed
along a path, and simulate twelve NB counts recursively starting from the last
observed training count. Future observed counts never enter this recursion.
The predictive PMF at horizon h is the mixture of the NB conditional masses
over posterior draws and the simulated preceding counts. Evaluate it in the
log domain, without moment matching, clipping, winsorization or a probability
floor. Preserve matched paths when aggregating categories and months.

Prior/posterior predictive checks recursively replicate the training period
conditional on its first count. They are not out-of-sample calibration checks.
Before admission, check the conditional likelihood independently, native/compiled
gradients, recursive forecasts and their marginal PMFs, posterior independence,
all-zero behavior, and ensemble-size sensitivity under the existing rule.

The replacement applies only to this initial control. Proposed-model synthetic
recovery and full factor/regime inference remain later stages. The 2024-2025
holdout, target masks, scoring rules, named baseline suite, comparison family
and all other probability laws are unchanged. This amendment supersedes the
stationary-scale revision as the Stage 6 primary independent-control choice;
that revision remains executable only as an explicitly named diagnostic variant.
