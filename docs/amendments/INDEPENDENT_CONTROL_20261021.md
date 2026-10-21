# Independent NB AR(1) control: stationary-scale prior

Decision dated 2026-10-21, under the user's instruction to finish Stage 6 and
resolve the independent Bayesian control using best practices or the literature.
This is a substantive development amendment, not a claim that the original
freeze already specified this prior. No independent-control development scores
or held-out scores have been computed. Other controls' development scores are
available but were not used to choose this prior.

## Reason

The frozen independent control specified `phi ~ Uniform(-1,1)` independently of
`tau ~ HalfNormal(0.3)`, with stationary initial variance `tau²/(1-phi²)`.
This permits enormous latent-state variance near unit persistence. A seeded
initial-window prior simulation produced log count rates above 160, outside
the integer count simulator's range. More generally, for any fixed positive
tau the mixture over phi has infinite `E[exp(u)]`: the exponential of
`tau²/[2(1-phi²)]` is not integrable at the endpoints. This does not prove that
the posterior is improper or that its moments are infinite. It shows that the
prior predictive count scale is unsuitable for this control.

Pure coordinate changes preserve that prior and cannot fix its prior-predictive
implications. Failed computational probes remain historical evidence; their
posterior draws are not admitted as forecasts.

## Revised law

For each category independently:

- `m ~ Normal(log(initial-window mean count + 0.5), 2)`;
- `phi ~ Uniform(-1,1)`;
- `s ~ HalfNormal(0.75)`, independently of m and phi;
- `tau = s * sqrt(1-phi²)`;
- `u[0] ~ Normal(0,s)` and `u[t] ~ Normal(phi*u[t-1],tau)`;
- `log(psi) ~ Normal(0,1)`;
- `N[t] ~ NegativeBinomial2(exp(m+u[t]), psi)`.

Here Normal's second argument is its standard deviation. The transition remains
stationary, with `Cov(u[t],u[t+d]) = s²*phi^|d|`. Persistence no longer determines
the prior marginal amplitude. All-zero counts retain their informative NB
likelihood; there is no Bayesian point-mass or artificial intensity floor.

The numerical scale 0.75 is our explicit weakly informative design choice, not
a value established by the literature. Its 95th percentile for the stationary
log-rate SD is about 1.47 (a one-SD multiplicative factor of about 4.35). Together
with the retained broad mean and dispersion priors it allows substantial count
variation. The amended prior predictive mean is finite:
`E[lambda] = exp(m0+2)/sqrt(1-0.75²)`. This statement concerns the first moment;
it does not assert that every prior-predictive moment is finite.

The approach follows prior-predictive model criticism in
[Gelman et al., Bayesian Workflow](https://arxiv.org/abs/2011.01808) and the
[Stan prior-predictive guidance](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html).
These sources support the workflow, not this exact cyber-count model or scale.
Exact centered/non-centered transformations follow the complementary
parameterization principle of
[Papaspiliopoulos, Roberts and Sköld (2007)](https://doi.org/10.1214/088342307000000014).

## Execution and scope

Sample category posteriors independently, each with four chains and the frozen
three-budget retry sequence. Reuse compiled numerical code within an origin;
start fresh chains and adaptation for every category and every origin. The
posterior factorizes exactly, so this does not introduce a posterior
approximation. Admit an origin only when every category passes all existing
diagnostic gates. Preserve failed attempts and vary no prior by origin.

This amendment changes only the independent NB AR(1) initial control's scale
prior. It leaves the proposed factor/regime model, pooled control, eight named
comparators, target masks, selection grids, score definitions, sampling budgets,
diagnostic thresholds and 2024-2025 held-out period unchanged. The original
Stage 2 freeze stays immutable. The companion JSON freezes this supplement;
new independent-control runs must record its hash and use the revised prior
for fitting and prior-predictive checks alike.
