# Wider-dynamics fallback: known exploration issue

Recorded on 2026-10-22 after retry 1 failed and before retry 2 completed.
This is the `dynamics_wide` candidate on fallback seed 7101, a different posterior
target from the primary prior. Its own failed attempt has maximum R-hat 1.5289,
minimum bulk ESS 7.21, minimum tail ESS 20.41, minimum BFMI 0.752 and zero
retained divergences/depth saturations. The worst R-hat is second persistence.

The [trace](chain_mixing.png) and [region counts](../dynamics_wide_attempts01.json)
show chain 1 staying near unit persistence, while chains 2-4 spend over 99% of
draws below 0.8. Only chain 1 visits the high-persistence regions with positive
or negative factor time mean. These frequencies are not reliable posterior
region weights.

For this candidate's final retry, review its own documented regions: second
persistence below 0.8, or above 0.95 with time mean above 0.5 or below -0.5.
The markers describe this observed failure and supplement the existing
exploration requirement. They are not new priors, significance tests or recovery
thresholds. An apparently converged retry that stays in only one documented
region leaves exploration unresolved. No conclusion relies on assuming the
primary prior and this sensitivity candidate have equal region probabilities.

The [figure audit](figure_audit.json) binds the failed attempt and posterior.
Neither generating truth nor recovery scores were consulted for this review.
