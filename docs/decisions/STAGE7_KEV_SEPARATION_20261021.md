# KEV slope review is separate from count-level regularization

2026-10-21. The level amendment does not activate a fitted KEV channel.
Stage 7's auxiliary fixture uses a known coefficient 0.5; neither the prior
count audit nor that conditional fixture validates an estimated elasticity.

The v1 proposal `beta_V ~ Normal(0,1)` truncated to [-1,1] is a substantive
elasticity restriction. Its mathematical role is distinct: bounding beta keeps
`E[exp(beta*eta)]` finite under the bounded conditional-Gaussian variance of eta.
An unbounded Gaussian beta times a Gaussian eta can lose its exponential moment;
moving eta's location or tightening its variance is not a general cure.

Carry this as a pending version-G decision. Before fitting G, specify its
publication/exploitation cohort relationship, exclude unmatched fallback KEV
additions from the count-linked likelihood (or provide the separately specified
intercept-only component), and declare slope sensitivity. A finite wider bound
is mathematically possible but needs predictive plausibility checks on KEV rates;
it is not automatically admitted by count-only checks. Inspect posterior mass
near the selected slope bounds and report the dependence on those restrictions.
Do not claim the current [-1,1] values are recommended by a source or identified
by Stage 7. Preserve the unmatched additions and their provenance in the panel.
