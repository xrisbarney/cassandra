# Check the repair hypotheses before further fitting

The [audit](audit.json) reads the preserved generating loading matrix and failed
fit diagnostic tables. It does not fit or change a model.

The representative fixture's current anchor rows, 1 and 2, already have the
largest absolute loading in their respective columns (0.592157 and 0.464758).
They also maximize the absolute two-row determinant across all 45 pairs
(0.275209). Thus neither columnwise magnitude selection nor maximum-volume
anchor selection changes this fixture. Selecting the two largest row norms
would order the same pair differently, which cannot establish a weak-anchor
cause. Density ordering is not identified as the cause of the failure.

With distinct diagonal factor persistence, an arbitrary factor rotation also
rotates the transition matrix: Q' diag(phi) Q is generally not diagonal. A new
triangular loading constraint after row permutation is therefore not automatically
an equivalent representation of this dynamic model. Its truncated loading priors
also require an explicit transformation if preserving the original prior.
Order dependence of common triangular factor priors is discussed by
[Leung and Drton](https://arxiv.org/abs/1409.7672); that static-factor result does
not by itself identify the mechanism of this dynamic count model's failure.

In the primary fallback seed-7101 final attempt, the fallback log level has
R-hat 1.004482, bulk ESS 830.689 and tail ESS 2200.511. The second factor's
persistence has R-hat 1.272287 and bulk ESS 11.170; known-category levels and
factor contributions also mix poorly. The overall fit still fails. These
descriptive diagnostics refute a fallback-level-only account, while leaving
possible indirect coupling open.

Proper priors and some well-mixed marginals do not certify the joint posterior;
see the [Stan guide on problematic posteriors](https://mc-stan.org/docs/stan-users-guide/label-switching-problematic.html).
Masking the fallback count likelihood would be a diagnostic intervention that
removes the information in its zeros. It would not decide the distinct question
of how unmatched fallback KEV additions should enter an auxiliary likelihood.
