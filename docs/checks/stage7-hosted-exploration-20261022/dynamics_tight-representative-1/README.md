# Tightened dynamics: representative retry 1

Reviewed after the second prescribed attempt and before the final hosted retry
starts. Maximum R-hat is 1.0385, minimum bulk ESS 106.61 and minimum tail ESS
150.10, with no retained divergences or depth hits. This attempt fails.

The [visually inspected trace](chains.png) and [record](review.json) show all
chains visiting lower and high persistence, with unequal frequencies. The
lower-persistence fractions are 12.7%, 1.2%, 23.6% and 15.6%; its indicator
R-hat is 1.057 and bulk ESS 66.03. Both high-persistence factor-mean regions
are visited, but that alone does not establish adequate exploration.

The third chain also has a short segment near negative unit persistence.
A [supplementary descriptive record](negative_boundary.json) binds the same
posterior: there are 0, 0, 44 and 2 draws below -0.95 across the four chains,
and the minimum is -0.99999237. These are observations from an unsuccessful
attempt, not posterior region probabilities or new admission thresholds.
The negative-boundary numerical check is recorded separately.

Review the final retry against this candidate's own evidence. No recovery
estimate is admitted, and the original budgets and statistical gates remain.
