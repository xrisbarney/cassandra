# Matched count-coordinate recovery comparisons

Declared 2026-10-22 before these additional fits. The conditional count
transformation has passed the complete density/Jacobian, derivative and fused
sampler checks. Effort-proxy seed 7101 was admitted on its first budget. In
contrast, the original oracle-effort and no-effort amplitude fits have failures;
no-effort seed 7103 exhausted its schedule with final BFMI 0.245 despite adequate
R-hat and ESS. Seed 7102 of that original no-effort configuration is still running.
The sparse-factor sensitivity cases also have two failed schedules.

Use the already verified exact count transformation, scalar block solver and
dense metric for the following matched comparisons, with all existing priors,
initialization rules, seeds, budgets and diagnostic gates unchanged:

- all three oracle-effort and no-effort seeds, alongside the three already
  declared conditional effort-proxy fits;
- all three primary sparse-factor seeds and seed 7101 under each of the four
  declared sensitivity priors.

Repeat accepted original cases as well, so comparisons do not select a
representation separately according to a recovery score or sensitivity result.
These are repeated computations on the existing datasets, not new independent
replications. Offsets and observed zero counts are already included in the
verified transformation; the likelihood remains the exact NB model. Numerical
backends are recorded per run and their timing is not a controlled comparison.

Preserve the original results and interrupt the unfinished original no-effort
seed 7102, if still running, before its full replacement schedule. Keep its
status distinct from an exhausted schedule. Every replacement runs in a fresh
directory. Any replacement that exhausts its gates remains a failure; do not
waive BFMI or retain its posterior as recovery evidence. No recovery metric,
held-out observation or forecast score is used to choose these coordinates.
The purpose is to complete the planned comparisons with a verified computational
method that has already admitted the corresponding effort-proxy submodel.
