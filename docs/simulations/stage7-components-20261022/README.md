# Completed component subset

This interim snapshot covers 33 primary cases, not the complete 43-case
campaign or its 12 sensitivity cases. It has 25 numerical passes and 24 fits
eligible for recovery interpretation. The kernel-present seed 7101 numerical
pass remains excluded because of documented unresolved modes. Failed cases
remain in the denominator; the [audit](summary/audit.json) binds each route,
result, observed dataset and exclusion evidence.

The one-factor sparse fixtures pass in all three seeds. Their pointwise 90%
log-rate interval coverage is 0.893–0.951, with RMSE 0.232–0.277. Zero-lag
covariance interval coverage is 0.875–1.000 across the 4-by-4 covariance
entries; those entries and time cells are dependent. Long-lag covariance is
less precisely constrained. The group-level scale has 0.165–0.233 posterior
probability in its upper 5% support region, so the declared scale-sensitivity
comparisons matter. This deliberately sparse known category is distinct
from the separately modelled fallback-zero fixture still being fitted.

All three severity/missingness fixtures pass numerically. Their severity
location's 90% interval coverage is 0.813–0.885, and RMSE is 0.039–0.047 on
the normalized severity scale. The exposure and auxiliary coefficients
are conditioned on their synthetic generating values: this validates neither
real auxiliary-channel eligibility nor estimation of those coefficients.

Two of three regime fits pass. Both recover the prescribed regime path's
classification in this fixed-path stress case. This is retrospective inference
with an obvious synthetic shift, not an out-of-sample alarm experiment. The
failed seed remains a computational limitation. Likewise the standalone latent
AR model passes two of three seeds, and independent counts pass all three.

No kernel case supplies validated marginal posterior recovery: all absent
cases and two present cases fail, and the remaining present case has the
unresolved-mode exclusion. There is consequently no verified width-recovery
or absent-kernel-shrinkage claim. Keep that optional extension unvalidated.

The [noise-separation review](../stage7-noise-20261022/README.md) details all six
admitted conditional noise experiments. Effort conclusions remain pending
the documented proxy sign-coordinate refits; the underlying physical
nonidentification of count intensity and reporting effort is unchanged.

These are fixed-truth descriptive results. The [covariance and contraction
analysis](analysis/details.json) does not establish posterior calibration,
and deterministic zero covariance in independent/no-iid fixtures is not
successful covariance learning. Final full-campaign reporting will supersede
this subset without altering its saved files.
