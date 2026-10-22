# Conditional noise-separation experiment

All six replacement fits pass the unchanged numerical gates. The exact iid
conditional coordinates retain the prior and negative-binomial likelihood;
the [coordinate decision](../../decisions/STAGE7_IID_COORDINATES_20261022.md)
and [verification audit](../../checks/stage7-iid-coordinates-20261022/audit.json)
record that change. Original unsuccessful attempts remain preserved.

These fixtures condition on the true category level and isolate iid latent
noise from negative-binomial dispersion. They give separation a favorable
setting; they do not establish identification in the full model.

Across the three dense seeds, posterior correlation between tau and log(psi)
is 0.749–0.812. It is positive: increasing psi reduces observation variance,
allowing greater latent variance. Tau posterior SD contracts to 0.208–0.360
of its marginal prior SD; log(psi) contracts to 0.537–0.699. The true values
(tau=0.35, psi=12) lie in their respective central 90% intervals in all six
fixtures, but psi remains broad: dense intervals range from [5.05, 27.17]
to [6.85, 66.27]. These are descriptive three-seed results, not a coverage
rate estimate or simulation-based calibration.

Sparse fits are less informative. Tau SD ratios are 0.453–0.589, and log(psi)
ratios are 0.816–0.905. Sparse log-rate interval coverage ranges from 0.724
to 0.953 across seeds, despite the numerical gates passing. No sampled tau
is in the recorded upper-support-edge region. The combined relative
overdispersion exp(tau²)*(1+1/psi)-1 has its generating value inside the
90% interval in all six cases; sparse intervals are substantially wider.

Consequently, the implementation can sample this isolated model, but these
experiments do not justify freely interpreting both noise components in the
production model. Keep iid noise optional, retain the parsimonious no-iid
pilot, and require the predeclared development comparison and sensitivity
checks before admitting it. Do not change the prior or score held-out data
to resolve the ambiguity. Positive-lag covariance is zero by construction
here and is not evidence of successful covariance learning.

Machine-readable results and their hashes are in [summary](summary/audit.json)
and [analysis](analysis/details.json). Training posterior predictive checks
reuse fitted observations; synthetic future draws are separately labelled.
