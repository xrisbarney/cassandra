# Completed matched effort comparison

**Superseded interpretation, 2026-10-22:** the subsequent
[mode audit](../../checks/stage7-proxy-mode-20261022/audit.json) found that the
accepted proxy seed 7102 retry stayed in one of two documented orientations.
The numerical gate counts below remain accurate, but its recovery values must
be read as mode-conditional until the
[working sign anchor](../../decisions/STAGE7_SIGN_GAUGE_20261022.md) is validated.
The data and original summaries are preserved; they are not full-posterior
evidence for that case. The analytic nonidentification argument still applies.

These nine scenario/seed cases reuse three observed count panels, one per seed.
They are not nine independent datasets. The explicit routes were recorded before
final recovery analysis. `summary/audit.json` records the source, data and result
hashes; `analysis/details.json` binds the analysis to that summary. Eight fits
are admitted. Oracle seed 7103 exhausted the three-attempt schedule with a
retained divergence and contributes no recovery measurements.

The table gives ranges across admitted seeds, not confidence intervals. Its
coverage is coverage of the generating latent trajectory by pointwise 90%
posterior intervals, **not forecast-interval calibration**. Category-month
cells are dependent, and these are three fixed-truth cases rather than SBC.

| Offset | Admitted / attempted | Operational log-intensity coverage | Generating physical log-intensity coverage | Operational log-intensity RMSE |
|---|---:|---:|---:|---:|
| None | 3 / 3 | .638--.674 | .289--.326 | .341--.406 |
| Count-derived proxy | 3 / 3 | .607--.693 | .227--.266 | .270--.348 |
| Known generating effort (oracle) | 2 / 3 | .813--.906 | .813--.906 | .164--.199 |

The proxy's expected-count RMSE is lower than the no-offset model on each of the
three common seeds (65.20 vs 82.33, 39.53 vs 72.55, and 41.90 vs 56.09). Those
are in-sample latent-mean recovery measurements, not held-out forecast scores.
They do not rescue physical interpretation: the proxy's intervals cover only
about 23--27% of the generating physical log intensities. Report both targets,
not just the more favorable count or operational measure. The oracle's better
recovery on its two admitted seeds does not conceal its third fit's failure.

The same observations also admit an equal shift in true log intensity with no
reporting shift. Counts identify their sum, not the decomposition. The analysis
records both physical interpretations without generating another dataset.
No posterior from these counts alone establishes which interpretation is true.

`analysis/covariance.csv` compares the fitted model's stationary stochastic
log-state component (conditional on each parameter draw and fixed regime) with
the generating component covariance. It excludes level, regime and effort
uncertainty. For the count-derived offset, this is not a claim to identify the
physical covariance conditional on that data-dependent preprocessing, nor is
it the covariance of the fitted latent states conditional on observations.
Contraction summaries compare matching marginal prior and posterior SDs.

All evidence is synthetic. The final whole-campaign audit must still verify
these artifacts and independently recompute accepted-posterior diagnostics.
January 2024--December 2025 remains unscored.
