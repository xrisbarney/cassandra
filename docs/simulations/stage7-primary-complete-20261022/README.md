# Completed primary synthetic campaign

All 43 primary scenario/seed cases have terminal results. The
[summary audit](summary/audit.json) binds the explicit routes, immutable results
and recovery exclusions. There are 28 numerical passes and 25 recovery-eligible
fits. Fifteen cases fail their diagnostic schedules; three numerical passes
are excluded for unresolved exploration. The 12 sensitivity cases are a
separate campaign and were not all complete when this primary summary was made.
The exact [routing snapshot](../stage7-routing-snapshots/a23cfc9d55c2dfafc862f5047b5bb07191b816e6764d1c82bb9375f8b7460218.json)
is preserved alongside its earlier versions; its hash matches the summary audit.

| Scenario | Cases | Numerical passes | Recovery eligible |
|---|---:|---:|---:|
| Independent counts | 3 | 3 | 3 |
| Latent AR | 3 | 2 | 2 |
| One factor, sparse category | 3 | 3 | 3 |
| Severity and missing channels | 3 | 3 | 3 |
| Regimes | 3 | 2 | 2 |
| Effort proxy | 3 | 3 | 1 |
| Oracle effort | 3 | 2 | 2 |
| No effort adjustment | 3 | 3 | 3 |
| Kernel present | 3 | 1 | 0 |
| Kernel absent | 3 | 0 | 0 |
| Dense conditional noise separation | 3 | 3 | 3 |
| Sparse conditional noise separation | 3 | 3 | 3 |
| Misspecified observations | 3 | 0 | 0 |
| Representative two-factor fixture | 1 | 0 | 0 |
| Independent fallback level | 3 | 0 | 0 |
| Total | 43 | 28 | 25 |

The three effort configurations reuse the same observations within each seed.
These are fixed-truth experiments, not 43 independent datasets or SBC. All
failures remain in the denominators. The excluded numerical passes are effort
proxy seeds 7101 and 7103 and kernel-present seed 7101, with evidence linked
from the summary audit. The [analysis](analysis/details.json) uses only eligible
fits and preserves covariance and prior-contraction results.

For the three one-factor sparse fixtures, log-intensity RMSE is 0.232--0.277
and pointwise 90% coverage is 0.893--0.951. Zero-lag covariance-entry coverage
is 0.875--1.000. Time cells and covariance entries are dependent; these are
descriptions of three cases, not general calibration estimates. Severity
location RMSE is 0.039--0.047 on its normalized scale and 90% coverage is
0.813--0.885. Its synthetic auxiliary coefficients are fixed at known values.

The [reviewed effort comparison](../stage7-effort-reviewed-20261022/README.md)
has only one seed with all three configurations eligible. It supports neither
identification of physical reporting effort from counts nor a calibrated
malicious-activity interpretation. Regime results are retrospective recovery
of fixed synthetic paths. Conditional noise-separation findings require the
[fixed-level qualification](../stage7-noise-20261022/README.md).

There is no validated kernel recovery or misspecification-robustness result.
The ten-category, two-factor representative fixture and all three fallback
fixtures fail, so this campaign does not admit the prescribed two-factor
production pilot. Completed experiments and mathematically verified code do
not substitute for a converged posterior. No real-data proposed-model fit or
January 2024--December 2025 score is produced by this stage.

The [fixed-seed figures](../stage7-figures-20261022/README.md) retain the kernel
exclusion. Their four result hashes and eligibility decisions were checked
against this completed primary campaign.
