# Reviewed effort experiment

Nine fixed-truth cases reuse three generated count panels, with no offset,
an oracle offset, or a count-derived training-only proxy. This is a synthetic
identification check, not held-out forecast evaluation or SBC.

Eight cases pass the numerical diagnostic gates. Six are eligible for recovery:
all three no-offset fits, oracle seeds 7101 and 7102, and proxy seed 7102.
Oracle seed 7103 exhausts its diagnostic schedule. Proxy seeds 7101 and 7103
pass numerical gates but do not explore alternative canonical loading regions
already found in retained earlier attempts. They are explicitly excluded,
not silently dropped from denominators or declared converged globally.

The [canonical sign-pattern audit](../../checks/stage7-sign-exploration-20261022/pattern_audit.json)
shows why raw working-sign crossings alone are insufficient. Proxy seed 7102
visits its documented alternative region in all four chains; this is evidence
about that region, not a proof that no other modes exist. The hash-bound
[summary audit](summary/audit.json) and [analysis](analysis/details.json)
retain the exact selected records and exclusions. Earlier effort summaries
remain historical, mode-conditional outputs and are superseded for recovery.

| Eligible configuration | Cases | Generating log-intensity 90% coverage | Log-intensity RMSE | Expected-count RMSE |
| --- | ---: | ---: | ---: | ---: |
| No offset | 3/3 | 0.289-0.326 | 0.897-0.972 | 56.09-82.33 |
| Oracle offset | 2/3 | 0.813-0.906 | 0.164-0.199 | 41.06-51.33 |
| Count-derived proxy | 1/3 | 0.266 | 0.889 | 39.50 |

These are pointwise summaries across dependent category-month cells in each
eligible fixed dataset. They are not repeated-sampling coverage estimates.
For the one eligible proxy case, operational log-intensity coverage is 0.695
and RMSE is 0.270. Its training posterior predictive count coverage is 0.961;
that does not establish recovery of physical threat activity. Only seed 7102
provides an eligible paired comparison across all three configurations.

For that seed, expected-count RMSE is 39.50 (proxy), 41.06 (oracle), and 72.55
(no offset), while generating log-intensity RMSE is 0.889, 0.199 and 0.897.
The proxy's observable fit can improve while its physical interpretation
remains poor. The identical count likelihood can equally represent a reporting
shift or the same shift in threat intensity. Under the alternative threat-shift
truth, the proxy case's 90% log-intensity coverage is 0.164 and RMSE is 1.025.
No analysis of the count likelihood alone distinguishes those interpretations.

Carry forward the proxy as an operational publication-volume adjustment only.
Do not claim that it separates true malicious activity from discovery effort.
The auxiliary-channel and independent reporting information needed for stronger
identification remain Stage 8 design questions. The proxy sampling difficulty
also remains an explicit component limitation; this experiment does not certify
that all proposed-model configurations will fit reliably.
