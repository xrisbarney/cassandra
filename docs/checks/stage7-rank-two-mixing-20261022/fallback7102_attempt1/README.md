# Fallback seed 7102: known exploration issue

Recorded on 2026-10-22 after retry 1 failed and before retry 2 completed.
Retry 1 has maximum R-hat 1.1213, minimum bulk ESS 21.30, minimum tail ESS
11.30, minimum BFMI 0.841, and zero retained divergences/depth saturations.
The worst R-hat belongs to the second persistence coefficient.

The [trace](chain_mixing.png) shows most draws at lower persistence with
second-factor time mean near zero. Chain 4 enters a near-unit-persistence
region late in the run and remains there; its factor time mean also visits
positive and negative regions. Chain 3 makes a much shorter high-persistence
visit. This is a failed fit, not an estimate of those regions' posterior mass.

Review the final attempt against these observed regions using the same
descriptive markers already used for seed 7101: second persistence below 0.8,
or above 0.95 with factor time mean respectively above 0.5 or below -0.5.
These thresholds describe this failed trace; they are not new priors, fitted
regimes, significance tests or recovery thresholds. Their review supplements
the existing requirement to resolve known exploration issues. All four chains
staying in one region would not by itself establish recovery eligibility.
Do not discard chains or infer region weights from this failed attempt.

The [figure audit](figure_audit.json) binds the failed attempt and posterior.
Neither generating truth nor recovery scores were used to choose these markers.
