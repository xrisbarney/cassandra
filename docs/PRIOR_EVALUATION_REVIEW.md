# What carries forward from the supplied evaluation

Updated 2026-10-20 following the user's clarification. The filename is retained so existing document links continue to work.

Use the supplied failed evaluation only to identify the requested baselines and measurement families. Do not carry forward its numerical results, rankings, inferred model choices, dataset sizes, or narrative conclusions. Do not assume its stated implementation procedure describes the new experiment.

The relevant original passages are now saved in [sources/USER_EVALUATION_EXTRACTS.md](sources/USER_EVALUATION_EXTRACTS.md). The exact adapted models and primary-source records are in [BASELINE_SPECIFICATIONS.md](BASELINE_SPECIFICATIONS.md); statistical calculations are fixed in [STATISTICAL_ANALYSIS_PLAN.md](STATISTICAL_ANALYSIS_PLAN.md).

## Required comparators

- RF, following the Kia-style approach.
- ARIMA.
- ETS.
- Seasonal naive.
- BSTS-U.
- Vuln4Cast.
- Almahmoud B-LSTM.
- Bakdash BSSM.

Verify the original methods and available implementations before specifying adaptations. Match targets, forecast horizons, information cutoffs, and update schedules. Simple diagnostic controls supplement this list rather than replace it.

## Measurement families to retain

- CRPS and predictive log score.
- MAE and RMSE.
- Central 50% and 90% interval coverage, with interval width and interval score.
- Randomized PIT diagnostics for count forecasts.
- Paired forecast comparisons, including DM where a defensible dependence-aware implementation and adequate series length permit it.
- Performance at 1-, 3-, 6-, and 12-month horizons.
- Component ablations and computational diagnostics.
- Separately defined economic validation with independently verified event evidence.

Re-specify and validate the calculations. Retaining a measurement name does not mean retaining an invalid approximation, independence assumption, or probability law. All values, tables, figures, and conclusions must come from the new saved forecast artifacts.

## New temporal split

The user requested January 2024–December 2025 as the held-out forecast period. Initial training remains 2010–2018; development and model selection use 2019–2023. Freeze design before inspecting the new held-out scores. At each rolling origin, fit only through that origin; earlier elapsed test months may enter subsequent training fits.

See [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) for the exact origin schedules, verified source availability, scoring rules, and audit gates. This is a historical forecast evaluation. No old score is a target to match or beat.
