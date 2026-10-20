# Decisions carried from panel review into Stages 6 and 8

Accepted with the user on 2026-10-20. This supplements the immutable Stage 5
[panel dictionary](PANEL_DICTIONARY.md); its original hash remains valid.

1. Panel IDs are NMF component IDs, not loading-matrix row positions. Compute the
   model order from initial-window counts, descending, ties by panel ID, and keep
   unknown last. This snapshot gives **8, 2, 6, 1, 4, 9, 3, 5, 7, 10**. Apply that
   permutation before triangular loading constraints and invert it before storing
   category forecasts. Never derive the order from development or test counts.
2. Count category 10 is an observed all-zero fallback series. Retain it in the
   protocol's equal-category primary score, and report its actual contribution
   separately. A low score is not evidence of forecasting a substantive threat.
   It may compress mean differences; near-zero scores are not guaranteed for
   every predictive law. The active stratum excludes it. The realized aggregate
   count is unchanged by adding zeros, but predicted aggregate counts can include
   positive unknown-category draws.
3. The nine learned categories are vulnerability-description topics. Three
   prominently reflect Oracle/Adobe/template wording. Carry that limitation into
   the manuscript rather than describing all nine as distinct threat mechanisms.
4. The all-zero negative-binomial count series informs its latent intensity
   toward low values; the prior regularizes it, without implying a literal hard
   floor for eta. The 21 unknown-category KEV additions concern CVEs outside the
   cohort defining that intensity. Before version G, choose explicitly between
   **masking unknown in the KEV likelihood** and **a separate intercept-only KEV
   rate independent of eta**. Neither alternative is selected in Stage 6. Preserve
   all additions, assignments and source hashes in the existing panel either way.

Stage 6 builds and validates the runner on development outcomes only. It does not
clear the held-out scoring gate or enable auxiliary likelihoods. Stage 8 must
record the chosen unknown-KEV treatment before fitting version G.
