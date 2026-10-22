# Wider levels: representative retry 1

This complete attempt fails with maximum R-hat 1.0503, minimum bulk ESS 95.72
and minimum tail ESS 522.82. There are no retained divergences or depth hits.
The [trace](chains.png) was visually inspected; [review.json](review.json)
binds the posterior, inputs and descriptive region statistics.

Only the second chain enters the lower-persistence marker, for 1.5% of its
draws. All chains visit both high-persistence mean regions; the positive-region
indicator still has R-hat 1.022 and bulk ESS 231.18. Visits and failed-chain
frequencies do not establish adequate mixing or posterior region weights.

The final retry is now starting and must be reviewed against this candidate's
own observed regions. The prior is a declared sensitivity stress case, not an
admitted default prior. This review changes no thresholds or recovery status.
