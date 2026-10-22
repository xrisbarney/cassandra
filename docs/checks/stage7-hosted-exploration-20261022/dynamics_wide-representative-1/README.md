# Wider dynamics: representative retry 1

This complete attempt fails with maximum R-hat 1.3260, minimum bulk ESS 9.87
and minimum tail ESS 13.42. There are no retained divergences or depth hits.
The [trace](chains.png) was visually inspected; [review.json](review.json)
binds the posterior, inputs and descriptive region statistics.

The second chain spends 57.2% of its draws in the lower-persistence marker,
which the other three never enter. The high-positive-mean region fractions are
74.7%, 24.7%, 40.1% and 79.7%; its indicator R-hat is 1.228 and bulk ESS 13.33.
These are signs of poor exploration, not estimates of posterior region weights.

The final retry is now starting and must be reviewed against this candidate's
own observed regions. The prior remains a declared sensitivity stress case,
with its prior-plausibility classification separately recorded. This review
changes no thresholds or recovery status.
