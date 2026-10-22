# Tightened levels: representative retry 1

Reviewed after the second prescribed attempt and before the final hosted retry
starts. The complete attempt fails: maximum R-hat 1.0270, minimum bulk ESS
106.67 and minimum tail ESS 281.28, with no retained divergences or depth hits.

The [visually inspected trace](chains.png) and [record](review.json) show both
high-persistence factor-mean regions. The negative-region fractions are 16.7%,
24.6%, 17.3% and 11.0%; its indicator R-hat is 1.017 and bulk ESS 108.70.
The positive indicator has R-hat 1.023 and bulk ESS 112.35. No chain enters
the lower-persistence marker that one chain visited in the first attempt.

Numerical improvement over the first attempt is insufficient for admission.
Review the final retry against both attempts' own-target evidence. These
fractions are descriptive exploration diagnostics, not posterior region
probabilities; no posterior recovery estimate is admitted.
