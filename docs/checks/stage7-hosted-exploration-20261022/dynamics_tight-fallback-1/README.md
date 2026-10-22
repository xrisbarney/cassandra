# Tightened dynamics: fallback retry 1

Reviewed after the second prescribed attempt and before the final hosted retry
starts. The complete attempt fails: maximum R-hat 1.0839, minimum bulk ESS
46.41 and minimum tail ESS 192.39, with no retained divergences or depth hits.

The [visually inspected trace](chains.png) and [record](review.json) show all
chains visiting the three previously defined descriptive regions. Frequencies
remain unequal: lower-persistence fractions are 22.1%, 13.9%, 1.5% and 18.3%;
high-negative-mean fractions are 20.7%, 42.8%, 21.6% and 13.5%. The latter
indicator has R-hat 1.090 and bulk ESS 55.05. These visits do not establish
mixing or posterior region weights.

This is evidence under this candidate's own posterior. Review any later
numerical pass against these observed regions; neither the thresholds nor the
sampling schedule changes. No recovery estimate is admitted from this attempt.
