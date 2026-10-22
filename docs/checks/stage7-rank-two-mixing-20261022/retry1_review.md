# Fallback seed 7101, retry 1

Recorded on 2026-10-22 before the final retry completed. The second attempt
failed the unchanged numerical gate: maximum R-hat 1.3001, minimum bulk ESS
10.62, minimum tail ESS 31.71, minimum BFMI 0.875, no retained divergences or
depth saturations, and no unresolved warnings.

The [region review](regions_attempts01.json) applies the markers documented
after retry 0. All four chains visit each marked region, but the lower-persistence
fractions differ from 1.4% to 74.7%, with only 2-4 changes per chain. Its indicator
R-hat is 1.685 and bulk ESS 6.51. Mere visits do not demonstrate adequate mixing.
Positive/negative high-persistence regions also have unequal chain frequencies.
These are descriptive frequencies in a failed fit, not posterior region weights.

The final prescribed retry proceeds unchanged. This failed attempt contributes
no recovery result and does not justify changing priors or acceptance thresholds.
