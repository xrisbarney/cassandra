# Hosted rank-two exploration reviews

These reviews describe retained chains from the already declared synthetic
cases. They do not change priors, seeds, budgets, numerical gates or recovery
eligibility. Different sensitivity priors define different posterior targets;
failed-chain frequencies are not posterior region probabilities.

## Tightened levels, fallback seed 7101, first attempt

Recorded after hosted attempt 0 and before its next retry starts. The attempt
has no retained divergences or depth saturations, but maximum R-hat is 1.3083,
minimum bulk ESS 10.22 and minimum tail ESS 12.54. It is unsuccessful.

The [trace](level_tight-fallback-0/chains.png) and
[review](level_tight-fallback-0/review.json) show two level/factor-mean regions
under this candidate's own target. Three chains spend 95.2--97.4% of retained
draws in the high-persistence/positive-mean region. The fourth spends 43.9%
there and 48.1% in the high-persistence/negative-mean region. The positive-region
indicator has R-hat 1.958 and bulk ESS 5.51. Its occasional transitions do not
establish good mixing. All chains' mean second-factor persistence exceeds .99.

Any later numerical pass for this case needs an exploration review against
these documented regions, with the original diagnostic and recovery-eligibility
rules retained. Do not infer region weights from this failed attempt or silently
discard its negative-mean region. Markers are descriptive: persistence below .8,
or persistence above .95 with factor time mean above .5 or below -.5.

The figure was visually inspected. The review script and all input hashes are
recorded alongside the plot; no truth-recovery result is derived from this fit.

## Tightened levels, representative seed 7101, first attempt

Also recorded before the next retry starts. This attempt has maximum R-hat
1.1823, minimum bulk ESS 17.31 and minimum tail ESS 81.38, with no retained
divergences or depth saturations. All chains visit both high-persistence mean
regions, but their frequencies differ substantially. The negative-region
fractions are 46.8%, 17.2%, 10.8% and 10.9%; the positive-region fractions are
32.7%, 63.9%, 75.0% and 76.2%. Both indicator R-hats are about 1.16, with bulk
ESS about 18. Visits alone therefore do not establish mixing. Only the second
chain visits the lower-persistence marker, spending 2.8% of its draws there.

The [trace](level_tight-representative-0/chains.png) was visually inspected;
the [review](level_tight-representative-0/review.json) retains the exact input
hashes and marker statistics. Review later numerical passes against this
candidate's own evidence. No region weights or recovery estimates are admitted.
