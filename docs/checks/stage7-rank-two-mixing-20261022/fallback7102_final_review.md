# Final prescribed retry, 2026-10-22

The full three-attempt schedule is now exhausted. Retry 2 completed all four
chains with 2,000 retained draws each and no retained divergence or depth hit,
but maximum R-hat is 1.2546, minimum bulk ESS 11.45 and minimum tail ESS 22.66.
The saved posterior diagnostics were independently recomputed and agree.

The [three-attempt region review](fallback7102_attempts012.json) confirms
continued poor exploration. The lower-persistence region contains 93.1%,
99.7%, 29.15% and 94.6% of the four chains' final retained draws. Its indicator
R-hat is 1.760 and bulk ESS 6.06; one chain never visits either recorded
high-persistence mean region. These fractions are diagnostic observations,
not estimates of posterior region probabilities. This case is a terminal
statistical failure and contributes no recovery estimate.
