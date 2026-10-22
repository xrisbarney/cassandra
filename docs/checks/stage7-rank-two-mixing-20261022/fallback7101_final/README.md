# Primary fallback seed 7101: final failure

All three prescribed attempts are exhausted. The final attempt completes
four chains with 2,000 retained draws each, with maximum R-hat 1.2723,
minimum bulk ESS 11.17, minimum tail ESS 28.01 and minimum BFMI 0.872.
There are no retained divergences or depth saturations. This is a diagnostic
failure and contributes no posterior recovery estimate.
Independent recomputation from the saved posterior matches every recorded
diagnostic at the original relative and absolute comparison tolerances of 1e-10.

The [visually inspected trace](chains.png) and [review](review.json) confirm
poor exploration of the previously documented regions. The lower-persistence
fractions are 24.1%, 2.0%, 49.7% and 15.35%. Every chain visits that marker,
but its indicator R-hat is 1.739 and bulk ESS 6.31. Visits are insufficient
to establish mixing, and these frequencies are not posterior region weights.

The original sampling records, earlier attempts and marker definitions remain
preserved. The review binds its exact posterior, attempt, data and source files.
