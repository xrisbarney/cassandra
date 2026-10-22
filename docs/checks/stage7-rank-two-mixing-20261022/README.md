# Rank-two mixing diagnostic

The first joint-coordinate fallback fit (seed 7101) completed four chains but
failed admission: maximum R-hat 1.248, minimum bulk ESS 12.23, minimum tail ESS
32.12, no retained divergences and minimum BFMI 0.854. Its saved posterior is
used only to diagnose the failure. It is not recovery evidence.

The [trace figure](chain_mixing.png) shows long visits to different persistence
and factor-mean regions. The near-unit-persistence region also permits positive
and negative second-factor means. Category levels and their scale change with
these visits. Free loading signs vary within chains, so merely flipping a sign
or relabeling the chains is not a demonstrated correction.

Before any rank-two case is admitted, use these diagnostic markers when reviewing
later attempts of this same dataset: second-factor persistence below 0.8, and
persistence above 0.95 with its mean over observed months respectively above 0.5
or below -0.5. These identify the separated regions visible in this failed
attempt; they are not fitted regimes, prior restrictions or recovery thresholds.
Inspect visits and transitions in each chain, and indicator diagnostics where
estimable. Do not estimate posterior region weights from the failed chains.

This supplements the existing recovery-eligibility rule: a numerical pass does
not resolve known alternative posterior regions merely because all new chains
stay together in one of them. Record unresolved exploration explicitly rather
than discarding older chains or treating their frequencies as true posterior
weights. The prescribed retries and prior-sensitivity grid continue unchanged.
No truth or recovery score is used to select these diagnostic markers.
