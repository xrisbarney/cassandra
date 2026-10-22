# Bounded Stage 7 convergence supplement: execution record

Status, 2026-10-22: **bounded round complete; 0/4 numerical passes**.
The [declaration](STAGE7_REPAIR_SUPPLEMENT.md) fixes four cases and one attempt
per case. The [activation](decisions/STAGE7_REPAIR_ACTIVATION_20261022.json)
binds the code, fixtures, prior audit and numerical checks. Stage 8 and held-out
scoring remain closed; the original 55-case campaign remains unchanged.

The [preflight record](checks/stage7-supplement-20261022/README.md) contains four
successful full-model density/gradient comparisons, a passing one-factor prior
plausibility audit, and 39 passing distinct tests. The masked fixture preserves
the prior anchors from the original unmasked counts. The matched fixture uses
the declared one-factor law and RNG seeds. The other two reduced fits retain
the original two-factor truth and are approximation checks.

All fits use four sequential chains, 2,000 warmup and 2,000 retained draws per
chain, target acceptance 0.99 and depth limit 15. Each case has a single reserved
directory, with no overwrite or sampler retry. Hosted reruns are refused.
The existing irreversible retained-failure rule stops an ineligible attempt
early and preserves its state and progress. Partial attempts are not recovery
evidence. Final numerical admission must be followed by exploration review.

The single [hosted execution](https://github.com/xrisbarney/cassandra/actions/runs/39830813971)
uses fitting commit `e164dd753c00e3af79a76e37687be17063a8773d` and clean source
provenance. The [execution record](decisions/STAGE7_REPAIR_EXECUTION_20261022.json)
reserves exactly one attempt for each case. Verified imports are saved under
[the supplement evidence directory](simulations/stage7-supplement-20261022/).

The matched one-factor fallback stopped at its first retained divergence:
chain 0, draw index 3470 (1,471 retained draws after 2,000 warmup). No complete
four-chain posterior exists for this attempt. The reduced representative
completed all four chains without divergences or depth saturations. Its
maximum R-hat is 1.007630, minimum bulk ESS 824.400 and minimum BFMI 0.884124,
but minimum tail ESS is 310.386, below the required value of 400. Independent
recomputation confirms this failure. Low tail ESS occurs in loadings,
persistence and levels; it is not confined to a duplicate deterministic
variable. Its diagnostic plots remain explicitly labeled as a failed fit.
Neither case supplies recovery evidence.

The reduced fallback also completes all four chains without divergences or
depth saturations, but fails maximum R-hat (1.066902) and minimum bulk ESS
(94.532). Minimum tail ESS is 423.935 and BFMI is 0.860968. Its worst R-hat and
bulk ESS occur in a factor state; the independent fallback level itself has
R-hat 1.033735 and bulk ESS 168.238. The saved chain plots show long visits to
high-persistence values with different occupancy across chains, alongside
changes in the level scale. These failed-chain displays are diagnostic evidence,
not estimates of posterior region weights or proof of a unique failure cause.
All three one-factor cases fail admission, and none receives a recovery summary.

The masked two-factor fallback stops on a retained divergence in chain 3,
draw index 3455: the fourth chain had completed 1,456 retained draws, following
three complete chains. Its 7,456 retained transitions do not form an admitted
four-chain posterior. Masking the zero-count channel did not produce an admitted
fit under this attempt, so it supplies no basis for removing category 10 from
production. It also does not settle the separate unmatched-KEV treatment.

| Case | Complete four-chain posterior | Numerical outcome |
|---|---|---|
| Masked fallback, two factors | No | Retained divergence; early stop |
| Reduced representative, one factor | Yes | Tail ESS 310.386, below 400 |
| Reduced fallback, one factor | Yes | R-hat 1.066902; bulk ESS 94.532 |
| Matched one-factor fallback | No | Retained divergence; early stop |

The [closure audit](simulations/stage7-supplement-20261022/audit.json) verifies
all four imported inventories and their activation, fixture and source hashes.
The two complete posteriors had all diagnostic aggregates independently
recomputed at import; the two partial attempts preserve failure markers, last
states and timing, with unavailable R-hat/ESS reported as null. No complete
posterior is invented for them. There are zero recovery-eligible cases and
no recovery or forecast-calibration claims from this supplement. The original
55-case campaign and its 27 eligible cases remain unchanged.

Exactly four attempts consumed 26,000 warmup and 24,927 retained transitions,
including transitions in failed attempts. The latter are not usable forecast
draws. Sampling calls took 2,765.982, 734.609, 2,119.457 and 582.566 seconds in
the table's order. The audit separately records full-target preflight compilation,
NUTS step construction, within-chain warmup and retained step time, diagnostics
and serialization. Within-chain times are nested in sampling-call times and
must not be added again. Concurrent worker costs are not elapsed campaign time;
these failed synthetic fits do not establish a production rolling-fit budget.

The original production configuration is unchanged. The condition requiring
all three one-factor cases to pass has failed, so this round cannot authorize
a one-factor pilot amendment. All four declared outcomes have now been audited;
the round stops here, with no retry, extra seed, prior change or new repair round.
This is a failure of the declared attempts, not a proof that a one-factor
model cannot be sampled under any alternative design. Stage 8 stays closed.
Kernel and misspecified extensions remain unvalidated. January 2024 through
December 2025 remains unscored. The implementation/preflight work is complete;
production-pilot posterior validation remains unresolved.
