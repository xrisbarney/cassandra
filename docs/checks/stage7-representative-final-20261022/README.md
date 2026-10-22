# Representative primary case: final failure

The fixed ten-category, two-factor, 108-month count fixture at seed 7101 has
exhausted all three prescribed attempts. The [audit](audit.json) checks its
terminal result, immutable artifacts and original budget schedule.

Attempt 0 stops on a retained divergence in the fourth chain. Attempt 1
completes all four chains but fails mixing: maximum R-hat 1.0415, minimum bulk
ESS 87.62 and minimum tail ESS 323.05. Its saved diagnostics were independently
recomputed earlier. Attempt 2 stops on a retained divergence at zero-based
draw index 2293 in the fourth chain, after that chain's 2,000 warmup draws.

The partial final attempt is not a complete posterior and supplies no recovery
estimate. Its recorded diagnostic counts are lower bounds. Raw warnings from
the older running process are preserved unchanged; the statistical failures
already prevent admission regardless of cache-warning classification.

This completes the declared representative experiment, but does not validate
the two-factor pilot or provide a successful-fit runtime for extrapolating a
rolling-backtest budget.
