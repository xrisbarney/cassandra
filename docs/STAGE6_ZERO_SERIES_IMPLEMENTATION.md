# Approved zero-series rule: implementation and verification

The user accepted the methodological change on 2026-10-21. The
[approved amendment](amendments/ZERO_SERIES_20261021.md) is active; its
[separate freeze](amendments/ZERO_SERIES_20261021.json) has SHA-256
`c4e73555e034bc9d9d540a2dcad2cd8ba0ce5b1e0744beda80750fa026c08bbd`.
The original Stage 2 freeze remains
`7597b76276b837f8db9bc2baff63cb035e14cdfa8656885c782795d38cb598e7`.

ARIMA now applies the declared point mass before Gaussian fitting when a
complete observed training series is entirely zero. The shared boundary
predicate also specifies the rule for the future ETS adapter; this does not
claim that the rest of ETS is implemented. Fit records identify every affected
category, the rule, training cutoff and absence of optimizer attempts.
Nonzero training histories retain the original frozen optimizer/retry sequence.

Forecast laws, count PMFs/CDFs, quantiles and scoring all use the same point
mass. Infinite negative log scores are preserved through category and origin
averaging. Category 10 remains in the panel; no unknown-category or KEV records
were edited. Historical runs remain unchanged.

New ARIMA runs require the active supplement and verify its text and parent
freeze. Run records retain the supplement; effective configuration hashes also
include its hash. Screening records carry the amendment hash, training cutoff,
development target, forecast hash, fit status and boundary-category list.
Run provenance now includes orchestration-script hashes as well as package code.

Verification:

- All 65 frozen ARIMA candidates bypass optimization on all-zero test series,
  including a category whose ID is not 10.
- Missing and invalid counts cannot qualify as zeros; a positive training count
  disables the boundary rule. Other fitting failures remain failures.
- Positive outcomes receive infinite negative log score, and that infinity
  survives both levels of summary averaging.
- Missing, altered or mismatched amendment records are rejected.
- Full suite: 90 passed. After the final provenance changes, the affected
  backtest, boundary and scaffold tests also passed: 29 passed.
- Ruff and the base dependency lock check pass. The documented missing-BLAS
  performance warning remains; it does not invalidate the boundary checks.

The frozen ARIMA screen is enabled: 65 candidates, 20 quarterly development
origins, 3,000 paths, unchanged fit budgets and selection rules. Worker count
changes execution concurrency only. Two valid finalists must subsequently be
evaluated on all 60 monthly origins. No winner is implied by enabling screening.
The pooled Bayesian control has a separate 60-origin development configuration.

Stage 6 is still in progress. Bayesian convergence/rolling checks, ARIMA
selection, ensemble checks and the later main-comparison gates remain explicit.
The 2024-2025 held-out scoring gate remains closed.
