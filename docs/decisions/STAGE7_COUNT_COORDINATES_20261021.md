# Exact conditional Gaussian coordinates for counts

Declared 2026-10-21 before any fit in these coordinates. The initial v2
effort-proxy seed exhausted its retries. Rank-two sensitivity checkpoints
included factor persistence close to one, followed by extremely slow warmup.
These checkpoints are not posterior estimates. Preserve all original runs.

Reuse the verified conditional severity transformation for stationary count
factors, with a working proxy `log(N + 0.5)` and variance
`1/(N + 0.5) + 1/psi`. Subtract the category level and supplied effort offset.
Missing counts contribute zero precision; observed zeros remain observations.
This working approximation chooses coordinates only. The exact negative
binomial likelihood, stationary Gaussian prior, every hyperprior, and the
parameter-dependent log Jacobian remain in the target.

For stacked factor states, form `P=P0+H'WH=L L'`, `b=H'W(proxy-level-effort)`
and `f=P^-1 b + L^-T u`. Sample u against a Flat base measure and add
`log p(f|phi) - sum(log(diag(L)))`, as for severity. Limit this implementation
to one regime, no residual process and no temporal kernel. Other combinations
require their own construction and validation. The transformation does not
claim that counts are Gaussian or that the proxy supplies inference evidence.

Before fitting, independently verify the full target density/Jacobian and
derivatives, including two factors, observed zeros, missingness and dispersion-
dependent weights. Preserve the existing severity equivalence test after
sharing the transformation code.

First compare seed 7101 for the primary representative and effort-proxy cases
under the same four-chain, three-budget schedule. Choose by diagnostic
admission and cost, not recovery scores. If admitted, apply the same coordinates
to all seeds of that case. For the rank-two representative/fallback comparison,
use the same coordinates across all five predeclared priors if the primary
representative passes; do not select coordinates separately by sensitivity
outcome. Keep original attempts visible. Repeated fits are computational
repairs on the same datasets, not independent replications. No held-out data,
prior, sampling budget or diagnostic threshold changes.

Execution update, before replacement fits: stop the three stalled rank-two
sensitivity retries (tight-level representative, wide-level representative,
wide-level fallback) and their coordinators to release compute capacity for
verification. Preserve their files and record these as interrupted engineering
attempts, not exhausted diagnostic schedules. The completed sparse-factor
sensitivity cases remain unchanged. If the replacement coordinates are adopted,
every replacement sensitivity case receives its full declared schedule in a new
directory. Completion requires an explicit accounting of both interrupted and
replacement runs; interruption alone cannot close a required case.

This extends the algebra in the
[severity decision](STAGE7_SEVERITY_COORDINATES_20261021.md); its Gaussian proxy
is an implementation choice whose usefulness must be established by the fits.
