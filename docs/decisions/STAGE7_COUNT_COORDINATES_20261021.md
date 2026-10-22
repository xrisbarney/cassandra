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

Numerical optimization, declared before block-solver fits: the dense rank-two
preflight measured about 0.0253 seconds per gradient, versus 0.000716 seconds
in amplitude coordinates under the current concurrent workload. Its precision
is block tridiagonal. Implement block Cholesky followed by forward/backward
triangular solves, using the same time ordering and log determinant. This is
the same transformation with linear rather than cubic work in the number of
months at fixed rank. Verify states, Jacobian and derivatives against dense
SciPy algebra for one and two factors, and repeat the complete count-target
test. Keep the dense solver available as an oracle and label each run's solver.
Existing dense-solver attempts are preserved; no timing-based diagnostic waiver
or change to the mathematical target is authorized.

Compiler follow-up, 2026-10-22: stack inspections with py-spy 0.4.1 located
the first block test in PyTensor rewrites and then LLVM optimization. Preserve
the unfinished default-level-3 test log/stack and repeat the entire block-test
set with `NUMBA_OPT=1`. This
[documented Numba setting](https://numba.readthedocs.io/en/stable/reference/envvars.html#envvar-NUMBA_OPT)
trades compiler optimization effort against execution cost. Keep native/NUMBA
value and gradient agreement as a prerequisite; record the compiler level in
each subsequent fit. Do not enable fast-math or suppress numerical warnings.

The symbolic Scan implementation remains available as a reference, but its
compilation was still too expensive under level 1. The production candidate
therefore implements the same forward block factorization and reverse
derivative as cached deterministic Numba kernels, wrapped in a PyTensor Op,
following the existing AR/HMM components. Reverse the backward solve, then
the forward solve and each local Cholesky operation. This supplies exact first
derivatives, not finite-difference derivatives to NUTS. Keep the independent
dense/finite-difference tests as the admission check for this implementation.
The interrupted symbolic test attempts are retained and are not passing tests.

Verification outcome: the cached primitive passed dense state/Jacobian,
independent finite-difference reverse-derivative, native/NUMBA and complete
count-target tests. Additional tests at persistence 0.9998 (and -0.995 in the
second factor), including Fortran-strided inputs, passed. The unfinished
amplitude-coordinate representative fit and the two dense conditional count
engineering fits are now superseded by fresh cached-block fits under the full
same schedules. Retain their partial attempts and hashes; none is labelled an
exhausted schedule or successful recovery experiment. The nearly completed
severity attempt continues; the verified block solver can replace its dense
linear algebra on subsequent seeds without changing its transformation.

Integration correction: the first cached-block launches failed before sampling
when the fused NUTS log-density/gradient graph received a native scalar Jacobian
from an Op declared to return a Tensor scalar. Return a zero-dimensional array
instead. Numerical value/Jacobian and separate-gradient tests did not exercise
that fused interface, so add combined log-density/gradient checks for both
count and censored-severity models. Retain the two execution failures separately;
they are not exhausted statistical fits or convergence failures.

This extends the algebra in the
[severity decision](STAGE7_SEVERITY_COORDINATES_20261021.md); its Gaussian proxy
is an implementation choice whose usefulness must be established by the fits.

First effort-proxy outcome, recorded before reading its recovery scores:
`count-block-v3/primary/effort_proxy-7101` passed the first four-chain budget
with R-hat 1.00874, minimum bulk/tail ESS 792/640, minimum BFMI 0.94095,
zero retained divergences/depth saturations, and no unresolved warnings.
Apply the same conditional count coordinates, cached block solver and dense
metric to seeds 7102 and 7103. The original three failed amplitude-coordinate
schedules remain part of the record. This does not select a new prior or
claim that the effort proxy identifies physical threat activity.

Allocation profile, 2026-10-22: at 109 states and rank two, the original cached
forward and reverse kernels took median 0.000530 and 0.002207 seconds in a
100-call microbenchmark under concurrent load. The reverse includes another
forward evaluation. Specialize the one- and two-factor triangular algebra into
scalar operations and preallocated arrays, retaining the general solver for
higher rank. This removes per-month tiny-matrix allocations; it does not alter
the precision, working likelihood, normalization or Jacobian. Repeat dense
reference, finite-difference, near-unit/strided and complete fused target checks
before using the optimization as validated evidence. Report a fresh full-gradient
timing, since primitive speed alone is not a sampler speedup estimate. Existing
processes retain their already loaded implementation and provenance.

The specialized implementation passed all six focused checks (882.11 seconds):
the four independent block/reference/gradient/near-unit tests and the complete
fused count and censored-severity targets. The two dense-oracle variants are
also included in the separate full-suite run. In the same 100-call primitive
microbenchmark, forward/reverse medians became 0.0000327/0.0000852 seconds.
These measurements were made under concurrent load and do not isolate a sampler
speedup. Use the verified scalar implementation for subsequent block fits;
existing partial attempts remain identifiable by their source hashes.

Misspecification follow-up, declared 2026-10-22 before any `misspecified` fit:
use the same verified conditional count coordinates, scalar block solver and
dense metric for all three seeds of that compatible one-regime, one-factor
fixture. The corresponding effort and sparse-factor submodels have already
produced admitted posteriors. The generator retains its 30% zero inflation,
while the fitted likelihood remains NB. This avoids conflating a deliberate
observation-model misspecification with the superseded sampling geometry.
It does not make the fitted model correctly specified or waive any diagnostic
gate. Preserve the original generator, truths, priors, seeds and budgets.
