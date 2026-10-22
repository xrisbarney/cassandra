# Joint level/factor coordinates for the representative case

Declared before implementation or joint-coordinate fitting, 2026-10-22.
The [unadmitted-chain diagnostic](../checks/stage7-joint-geometry-20261022/diagnostic.json)
shows strong dependence between category levels and factor time means
(correlations about -0.87 and -0.76 for the leading categories). The second
factor persistence is near 0.998 across chains, and its time mean differs
substantially between chains. These are computational observations, not
admitted parameter estimates or recovery evidence. The current transformation
whitens factors conditional on levels and leaves this dependence untreated.

Investigate an exact joint Gaussian-coordinate transform for the revised
rank-two, one-regime, factor-only count submodel. Keep the current prior,
likelihood, data, category order, chains, budgets and gates. Do not narrow phi
or drop poorly mixing states from the diagnostics. Continue existing metric
attempts while verifying this candidate.

Let v=m-c have the existing Gaussian prior covariance C, including the
independent fallback block where declared. Let F have the existing stationary
factor prior precision P0. The existing working count Gaussian approximation
has weights W and centered log-count proxy y. Its joint conditional precision
and natural parameter are

    Q = [[C^-1 + D, B], [B', P]],   b = [b_m, b_F]
    P = P0 + H' W H
    D = diagonal category weight sums
    B = category-to-state weighted loading blocks
    b_m = category weighted proxy sums, b_F = H' W y.

Use the Schur complement S=C^-1+D-B P^-1 B'. Draw v from the working
marginal with precision S and natural parameter b_m-B P^-1 b_F, using a
Flat whitened coordinate and a Cholesky solve. Draw F conditionally on that
v with precision P and natural parameter b_F-B'v. The exact transformation
Jacobian is -logdet(chol(S))-logdet(chol(P)). Retain the original normalized
Gaussian level and state priors plus this Jacobian, and evaluate the actual
negative-binomial likelihood. The working observation approximation is never
an added likelihood. Evaluate block solves using the independently verified
primitive; do not add jitter or change any statistical tolerance.

Verify against an independent dense joint Gaussian calculation, complete
old/new target equality with the expected coordinate Jacobian, native/fused
Numba gradients, finite differences and canonical level/state reconstruction.
Include fallback and near-unit-persistence cases. Measure the complete
gradient, rather than infer cost from the primitive alone.

Only after verification, fit the primary representative seed 7101 in a fresh
directory with diagonal adaptation (the complete working joint covariance
has been whitened). Use its complete prescribed retry schedule. If it passes
all gates, use this exact coordinate system and fixed metric uniformly for
the rank-two fallback and sensitivity grid. Preserve earlier complete attempts;
any superseded unfinished metric attempts are explicitly interrupted trials,
not failed schedules. If a pre-existing representative attempt passes first,
retain its original metric decision and do not add an unnecessary posterior
refit. If the joint candidate fails, preserve that outcome and reassess before
expansion. Recovery scores cannot select coordinates or metrics.

This computational follow-up does not establish that near-unit persistence
is physically meaningful or resolve decomposition identification. Those
questions remain part of the recovery and prior-sensitivity interpretation.

Compiler follow-up, before fitting: the standalone candidate passed both
dense-reference/full-density/gradient cases under NUMBA_OPT=1 in 986.44 seconds
under concurrent load. A stack inspection located the long portion in LLVM's
final optimization of the fused density/gradient graph. Check NUMBA_OPT=0 on
the same equations and on the integrated model-builder target. Use that setting
for the joint representative and dependent grid only if all integrated checks
pass. Record it with every run and retain the representative's full-gradient
preflight. Neither setting enables fast-math or changes the numerical gates.
These overlapping checks have different cache histories; their wall times are
not a controlled compiler-speed comparison. The unfinished standalone OPT0
test was explicitly superseded by the integrated six-test run, which includes
both original cases plus direct builder equality and unsupported-model guards.
