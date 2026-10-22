# Stage 7 synthetic validation record

Status, 2026-10-22: **posterior campaign in progress; not yet complete**.
The active laws are `anchored_levels_v2`. The primary prior-plausibility gates
pass, and 175 distinct tests pass. The remaining work is the full posterior
campaign, sensitivity analysis and final component decisions. No real-data
proposed-model fitting or January 2024--December 2025 scoring has started.

## Scope and evidence

The [design](STAGE7_DESIGN.md) and
[level amendment](amendments/STATE_SPACE_LEVELS_20261021.md) prescribe 43 primary
scenario/seed cases and 12 sensitivity cases. The three effort variants share
the same observations for each seed. Repeated coordinate fits and sensitivity
fits are not additional independent datasets. These are fixed-truth recovery
experiments, **not simulation-based calibration**. Neither three seeds nor
coverage over dependent category-month cells establishes general calibration.
The historical "40 datasets" wording in the preserved design/amendment means
40 scenario/seed cases; it does not override their explicit effort-data reuse.

The small factor fixtures use four categories and 96 months; kernel fixtures
use 144 months, dispersion fixtures one category and 192 months, and the
representative/fallback fixtures ten categories, two factors and 108 months.
The generator uses NumPy/SciPy separately from the PyMC likelihood. All
posterior fixtures and future targets are synthetic. Only the prior-plausibility
check reads the real initial-window counts, January 2010--December 2018.

## Prior correction

The original laws gave infinite prior rate moments and generated counts beyond
the integer range; see the [tail finding](STAGE7_PRIOR_FINDING.md). The first
amendment bounded the dynamic variance but left the level contribution broad.
The second amendment centers the known-category hierarchy on the training-only
log mean plus 0.5, with hypermean SD 1 and category scale HalfNormal(1) truncated
to (0,2). The independently recomputed real-panel anchor is 4.301907835285142;
4.1972994 includes the fallback and is not the nine-category anchor. Synthetic
fits derive their own anchor from observed synthetic training counts.

The declared fallback has its own Normal level centered on its initial mean
plus 0.5. It keeps the count likelihood, shared dynamics and forecast/scoring
role. This is neither a point mass at zero nor an intensity floor. An all-zero
series informs the rate, but cannot determine arbitrarily small positive rates.
Its independent level prevents the known-category scale from directly absorbing
that separation. Shared factors still permit indirect dependence.

Stationary unit-variance factors, bounded loading/noise amplitudes, removal of
the former 0.01 noise floor and the level revision are substantive changes to
the original prior. The current conditional Gaussian variance envelope is 11
before independent regime shifts; the moment argument is in the amendments.
The original freeze and both amendments remain preserved.

The [prior audit](checks/stage7-levels-20261021/README.md) used 2,000 trajectories
and whole-trajectory bootstrap intervals. At zero effort the primary pilot has
median count 68, 99.9th percentile 20,794, and zero probability 0.00838 for known
categories. Their 95% bootstrap intervals are 65--72, 17,356--25,649 and
0.00746--0.00919. All declared pilot gates pass. Along the initial-window effort
trajectory the median is 65 and the 99.9th percentile 43,763; the separately
specified longitudinal finite-simulation/envelope gate passes.

Both tighter candidates pass the prior gates. `level_wide` fails the tail gate.
`dynamics_wide` remains unresolved after the prescribed 8,000-trajectory repeat:
its 99.9th percentile is 50,854 with interval 47,678--54,119 crossing the 50,000
threshold. The original classifier's incorrect fail label is preserved with a
correction record. These wide candidates remain labelled sensitivity stress
cases, not admitted priors. The R=3 + iid + kernel joint prior is broader
(q99.9 about 147,120 at reference effort); the pilot gate does not validate that
extension. The maximum-effort/highest-regime envelope is stress-only.

The generic prior-predictive Potential warning is retained. This audit uses
innovation coordinates with proper generative priors; the ignored Potential
is the observation likelihood, which must not condition a prior draw.

## Numerical correctness and computation

Independent checks cover NB2 and censored mark densities, missing masks,
endpoint classification, HMM marginalization against exhaustive enumeration
through six months/three regimes, forward/backward state probabilities, analytic
and finite-difference gradients, covariance and posterior forecast simulation.
The [numerical audit](checks/stage7-numerics-20261021/numerics.json) distinguishes
exact discrete coverage from nominal interval levels. Its finite kernel grid
finds maximum covariance error about 0.0002384 at width 18, lag 34; this is not
a uniform mathematical error bound over all widths/lags.

The hierarchical Gaussian hypermean is integrated out exactly, giving
`s_m^2 I + hyper_sd^2 11'`. Conditional Gaussian coordinates are exact changes
of variables, with their normalized state prior and Jacobian retained. The
working Gaussian count/mark approximation chooses coordinates only; the NB2
and censored mark likelihoods remain unchanged. Dense independent calculations,
finite differences and native/Numba complete-target comparisons verify both
count and severity transformations.

A block-tridiagonal implementation and scalar rank-one/rank-two specialization
remove unnecessary dense operations and small-array allocations. No jitter,
fast-math approximation or statistical tolerance was added. Near-unit
persistence and non-contiguous arrays are tested. The first representative
full-gradient preflight improved from about 0.00252 to 0.00111 seconds under
concurrent load; this is not an end-to-end sampling speedup estimate.

Computational changes and original attempts are recorded in the
[count coordinate decision](decisions/STAGE7_COUNT_COORDINATES_20261021.md),
[severity coordinate decision](decisions/STAGE7_SEVERITY_COORDINATES_20261021.md),
[metric decision](decisions/STAGE7_WHITENED_METRIC_20261022.md) and
[matched-fit decision](decisions/STAGE7_MATCHED_COUNT_REPAIRS_20261022.md).
The matched count comparisons repeat accepted cases as well as failed ones;
coordinates are not chosen separately by recovery scores. Figures use the
explicit routed records and fixed seed 7101, displaying failures as failures.

## Admission and recovery

Every admitted posterior requires four complete chains, maximum R-hat below
1.01, minimum bulk/tail ESS above 400, minimum BFMI above 0.3, finite diagnostics,
zero retained divergences/depth saturations and no unresolved numerical warnings.
The fixed attempts use warmup/retained draws 1000/1000, 2000/1000 and 2000/2000
per chain, target acceptance .95/.99/.99 and depth limits 12/15/15. Early
termination after an irreversible retained violation preserves the partial
attempt and advances only through the prescribed schedule. Partial counts
are lower bounds and partial chains are not recovery evidence.

**Final case counts, recovery tables, sensitivity findings and component
admission decisions are pending completion of the declared fits.** A numerical
diagnostic pass with unresolved warnings does not count as an admitted fit.
Interrupted engineering trials and execution errors remain separate from
fully exhausted diagnostic schedules. The final audit must independently
recompute diagnostics from each admitted saved posterior.

The [mode diagnostic](checks/stage7-mode-diagnostic-20261022/audit.json) documents
opposite loading/decomposition regions in unsuccessful factor/kernel attempts.
Because the positive anchor constraint is retained, these are not exact
sign-equivalent labelings that can be repaired by relabeling a chain. Unmixed
chains provide no reliable posterior mode weights. No chain is discarded or
relabeled to satisfy the admission gates.

Recovery targets include operational log intensity, generating interpretation,
expected observed counts, censored-severity location and implied covariance.
The effort-only and threat-only interpretations can produce the same count
likelihood; good operational recovery cannot resolve that identification issue.
Regime probabilities inferred with full synthetic records are retrospective,
not evidence of a successful early-warning system. The deliberately misspecified
fixture has 30% zero inflation; its observable mean includes that reduction and
is distinct from its underlying NB-component intensity.

## Verification and limits

The [test record](checks/stage7-test-suite-20261022/audit.json) contains **175
distinct passing tests** across 20 files. Each file ran in a fresh process;
the subsequently expanded audit file's five tests replace its earlier four.
The interrupted single-process test invocation is not counted as passing.
The test result is software verification, not evidence that every posterior
converges or every component is identified.

Final lint, freeze/provenance checks and the completion audit remain to be
recorded alongside the completed posterior tables. The original Stage 2 freeze,
Stage 5 panel and Stage 6 results remain preserved. Stage 8 production integration,
real auxiliary-channel eligibility, economic calibration and the six remaining
named comparison systems are outside this stage. The KEV slope and unmatched
fallback-KEV treatment remain separately pending; synthetic fixtures with known
auxiliary coefficients do not resolve them. January 2024--December 2025 remains
unscored.
