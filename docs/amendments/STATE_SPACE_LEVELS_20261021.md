# Stage 7 second prior amendment: anchored levels and a separate fallback

Declared 2026-10-21 before revised prior simulations or posterior fits.
Authority: user, "okay, treat all and proceed", followed by the instruction to
take Stage 7 to completion. Identity: `anchored_levels_v2`. This supersedes the
intensity-level rows of `stationary_bounded_v1`; its other count/severity laws
remain in force. Preserve both earlier amendments, all fits, and the Stage 2
freeze. Held-out scoring remains disabled.

## Laws and reference

Normal parameters are mean and standard deviation. For known categories:

    m_bar ~ Normal(c, 1)
    s_m ~ HalfNormal(1) conditional on 0 < s_m < 2
    m_k | m_bar,s_m ~ Normal(m_bar,s_m)

Integrating out m_bar exactly gives covariance `s_m**2 I + 11'`.
For the fallback category only, independently of that hierarchy:

    m_fallback ~ Normal(log(mean_initial_fallback_count + 0.5), 1)

This is a stochastic level with unbounded support, not a zero point mass or
intensity floor. Keep its observed-count likelihood, shared dynamic components,
forecasts, and primary scoring contribution. Its level does not update s_m or
m_bar directly; shared factors can still transmit information. Apply this rule
by declared fallback identity, never by automatically removing every zero series.
It does not resolve which KEV observations may depend on that intensity.

For the real panel, c is log of the arithmetic mean across all 9 known categories
and 108 initial-window months, plus 0.5: **4.301907835285142**. The value 4.1972994
includes category 10 and is not this anchor. Use January 2010--December 2018 only,
fixed across all later origins. This is a training-derived empirical-Bayes
regularization choice; do not describe the check as an externally elicited prior.
The baseline precedent is training-only anchoring, not an identical hierarchy.

c is a rough location on the log-count scale at reference effort zero, not an
identity between a count mean and latent median. Do not subtract an unrecorded
Jensen correction. Check the observation distribution at zero effort and along
the initial-window standardized filtered effort trajectory. Origin-specific
effort centering remains as frozen; disclose its origin-dependent reference.

Synthetic cases derive their own c from observed counts in the first
min(108,T) training months, using no truth or future values. The original 40
datasets and truths are preserved. The original sparse category is a known
category, not relabelled fallback. Add three `fallback_zero` fixtures with
10 categories, 108 months, two factors, last category explicitly fallback,
true last baseline rate 0.0001. Generate NB observations without forcing zeros;
report the realized zero/nonzero count. This tests an extremely rare category
while preserving a positive true rate. No real counts enter posterior fixtures.

The conditional Gaussian variance envelope for known log intensity is now
`1 + 4 + 4 + 1 + 1 = 11`, before independent regime shifts. All finite-moment
arguments of v1 continue to hold. This is an upper bound, not typical variance.

## Prior-predictive gates, declared before the checks

Use 2,000 independent parameter/trajectory draws per case and retain complete
trajectory dependence. Report percentile uncertainty with 200 bootstrap
resamples of whole trajectories. Sample regime occupancy from its transition
matrix, starting uniformly. Never stack maxima and call them the joint prior.

The primary R=1, two-factor, no-iid/no-kernel pilot at reference effort must have:

- known-category median count in [30,175], rounding outward the independently
  verified initial category-mean range [30.68,173.01];
- known-category zero probability at most 0.05;
- known-category 99.9th percentile at most 50,000 publications/month: a deliberately
  permissive 500-fold threshold relative to a 100-count reference, not a physical
  maximum or literature constant;
- no nonfinite rates or count-generator failures. Failed draws are never omitted
  from a passing result.

These are project plausibility criteria, not claims of prior calibration. All
three point-estimate gates must pass; if a 95% trajectory-bootstrap interval
straddles a threshold, label the result unresolved and repeat with 8,000 draws
using the same seeds/prefix, without changing thresholds.

For the same pilot along the fixed training-only standardized effort path,
report all metrics and require finite simulation. Its zero proportion is not
subject to the reference-effort 5% gate, because low-effort months deliberately
lower the count rate. Require q99.9 <= 50,000*exp(max(e)) as an explicit generous
envelope (not a calibrated probability bound). Report this alongside the tighter
reference gate so a permissive longitudinal threshold cannot conceal breadth.

Also audit a joint R=3 + iid + kernel case with actual sampled occupancy, both
at reference effort and along that path. Report its spread; extension admission
requires its own Stage 8 assessment, not a claim that the R=1 gate validates it.
Retain an e=3, highest-regime envelope as stress-only. A higher envelope under
an upward location correction is not itself evidence against that correction.

## Sensitivity and posterior work

Predeclare five candidates, changing one block at a time:

| Candidate | Level hyper-SD / s_m scale / upper bound | Dynamic amplitude multiplier |
|---|---|---|
| primary | 1 / 1 / 2 | 1 |
| level_tight | .75 / .75 / 1.5 | 1 |
| level_wide | 1.5 / 1.5 / 3 | 1 |
| dynamics_tight | 1 / 1 / 2 | .75 |
| dynamics_wide | 1 / 1 / 2 | 1.5 |

The dynamic multiplier scales both SD and bound of intensity/severity loadings,
iid/standalone-AR stationary SD, and kernel amplitudes (including their hyper-SD).
It leaves persistence, NB dispersion, regime shifts and the fallback level law
unchanged. Sensitivity candidates are stress comparisons, not a tournament
selected by recovery scores. Report failed prior gates as such.

Run the original 40 fixed-truth cases plus the three fallback fixtures under
the primary candidate. Use amplitude coordinates and dense mass adaptation,
whose exact-density checks and first-seed v1 diagnostics passed for factors and
latent AR. Keep the four-chain budgets, all convergence gates, retry seeds,
fail-fast policy, and artifact preservation unchanged. Run all four sensitivity
candidates on `factors_sparse`, `representative`, and `fallback_zero`, seed 7101,
with the same diagnostics. Compare predictive and covariance recovery, known
levels, fallback rate and scale-bound occupancy. Failed posteriors yield no
recovery evidence. Three fixed-truth seeds are not SBC; no general posterior
calibration claim is permitted. Report computational failures with denominators.

## KEV and manuscript

The v1 bounded KEV slope remains a separately pending Stage 8 choice. This
amendment neither widens it nor uses it to justify count priors. Stage 7's known
auxiliary coefficient fixture does not validate an estimated KEV elasticity.
Update the manuscript change record with the new laws and empirical anchor;
retain the v1 covariance/noise/loading changes as substantive prior revisions.

Reference workflow: [Stan prior-predictive checks](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html).
All numeric gates and candidate scales above are project decisions.
