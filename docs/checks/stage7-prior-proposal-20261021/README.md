# Prior proposal audit, 2026-10-21

This checks [the proposed amendment](../../amendments/STATE_SPACE_PRIORS_20261021.md),
not an activated inference model. The [JSON output](audit-final.json) binds the
proposal and script by SHA-256. Seed 72100; 20,000 parameter/state draws per
case. No real panel was read and no posterior was sampled.

| Case | Count draws | Count-generator failures | KEV-generator failures | Count log means >20 | Largest count |
|---|---:|---:|---:|---:|---:|
| Four categories, one factor | 80,000 | 0 | 0 | 0 | 12,156,646 |
| Ten categories, two factors | 200,000 | 0 | 0 | 0 | 120,040,522 |
| All components, effort 0 | 200,000 | 0 | 0 | 0 | 121,671,918 |
| All components, effort 3 | 200,000 | 0 | 0 | 4 | 1,228,341,383 |

There were 680,000 publication-count draws and another 680,000 KEV-count draws,
with no numerical simulation failure in this finite sample. The maximum
publication log mean was 21.40935. The all-component cases use the highest
ordered regime shift as an envelope, not simulated regime occupancy. Their
results are deliberately conservative stress cases, not the exact joint
longitudinal prior predictive distribution.

**The remaining spread matters.** Removing infinite moments does not make
these priors narrowly plausible. The retained Normal global level and broad
category offsets still allow enormous publication counts, as the maxima show.
These draws must remain visible in prior review. Do not call zero numerical
failures a scientific validation, and do not clip the four log means above 20.
The new prior does not cap counts or eta; it caps selected scale parameters.

Numerical integration gives `log E[exp(m)] = 6.5042857` and
`log E[exp(2*m)] = 31.7113618` for the category-level prior. These are below
their analytic variance-bound values 9 and 36. The broad second moment is
another reason not to rely on a small Monte Carlo sample to estimate tail
functionals. For the rank-one loading, the closed-form first rate multiplier
1.6718382009405384 agrees with quadrature to floating-point precision.

The stationary transition variance identity was evaluated at phi
-0.9999, -0.9, 0, 0.65 and 0.9999; this is an algebra check, not a sampled
transition/forecast implementation test. The full conditional dynamic-variance
bound is 6; the largest sampled value was below 4.71. The amendment contains
the mathematical finite-moment proof. Longitudinal prior checks, implementation
gradient/forecast checks, posterior diagnostics and recovery remain required.
