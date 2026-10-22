# Exact working sign anchor for the proxy factor

Declared 2026-10-22 before implementation and replacement fits. The
[mode audit](../checks/stage7-proxy-mode-20261022/audit.json) shows that the
numerically admitted proxy seed 7102 retry stayed in one orientation. An earlier
attempt found another orientation, but its inter-chain split is not a posterior
mode-weight estimate. R-hat within a common region is insufficient to resolve
this finding. Preserve both attempts and treat their recovery as mode-conditional.

## Probability-law identity

Restrict this change to the revised, one-factor, conditionally whitened count
submodel. Each loading has the same symmetric truncated-Normal density g on
(-b,b), except the canonical first loading, whose positive restriction has
density 2g on (0,b). The zero-mean stationary factor law is invariant under
reflection of its entire trajectory. There are no above-diagonal zeros to move
in a one-factor loading column.

Let working loadings B instead have their positive restriction at row j, with
all other rows symmetric. For s=sign(B[0]), map to canonical variables by

    Gamma = s*B
    f = s*F_working

The canonical first loading is positive, and Gamma*f equals B*F_working. The
map is bijective except at measure-zero boundaries: its inverse uses
s=sign(Gamma[j]). Its absolute Jacobian is one on each region. Both prior
densities are 2 times the product of the same symmetric loading densities,
and the factor prior is unchanged by reflection. Thus the observation target,
priors and canonical posterior are identical. This is not a category reorder,
a new positive-correlation restriction, or relabeling of trapped chains.

Evaluate the likelihood, state prior and Gaussian-coordinate Jacobian in the
smooth working variables. Apply s only to stored canonical loadings and states,
so forecasts retain the original positive-first-loading convention. A working
first loading can cross zero without moving the other well-identified loadings
through zero at the same time. Do not put the discontinuous canonical sign map
inside the sampling target. Keep the original triangular constraint and priors
for all other models, including the rank-two cases.

## Verification and execution

Use working anchor j=3 (zero-based, the fourth synthetic category), whose loading
is large in both documented orientations. This choice uses computational
diagnostics, not generating truth or recovery scores. Fix it for all three
proxy seeds. Verify complete canonical/working density equality on both sides
of zero, observable and terminal-state reconstruction, native/fused gradients,
and finite differences through the zero crossing before fitting.

Start with the diagnosed seed 7102, preserving priors, data, dense metric, seeds,
four chains, budgets and numerical gates. Inspect retained orientation changes,
per-chain negative-orientation frequencies and indicator diagnostics; do not
declare the issue resolved solely because ordinary R-hat passes. If the verified
chart yields an admitted fit that explores the documented regions, apply it
unchanged to seeds 7101 and 7103. Otherwise investigate before expansion.
All original fits remain available. No known-mode posterior weights are inferred
from how many old chains happened to initialize in each region.

The count-derived effort decomposition remains physically unidentified even if
this computational issue is resolved. Rebuild its recovery analysis from the
replacement posteriors before presenting those numbers as marginal posterior
results. No real-data or held-out scores enter this decision.
