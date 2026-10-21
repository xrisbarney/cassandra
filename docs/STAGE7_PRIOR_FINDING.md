# Stage 7: frozen prior predictive tails

This finding concerns the proposed state-space model's frozen priors, not the
Stage 6 independent control that was replaced by amendment. The prior audit
does not alter either model or any historical score.

The finite [stress audit](checks/stage7-numerics-20261021/prior_predictive.json)
drew 20,000 parameter/state samples for four categories at a stationary month,
using one factor, no iid residual and no effort offset. Among 80,000 category
draws, 64 log rates exceeded the specification's diagnostic flag of 20 and 16
NB count simulations failed the integer-range limit. The largest log rate was
142.1073. Nothing was clipped, replaced or silently redrawn. The generator saved
the indices and log rates of the failures.

There is also an analytic reason to review these priors. For the leading
category, the loading G has HalfNormal(1) prior and a stationary factor F has
variance v=1/(1-phi^2), at least one. Conditional on G and phi,

`E[exp(G*F) | G,phi] = exp(v*G^2/2)`.

Integrating against the half-normal density requires integrating
`exp((v-1)*G^2/2)` on the positive half-line, which diverges even at v=1.
Thus this factor contribution has an infinite prior mean on the rate scale.
This is not only a numerical effect near the AR stability boundary.

The hierarchical level prior has a separate problem for the same moment.
With `m=group_mean+s*Z`, `Z~Normal(0,1)` and `s~HalfNormal(2)`, integrating Z
gives `exp(s^2/2)`; the prior density of s contributes `exp(-s^2/8)`.
Their product is nonintegrable. Both calculations follow directly from the
frozen specification, without fitting or examining real outcomes.

These are proper distributions with an infinite rate-scale first moment.
Properness alone does not make a prior predictive mean useful, and a finite
Monte Carlo mean is not evidence that the expectation exists. The fixed-truth
recovery experiments can still test the likelihood and posterior computation
in their declared conditions. They do not validate these priors globally or
establish existence of all posterior predictive moments.

**Carry-forward gate:** before a real-data proposed-model pilot, document and
validate a substantive prior amendment addressing hierarchical level scales
and stationary factor-loading scales jointly. Merely bounding phi away from
one does not fix the stated HalfNormal(1) loading calculation. Recheck all
enabled components, including iid residuals and the kernel, rather than fixing
one tail source and assuming the whole model is resolved. Retain this frozen-
prior audit and rerun appropriate prior-predictive and recovery checks for the
amended model. Do not change priors to obtain better held-out scores.
