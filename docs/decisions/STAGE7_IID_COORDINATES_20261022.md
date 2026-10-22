# Conditional coordinates for the noise-separation experiments

Declared 2026-10-22 before replacement fits. The dense-count seed 7101 original
coordinate fits have BFMI 0.079 and 0.091 in their first two complete attempts;
the remaining schedule is retained. These failures prevent using their draws
to assess separation of latent iid noise and negative-binomial dispersion.
No recovery score is used for this computational decision.

For the isolated R=1, no-factor, no-kernel, fixed-level count experiments, retain
`u = tau*x`, `x ~ Normal(0,1)` and the exact NB likelihood. Define a Gaussian
working observation with residual `b=log(N+0.5)-m-e` and precision
`w=1/(1/(N+0.5)+1/psi)` on observed cells, zero on missing cells. Then set

    v = 1/(1 + tau^2*w)
    q = tau*w*b*v
    x = q + sqrt(v)*z

Use a Flat working variable z, the normalized standard-Normal density of x,
and the Jacobian `0.5*sum(log(v))`. This is an invertible change of coordinates
for every positive tau and psi, not a Gaussian replacement likelihood. It has
a finite limit as tau tends to zero and introduces no noise floor.

First compare complete old/new target densities, independent Gaussian algebra,
finite-difference derivatives, native/Numba fused targets, observed zeros and
missing values. If those checks pass, repeat dense seed 7101 under the unchanged
four-chain schedule and dense metric. Only if that fit is admitted, extend the
same coordinates to all three dense and all three sparse seeds. Keep the
original fits and record any interrupted or unstarted originals separately.
All replacement priors, synthetic observations/truths, retry seeds, budgets and
diagnostic thresholds are unchanged. There are still six separation cases,
not six additional independent datasets. If the replacement fails all budgets,
do not use its posterior to claim identification and retain the original gate.

Posterior separation remains an empirical question after convergence. The
pre-existing rule to omit iid latent noise from the initial factor pilot when
separation is unsupported remains in force. This coordinate check cannot
establish structural nonidentifiability or make the optional component valid.
