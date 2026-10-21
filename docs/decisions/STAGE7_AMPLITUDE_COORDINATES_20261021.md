# Stage 7: center the dynamic amplitudes without changing their priors

Recorded before any fit in these coordinates, 2026-10-21. The first amended
sparse-factor attempts had minimum BFMI .245 and .167, and the first latent-AR
attempt .168. Thus eliminating infinite prior moments has not ensured good
energy mixing. Failed-posterior diagnostic exploration found energy associated
with factor amplitude, persistence and dispersion. This motivates an exact
coordinate change, not a new prior or relaxed admission threshold.

For factor j, let `a_j=Gamma[j,j]>0` and sample `q[t,j]=a_j*f[t,j]`:

```
q[0,j] ~ Normal(0,a_j)
q[t,j] ~ Normal(phi_j*q[t-1,j], a_j*sqrt(1-phi_j**2))
f[t,j] = q[t,j]/a_j
```

All priors on Gamma and phi remain those of the activated amendment. Keeping
the normalized initial/transition densities supplies the entire change-of-
variables term: `log p(q|a,phi) = log p(f|phi) - (T+1)*sum(log a)`.
The leading category directly measures q for its leading factor, which may
reduce the likelihood's multiplicative coupling. Other categories retain the
same loadings and reconstructed unit-variance factor states. The severity
factor uses the identical transformation. For a standalone AR residual, sample
u directly under stationary SD tau and innovation SD `tau*sqrt(1-phi_u**2)`.
The optional iid residual stays in its original noncentered coordinates.

This is an application of centered/noncentered reparameterization, whose value
depends on the information in the observations; it does not promise improvement.
[Stan's efficiency guidance](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html)
describes both choices and the need to preserve the target distribution.
The particular positive-loading transformation here is derived above.

Validate density/Jacobian equality and gradients before fitting. Use the same
first seed 7101 and frozen three-budget schedule for diagnostic comparisons
of sparse factors and latent AR, with all failed attempts retained separately.
Choose coordinates using diagnostic admission and sampling cost only, never
recovery accuracy or real-data scores. An adopted coordinate choice must then
be evaluated on all three seeds; do not substitute an easier seed. Existing
v4 fits continue their recorded retries. No observation, generating truth,
prior bound, chain count, budget or held-out rule changes.
