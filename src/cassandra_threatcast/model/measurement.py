"NumPyro likelihood functions for the three observation channels:"
from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def _soft_clip(x: jnp.ndarray, bound: float = 30.0) -> jnp.ndarray:
    "Smoothly saturate x into (-bound, bound), preserving a nonzero"
    return bound * jnp.tanh(x / bound)


def vulnerability_obs(
    lambda_kt: jnp.ndarray,     # (K, T) log-intensities
    e_t: jnp.ndarray,           # (T,) effort covariate
    psi_k: jnp.ndarray,         # (K,) NegBin dispersion per topic
    N_obs: jnp.ndarray | None,  # (K, T) observed counts or None
) -> jnp.ndarray:
    "NegBin likelihood for CVE counts."
    # Broadcast e_t across topics: e_t is (T,), lambda_kt is (K, T)
    log_mu_kt = lambda_kt + e_t[None, :]  # (K, T)

    # Soft-clamp before exp: guards against overflow (mu -> inf makes the...
    mu_kt = jnp.exp(_soft_clip(log_mu_kt))  # (K, T)

    # NegativeBinomial2 uses mean and concentration.
    concentration = psi_k[:, None] * jnp.ones_like(mu_kt)  # (K, T)

    numpyro.sample(
        "N_obs",
        dist.NegativeBinomial2(mean=mu_kt, concentration=concentration),
        obs=N_obs,
    )

    return log_mu_kt


def exploitation_obs(
    lambda_kt: jnp.ndarray,     # (K, T)
    alpha_k: jnp.ndarray,       # (K,) intercept
    beta_k: jnp.ndarray,        # (K,) slope
    varsigma_k: jnp.ndarray,    # (K,) obs noise std
    E_obs: jnp.ndarray | None,  # (K, T) observed mean EPSS in [0,1], or None
) -> None:
    "Normal likelihood on the logit of the mean EPSS score."
    logit_E_kt = alpha_k[:, None] + beta_k[:, None] * lambda_kt  # (K, T) mean
    sigma_kt = varsigma_k[:, None] * jnp.ones_like(logit_E_kt)   # (K, T)

    if E_obs is None:
        numpyro.sample("E_obs", dist.Normal(logit_E_kt, sigma_kt))
        return

    E = jnp.asarray(E_obs)
    obs_mask = ~jnp.isnan(E)
    # Clip observed probabilities before applying logit.
    E_clip = jnp.clip(jnp.where(obs_mask, E, 0.5), 1e-4, 1.0 - 1e-4)
    logit_E_obs = jnp.log(E_clip) - jnp.log1p(-E_clip)

    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample(
            "E_obs",
            dist.Normal(logit_E_kt, sigma_kt),
            obs=logit_E_obs,
        )


def severity_obs(
    zeta_kt: jnp.ndarray,       # (K, T) latent log-severity
    kappa_k: jnp.ndarray,       # (K,) measurement-noise std
    B_obs: jnp.ndarray | None,  # (K, T) observed mean-CVSS marks or None
) -> None:
    "LogNormal likelihood for the mean-CVSS severity marks B_kt."
    scale_kt = kappa_k[:, None] * jnp.ones_like(zeta_kt)  # (K, T)

    if B_obs is None:
        numpyro.sample("B_obs", dist.LogNormal(zeta_kt, scale_kt))
        return

    B = jnp.asarray(B_obs)
    obs_mask = ~jnp.isnan(B)
    # Replace missing / non-positive marks with a dummy positive value;...
    B_safe = jnp.clip(jnp.where(obs_mask, B, 1.0), 0.1, 10.0)

    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("B_obs", dist.LogNormal(zeta_kt, scale_kt), obs=B_safe)


def incident_obs(
    lambda_kt: jnp.ndarray,    # (K, T)
    M_skt: jnp.ndarray,        # (S, K, T) exposure map
    pi_st: jnp.ndarray,        # (S, T) sector baseline
    rho_s: jnp.ndarray,        # (S,) sector scaling
    D_obs: jnp.ndarray | None, # (S, T) observed 8-K counts or None
    dispersion: jnp.ndarray | None = None,  # (S,) NegBin concentration, or None
) -> None:
    "Likelihood for 8-K cybersecurity incident disclosures."
    # Broadcast exp(lambda_kt) against M_skt.
    exp_lambda = jnp.exp(_soft_clip(lambda_kt))[None, :, :]  # (1, K, T)

    # Weighted sum over topics: (S, T)
    weighted_sum = jnp.sum(M_skt * exp_lambda, axis=1)  # (S, T)

    # rho_s: (S,) → (S, 1)
    rate_st = rho_s[:, None] * weighted_sum + pi_st  # (S, T)

    # Guard against negative rates (numerical safety)
    rate_st = jnp.clip(rate_st, 1e-8)  # positional min (JAX dropped a_min kwarg)

    if dispersion is None:
        numpyro.sample("D_obs", dist.Poisson(rate_st), obs=D_obs)
    else:
        concentration = dispersion[:, None] * jnp.ones_like(rate_st)  # (S, T)
        numpyro.sample(
            "D_obs",
            dist.NegativeBinomial2(mean=rate_st, concentration=concentration),
            obs=D_obs,
        )
