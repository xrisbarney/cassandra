"""
measurement.py
==============
NumPyro likelihood functions for the three observation channels:
  1. CVE counts     (NegBin)
  2. EPSS scores    (Normal on logit scale)
  3. 8-K incidents  (Poisson)
"""
from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def vulnerability_obs(
    lambda_kt: jnp.ndarray,     # (K, T) log-intensities
    e_t: jnp.ndarray,           # (T,) effort covariate
    psi_k: jnp.ndarray,         # (K,) NegBin dispersion per topic
    N_obs: jnp.ndarray | None,  # (K, T) observed counts or None
) -> jnp.ndarray:
    """
    NegBin likelihood for CVE counts.

    mu_kt  = exp(lambda_kt + e_t)      (K, T)
    N_kt   ~ NegativeBinomial(mu_kt, psi_k)

    Registers numpyro.sample("N_obs", ...) conditioned on N_obs if provided.

    Returns log_mu_kt of shape (K, T).
    """
    # Broadcast e_t across topics: e_t is (T,), lambda_kt is (K, T)
    log_mu_kt = lambda_kt + e_t[None, :]  # (K, T)

    mu_kt = jnp.exp(log_mu_kt)  # (K, T)

    # NumPyro's NegativeBinomial2 uses mean + concentration parameterisation.
    # concentration = psi_k (total_count in the overdispersion sense).
    # psi_k shape (K,) → broadcast to (K, T)
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
    E_obs: jnp.ndarray | None,  # (K, T) observed logit-EPSS or None
) -> None:
    """
    Normal likelihood on logit(E_kt).

    logit_E_kt  = alpha_k + beta_k * lambda_kt   (K, T)
    E_kt        ~ Normal(logit_E_kt, varsigma_k)
    """
    # alpha_k: (K,) → (K, 1), beta_k: (K,) → (K, 1)
    logit_E_kt = alpha_k[:, None] + beta_k[:, None] * lambda_kt  # (K, T)
    sigma_kt = varsigma_k[:, None] * jnp.ones_like(logit_E_kt)   # (K, T)

    numpyro.sample(
        "E_obs",
        dist.Normal(logit_E_kt, sigma_kt),
        obs=E_obs,
    )


def severity_obs(
    zeta_kt: jnp.ndarray,       # (K, T) latent log-severity
    kappa_k: jnp.ndarray,       # (K,) measurement-noise std
    B_obs: jnp.ndarray | None,  # (K, T) observed mean-CVSS marks or None
) -> None:
    """
    LogNormal likelihood for the mean-CVSS severity marks B_kt.

    log B_kt ~ Normal(zeta_kt, kappa_k),  so sigma_kt = exp(zeta_kt) is the
    latent severity on the CVSS scale.

    Empty (topic, month) cells carry NaN in B (no CVEs that month); those
    entries are masked out of the likelihood so they contribute nothing.
    """
    scale_kt = kappa_k[:, None] * jnp.ones_like(zeta_kt)  # (K, T)

    if B_obs is None:
        numpyro.sample("B_obs", dist.LogNormal(zeta_kt, scale_kt))
        return

    B = jnp.asarray(B_obs)
    obs_mask = ~jnp.isnan(B)
    # Replace missing / non-positive marks with a dummy positive value; the mask
    # zeroes their contribution to the log-density.
    B_safe = jnp.clip(jnp.where(obs_mask, B, 1.0), 0.1, 10.0)

    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("B_obs", dist.LogNormal(zeta_kt, scale_kt), obs=B_safe)


def incident_obs(
    lambda_kt: jnp.ndarray,    # (K, T)
    M_skt: jnp.ndarray,        # (S, K, T) exposure map
    pi_st: jnp.ndarray,        # (S, T) sector baseline
    rho_s: jnp.ndarray,        # (S,) sector scaling
    D_obs: jnp.ndarray | None, # (S, T) observed 8-K counts or None
) -> None:
    """
    Poisson likelihood for 8-K cybersecurity incident disclosures.

    rate_st = rho_s * sum_k M_skt * exp(lambda_kt) + pi_st
    D_st    ~ Poisson(rate_st)

    Parameters
    ----------
    lambda_kt : (K, T) log-intensities.
    M_skt     : (S, K, T) sector-topic-time exposure weights.
    pi_st     : (S, T) sector-specific baseline disclosure rate.
    rho_s     : (S,) sector-level scaling of cyber-to-disclosure elasticity.
    D_obs     : (S, T) observed incident counts or None.
    """
    # exp(lambda_kt): (K, T) → (1, K, T) for broadcasting with M_skt (S, K, T)
    exp_lambda = jnp.exp(lambda_kt)[None, :, :]  # (1, K, T)

    # Weighted sum over topics: (S, T)
    weighted_sum = jnp.sum(M_skt * exp_lambda, axis=1)  # (S, T)

    # rho_s: (S,) → (S, 1)
    rate_st = rho_s[:, None] * weighted_sum + pi_st  # (S, T)

    # Guard against negative rates (numerical safety)
    rate_st = jnp.clip(rate_st, 1e-8)  # positional min (JAX dropped a_min kwarg)

    numpyro.sample(
        "D_obs",
        dist.Poisson(rate_st),
        obs=D_obs,
    )
