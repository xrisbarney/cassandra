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


def _soft_clip(x: jnp.ndarray, bound: float = 30.0) -> jnp.ndarray:
    """Smoothly saturate x into (-bound, bound), preserving a nonzero
    gradient everywhere. A hard jnp.clip has a zero-gradient plateau beyond
    its bounds — if an HMC trajectory's momentum ever carries a differentiable
    quantity past that plateau, there is no gradient signal to pull it back,
    which can destabilize the leapfrog integrator and collapse NUTS's
    step-size adaptation. tanh saturates smoothly instead, with a gradient
    that decays but never vanishes."""
    return bound * jnp.tanh(x / bound)


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

    # Soft-clamp before exp: guards against overflow (mu -> inf makes the NegBin
    # rate concentration/mu collapse to 0, an invalid Gamma rate) while keeping a
    # nonzero gradient everywhere. exp(30) is already an implausibly large
    # monthly count, so this never bites real, well-behaved data.
    mu_kt = jnp.exp(_soft_clip(log_mu_kt))  # (K, T)

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
    E_obs: jnp.ndarray | None,  # (K, T) observed mean EPSS in [0,1], or None
) -> None:
    """
    Normal likelihood on the logit of the mean EPSS score.

    logit_E_kt   = alpha_k + beta_k * lambda_kt          (K, T)  -- the mean
    logit(E_kt)  ~ Normal(logit_E_kt, varsigma_k)

    The observed EPSS marks are probabilities in [0, 1]; they are logit-
    transformed here to match the model's (unbounded) logit-scale mean. Empty
    (topic, month) cells carry NaN and are masked out of the likelihood.
    """
    logit_E_kt = alpha_k[:, None] + beta_k[:, None] * lambda_kt  # (K, T) mean
    sigma_kt = varsigma_k[:, None] * jnp.ones_like(logit_E_kt)   # (K, T)

    if E_obs is None:
        numpyro.sample("E_obs", dist.Normal(logit_E_kt, sigma_kt))
        return

    E = jnp.asarray(E_obs)
    obs_mask = ~jnp.isnan(E)
    # logit of the observed probability, clipped away from 0/1 to stay finite.
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
    dispersion: jnp.ndarray | None = None,  # (S,) NegBin concentration, or None
) -> None:
    """
    Likelihood for 8-K cybersecurity incident disclosures.

    rate_st = rho_s * sum_k M_skt * exp(lambda_kt) + pi_st

    * dispersion is None  -> D_st ~ Poisson(rate_st)                (paper-native)
    * dispersion given     -> D_st ~ NegBinomial2(rate_st, dispersion)  (enhanced;
      overdispersed, since mandatory 8-K disclosure counts are bursty)

    Parameters
    ----------
    lambda_kt  : (K, T) log-intensities.
    M_skt      : (S, K, T) sector-topic-time exposure weights.
    pi_st      : (S, T) sector-specific baseline disclosure rate.
    rho_s      : (S,) sector-level scaling of cyber-to-disclosure elasticity.
    D_obs      : (S, T) observed incident counts or None.
    dispersion : (S,) per-sector NegBin concentration, or None for Poisson.
    """
    # exp(lambda_kt): (K, T) → (1, K, T) for broadcasting with M_skt (S, K, T)
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
