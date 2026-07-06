"""
full.py
=======
Combined NumPyro model: latent dynamics + all three measurement likelihoods,
plus predictive utilities.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.primitives import deterministic

from cassandra_threatcast.model.measurement import vulnerability_obs, exploitation_obs, incident_obs
from cassandra_threatcast.model.economic import (
    DamageFunctionParams,
    damage_function,
    leontief_propagation,
)


# ---------------------------------------------------------------------------
# Full combined model
# ---------------------------------------------------------------------------

def full_model(data: dict, config: dict) -> None:
    """
    Combined NumPyro model: latent dynamics + all three measurement likelihoods.

    data keys
    ---------
    N      : (K, T) observed CVE counts, or None
    E      : (K, T) observed logit-EPSS scores, or None
    D      : (S, T) observed 8-K incident counts, or None
    M_skt  : (S, K, T) exposure map
    e_t    : (T,) effort covariate
    x_s    : (S,) gross output by sector
    Lambda_L : (S, S) Leontief inverse

    config keys (nested under "model")
    -----------------------------------
    K, S, r, R, vi_rank
    """
    model_cfg = config.get("model", config)
    K: int = int(model_cfg["K"])
    S: int = int(model_cfg["S"])
    r: int = int(model_cfg["r"])
    R: int = int(model_cfg["R"])

    T: int = int(data["e_t"].shape[0])

    e_t = jnp.asarray(data["e_t"])          # (T,)
    M_skt = jnp.asarray(data["M_skt"])      # (S, K, T)

    # ------------------------------------------------------------------ #
    # 1. Latent dynamics priors
    # ------------------------------------------------------------------ #
    mu_r = numpyro.sample(
        "mu_r",
        dist.Normal(jnp.zeros((R, K)), 2.0 * jnp.ones((R, K))),
    )  # (R, K)

    Gamma = numpyro.sample(
        "Gamma",
        dist.Normal(jnp.zeros((K, r)), jnp.ones((K, r))),
    )  # (K, r)

    Phi_raw = numpyro.sample(
        "Phi_raw",
        dist.Normal(jnp.zeros((R, r, r)), 0.3 * jnp.ones((R, r, r))),
    )
    tril_mask = jnp.tril(jnp.ones((r, r)))
    Phi_r = deterministic("Phi_r", Phi_raw * tril_mask[None, :, :])  # (R, r, r)

    Q_r = numpyro.sample(
        "Q_r",
        dist.HalfNormal(0.5 * jnp.ones((R, r))),
    )  # (R, r)

    Pi = numpyro.sample(
        "Pi",
        dist.Dirichlet(2.0 * jnp.ones((R, R))),
    )  # (R, R)

    tau_k = numpyro.sample(
        "tau_k",
        dist.HalfNormal(0.5 * jnp.ones(K)),
    )  # (K,)

    f_init = numpyro.sample("f_init", dist.Normal(jnp.zeros(r), jnp.ones(r)))
    z_init = numpyro.sample("z_init", dist.Categorical(probs=jnp.ones(R) / R))

    def _latent_step(carry, _):
        f_prev, z_prev = carry  # (r,), ()

        z_t = numpyro.sample("z_t", dist.Categorical(probs=Pi[z_prev]))
        eps = numpyro.sample("eps_f", dist.Normal(jnp.zeros(r), jnp.ones(r)))

        Phi_t = Phi_r[z_t]           # (r, r)
        Q_chol_t = jnp.diag(Q_r[z_t])  # (r, r)
        f_t = Phi_t @ f_prev + Q_chol_t @ eps  # (r,)

        mean_eta = mu_r[z_t] + f_t @ Gamma.T  # (K,)
        eta_noise = numpyro.sample("eta_noise", dist.Normal(jnp.zeros(K), tau_k))
        eta_t = mean_eta + eta_noise  # (K,)

        return (f_t, z_t), (f_t, z_t, eta_t)

    _, (f_seq, z_seq, eta_seq) = scan(
        _latent_step,
        (f_init, z_init),
        xs=None,
        length=T,
    )
    # f_seq: (T, r), z_seq: (T,), eta_seq: (T, K)

    deterministic("f_t", f_seq)
    deterministic("z_t", z_seq)
    deterministic("eta_t", eta_seq)

    # lambda_kt: (K, T) â€” transpose from (T, K)
    lambda_kt = eta_seq.T  # (K, T)

    # ------------------------------------------------------------------ #
    # 2. Measurement parameters
    # ------------------------------------------------------------------ #
    psi_k = numpyro.sample(
        "psi_k",
        dist.HalfNormal(10.0 * jnp.ones(K)),
    )  # (K,) NegBin dispersion

    alpha_k = numpyro.sample(
        "alpha_k",
        dist.Normal(jnp.zeros(K), 2.0 * jnp.ones(K)),
    )  # (K,) EPSS intercept

    beta_k = numpyro.sample(
        "beta_k",
        dist.Normal(jnp.zeros(K), jnp.ones(K)),
    )  # (K,) EPSS slope

    varsigma_k = numpyro.sample(
        "varsigma_k",
        dist.HalfNormal(jnp.ones(K)),
    )  # (K,) EPSS obs noise

    rho_s = numpyro.sample(
        "rho_s",
        dist.HalfNormal(jnp.ones(S)),
    )  # (S,) sector scaling

    pi_st = numpyro.sample(
        "pi_st",
        dist.HalfNormal(0.5 * jnp.ones((S, T))),
    )  # (S, T) sector baseline

    # ------------------------------------------------------------------ #
    # 3. Observation likelihoods
    # ------------------------------------------------------------------ #
    N_obs = jnp.asarray(data["N"]) if data.get("N") is not None else None
    vulnerability_obs(lambda_kt, e_t, psi_k, N_obs)

    E_obs = jnp.asarray(data["E"]) if data.get("E") is not None else None
    exploitation_obs(lambda_kt, alpha_k, beta_k, varsigma_k, E_obs)

    D_obs = jnp.asarray(data["D"]) if data.get("D") is not None else None
    incident_obs(lambda_kt, M_skt, pi_st, rho_s, D_obs)


# ---------------------------------------------------------------------------
# Predictive function
# ---------------------------------------------------------------------------

def predict(
    posterior_samples: dict,
    data: dict,
    horizon: int,
    Lambda_L: np.ndarray,
    x_s: np.ndarray,
    M_future: np.ndarray,        # (S, K, horizon)
    damage_params: "DamageFunctionParams",
) -> dict:
    """
    Generate h-step-ahead predictive draws.

    For each posterior sample:
      1. Extend factor dynamics h steps forward
      2. Sample future eta (log-intensities)
      3. Sample future N, E, D from measurement models
      4. Compute g_s via damage_function
      5. Compute ell via leontief_propagation

    Returns
    -------
    dict with keys:
        lambda_pred : (n_samples, K, horizon)
        N_pred      : (n_samples, K, horizon)
        D_pred      : (n_samples, S, horizon)
        ell_pred    : (n_samples, S, horizon)
        ell_agg     : (n_samples, horizon)
    """
    rng = np.random.default_rng(seed=42)

    n_samples = next(iter(posterior_samples.values())).shape[0]
    K = posterior_samples["tau_k"].shape[-1]
    S = posterior_samples["rho_s"].shape[-1]
    r = posterior_samples["f_init"].shape[-1]
    R = posterior_samples["mu_r"].shape[-2]

    lambda_pred = np.zeros((n_samples, K, horizon))
    N_pred = np.zeros((n_samples, K, horizon))
    D_pred = np.zeros((n_samples, S, horizon))
    ell_pred = np.zeros((n_samples, S, horizon))
    ell_agg = np.zeros((n_samples, horizon))

    Lambda_L_j = jnp.asarray(Lambda_L)
    x_s_j = jnp.asarray(x_s)
    M_future_j = jnp.asarray(M_future)  # (S, K, horizon)

    for i in range(n_samples):
        # Extract parameters for sample i
        mu_r_i = posterior_samples["mu_r"][i]       # (R, K)
        Gamma_i = posterior_samples["Gamma"][i]      # (K, r)
        Phi_r_i = posterior_samples["Phi_r"][i]      # (R, r, r)
        Q_r_i = posterior_samples["Q_r"][i]          # (R, r)
        Pi_i = posterior_samples["Pi"][i]             # (R, R)
        tau_k_i = posterior_samples["tau_k"][i]      # (K,)
        psi_k_i = posterior_samples["psi_k"][i]      # (K,)
        alpha_k_i = posterior_samples["alpha_k"][i]  # (K,)
        beta_k_i = posterior_samples["beta_k"][i]    # (K,)
        varsigma_k_i = posterior_samples["varsigma_k"][i]  # (K,)
        rho_s_i = posterior_samples["rho_s"][i]      # (S,)

        # Last factor and regime from posterior
        f_last = np.array(posterior_samples["f_t"][i, -1, :])   # (r,)
        z_last = int(posterior_samples["z_t"][i, -1])

        for h in range(horizon):
            # Regime transition
            z_probs = np.array(Pi_i[z_last])
            z_curr = int(rng.choice(R, p=z_probs / z_probs.sum()))

            # Factor update
            eps = rng.standard_normal(r)
            Q_chol = np.diag(np.array(Q_r_i[z_curr]))
            Phi = np.array(Phi_r_i[z_curr])
            f_curr = Phi @ f_last + Q_chol @ eps

            # Log-intensity
            mean_eta = np.array(mu_r_i[z_curr]) + f_curr @ np.array(Gamma_i).T  # (K,)
            eta_noise = rng.standard_normal(K) * np.array(tau_k_i)
            eta_h = mean_eta + eta_noise  # (K,)

            lambda_pred[i, :, h] = eta_h

            # CVE counts
            mu_kt_h = np.exp(eta_h)
            psi_arr = np.array(psi_k_i)
            for k in range(K):
                p_nb = psi_arr[k] / (psi_arr[k] + mu_kt_h[k] + 1e-12)
                N_pred[i, k, h] = rng.negative_binomial(psi_arr[k], p_nb)

            # Incident counts
            M_h = np.array(M_future_j[:, :, h])  # (S, K)
            exp_lam = np.exp(eta_h)               # (K,)
            rho_arr = np.array(rho_s_i)           # (S,)
            rate_s = rho_arr * (M_h @ exp_lam)    # (S,)
            rate_s = np.clip(rate_s, 1e-8, None)
            D_pred[i, :, h] = rng.poisson(rate_s)

            # Economic losses
            shock_load = jnp.asarray(rate_s / (np.array(x_s) + 1e-12))
            g_s = np.array(damage_function(shock_load, damage_params))
            d_s, ell = leontief_propagation(
                jnp.asarray(g_s), x_s_j, Lambda_L_j
            )
            ell_pred[i, :, h] = np.array(ell)
            ell_agg[i, h] = float(jnp.sum(ell))

            f_last = f_curr
            z_last = z_curr

    return {
        "lambda_pred": lambda_pred,
        "N_pred": N_pred,
        "D_pred": D_pred,
        "ell_pred": ell_pred,
        "ell_agg": ell_agg,
    }
