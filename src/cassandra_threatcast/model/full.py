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

from cassandra_threatcast.model.measurement import (
    vulnerability_obs,
    exploitation_obs,
    incident_obs,
    severity_obs,
)
from cassandra_threatcast.model.economic import (
    DamageFunctionParams,
    damage_function,
    leontief_propagation,
)


# ---------------------------------------------------------------------------
# Identifiability helper
# ---------------------------------------------------------------------------
def _positive_lower_triangular(raw: jnp.ndarray, d: int) -> jnp.ndarray:
    """Impose the standard factor-model identification constraint on a (K, d)
    loading matrix: the top d x d block is lower-triangular with a positive
    diagonal (rows below are free). This removes the rotation/scale/sign
    indeterminacy between the loadings and the latent factors that otherwise
    leaves the posterior non-identified (flat ridges -> terrible HMC mixing)."""
    lower = jnp.tril(jnp.ones((d, d)))
    top = raw[:d] * lower
    top = top.at[jnp.diag_indices(d)].set(jax.nn.softplus(jnp.diagonal(raw[:d])))
    return jnp.concatenate([top, raw[d:]], axis=0)


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

    # Enhanced mode (opt-in, --enhanced-mode): heavy-tailed latent innovations
    # (Student-t) and an overdispersed Negative-Binomial incident channel.
    # Default is the paper-native model (Gaussian innovations, Poisson incidents).
    enh_cfg = config.get("enhanced", {})
    enhanced: bool = bool(enh_cfg.get("enabled", False))
    student_t_df: float = float(enh_cfg.get("student_t_df", 4.0))

    def _innovation(name, loc, scale):
        """Latent innovation: Student-t in enhanced mode, else Normal."""
        if enhanced:
            return numpyro.sample(name, dist.StudentT(student_t_df, loc, scale))
        return numpyro.sample(name, dist.Normal(loc, scale))

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

    Gamma_raw = numpyro.sample(
        "Gamma_raw",
        dist.Normal(jnp.zeros((K, r)), jnp.ones((K, r))),
    )  # (K, r)
    # Identified loadings: positive-lower-triangular top block (see helper).
    Gamma = deterministic("Gamma", _positive_lower_triangular(Gamma_raw, r))  # (K, r)

    Phi_raw = numpyro.sample(
        "Phi_raw",
        dist.Normal(jnp.zeros((R, r, r)), 0.3 * jnp.ones((R, r, r))),
    )
    tril_mask = jnp.tril(jnp.ones((r, r)))
    # tanh bounds every entry (hence the diagonal = eigenvalues of a triangular
    # matrix) to (-1, 1), guaranteeing a stationary AR so factors cannot explode
    # over long series (T can be ~180 months).
    Phi_r = deterministic("Phi_r", jnp.tanh(Phi_raw) * tril_mask[None, :, :])  # (R, r, r)

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

    # ------------------------------------------------------------------ #
    # 1b. Latent severity process priors (log sigma_{k,t}), analogous to the
    #     intensity: regime-dependent level + lower-dimensional AR factors.
    # ------------------------------------------------------------------ #
    r_sig: int = int(model_cfg.get("r_sigma", r))
    tril_sig = jnp.tril(jnp.ones((r_sig, r_sig)))

    nu_r = numpyro.sample(
        "nu_r",
        dist.Normal(1.5 * jnp.ones((R, K)), 1.0 * jnp.ones((R, K))),
    )  # (R, K) regime-dependent severity level (log-CVSS scale)
    Psi_raw = numpyro.sample(
        "Psi_raw",
        dist.Normal(jnp.zeros((K, r_sig)), jnp.ones((K, r_sig))),
    )  # (K, r_sig)
    # Same identifiability constraint for the severity loadings.
    Psi = deterministic("Psi", _positive_lower_triangular(Psi_raw, r_sig))  # (K, r_sig)
    A_h_raw = numpyro.sample(
        "A_h_raw",
        dist.Normal(jnp.zeros((r_sig, r_sig)), 0.3 * jnp.ones((r_sig, r_sig))),
    )
    A_h = deterministic("A_h", jnp.tanh(A_h_raw) * tril_sig)  # (r_sig, r_sig) stationary
    Q_h = numpyro.sample("Q_h", dist.HalfNormal(0.5 * jnp.ones(r_sig)))   # (r_sig,)
    omega_k = numpyro.sample("omega_k", dist.HalfNormal(0.5 * jnp.ones(K)))  # (K,) idiosyncratic
    kappa_k = numpyro.sample("kappa_k", dist.HalfNormal(0.5 * jnp.ones(K)))  # (K,) severity meas. noise

    f_init = numpyro.sample("f_init", dist.Normal(jnp.zeros(r), jnp.ones(r)))
    z_init = numpyro.sample("z_init", dist.Categorical(probs=jnp.ones(R) / R))
    h_init = numpyro.sample("h_init", dist.Normal(jnp.zeros(r_sig), jnp.ones(r_sig)))

    def _latent_step(carry, _):
        f_prev, z_prev, h_prev = carry  # (r,), (), (r_sig,)

        z_t = numpyro.sample("z_t", dist.Categorical(probs=Pi[z_prev]))
        eps = _innovation("eps_f", jnp.zeros(r), jnp.ones(r))

        Phi_t = Phi_r[z_t]           # (r, r)
        Q_chol_t = jnp.diag(Q_r[z_t])  # (r, r)
        f_t = Phi_t @ f_prev + Q_chol_t @ eps  # (r,)

        mean_eta = mu_r[z_t] + f_t @ Gamma.T  # (K,)
        # Non-centered idiosyncratic noise (eps ~ N(0,1), scaled by tau_k) to
        # avoid the funnel geometry of sampling eta_noise ~ N(0, tau_k) directly.
        eps_eta = _innovation("eps_eta", jnp.zeros(K), jnp.ones(K))
        eta_t = mean_eta + tau_k * eps_eta  # (K,)

        # Severity factor evolution and log-severity (shares the regime z_t).
        xi_h = _innovation("xi_h", jnp.zeros(r_sig), jnp.ones(r_sig))
        h_t = A_h @ h_prev + jnp.diag(Q_h) @ xi_h  # (r_sig,)
        eps_v = _innovation("eps_v", jnp.zeros(K), jnp.ones(K))  # non-centered
        zeta_t = nu_r[z_t] + h_t @ Psi.T + omega_k * eps_v  # (K,)

        return (f_t, z_t, h_t), (f_t, z_t, eta_t, h_t, zeta_t)

    _, (f_seq, z_seq, eta_seq, h_seq, zeta_seq) = scan(
        _latent_step,
        (f_init, z_init, h_init),
        xs=None,
        length=T,
    )
    # f_seq: (T, r), z_seq: (T,), eta_seq: (T, K), h_seq: (T, r_sig), zeta_seq: (T, K)

    deterministic("f_t", f_seq)
    # "z_t" is already recorded by scan as a sampled (T,) site; re-declaring it
    # as deterministic duplicates the site name and crashes the model.
    deterministic("eta_t", eta_seq)
    deterministic("h_t", h_seq)
    deterministic("zeta_t", zeta_seq)

    # transpose (T, K) -> (K, T)
    lambda_kt = eta_seq.T   # (K, T) log-intensities
    zeta_kt = zeta_seq.T    # (K, T) log-severity

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

    # Sector-specific baseline disclosure rate. One parameter per sector (S),
    # constant over time, rather than one per sector-month (S*T ~ 2000 nuisance
    # parameters). The incident channel is ~99% zeros, so per-month baselines
    # were pinned against zero and destroyed the sampling geometry (step size
    # collapse). Broadcast over T when forming the Poisson rate.
    pi_s = numpyro.sample(
        "pi_s",
        dist.HalfNormal(0.5 * jnp.ones(S)),
    )  # (S,) sector baseline

    # ------------------------------------------------------------------ #
    # 3. Observation likelihoods
    # ------------------------------------------------------------------ #
    N_obs = jnp.asarray(data["N"]) if data.get("N") is not None else None
    vulnerability_obs(lambda_kt, e_t, psi_k, N_obs)

    E_obs = jnp.asarray(data["E"]) if data.get("E") is not None else None
    exploitation_obs(lambda_kt, alpha_k, beta_k, varsigma_k, E_obs)

    D_obs = jnp.asarray(data["D"]) if data.get("D") is not None else None
    pi_st = pi_s[:, None]  # (S, 1) broadcasts over months
    if enhanced:
        phi_D = numpyro.sample("phi_D", dist.HalfNormal(10.0 * jnp.ones(S)))  # (S,) NegBin dispersion
        incident_obs(lambda_kt, M_skt, pi_st, rho_s, D_obs, dispersion=phi_D)
    else:
        incident_obs(lambda_kt, M_skt, pi_st, rho_s, D_obs)

    B_obs = jnp.asarray(data["B"]) if data.get("B") is not None else None
    severity_obs(zeta_kt, kappa_k, B_obs)


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
    enhanced: bool = False,
    student_t_df: float = 4.0,
    start_t: int | None = None,  # forecast from this time index (default: series end)
) -> dict:
    """
    Generate h-step-ahead predictive draws.

    For each posterior sample:
      1. Extend factor dynamics h steps forward
      2. Sample future eta (log-intensities)
      3. Sample future N, E, D from measurement models
      4. Compute g_s via damage_function
      5. Compute ell via leontief_propagation

    Set ``enhanced=True`` (matching how the model was fit) to draw latent
    innovations from Student-t and incident counts from Negative Binomial.

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

    def _draw(size):
        """Innovation draw: Student-t in enhanced mode, else standard normal."""
        if enhanced:
            return rng.standard_t(student_t_df, size)
        return rng.standard_normal(size)

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

    r_sig = posterior_samples["h_init"].shape[-1]

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

        # Severity-process parameters for sample i
        nu_r_i = posterior_samples["nu_r"][i]        # (R, K)
        Psi_i = np.array(posterior_samples["Psi"][i])        # (K, r_sig)
        A_h_i = np.array(posterior_samples["A_h"][i])        # (r_sig, r_sig)
        Q_h_i = np.array(posterior_samples["Q_h"][i])        # (r_sig,)
        omega_k_i = np.array(posterior_samples["omega_k"][i])  # (K,)

        # Latent state to forecast from: the series end by default, or the state
        # at start_t-1 when evaluating a forecast issued at time start_t.
        s_idx = -1 if start_t is None else start_t - 1
        f_last = np.array(posterior_samples["f_t"][i, s_idx, :])   # (r,)
        z_last = int(posterior_samples["z_t"][i, s_idx])
        h_last = np.array(posterior_samples["h_t"][i, s_idx, :])   # (r_sig,)

        for h in range(horizon):
            # Regime transition
            z_probs = np.array(Pi_i[z_last])
            z_curr = int(rng.choice(R, p=z_probs / z_probs.sum()))

            # Factor update
            eps = _draw(r)
            Q_chol = np.diag(np.array(Q_r_i[z_curr]))
            Phi = np.array(Phi_r_i[z_curr])
            f_curr = Phi @ f_last + Q_chol @ eps

            # Log-intensity
            mean_eta = np.array(mu_r_i[z_curr]) + f_curr @ np.array(Gamma_i).T  # (K,)
            eta_noise = _draw(K) * np.array(tau_k_i)
            eta_h = np.clip(mean_eta + eta_noise, -30.0, 30.0)  # (K,), clamp vs overflow

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
            if enhanced and "phi_D" in posterior_samples:
                phi_D_i = np.array(posterior_samples["phi_D"][i])       # (S,)
                p_nb_D = phi_D_i / (phi_D_i + rate_s)
                D_pred[i, :, h] = rng.negative_binomial(phi_D_i, p_nb_D)
            else:
                D_pred[i, :, h] = rng.poisson(rate_s)

            # Severity forecast: evolve the severity factor and draw log-severity
            # (shares the regime z_curr), then sigma_k = exp(zeta_k).
            xi_h = _draw(r_sig)
            h_curr = A_h_i @ h_last + np.diag(Q_h_i) @ xi_h        # (r_sig,)
            v_zeta = _draw(K) * omega_k_i                          # (K,)
            zeta_h = np.clip(np.array(nu_r_i[z_curr]) + h_curr @ Psi_i.T + v_zeta, -30.0, 30.0)  # (K,)
            sigma_h = np.exp(zeta_h)                              # (K,) latent severity

            # Economic shock load per Eq. (1): sum_k M_{s,k} * lambda_k * sigma_k
            # (exposure-weighted intensity x severity). rho_s and 1/x_s are NOT
            # applied here — rho_s belongs to the incident-disclosure channel and
            # gross output x_s enters later via direct losses d_s = g_s * x_s.
            shock_load = jnp.asarray(M_h @ (exp_lam * sigma_h))  # (S,)
            g_s = np.array(damage_function(shock_load, damage_params))
            d_s, ell = leontief_propagation(
                jnp.asarray(g_s), x_s_j, Lambda_L_j
            )
            ell_pred[i, :, h] = np.array(ell)
            ell_agg[i, h] = float(jnp.sum(ell))

            f_last = f_curr
            h_last = h_curr
            z_last = z_curr

    return {
        "lambda_pred": lambda_pred,
        "N_pred": N_pred,
        "D_pred": D_pred,
        "ell_pred": ell_pred,
        "ell_agg": ell_agg,
    }
