"""Paper-conformant four-layer Bayesian state-space model.

This module follows Steps 5--24 of the manuscript literally.  In particular,
factor transitions depend on the sampled Markov regime and the incident
channel uses the multiplicative, time-varying reporting propensity ``pi_st``.
The discrete path is supplied by the blocked NUTS/FFBS driver.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan


def _sample_array(name: str, distribution: dist.Distribution, shape: tuple[int, ...]):
    """Draw an independent array while declaring event dimensions explicitly."""
    return numpyro.sample(name, distribution.expand(shape).to_event(len(shape)))


def _ordered_regime_means(raw: jnp.ndarray) -> jnp.ndarray:
    """Apply Step 5's mean-level ordering to make regime labels stable."""
    return raw[jnp.argsort(jnp.mean(raw, axis=1))]


def _stationary_matrix(raw: jnp.ndarray) -> jnp.ndarray:
    """Bound triangular VAR eigenvalues without changing the paper's structure."""
    return jnp.tanh(raw) * jnp.tril(jnp.ones(raw.shape[-2:]))


def reporting_propensity(
    logit_base_s: jnp.ndarray,
    reporting_rw_t: jnp.ndarray,
    mandatory_effect: jnp.ndarray,
    mandatory_t: jnp.ndarray,
) -> jnp.ndarray:
    """Step 14: sector/time propensity with a December-2023 regime indicator."""
    logits = logit_base_s[:, None] + reporting_rw_t[None, :]
    logits = logits + mandatory_effect * mandatory_t[None, :]
    return jax.nn.sigmoid(logits)


def paper_model(data: dict, config: dict, z_path: jnp.ndarray) -> None:
    """Joint continuous conditional posterior for a fixed regime path.

    The outer sampler alternates this conditional model (NUTS, Step 22) with
    FFBS draws of ``z_path`` (Step 23).
    """
    cfg = config.get("model", config)
    K, S = int(cfg["K"]), int(cfg["S"])
    r, r_sig, R = int(cfg["r"]), int(cfg.get("r_sigma", cfg["r"])), int(cfg["R"])
    T = int(np.shape(data["N"])[1])
    z_path = jnp.asarray(z_path, dtype=jnp.int32)

    # Steps 5--9: Markov switching factors, intensities, severities, hierarchy.
    mu_raw = _sample_array("mu_raw", dist.Normal(0.0, 5.0), (R, K))
    mu_r = numpyro.deterministic("mu_r", _ordered_regime_means(mu_raw))
    nu_raw = _sample_array("nu_raw", dist.Normal(1.5, 1.0), (R, K))
    # Severity regimes share the intensity ordering, as both use the same z_t.
    order = jnp.argsort(jnp.mean(mu_raw, axis=1))
    nu_r = numpyro.deterministic("nu_r", nu_raw[order])

    gamma0 = _sample_array("gamma0", dist.Normal(0.0, 1.0), (r,))
    gamma_scale = _sample_array("gamma_scale", dist.HalfNormal(1.0), (r,))
    Gamma = numpyro.sample(
        "Gamma", dist.Normal(gamma0, gamma_scale).expand((K, r)).to_event(2)
    )
    psi0 = _sample_array("psi0", dist.Normal(0.0, 1.0), (r_sig,))
    psi_scale = _sample_array("psi_scale", dist.HalfNormal(1.0), (r_sig,))
    Psi = numpyro.sample(
        "Psi", dist.Normal(psi0, psi_scale).expand((K, r_sig)).to_event(2)
    )

    a_tau = float(cfg.get("a_tau", 3.0))
    b_tau = float(cfg.get("b_tau", 0.5))
    tau2_k = _sample_array("tau2_k", dist.InverseGamma(a_tau, b_tau), (K,))
    tau_k = numpyro.deterministic("tau_k", jnp.sqrt(tau2_k))
    omega2_k = _sample_array("omega2_k", dist.InverseGamma(a_tau, b_tau), (K,))
    omega_k = numpyro.deterministic("omega_k", jnp.sqrt(omega2_k))

    Phi_raw = _sample_array("Phi_raw", dist.Normal(0.0, 0.3), (R, r, r))
    Phi_r = numpyro.deterministic("Phi_r", _stationary_matrix(Phi_raw))
    Q_r = numpyro.deterministic(
        "Q_r", _sample_array("Q_r_raw", dist.HalfNormal(0.5), (R, r)) + 1e-3
    )
    A_raw = _sample_array("A_sigma_raw", dist.Normal(0.0, 0.3), (R, r_sig, r_sig))
    A_sigma_r = numpyro.deterministic("A_sigma_r", _stationary_matrix(A_raw))
    Q_sigma_r = numpyro.deterministic(
        "Q_sigma_r",
        _sample_array("Q_sigma_r_raw", dist.HalfNormal(0.5), (R, r_sig)) + 1e-3,
    )
    Pi = numpyro.sample("Pi", dist.Dirichlet(2.0 * jnp.ones(R)).expand((R,)).to_event(1))
    numpyro.factor("z_transition", jnp.log(1.0 / R) + jnp.log(Pi[z_path[:-1], z_path[1:]]).sum())

    # Steps 11--15: latent effort, three conditionally independent channels.
    effort_proxy = jnp.asarray(data.get("e_t", np.zeros(T)))
    effort_rw_scale = numpyro.sample("effort_rw_scale", dist.HalfNormal(0.15))
    effort_eps = _sample_array("effort_eps", dist.Normal(0.0, 1.0), (T,))
    log_effort_t = numpyro.deterministic(
        "log_effort_t", effort_proxy + effort_rw_scale * jnp.cumsum(effort_eps)
    )
    psi_k = numpyro.deterministic(
        "psi_k", _sample_array("psi_k_raw", dist.HalfNormal(10.0), (K,)) + 1e-3
    )
    alpha_k = _sample_array("alpha_k", dist.Normal(0.0, 2.0), (K,))
    beta_k = _sample_array("beta_k", dist.Normal(0.0, 1.0), (K,))
    varsigma_k = numpyro.deterministic(
        "varsigma_k", _sample_array("varsigma_k_raw", dist.HalfNormal(1.0), (K,)) + 1e-3
    )
    kappa_k = numpyro.deterministic(
        "kappa_k", _sample_array("kappa_k_raw", dist.HalfNormal(0.5), (K,)) + 1e-3
    )
    rho_s = numpyro.deterministic(
        "rho_s", _sample_array("rho_s_raw", dist.HalfNormal(1.0), (S,)) + 1e-6
    )

    # Step 10 is opt-in: a non-negative cross-excitation matrix augments the
    hawkes_on = bool(config.get("hawkes", {}).get("enabled", False))
    if hawkes_on:
        hawkes_alpha = _sample_array("hawkes_alpha", dist.HalfNormal(0.05), (K, K))
        hawkes_decay = numpyro.sample("hawkes_decay", dist.Beta(2.0, 2.0))
    else:
        hawkes_alpha = jnp.zeros((K, K))
        hawkes_decay = 0.0

    report_base = _sample_array("report_logit_base_s", dist.Normal(-3.0, 1.5), (S,))
    report_rw_scale = numpyro.sample("report_rw_scale", dist.HalfNormal(0.15))
    report_eps = _sample_array("report_eps", dist.Normal(0.0, 1.0), (T,))
    report_rw = report_rw_scale * jnp.cumsum(report_eps)
    mandatory_effect = numpyro.sample("mandatory_effect", dist.Normal(1.0, 0.75))
    mandatory_t = jnp.asarray(data.get("mandatory_t", np.zeros(T)))
    pi_st = numpyro.deterministic(
        "pi_st", reporting_propensity(report_base, report_rw, mandatory_effect, mandatory_t)
    )

    N = jnp.asarray(data["N"]).T
    B = jnp.asarray(data["B"]).T
    E = jnp.asarray(data["E"]).T
    D = jnp.asarray(data["D"]).T
    M = jnp.moveaxis(jnp.asarray(data["M_skt"]), -1, 0)

    f0 = _sample_array("f_init", dist.Normal(0.0, 1.0), (r,))
    h0 = _sample_array("h_init", dist.Normal(0.0, 1.0), (r_sig,))

    def transition(carry, values):
        f_prev, h_prev, eta_prev = carry
        z_t, n_t, b_t, e_obs_t, d_t, m_t, log_e_t, pi_t = values
        f_t = numpyro.sample("f_t", dist.Normal(Phi_r[z_t] @ f_prev, Q_r[z_t]).to_event(1))
        h_t = numpyro.sample(
            "h_t", dist.Normal(A_sigma_r[z_t] @ h_prev, Q_sigma_r[z_t]).to_event(1)
        )
        excitation = jnp.log1p(hawkes_decay * (hawkes_alpha @ jnp.exp(jnp.clip(eta_prev, -20, 20))))
        eta_t = numpyro.sample(
            "eta_t", dist.Normal(mu_r[z_t] + Gamma @ f_t + excitation, tau_k).to_event(1)
        )
        zeta_t = numpyro.sample(
            "zeta_t", dist.Normal(nu_r[z_t] + Psi @ h_t, omega_k).to_event(1)
        )
        lam_t = jnp.exp(jnp.clip(eta_t, -30.0, 30.0))

        # Step 12: e_t * lambda_kt, with log_effort_t = log(e_t).
        numpyro.sample(
            "N_obs", dist.NegativeBinomial2(lam_t * jnp.exp(log_e_t), psi_k).to_event(1), obs=n_t
        )
        e_mask = ~jnp.isnan(e_obs_t)
        e_safe = jnp.clip(jnp.where(e_mask, e_obs_t, 0.5), 1e-4, 1 - 1e-4)
        with numpyro.handlers.mask(mask=e_mask):
            numpyro.sample(
                "E_obs",
                dist.Normal(alpha_k + beta_k * eta_t, varsigma_k).to_event(1),
                obs=jnp.log(e_safe) - jnp.log1p(-e_safe),
            )
        b_mask = ~jnp.isnan(b_t)
        with numpyro.handlers.mask(mask=b_mask):
            numpyro.sample(
                "B_obs", dist.LogNormal(zeta_t, kappa_k).to_event(1),
                obs=jnp.clip(jnp.where(b_mask, b_t, 1.0), 0.1, 10.0),
            )
        # Step 15 is multiplicative: pi_st * rho_s * sum_k M_skt lambda_kt.
        incident_rate = jnp.clip(pi_t * rho_s * (m_t @ lam_t), 1e-8)
        numpyro.sample("D_obs", dist.Poisson(incident_rate).to_event(1), obs=d_t)
        return (f_t, h_t, eta_t), (eta_t, zeta_t)

    _, (eta, zeta) = scan(
        transition,
        (f0, h0, jnp.zeros(K)),
        (z_path, N, B, E, D, M, log_effort_t, pi_st.T),
        length=T,
    )
    numpyro.deterministic("lambda_t", jnp.exp(jnp.clip(eta, -30.0, 30.0)))
    numpyro.deterministic("sigma_t", jnp.exp(jnp.clip(zeta, -30.0, 30.0)))


def regime_log_potentials(draw: dict, data: dict) -> np.ndarray:
    """Compute local FFBS potentials p(states_t, observations_t | z_t=r)."""
    mu, nu = np.asarray(draw["mu_r"]), np.asarray(draw["nu_r"])
    phi, q = np.asarray(draw["Phi_r"]), np.asarray(draw["Q_r"])
    a_sig, q_sig = np.asarray(draw["A_sigma_r"]), np.asarray(draw["Q_sigma_r"])
    gamma, psi_load = np.asarray(draw["Gamma"]), np.asarray(draw["Psi"])
    f, h = np.asarray(draw["f_t"]), np.asarray(draw["h_t"])
    eta, zeta = np.asarray(draw["eta_t"]), np.asarray(draw["zeta_t"])
    f_prev = np.vstack([np.asarray(draw["f_init"])[None], f[:-1]])
    h_prev = np.vstack([np.asarray(draw["h_init"])[None], h[:-1]])
    T, R = f.shape[0], mu.shape[0]

    from scipy.stats import nbinom, norm, poisson

    out = np.zeros((T, R))
    N, B, E, D = (np.asarray(data[k]).T for k in ("N", "B", "E", "D"))
    M = np.moveaxis(np.asarray(data["M_skt"]), -1, 0)
    log_effort, pi_st = np.asarray(draw["log_effort_t"]), np.asarray(draw["pi_st"])
    for t in range(T):
        for z in range(R):
            value = norm.logpdf(f[t], phi[z] @ f_prev[t], q[z]).sum()
            value += norm.logpdf(h[t], a_sig[z] @ h_prev[t], q_sig[z]).sum()
            excitation = 0.0
            if "hawkes_alpha" in draw:
                previous_eta = np.zeros(eta.shape[1]) if t == 0 else eta[t - 1]
                excitation = np.log1p(
                    float(draw["hawkes_decay"])
                    * (np.asarray(draw["hawkes_alpha"]) @ np.exp(np.clip(previous_eta, -20, 20)))
                )
            value += norm.logpdf(
                eta[t], mu[z] + gamma @ f[t] + excitation, np.asarray(draw["tau_k"])
            ).sum()
            value += norm.logpdf(zeta[t], nu[z] + psi_load @ h[t], np.asarray(draw["omega_k"])).sum()
            lam = np.exp(np.clip(eta[t], -30, 30))
            mean_n = lam * np.exp(log_effort[t])
            conc = np.asarray(draw["psi_k"])
            value += nbinom.logpmf(N[t], conc, conc / (conc + mean_n)).sum()
            emask = np.isfinite(E[t])
            if emask.any():
                elogit = np.log(np.clip(E[t, emask], 1e-4, 1 - 1e-4) / np.clip(1 - E[t, emask], 1e-4, 1))
                value += norm.logpdf(
                    elogit,
                    np.asarray(draw["alpha_k"])[emask] + np.asarray(draw["beta_k"])[emask] * eta[t, emask],
                    np.asarray(draw["varsigma_k"])[emask],
                ).sum()
            bmask = np.isfinite(B[t])
            if bmask.any():
                value += norm.logpdf(np.log(np.clip(B[t, bmask], 0.1, 10)), zeta[t, bmask], np.asarray(draw["kappa_k"])[bmask]).sum()
            rate = pi_st[:, t] * np.asarray(draw["rho_s"]) * (M[t] @ lam)
            value += poisson.logpmf(D[t], np.clip(rate, 1e-8, None)).sum()
            out[t, z] = value
    return out


def predict(
    posterior: dict,
    data: dict,
    horizon: int,
    Lambda_L: np.ndarray,
    x_s: np.ndarray,
    M_future: np.ndarray,
    damage_params,
    *,
    damage_draws: dict | None = None,
    exposure_concentration: float = 200.0,
    seed: int = 42,
    start_t: int | None = None,
) -> dict[str, np.ndarray]:
    """Execute paper Steps 25--28 for every joint posterior draw."""
    from cassandra_threatcast.model.economic import DamageFunctionParams, damage_function

    rng = np.random.default_rng(seed)
    n = next(iter(posterior.values())).shape[0]
    K, S = posterior["tau_k"].shape[-1], posterior["rho_s"].shape[-1]
    r, r_sig = posterior["f_init"].shape[-1], posterior["h_init"].shape[-1]
    R = posterior["Pi"].shape[-1]
    eta_out = np.zeros((n, K, horizon))
    sigma_out = np.zeros((n, K, horizon))
    N_out = np.zeros((n, K, horizon))
    E_out = np.zeros((n, K, horizon))
    B_out = np.zeros((n, K, horizon))
    D_out = np.zeros((n, S, horizon))
    g_out = np.zeros((n, S, horizon))
    ell_out = np.zeros((n, S, horizon))
    M_out = np.zeros((n, S, K, horizon))

    for i in range(n):
        terminal = -1 if start_t is None else max(0, start_t - 1)
        history_stop = posterior["report_eps"].shape[1] if start_t is None else terminal + 1
        f = np.asarray(posterior["f_t"][i, terminal]).copy()
        h_state = np.asarray(posterior["h_t"][i, terminal]).copy()
        z = int(np.asarray(posterior["z_t"][i, terminal]))
        eta_previous = np.asarray(posterior["eta_t"][i, terminal]).copy()
        effort_rw = float(np.asarray(posterior["log_effort_t"][i, terminal]))
        report_rw = float(
            np.asarray(posterior["report_rw_scale"][i])
            * np.asarray(posterior["report_eps"][i, :history_stop]).sum()
        )
        for step in range(horizon):
            Pi = np.asarray(posterior["Pi"][i])
            z = int(rng.choice(R, p=Pi[z] / Pi[z].sum()))
            f = np.asarray(posterior["Phi_r"][i, z]) @ f
            f += np.asarray(posterior["Q_r"][i, z]) * rng.standard_normal(r)
            h_state = np.asarray(posterior["A_sigma_r"][i, z]) @ h_state
            h_state += np.asarray(posterior["Q_sigma_r"][i, z]) * rng.standard_normal(r_sig)
            eta = np.asarray(posterior["mu_r"][i, z])
            eta += np.asarray(posterior["Gamma"][i]) @ f
            if "hawkes_alpha" in posterior:
                eta += np.log1p(
                    float(posterior["hawkes_decay"][i])
                    * (np.asarray(posterior["hawkes_alpha"][i]) @ np.exp(np.clip(eta_previous, -20, 20)))
                )
            eta += np.asarray(posterior["tau_k"][i]) * rng.standard_normal(K)
            zeta = np.asarray(posterior["nu_r"][i, z])
            zeta += np.asarray(posterior["Psi"][i]) @ h_state
            zeta += np.asarray(posterior["omega_k"][i]) * rng.standard_normal(K)
            lam, severity = np.exp(np.clip(eta, -30, 30)), np.exp(np.clip(zeta, -30, 30))

            effort_rw += float(posterior["effort_rw_scale"][i]) * rng.standard_normal()
            report_rw += float(posterior["report_rw_scale"][i]) * rng.standard_normal()
            report_logit = np.asarray(posterior["report_logit_base_s"][i]) + report_rw
            report_logit += float(posterior["mandatory_effect"][i])
            pi_s = 1.0 / (1.0 + np.exp(-np.clip(report_logit, -30, 30)))

            base_M = np.asarray(M_future[:, :, min(step, M_future.shape[2] - 1)])
            # Step 28: integrate exposure-map uncertainty jointly per topic.
            M_draw = np.column_stack([
                rng.dirichlet(np.clip(base_M[:, k], 1e-6, None) * exposure_concentration)
                for k in range(K)
            ])
            M_out[i, :, :, step] = M_draw

            conc = np.asarray(posterior["psi_k"][i])
            mean_n = lam * np.exp(np.clip(effort_rw, -20, 20))
            N_out[i, :, step] = rng.negative_binomial(conc, conc / (conc + mean_n))
            logit_e = np.asarray(posterior["alpha_k"][i]) + np.asarray(posterior["beta_k"][i]) * eta
            E_out[i, :, step] = 1.0 / (1.0 + np.exp(-rng.normal(logit_e, posterior["varsigma_k"][i])))
            B_out[i, :, step] = rng.lognormal(zeta, np.asarray(posterior["kappa_k"][i]))
            rate_d = pi_s * np.asarray(posterior["rho_s"][i]) * (M_draw @ lam)
            D_out[i, :, step] = rng.poisson(np.clip(rate_d, 1e-8, None))

            if damage_draws:
                j = i % len(np.asarray(damage_draws["max_damage"]))
                params = DamageFunctionParams(
                    shape=np.asarray(damage_draws["shape"])[j],
                    scale=np.asarray(damage_draws["scale"])[j],
                    max_damage=np.asarray(damage_draws["max_damage"])[j],
                )
            else:
                params = damage_params
            shock = M_draw @ (lam * severity)
            g = np.asarray(damage_function(jnp.asarray(shock), params))
            direct = g * np.asarray(x_s)
            ell = np.asarray(Lambda_L) @ direct
            eta_out[i, :, step], sigma_out[i, :, step] = eta, severity
            g_out[i, :, step], ell_out[i, :, step] = g, ell
            eta_previous = eta

    return {
        "lambda_pred": eta_out,
        "sigma_pred": sigma_out,
        "N_pred": N_out,
        "E_pred": E_out,
        "B_pred": B_out,
        "D_pred": D_out,
        "M_pred": M_out,
        "g_pred": g_out,
        "ell_pred": ell_out,
        "ell_agg": ell_out.sum(axis=1),
    }
