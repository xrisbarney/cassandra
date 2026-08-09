"""
full.py
=======
Combined NumPyro model: latent dynamics + all four measurement likelihoods,
plus predictive utilities.

Inference approach (see docs/PAPER_NOTES.md for the full history):
the discrete regime path z_t is marginalized analytically via the standard
HMM forward algorithm (explicit log-sum-exp over the R regime hypotheses at
each time step, injected with numpyro.factor), rather than sampled via
DiscreteHMCGibbs. This lets plain NUTS handle the entire continuous parameter
space -- no interleaved discrete Gibbs kernel, no funsor/config_enumerate
(which cannot correctly handle a regime that also drives the continuous
factor's transition matrix -- a switching linear dynamical system that
automatic enumeration is not designed for). DiscreteHMCGibbs was observed to
collapse NUTS's warmup step-size adaptation to numerical underflow on the
real data with high frequency; this manual marginalization has not shown
that failure mode in testing.

Consequence for the model: the factor/severity AR transition dynamics
(Phi, Q_f, A_h, Q_h) are regime-CONSTANT -- only the emission means
(mu_r[z_t], nu_r[z_t]) switch by regime. This is a "Markov-switching mean"
model rather than a full "Markov-switching VAR"; it is a further deviation
from the paper's Eq. (3)/(4) (documented in docs/PAPER_NOTES.md) made to
keep exact regime marginalization tractable.
"""
from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.primitives import deterministic

from cassandra_threatcast.inference.ffbs import forward_filter, backward_sample
from cassandra_threatcast.model.economic import (
    DamageFunctionParams,
    damage_function,
    leontief_propagation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SCALE_FLOOR = 0.02       # minimum value for latent-dynamics noise scales
_SOFT_CLIP_BOUND = 30.0   # saturation bound for log-intensity/log-severity


def _soft_clip(x: jnp.ndarray, bound: float = _SOFT_CLIP_BOUND) -> jnp.ndarray:
    """Smoothly saturate x into (-bound, bound), preserving a nonzero
    gradient everywhere (a hard jnp.clip has a zero-gradient plateau beyond
    its bounds, a known source of HMC leapfrog instability)."""
    return bound * jnp.tanh(x / bound)


def _plated_sample(name: str, base_dist: dist.Distribution, shape: tuple) -> jnp.ndarray:
    """Sample `name` ~ base_dist (a scalar/atomic distribution), broadcast to
    `shape` via nested numpyro.plates, one per axis. Explicit plates are not
    needed for the (removed) funsor enumeration path, but are kept here for
    shape-safety and to keep every prior a clean, independent draw."""
    with ExitStack() as stack:
        ndim = len(shape)
        for i, size in enumerate(shape):
            stack.enter_context(numpyro.plate(f"_{name}_ax{i}", size, dim=-(ndim - i)))
        return numpyro.sample(name, base_dist)


def _floored_scale(name: str, base_scale: float, shape: tuple) -> jnp.ndarray:
    """Sample a HalfNormal(base_scale) scale, floored away from exact zero.

    An unfloored HalfNormal scale can wander arbitrarily close to 0, making
    the corresponding state-transition conditionally near-deterministic
    (infinite curvature in that direction), which collapses HMC's step-size
    adaptation. This floor keeps the scale strictly positive.
    """
    raw = _plated_sample(f"{name}_raw", dist.HalfNormal(base_scale), shape)
    return deterministic(name, _SCALE_FLOOR + raw)


def _positive_lower_triangular(raw: jnp.ndarray, d: int) -> jnp.ndarray:
    """Impose the standard factor-model identification constraint on a (K, d)
    loading matrix: the top d x d block is lower-triangular with a positive
    diagonal (rows below are free). Removes the rotation/scale/sign
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
    Combined NumPyro model: latent dynamics + all four measurement likelihoods.
    The discrete regime z_t is marginalized via the HMM forward algorithm
    (see module docstring); plain NUTS samples everything else.

    data keys
    ---------
    N      : (K, T) observed CVE counts, or None
    B      : (K, T) observed mean-CVSS severity marks (NaN = unobserved), or None
    E      : (K, T) observed mean EPSS in [0,1] (NaN = unobserved), or None
    D      : (S, T) observed 8-K incident counts, or None
    M_skt  : (S, K, T) exposure map
    e_t    : (T,) effort covariate
    x_s    : (S,) gross output by sector [unused here; consumed by predict()]
    Lambda_L : (S, S) Leontief inverse [unused here; consumed by predict()]

    config keys (nested under "model")
    -----------------------------------
    K, S, r, R, r_sigma
    config["enhanced"]: {enabled, student_t_df} -- heavier-tailed innovations
    and an overdispersed (NegBin) incident channel when enabled.
    """
    model_cfg = config.get("model", config)
    K: int = int(model_cfg["K"])
    S: int = int(model_cfg["S"])
    r: int = int(model_cfg["r"])
    R: int = int(model_cfg["R"])
    r_sig: int = int(model_cfg.get("r_sigma", r))

    enh_cfg = config.get("enhanced", {})
    enhanced: bool = bool(enh_cfg.get("enabled", False))
    student_t_df: float = float(enh_cfg.get("student_t_df", 4.0))

    # Ablation flags (paper Table 4; driven by scripts/ablation.py).  Each
    # removes one model component while leaving everything else untouched:
    #   no_factors  : Gamma = 0 (independent topics; factors become nuisance)
    #   flat_priors : all prior scales x10 (no shrinkage toward group levels)
    #   drop_E      : exploitation channel excluded from the likelihood
    #   drop_D      : incident channel excluded from the likelihood
    #   no_effort   : e_t = 0 (raw counts treated as truth)
    # "- regimes" needs no flag: set model.R = 1 in the variant config.
    abl_cfg = config.get("ablation", {}) or {}
    _abl_no_factors: bool = bool(abl_cfg.get("no_factors", False))
    _abl_w_E: float = 0.0 if abl_cfg.get("drop_E", False) else 1.0
    _abl_w_D: float = 0.0 if abl_cfg.get("drop_D", False) else 1.0
    _pw: float = 10.0 if abl_cfg.get("flat_priors", False) else 1.0

    # Learned temporal kernel (config["kernel"]): a moving-window ("square")
    # covariance component with LEARNABLE width and amplitude.  See §9 of
    # docs/PAPER_NOTES.md.  g_t = sigma_g * sum_d w_d(h) u_{t-d} with iid
    # u ~ N(0,1) and smooth-edged box weights w_d(h) = sigmoid((h-|d|)/2),
    # unit-normalised.  Implied covariance Cov(g_t, g_{t+d}) =
    # sigma_g^2 * sum_j w_j w_{j+d}: nonzero exactly when two months fall
    # inside overlapping windows.  h quantifies "how close is close";
    # sigma_g quantifies how much local covariance matters beyond the AR
    # factors and global trends; per-topic loadings a_g say who feels it.
    krn_cfg = config.get("kernel", {}) or {}
    kernel_on: bool = bool(krn_cfg.get("enabled", False))
    _krn_D: int = int(krn_cfg.get("max_halfwidth", 60))   # window support, months

    def _innovation(name: str, shape: tuple) -> jnp.ndarray:
        """Standardized latent innovation: Student-t in enhanced mode, else
        standard Normal. Always drawn as one (T, *shape) block up front (not
        per-scan-step) so there is no per-step numpyro.sample call left for
        an eliminated discrete regime to interact with."""
        base = dist.StudentT(student_t_df, 0.0, 1.0) if enhanced else dist.Normal(0.0, 1.0)
        return _plated_sample(name, base, shape)

    T: int = int(data["e_t"].shape[0])
    e_t = jnp.asarray(data["e_t"])          # (T,)
    if abl_cfg.get("no_effort", False):
        e_t = jnp.zeros_like(e_t)
    M_skt = jnp.asarray(data["M_skt"])      # (S, K, T)

    # ------------------------------------------------------------------ #
    # 1. Latent dynamics priors
    # ------------------------------------------------------------------ #
    # Real CVE-topic counts span several orders of magnitude across topics
    # (e.g. mean ~25/month vs. ~450/month), so mu_r must be able to sit far
    # from 0 for high-volume topics; sd=5 keeps this weakly informative
    # while accommodating the observed dynamic range.
    mu_r = _plated_sample("mu_r", dist.Normal(0.0, 5.0 * _pw), (R, K))  # (R, K)

    Gamma_raw = _plated_sample("Gamma_raw", dist.Normal(0.0, 1.0 * _pw), (K, r))
    Gamma_ltri = _positive_lower_triangular(Gamma_raw, r)
    if _abl_no_factors:
        # Zero BEFORE the deterministic so the posterior "Gamma" is zero too
        # and every downstream consumer (predict, sequential eval) stays
        # consistent with the ablated likelihood.
        Gamma_ltri = jnp.zeros_like(Gamma_ltri)
    Gamma = deterministic("Gamma", Gamma_ltri)  # (K, r)

    # Factor AR dynamics are regime-CONSTANT (see module docstring): only the
    # emission mean mu_r[z_t] switches by regime.
    Phi_raw = _plated_sample("Phi_raw", dist.Normal(0.0, 0.3), (r, r))
    tril_mask = jnp.tril(jnp.ones((r, r)))
    # tanh bounds every entry (hence the diagonal = eigenvalues of a
    # triangular matrix) to (-1, 1), guaranteeing a stationary AR so factors
    # cannot explode over long series (T can be ~180 months).
    Phi = deterministic("Phi", jnp.tanh(Phi_raw) * tril_mask)  # (r, r)
    Q_f = _floored_scale("Q_f", 0.5 * _pw, (r,))  # (r,)

    Pi = _plated_sample("Pi", dist.Dirichlet(2.0 * jnp.ones(R)), (R,))  # (R, R)
    tau_k = _floored_scale("tau_k", 0.5 * _pw, (K,))  # (K,)

    # ------------------------------------------------------------------ #
    # 1b. Latent severity process priors (log sigma_{k,t}), analogous to the
    #     intensity: regime-dependent level + lower-dimensional AR factors
    #     (also regime-constant dynamics, sharing the same z_t marginalization).
    # ------------------------------------------------------------------ #
    nu_r = _plated_sample("nu_r", dist.Normal(1.5, 1.0 * _pw), (R, K))  # (R, K)
    Psi_raw = _plated_sample("Psi_raw", dist.Normal(0.0, 1.0 * _pw), (K, r_sig))
    Psi = deterministic("Psi", _positive_lower_triangular(Psi_raw, r_sig))  # (K, r_sig)
    A_h_raw = _plated_sample("A_h_raw", dist.Normal(0.0, 0.3), (r_sig, r_sig))
    tril_sig = jnp.tril(jnp.ones((r_sig, r_sig)))
    A_h = deterministic("A_h", jnp.tanh(A_h_raw) * tril_sig)  # (r_sig, r_sig)
    Q_h = _floored_scale("Q_h", 0.5 * _pw, (r_sig,))         # (r_sig,)
    omega_k = _floored_scale("omega_k", 0.5 * _pw, (K,))     # (K,) idiosyncratic
    kappa_k = _floored_scale("kappa_k", 0.5 * _pw, (K,))     # (K,) severity meas. noise

    # ------------------------------------------------------------------ #
    # 2. Measurement parameters
    # ------------------------------------------------------------------ #
    psi_k = _plated_sample("psi_k", dist.HalfNormal(10.0), (K,))      # NegBin dispersion
    alpha_k = _plated_sample("alpha_k", dist.Normal(0.0, 2.0), (K,))  # EPSS intercept
    beta_k = _plated_sample("beta_k", dist.Normal(0.0, 1.0), (K,))    # EPSS slope
    varsigma_k = _floored_scale("varsigma_k", 1.0, (K,))              # EPSS obs noise
    rho_s = _plated_sample("rho_s", dist.HalfNormal(1.0), (S,))       # sector scaling

    # Sector-specific baseline disclosure rate. One parameter per sector (S),
    # constant over time, rather than one per sector-month (S*T ~ 2000
    # nuisance parameters, which pinned against zero on the ~99%-zero
    # incident channel and destroyed the sampling geometry).
    pi_s = _plated_sample("pi_s", dist.HalfNormal(0.5), (S,))  # (S,)

    phi_D = None
    if enhanced:
        phi_D = _plated_sample("phi_D", dist.HalfNormal(10.0), (S,))  # NegBin dispersion

    # ------------------------------------------------------------------ #
    # 3. Initial state + per-step innovations (drawn once, up front)
    # ------------------------------------------------------------------ #
    f_init = _plated_sample("f_init", dist.Normal(0.0, 1.0), (r,))
    h_init = _plated_sample("h_init", dist.Normal(0.0, 1.0), (r_sig,))

    eps_f_all = _innovation("eps_f", (T, r))
    eps_eta_all = _innovation("eps_eta", (T, K))
    xi_h_all = _innovation("xi_h", (T, r_sig))
    eps_v_all = _innovation("eps_v", (T, K))

    # ------------------------------------------------------------------ #
    # 3b. Learned moving-window kernel factor (optional)
    # ------------------------------------------------------------------ #
    if kernel_on:
        # Half-width prior centred on 24 months (a "square" spanning ~4
        # years, e.g. 2016-2020, centred on the prediction point).
        g_halfwidth = numpyro.sample(
            "g_halfwidth", dist.LogNormal(jnp.log(float(krn_cfg.get("halfwidth_prior_months", 24.0))), 0.5))
        sigma_g = numpyro.sample("sigma_g", dist.HalfNormal(0.5))
        a_g = _plated_sample("a_g", dist.Normal(0.0, 1.0), (K,))   # per-topic loading
        u_g = _plated_sample("u_g", dist.Normal(0.0, 1.0), (T,))   # window innovations

        lags = jnp.arange(-_krn_D, _krn_D + 1)                      # (2D+1,)
        w_g = jax.nn.sigmoid((g_halfwidth - jnp.abs(lags)) / 2.0)
        w_g = w_g / jnp.sqrt(jnp.sum(w_g ** 2) + 1e-12)             # Var(g)=sigma_g^2
        deterministic("g_kernel_weights", w_g)
        # Explicit zero-pad + 'valid' keeps the output length exactly T and
        # the window unambiguously centred ('same' returns the LONGER input's
        # length when the window exceeds the series).
        u_pad = jnp.concatenate([jnp.zeros(_krn_D), u_g, jnp.zeros(_krn_D)])
        g_seq = sigma_g * jnp.convolve(u_pad, w_g, mode="valid")    # (T,)
        deterministic("g_t", g_seq)
    else:
        a_g = jnp.zeros(K)
        g_seq = jnp.zeros(T)

    # ------------------------------------------------------------------ #
    # 4. Data channels, prepared as (T, ...) sequences for scan
    # ------------------------------------------------------------------ #
    def _seq(key: str, fallback_shape: tuple) -> jnp.ndarray:
        arr = data.get(key)
        arr = jnp.asarray(arr) if arr is not None else jnp.full(fallback_shape, jnp.nan)
        return arr.T  # (K or S, T) -> (T, K or S)

    N_seq = _seq("N", (K, T))
    E_seq = _seq("E", (K, T))
    D_seq = _seq("D", (S, T))
    B_seq = _seq("B", (K, T))
    M_skt_seq = jnp.moveaxis(M_skt, -1, 0)  # (S, K, T) -> (T, S, K)

    log_pi0 = jnp.log(jnp.ones(R) / R)  # uniform initial regime distribution
    log_Pi = jnp.log(Pi + 1e-30)        # (R, R)

    # ------------------------------------------------------------------ #
    # 5. Scan: regime-constant continuous dynamics + exact HMM forward
    #    marginalization of the discrete regime, all four channels.
    # ------------------------------------------------------------------ #
    def step(carry, xs_t):
        f_prev, h_prev, log_alpha_prev = carry
        (e_t_t, M_skt_t, N_t, E_t, D_t, B_t,
         eps_f_t, eps_eta_t, xi_h_t, eps_v_t, g_t_t) = xs_t

        f_t = Phi @ f_prev + Q_f * eps_f_t
        h_t = A_h @ h_prev + Q_h * xi_h_t

        # NaN-safe transforms computed once (shared by every regime hypothesis
        # below and by the diagnostic observation sites further down) so a
        # missing entry never lets a NaN reach a log_prob call.
        obs_mask_E = ~jnp.isnan(E_t)
        E_clip = jnp.clip(jnp.where(obs_mask_E, E_t, 0.5), 1e-4, 1.0 - 1e-4)
        logit_E_obs_t = jnp.log(E_clip) - jnp.log1p(-E_clip)
        obs_mask_B = ~jnp.isnan(B_t)
        B_safe = jnp.clip(jnp.where(obs_mask_B, B_t, 1.0), 0.1, 10.0)

        def loglik_given_regime(r_idx):
            eta_t = mu_r[r_idx] + f_t @ Gamma.T + a_g * g_t_t + tau_k * eps_eta_t
            zeta_t = nu_r[r_idx] + h_t @ Psi.T + omega_k * eps_v_t

            # -- vulnerability (NegBin) --
            log_mu_t = eta_t + e_t_t
            mu_t = jnp.exp(_soft_clip(log_mu_t))
            ll_N = dist.NegativeBinomial2(mean=mu_t, concentration=psi_k).log_prob(N_t).sum()

            # -- exploitation (EPSS, NaN-masked) --
            logit_E_t = alpha_k + beta_k * eta_t
            ll_E = jnp.where(
                obs_mask_E, dist.Normal(logit_E_t, varsigma_k).log_prob(logit_E_obs_t), 0.0
            ).sum()

            # -- incidents (Poisson, or NegBin in enhanced mode) --
            exp_lam_t = jnp.exp(_soft_clip(eta_t))
            weighted_t = jnp.einsum("sk,k->s", M_skt_t, exp_lam_t)
            rate_s_t = jnp.clip(rho_s * weighted_t + pi_s, 1e-8)
            if enhanced:
                ll_D = dist.NegativeBinomial2(mean=rate_s_t, concentration=phi_D).log_prob(D_t).sum()
            else:
                ll_D = dist.Poisson(rate_s_t).log_prob(D_t).sum()

            # -- severity (LogNormal, NaN-masked) --
            ll_B = jnp.where(
                obs_mask_B, dist.LogNormal(zeta_t, kappa_k).log_prob(B_safe), 0.0
            ).sum()

            # Ablation channel weights are 1.0 unless a variant drops a channel.
            return ll_N + _abl_w_E * ll_E + _abl_w_D * ll_D + ll_B

        loglik_r = jax.vmap(loglik_given_regime)(jnp.arange(R))  # (R,)

        # Hamilton forward-filter update in log-space:
        # log_alpha_t[r] = logsumexp_{r'}(log_alpha_prev[r'] + log_Pi[r',r]) + loglik_r[r]
        log_predict = jax.scipy.special.logsumexp(log_alpha_prev[:, None] + log_Pi, axis=0)
        log_alpha_t = log_predict + loglik_r  # (R,)

        # One-step-ahead (pre-update) regime-forecast weights, used only for
        # the diagnostic quantities below -- NOT the likelihood, which is the
        # exact joint marginal already captured in log_alpha_t/loglik_r.
        predict_probs_t = jax.nn.softmax(log_predict)  # (R,)
        mu_bar_t = predict_probs_t @ mu_r   # (K,) regime-forecast-weighted level
        nu_bar_t = predict_probs_t @ nu_r   # (K,)
        eta_t = mu_bar_t + f_t @ Gamma.T + a_g * g_t_t + tau_k * eps_eta_t
        zeta_t = nu_bar_t + h_t @ Psi.T + omega_k * eps_v_t

        # Diagnostic-only observation sites: give downstream tooling
        # (Predictive(), posterior-predictive checks) real N_obs/E_obs/D_obs/
        # B_obs sample sites to draw from, built from the single
        # regime-forecast-weighted level above. mask(mask=False) forces their
        # log_prob contribution to exactly zero, so they cannot affect the
        # posterior or double-count the likelihood -- inference is driven
        # exclusively by the numpyro.factor() marginal below.
        log_mu_t = eta_t + e_t_t
        mu_t = jnp.exp(_soft_clip(log_mu_t))
        logit_E_t = alpha_k + beta_k * eta_t
        exp_lam_t = jnp.exp(_soft_clip(eta_t))
        weighted_t = jnp.einsum("sk,k->s", M_skt_t, exp_lam_t)
        rate_s_t = jnp.clip(rho_s * weighted_t + pi_s, 1e-8)
        with numpyro.handlers.mask(mask=False):
            numpyro.sample("N_obs", dist.NegativeBinomial2(mean=mu_t, concentration=psi_k), obs=N_t)
            numpyro.sample("E_obs", dist.Normal(logit_E_t, varsigma_k), obs=logit_E_obs_t)
            if enhanced:
                numpyro.sample(
                    "D_obs", dist.NegativeBinomial2(mean=rate_s_t, concentration=phi_D), obs=D_t
                )
            else:
                numpyro.sample("D_obs", dist.Poisson(rate_s_t), obs=D_t)
            numpyro.sample("B_obs", dist.LogNormal(zeta_t, kappa_k), obs=B_safe)

        return (f_t, h_t, log_alpha_t), (f_t, h_t, eta_t, zeta_t, loglik_r)

    (_, _, log_alpha_final), (f_seq, h_seq, eta_seq, zeta_seq, loglik_seq) = scan(
        step,
        (f_init, h_init, log_pi0),
        (e_t, M_skt_seq, N_seq, E_seq, D_seq, B_seq,
         eps_f_all, eps_eta_all, xi_h_all, eps_v_all, g_seq),
        length=T,
    )
    # f_seq: (T,r), h_seq: (T,r_sig), eta_seq/zeta_seq: (T,K), loglik_seq: (T,R)
    # log_alpha_final: (R,) final carry -- the exact forward-filter log-weights
    # log alpha_T[r] = log p(all observations, z_T = r)

    deterministic("f_t", f_seq)
    deterministic("h_t", h_seq)
    deterministic("eta_t", eta_seq)     # one-step-ahead regime-forecast-weighted diagnostic
    deterministic("zeta_t", zeta_seq)   # one-step-ahead regime-forecast-weighted diagnostic
    deterministic("loglik_regime_t", loglik_seq)  # (T,R) per-regime log-lik, for FFBS recovery

    # Total marginal log-likelihood over the discrete regime path: log sum_r
    # alpha_T[r]. This IS the model's actual likelihood contribution for the
    # N/E/D/B channels -- their exact per-regime densities were evaluated
    # inside the scan above via
    # plain dist.*.log_prob() calls (not numpyro.sample statements), so
    # injecting this single scalar factor is what registers that contribution
    # with NUTS.
    marginal_ll = jax.scipy.special.logsumexp(log_alpha_final)
    numpyro.factor("marginal_regime_lik", marginal_ll)


# ---------------------------------------------------------------------------
# Post-hoc regime-path recovery (for forecasting)
# ---------------------------------------------------------------------------

def recover_terminal_regime(
    loglik_regime_t: np.ndarray,  # (T, R) per-regime log-likelihood for one posterior draw
    Pi: np.ndarray,                # (R, R) transition matrix for the same draw
    rng: np.random.Generator,
) -> int:
    """Sample z_T from its EXACT filtered posterior given one posterior draw's
    per-timestep regime log-likelihoods, via the Hamilton forward filter
    (ffbs.forward_filter). Only the terminal regime is needed (to seed
    forecasting); the full backward-sampled path is available via
    ffbs.backward_sample if ever needed for diagnostics.
    """
    T, R = loglik_regime_t.shape
    pi0 = np.ones(R) / R
    filtered_probs, _ = forward_filter(loglik_regime_t, Pi, pi0)
    probs_T = filtered_probs[-1]
    probs_T = probs_T / probs_T.sum()
    return int(rng.choice(R, p=probs_T))


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
      1. Recover the terminal regime z_T from its exact filtered posterior
         (via the Hamilton forward filter on the per-draw regime log-likelihoods).
      2. Extend factor/severity dynamics h steps forward (regime-constant AR,
         but the regime-dependent EMISSION mean still switches via a sampled
         forward regime path).
      3. Sample future N, E, D from measurement models.
      4. Compute g_s via damage_function.
      5. Compute ell via leontief_propagation.

    Set ``enhanced=True`` (matching how the model was fit) to draw latent
    innovations from Student-t and incident counts from Negative Binomial.

    Covariates (see docs/PAPER_NOTES.md §7): the reporting-effort offset
    ``e_t`` -- which the fitted N-channel likelihood includes as
    ``log mu = eta + e_t`` -- is carried into the forecast at its last
    pre-forecast value (hold-last), taken from ``data["e_t"]``.  Omitting it
    (the previous behaviour) implicitly reset effort to its historical mean
    and under-predicted counts by exp(e_last) -- ~3.7x at end-2024 effort
    levels (PAPER_NOTES §5).  The incident rate likewise now includes the
    additive baseline disclosure rate ``pi_s`` present in the fitted
    D-channel likelihood.

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

    # Hold-last reporting-effort offset for the forecast window.
    e_last = 0.0
    e_arr = np.asarray(data.get("e_t", []), dtype=float) if isinstance(data, dict) else np.array([])
    if e_arr.size:
        e_idx = (e_arr.size - 1) if start_t is None else min(start_t - 1, e_arr.size - 1)
        e_last = float(e_arr[e_idx])

    def _draw(size):
        if enhanced:
            return rng.standard_t(student_t_df, size)
        return rng.standard_normal(size)

    n_samples = next(iter(posterior_samples.values())).shape[0]
    K = posterior_samples["tau_k"].shape[-1]
    S = posterior_samples["rho_s"].shape[-1]
    r = posterior_samples["f_init"].shape[-1]
    R = posterior_samples["mu_r"].shape[-2]
    r_sig = posterior_samples["h_init"].shape[-1]

    lambda_pred = np.zeros((n_samples, K, horizon))
    N_pred = np.zeros((n_samples, K, horizon))
    D_pred = np.zeros((n_samples, S, horizon))
    ell_pred = np.zeros((n_samples, S, horizon))
    ell_agg = np.zeros((n_samples, horizon))
    sigma_pred = np.zeros((n_samples, K, horizon))   # latent severity draws
    g_pred = np.zeros((n_samples, S, horizon))       # sectoral damage fractions

    Lambda_L_j = jnp.asarray(Lambda_L)
    x_s_j = jnp.asarray(x_s)
    M_future_j = jnp.asarray(M_future)  # (S, K, horizon)

    for i in range(n_samples):
        mu_r_i = posterior_samples["mu_r"][i]        # (R, K)
        Gamma_i = posterior_samples["Gamma"][i]       # (K, r)
        Phi_i = np.array(posterior_samples["Phi"][i])         # (r, r) regime-constant
        Q_f_i = np.array(posterior_samples["Q_f"][i])         # (r,)   regime-constant
        Pi_i = posterior_samples["Pi"][i]             # (R, R)
        tau_k_i = posterior_samples["tau_k"][i]       # (K,)
        psi_k_i = posterior_samples["psi_k"][i]       # (K,)
        alpha_k_i = posterior_samples["alpha_k"][i]   # (K,)
        beta_k_i = posterior_samples["beta_k"][i]     # (K,)
        varsigma_k_i = posterior_samples["varsigma_k"][i]  # (K,)
        rho_s_i = posterior_samples["rho_s"][i]       # (S,)

        nu_r_i = posterior_samples["nu_r"][i]         # (R, K)
        Psi_i = np.array(posterior_samples["Psi"][i])          # (K, r_sig)
        A_h_i = np.array(posterior_samples["A_h"][i])          # (r_sig, r_sig)
        Q_h_i = np.array(posterior_samples["Q_h"][i])          # (r_sig,)
        omega_k_i = np.array(posterior_samples["omega_k"][i])  # (K,)

        # Terminal continuous state: f_t/h_t are regime-CONSTANT sequences
        # (no branching), so they are exactly recoverable at any time index.
        s_idx = -1 if start_t is None else start_t - 1
        f_last = np.array(posterior_samples["f_t"][i, s_idx, :])   # (r,)
        h_last = np.array(posterior_samples["h_t"][i, s_idx, :])   # (r_sig,)

        # Learned moving-window kernel factor: extend g past the data by
        # convolving [observed innovations, fresh draws] with this draw's
        # learned window (see full_model §3b).
        if "u_g" in posterior_samples:
            w_i = np.array(posterior_samples["g_kernel_weights"][i])
            sg_i = float(posterior_samples["sigma_g"][i])
            ag_i = np.array(posterior_samples["a_g"][i])            # (K,)
            u_i = np.array(posterior_samples["u_g"][i])
            u_hist = u_i if start_t is None else u_i[:start_t]
            D_k = (len(w_i) - 1) // 2
            u_ext = np.concatenate([np.zeros(D_k), u_hist,
                                    rng.standard_normal(horizon + D_k)])
            g_fore = (sg_i * np.convolve(u_ext, w_i, mode="valid"))[
                len(u_hist): len(u_hist) + horizon]                 # (horizon,)
        else:
            ag_i = np.zeros(K)
            g_fore = np.zeros(horizon)

        # Terminal regime: recovered via the exact Hamilton filter on this
        # draw's per-timestep regime log-likelihoods (loglik_regime_t is
        # only available up to the training series length; for start_t before
        # the series end we filter only up to that point).
        loglik_i = np.array(posterior_samples["loglik_regime_t"][i])  # (T, R)
        loglik_upto = loglik_i if start_t is None else loglik_i[:start_t]
        z_last = recover_terminal_regime(loglik_upto, np.array(Pi_i), rng)

        for h in range(horizon):
            z_probs = np.array(Pi_i[z_last])
            z_curr = int(rng.choice(R, p=z_probs / z_probs.sum()))

            eps = _draw(r)
            f_curr = Phi_i @ f_last + Q_f_i * eps

            mean_eta = np.array(mu_r_i[z_curr]) + f_curr @ np.array(Gamma_i).T + ag_i * g_fore[h]
            eta_noise = _draw(K) * np.array(tau_k_i)
            eta_h = np.clip(mean_eta + eta_noise, -30.0, 30.0)
            lambda_pred[i, :, h] = eta_h

            mu_kt_h = np.exp(np.clip(eta_h + e_last, -30.0, 30.0))
            psi_arr = np.array(psi_k_i)
            for k in range(K):
                p_nb = psi_arr[k] / (psi_arr[k] + mu_kt_h[k] + 1e-12)
                N_pred[i, k, h] = rng.negative_binomial(psi_arr[k], p_nb)

            M_h = np.array(M_future_j[:, :, h])
            exp_lam = np.exp(eta_h)
            rho_arr = np.array(rho_s_i)
            pi_arr = (np.array(posterior_samples["pi_s"][i])
                      if "pi_s" in posterior_samples else np.zeros(S))
            rate_s = rho_arr * (M_h @ exp_lam) + pi_arr
            rate_s = np.clip(rate_s, 1e-8, None)
            if enhanced and "phi_D" in posterior_samples:
                phi_D_i = np.array(posterior_samples["phi_D"][i])
                p_nb_D = phi_D_i / (phi_D_i + rate_s)
                D_pred[i, :, h] = rng.negative_binomial(phi_D_i, p_nb_D)
            else:
                D_pred[i, :, h] = rng.poisson(rate_s)

            xi_h_draw = _draw(r_sig)
            h_curr = A_h_i @ h_last + Q_h_i * xi_h_draw
            v_zeta = _draw(K) * omega_k_i
            zeta_h = np.clip(np.array(nu_r_i[z_curr]) + h_curr @ Psi_i.T + v_zeta, -30.0, 30.0)
            sigma_h = np.exp(zeta_h)

            # Economic shock load per Eq. (1): sum_k M_{s,k} * lambda_k * sigma_k.
            shock_load = jnp.asarray(M_h @ (exp_lam * sigma_h))
            g_s = np.array(damage_function(shock_load, damage_params))
            d_s, ell = leontief_propagation(jnp.asarray(g_s), x_s_j, Lambda_L_j)
            ell_pred[i, :, h] = np.array(ell)
            ell_agg[i, h] = float(jnp.sum(ell))
            sigma_pred[i, :, h] = sigma_h
            g_pred[i, :, h] = g_s

            f_last = f_curr
            h_last = h_curr
            z_last = z_curr

    return {
        "lambda_pred": lambda_pred,
        "N_pred": N_pred,
        "D_pred": D_pred,
        "ell_pred": ell_pred,
        "ell_agg": ell_agg,
        "sigma_pred": sigma_pred,
        "g_pred": g_pred,
    }
