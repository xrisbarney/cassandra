"""
sequential.py
=============
Sequential (filtered) posterior-predictive machinery shared by the backtest,
the damage-function calibration (pre-event predictives), and the ablation
study.

The Hamilton filter runs from the first panel month (uniform initial regime
distribution), so the belief entering any month t is a recursive function of
ALL preceding months' data across all four observation channels.  The
one-step-ahead prediction for month t uses only P(z_t | y_{1:t-1}), the
factor state at t-1, and covariates lagged to t-1 -- never month t itself.

Documented caveats (see scripts/backtest.py and docs/PAPER_NOTES.md §7):
continuous factor states and static parameters come from the full-sample
MCMC posterior (the standard posterior-predictive check for MCMC-fitted
state-space models), and e_t is itself a two-sided full-sample estimate.
"""
from __future__ import annotations

import numpy as np


def soft_clip(x: np.ndarray, bound: float = 30.0) -> np.ndarray:
    """Numpy mirror of the model's jax soft clip: bound * tanh(x / bound)."""
    return bound * np.tanh(x / bound)


def batch_forward_filter(loglik: np.ndarray, Pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Hamilton filter vectorised across posterior draws.

    loglik : (n, T, R) log p(y_t | z_t=r) per draw
    Pi     : (n, R, R) transition matrices per draw

    Returns (filtered, predicted), each (n, T, R):
      predicted[:, t] = P(z_t | y_{1:t-1})   -- uses data through t-1 ONLY
      filtered[:, t]  = P(z_t | y_{1:t})
    Initial predicted distribution at t=0 is uniform, matching the model.
    """
    n, T, R = loglik.shape
    filtered = np.zeros((n, T, R))
    predicted = np.zeros((n, T, R))
    pred = np.full((n, R), 1.0 / R)
    for t in range(T):
        if t > 0:
            pred = np.einsum("nr,nrj->nj", filtered[:, t - 1], Pi)
        predicted[:, t] = pred
        log_joint = np.log(pred + 1e-300) + loglik[:, t]
        log_joint -= log_joint.max(axis=1, keepdims=True)
        joint = np.exp(log_joint)
        filtered[:, t] = joint / joint.sum(axis=1, keepdims=True)
    return filtered, predicted


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorised categorical draw: probs (n, R) -> (n,) integer labels."""
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    u = rng.random((probs.shape[0], 1))
    z = (u > np.cumsum(probs, axis=1)).sum(axis=1)
    # cumsum's last entry can round a few ULP below 1, which would let a
    # u drawn extremely close to 1 index one past the last regime.
    return np.minimum(z, probs.shape[1] - 1).astype(int)


def one_step_state_predictive(
    post: dict,
    t: int,
    rng: np.random.Generator,
    predicted_probs: np.ndarray | None = None,   # (n, T, R) from batch_forward_filter
    enhanced: bool = False,
    student_t_df: float = 4.0,
) -> dict:
    """
    Pre-month-t predictive draws of the latent states, conditioned on data
    through t-1 only (filtered regime belief; factor/severity states
    propagated one AR step from their t-1 posterior values with fresh noise).

    Returns dict with:
      eta   : (n, K) predictive log-intensity at t
      zeta  : (n, K) predictive log-severity at t
      z     : (n,)   sampled regime at t
    """
    Pi = np.asarray(post["Pi"], dtype=float)
    mu_r = np.asarray(post["mu_r"], dtype=float)
    nu_r = np.asarray(post["nu_r"], dtype=float)
    Gamma = np.asarray(post["Gamma"], dtype=float)
    Psi = np.asarray(post["Psi"], dtype=float)
    Phi = np.asarray(post["Phi"], dtype=float)
    A_h = np.asarray(post["A_h"], dtype=float)
    Q_f = np.asarray(post["Q_f"], dtype=float)
    Q_h = np.asarray(post["Q_h"], dtype=float)
    tau_k = np.asarray(post["tau_k"], dtype=float)
    omega_k = np.asarray(post["omega_k"], dtype=float)

    n = Pi.shape[0]
    K = mu_r.shape[-1]
    r_dim = Q_f.shape[-1]
    r_sig = Q_h.shape[-1]

    def _draw(shape):
        if enhanced:
            return rng.standard_t(student_t_df, shape)
        return rng.standard_normal(shape)

    if predicted_probs is not None:
        pred_z = predicted_probs[:, t]
    elif t > 0:
        loglik = np.asarray(post["loglik_regime_t"], dtype=float)
        filtered, _ = batch_forward_filter(loglik[:, :t], Pi)
        pred_z = np.einsum("nr,nrj->nj", filtered[:, -1], Pi)
    else:
        pred_z = np.full((n, Pi.shape[-1]), 1.0 / Pi.shape[-1])

    z = sample_categorical(pred_z, rng)
    idx = np.arange(n)

    f_prev = (np.asarray(post["f_t"], dtype=float)[:, t - 1]
              if t > 0 else np.asarray(post["f_init"], dtype=float))
    h_prev = (np.asarray(post["h_t"], dtype=float)[:, t - 1]
              if t > 0 else np.asarray(post["h_init"], dtype=float))
    f_curr = np.einsum("nij,nj->ni", Phi, f_prev) + Q_f * _draw((n, r_dim))
    h_curr = np.einsum("nij,nj->ni", A_h, h_prev) + Q_h * _draw((n, r_sig))

    eta = mu_r[idx, z] + np.einsum("nkr,nr->nk", Gamma, f_curr) + tau_k * _draw((n, K))
    zeta = nu_r[idx, z] + np.einsum("nkr,nr->nk", Psi, h_curr) + omega_k * _draw((n, K))
    return {"eta": eta, "zeta": zeta, "z": z}


def kernel_factor_predictive(
    post: dict,
    T_out: int,
    T_obs: int,
    h: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    h-step-ahead predictive of the learned moving-window kernel factor
    (model config["kernel"]; see full.py §3b).  For month t, conditions on
    the window innovations u only through t-h; in-window u's newer than
    that are replaced by their exact Gaussian predictive (fresh noise with
    variance 1 - sum of known weights^2, thanks to the unit-norm weights).

    Returns (g_pred, a_g): g_pred (n, T_out) kernel factor draws, a_g (n, K)
    per-topic loadings -- or (None, None) if the model has no kernel.
    """
    if "u_g" not in post:
        return None, None
    u = np.asarray(post["u_g"], dtype=float)                    # (n, T_obs)
    w_all = np.asarray(post["g_kernel_weights"], dtype=float)   # (n, 2D+1)
    sg = np.asarray(post["sigma_g"], dtype=float).reshape(-1)   # (n,)
    a_g = np.asarray(post["a_g"], dtype=float)                  # (n, K)
    n = u.shape[0]
    D = (w_all.shape[1] - 1) // 2

    g = np.zeros((n, T_out))
    for i in range(n):
        w = w_all[i]                                            # symmetric, lags -D..D
        for t in range(T_out):
            lo = max(t - D, 0)
            hi = min(t - h, T_obs - 1)
            if hi >= lo:
                d = t - np.arange(lo, hi + 1)                   # lags h..D within data
                wk = w[D + d]
                known = float(wk @ u[i, lo:hi + 1])
                var_missing = max(1.0 - float(wk @ wk), 0.0)
            else:
                known, var_missing = 0.0, 1.0
            g[i, t] = sg[i] * (known + np.sqrt(var_missing) * rng.standard_normal())
    return g, a_g


def sequential_h_step_predict(
    post: dict,
    e_t: np.ndarray,        # (T,) effort covariate
    M_skt: np.ndarray,      # (S, K, T) exposure map (unused for N; kept for symmetry)
    h: int,
    enhanced: bool = False,
    student_t_df: float = 4.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Rolling-origin h-step-ahead CVE-count predictive.

    Entry [:, :, t] (for t >= h) is the prediction of month t issued at
    origin o = t - h: regime belief = filtered at o advanced h steps through
    the sampled chain, factor state = posterior at o propagated h AR steps
    with fresh noise, covariates held at their origin values.  For h = 1
    this reduces to the one-step machinery.  Months t < h are NaN.

    Returns N_pred of shape (n, K, T).
    """
    rng = np.random.default_rng(seed=seed)

    loglik = np.asarray(post["loglik_regime_t"], dtype=float)
    Pi     = np.asarray(post["Pi"], dtype=float)
    mu_r   = np.asarray(post["mu_r"], dtype=float)
    Gamma  = np.asarray(post["Gamma"], dtype=float)
    Phi    = np.asarray(post["Phi"], dtype=float)
    Q_f    = np.asarray(post["Q_f"], dtype=float)
    tau_k  = np.asarray(post["tau_k"], dtype=float)
    psi_k  = np.asarray(post["psi_k"], dtype=float)
    f_post = np.asarray(post["f_t"], dtype=float)

    n, T, R = loglik.shape
    K = mu_r.shape[-1]
    r_dim = Q_f.shape[-1]
    idx = np.arange(n)

    def _draw(shape):
        if enhanced:
            return rng.standard_t(student_t_df, shape)
        return rng.standard_normal(shape)

    filtered, _ = batch_forward_filter(loglik, Pi)
    g_pred, a_g = kernel_factor_predictive(post, T, T, h, rng)

    N_pred = np.full((n, K, T), np.nan)
    for t in range(h, T):
        o = t - h
        z = sample_categorical(filtered[:, o], rng)
        f_prev = f_post[:, o]
        for _step in range(h):
            z = sample_categorical(Pi[idx, z], rng)
            f_prev = np.einsum("nij,nj->ni", Phi, f_prev) + Q_f * _draw((n, r_dim))
        eta = mu_r[idx, z] + np.einsum("nkr,nr->nk", Gamma, f_prev) + tau_k * _draw((n, K))
        if g_pred is not None:
            eta = eta + a_g * g_pred[:, t][:, None]
        mu_N = np.exp(soft_clip(eta + e_t[o]))
        p_nb = psi_k / (psi_k + mu_N)
        N_pred[:, :, t] = rng.negative_binomial(psi_k, p_nb)
    return N_pred


def sequential_one_step_predict(
    post: dict,
    e_t: np.ndarray,        # (T,) effort covariate (standardised)
    M_skt: np.ndarray,      # (S, K, T) exposure map
    tail_months: int,
    enhanced: bool,
    student_t_df: float,
    seed: int = 42,
) -> dict:
    """
    One-step-ahead predictive draws for every month of the panel, plus a
    pure-forecast tail after the data ends.  Fully vectorised across draws.
    See the module docstring for the conditioning guarantees.
    """
    rng = np.random.default_rng(seed=seed)

    loglik = np.asarray(post["loglik_regime_t"], dtype=float)  # (n, T, R)
    Pi     = np.asarray(post["Pi"], dtype=float)               # (n, R, R)
    mu_r   = np.asarray(post["mu_r"], dtype=float)             # (n, R, K)
    Gamma  = np.asarray(post["Gamma"], dtype=float)            # (n, K, r)
    Phi    = np.asarray(post["Phi"], dtype=float)              # (n, r, r)
    Q_f    = np.asarray(post["Q_f"], dtype=float)              # (n, r)
    tau_k  = np.asarray(post["tau_k"], dtype=float)            # (n, K)
    psi_k  = np.asarray(post["psi_k"], dtype=float)            # (n, K)
    rho_s  = np.asarray(post["rho_s"], dtype=float)            # (n, S)
    pi_s   = np.asarray(post["pi_s"], dtype=float)             # (n, S)
    f_post = np.asarray(post["f_t"], dtype=float)              # (n, T, r)
    f_init = np.asarray(post["f_init"], dtype=float)           # (n, r)

    n, T, R = loglik.shape
    K = mu_r.shape[-1]
    S = rho_s.shape[-1]
    r_dim = f_init.shape[-1]
    T_all = T + tail_months

    def _draw(shape):
        if enhanced:
            return rng.standard_t(student_t_df, shape)
        return rng.standard_normal(shape)

    print(f"      Hamilton filter over {T} months x {n} draws ...")
    filtered, predicted = batch_forward_filter(loglik, Pi)
    g_pred, a_g = kernel_factor_predictive(post, T_all, T, 1, rng)

    N_pred = np.zeros((n, K, T_all))
    D_pred = np.zeros((n, S, T_all))
    idx = np.arange(n)

    print(f"      One-step-ahead predictions for {T} observed months ...")
    for t in range(T_all):
        in_sample = t < T

        # --- regime, one step ahead of the data the model has seen -------
        if in_sample:
            # P(z_t | y_{1:t-1}): month t's own data is NOT in here.
            z_t = sample_categorical(predicted[:, t], rng)
        elif t == T:
            # First tail month: advance the last filtered belief once.
            z_t = sample_categorical(
                np.einsum("nr,nrj->nj", filtered[:, T - 1], Pi), rng)
        else:
            # Deeper tail: evolve the sampled chain, no more updates.
            z_t = sample_categorical(Pi[idx, z_prev], rng)
        z_prev = z_t

        # --- factor state propagated from t-1 ----------------------------
        if t == 0:
            f_prev = f_init
        elif t <= T:
            # month t's prediction starts from the posterior state at t-1;
            # at t == T this is f_post[:, T-1], the last observed month.
            f_prev = f_post[:, t - 1]
        # (deeper in the tail, f_prev carries over from the previous step)
        f_curr = np.einsum("nij,nj->ni", Phi, f_prev) + Q_f * _draw((n, r_dim))
        f_prev = f_curr

        # --- threat intensity and observation channels --------------------
        mean_eta = mu_r[idx, z_t] + np.einsum("nkr,nr->nk", Gamma, f_curr)
        if g_pred is not None:
            mean_eta = mean_eta + a_g * g_pred[:, t][:, None]
        eta = mean_eta + tau_k * _draw((n, K))

        # Covariates enter the PREDICTION lagged (hold-last): month t's own
        # e_t and M_skt values are month-t data, so the prediction for t may
        # only use their t-1 values.  (The belief UPDATE, via
        # loglik_regime_t, correctly uses month t's own values -- it happens
        # after the month is observed.)
        cov_idx = min(max(t - 1, 0), T - 1)
        mu_N = np.exp(soft_clip(eta + e_t[cov_idx]))
        p_nb = psi_k / (psi_k + mu_N)
        N_pred[:, :, t] = rng.negative_binomial(psi_k, p_nb)

        M_t = M_skt[:, :, cov_idx]
        exp_lam = np.exp(soft_clip(eta))
        rate_s = np.clip(np.einsum("sk,nk->ns", M_t, exp_lam) * rho_s + pi_s, 1e-8, None)
        if enhanced and "phi_D" in post:
            phi_D = np.asarray(post["phi_D"], dtype=float)      # (n, S)
            D_pred[:, :, t] = rng.negative_binomial(phi_D, phi_D / (phi_D + rate_s))
        else:
            D_pred[:, :, t] = rng.poisson(rate_s)

    return {"N_pred": N_pred, "D_pred": D_pred,
            "filtered": filtered, "predicted": predicted}
