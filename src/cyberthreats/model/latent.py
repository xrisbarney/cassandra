"""
latent.py
=========
NumPyro generative model for the latent factor / regime-switching layer.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.primitives import deterministic


# ---------------------------------------------------------------------------
# Helper: one-step AR(1) factor dynamics
# ---------------------------------------------------------------------------

def factor_dynamics(
    f_prev: jnp.ndarray,  # (r,)
    Phi: jnp.ndarray,     # (r, r)
    Q_chol: jnp.ndarray,  # (r, r) lower-triangular Cholesky of noise cov
    eps: jnp.ndarray,     # (r,) standard-normal noise
) -> jnp.ndarray:
    """
    One-step AR(1) factor dynamics.

    f_t = Phi @ f_{t-1} + Q_chol @ eps,  eps ~ N(0, I)

    Returns f_t of shape (r,).
    """
    return Phi @ f_prev + Q_chol @ eps


# ---------------------------------------------------------------------------
# Helper: regime transition
# ---------------------------------------------------------------------------

def regime_transition(z_prev: int, Pi: jnp.ndarray, rng_key) -> int:
    """
    Sample next regime from Pi[z_prev, :].

    Parameters
    ----------
    z_prev : current regime index (scalar int).
    Pi     : (R, R) Markov transition matrix.
    rng_key: JAX random key.

    Returns
    -------
    z_next : sampled next regime (scalar int).
    """
    probs = Pi[z_prev]
    z_next = jax.random.categorical(rng_key, jnp.log(probs + 1e-30))
    return z_next


# ---------------------------------------------------------------------------
# Full NumPyro generative model
# ---------------------------------------------------------------------------

def latent_dynamics_model(
    T: int,
    K: int,
    r: int,
    R: int,
    observed_eta=None,  # (T, K) or None
) -> None:
    """
    Full NumPyro generative model for the latent layer.

    Samples
    -------
    mu_r   : (R, K)    regime-specific means for log-intensity
    Gamma  : (K, r)    factor loadings
    Phi_r  : (R, r, r) per-regime AR matrices (lower-triangular for identifiability)
    Q_r    : (R, r)    per-regime factor noise std (diagonal Q)
    Pi     : (R, R)    Markov transition matrix (Dirichlet rows)
    tau_k  : (K,)      idiosyncratic std per topic
    f_t    : (T, r)    latent factors via scan
    z_t    : (T,)      regime path via scan
    eta_t  : (T, K)    log-intensities = mu_r[z_t] + f_t @ Gamma.T + noise
    """
    # ---- Global priors ----
    mu_r = numpyro.sample(
        "mu_r",
        dist.Normal(jnp.zeros((R, K)), jnp.ones((R, K)) * 2.0),
    )  # (R, K)

    Gamma_raw = numpyro.sample(
        "Gamma",
        dist.Normal(jnp.zeros((K, r)), jnp.ones((K, r))),
    )  # (K, r)

    # Per-regime AR matrices: lower-triangular for identifiability
    # We parameterise via a raw (R, r, r) matrix then mask upper triangle
    Phi_raw = numpyro.sample(
        "Phi_raw",
        dist.Normal(jnp.zeros((R, r, r)), jnp.ones((R, r, r)) * 0.3),
    )
    # Zero out upper triangle (above diagonal) for each regime
    tril_mask = jnp.tril(jnp.ones((r, r)))
    Phi_r = deterministic("Phi_r", Phi_raw * tril_mask[None, :, :])  # (R, r, r)

    # Per-regime factor noise standard deviations (positive)
    Q_r = numpyro.sample(
        "Q_r",
        dist.HalfNormal(jnp.ones((R, r)) * 0.5),
    )  # (R, r)

    # Markov transition matrix rows ~ Dirichlet
    Pi_rows = numpyro.sample(
        "Pi",
        dist.Dirichlet(jnp.ones((R, R)) * 2.0),
    )  # (R, R)

    # Idiosyncratic std per topic
    tau_k = numpyro.sample(
        "tau_k",
        dist.HalfNormal(jnp.ones(K) * 0.5),
    )  # (K,)

    # ---- Time-series scan ----
    # State: (f_t, z_t) — (r,) factor and scalar regime index

    def _transition(carry, _):
        f_prev, z_prev = carry  # (r,), ()

        # Sample regime for this step via the Categorical distribution
        z_t = numpyro.sample(
            "z_t",
            dist.Categorical(probs=Pi_rows[z_prev]),
        )

        # Sample factor noise
        eps = numpyro.sample("eps_f", dist.Normal(jnp.zeros(r), jnp.ones(r)))

        # Build diagonal Cholesky from Q_r[z_t]
        Q_chol_t = jnp.diag(Q_r[z_t])  # (r, r)
        Phi_t = Phi_r[z_t]              # (r, r)
        f_t = factor_dynamics(f_prev, Phi_t, Q_chol_t, eps)

        # Log-intensity
        mean_eta = mu_r[z_t] + f_t @ Gamma_raw.T  # (K,)
        eta_noise = numpyro.sample(
            "eta_noise", dist.Normal(jnp.zeros(K), tau_k)
        )
        eta_t = mean_eta + eta_noise  # (K,)

        # Condition on observed_eta if provided (passed via obs)
        # Observation is handled in full_model; here we just sample.
        return (f_t, z_t), (f_t, z_t, eta_t)

    # Initial state
    f_init = numpyro.sample("f_init", dist.Normal(jnp.zeros(r), jnp.ones(r)))
    z_init = numpyro.sample("z_init", dist.Categorical(probs=jnp.ones(R) / R))

    # Run scan over T steps
    _, (f_seq, z_seq, eta_seq) = scan(
        _transition,
        (f_init, z_init),
        xs=None,
        length=T,
    )
    # f_seq: (T, r), z_seq: (T,), eta_seq: (T, K)

    deterministic("f_t", f_seq)
    deterministic("z_t", z_seq)

    # Optionally condition on observed log-intensities
    if observed_eta is not None:
        numpyro.sample(
            "eta_t",
            dist.Normal(eta_seq, tau_k * 0.1),
            obs=jnp.asarray(observed_eta),
        )
    else:
        deterministic("eta_t", eta_seq)
