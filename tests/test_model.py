"""Tests for the Bayesian model components."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def make_synthetic_data(K: int = 3, S: int = 2, T: int = 12, r: int = 2, R: int = 2) -> dict:
    """Return a minimal data dict suitable for testing the model."""
    rng = np.random.default_rng(42)
    M_raw = rng.dirichlet(np.ones(K), size=(S, T))  # (S, T, K)
    M_skt = M_raw.transpose(0, 2, 1)               # (S, K, T)
    return {
        "N":        rng.poisson(10, (K, T)).astype(float),
        "B":        rng.uniform(3.0, 9.0, (K, T)),
        "E":        rng.uniform(0.01, 0.5, (K, T)),
        "D":        rng.poisson(2, (S, T)).astype(float),
        "M_skt":    M_skt.astype(float),
        "e_t":      rng.standard_normal(T),
        "x_s":      rng.uniform(1e9, 1e12, S),
        "Lambda_L": np.eye(S) + 0.1 * rng.uniform(0, 1, (S, S)),
    }


# ---------------------------------------------------------------------------
# full_model
# ---------------------------------------------------------------------------

def test_full_model_forward_pass():
    """full_model can be traced by NumPyro's Predictive without raising."""
    import jax
    from numpyro.infer import Predictive
    from cassandra_threatcast.model.full import full_model

    K, S, T, r, R = 3, 2, 12, 2, 2
    data   = make_synthetic_data(K, S, T, r, R)
    config = {"model": {"K": K, "S": S, "r": r, "R": R}}

    rng = jax.random.PRNGKey(0)
    pred = Predictive(full_model, num_samples=2)
    samples = pred(rng, data=data, config=config)
    # The prior predictive trace must return a non-empty dict
    assert isinstance(samples, dict)
    assert len(samples) > 0


def test_full_model_contains_observables():
    """Prior predictive must contain at least one observation site."""
    import jax
    from numpyro.infer import Predictive
    from cassandra_threatcast.model.full import full_model

    K, S, T, r, R = 3, 2, 12, 2, 2
    data   = make_synthetic_data(K, S, T, r, R)
    config = {"model": {"K": K, "S": S, "r": r, "R": R}}

    rng = jax.random.PRNGKey(1)
    pred = Predictive(full_model, num_samples=2)
    samples = pred(rng, data=data, config=config)

    # At least one of the expected observation sites must be present
    observable_sites = {"N_obs", "D_obs", "obs_N", "obs_D"}
    found = observable_sites & set(samples.keys())
    assert len(found) > 0, (
        f"No observable site found.  Available keys: {list(samples.keys())}"
    )


def test_full_model_sample_shapes():
    """Prior predictive samples have correct batch dimension."""
    import jax
    from numpyro.infer import Predictive
    from cassandra_threatcast.model.full import full_model

    K, S, T, r, R = 3, 2, 12, 2, 2
    data   = make_synthetic_data(K, S, T, r, R)
    config = {"model": {"K": K, "S": S, "r": r, "R": R}}

    n_samples = 4
    rng = jax.random.PRNGKey(2)
    pred = Predictive(full_model, num_samples=n_samples)
    samples = pred(rng, data=data, config=config)

    for key, val in samples.items():
        assert val.shape[0] == n_samples, (
            f"Sample {key} has batch size {val.shape[0]}, expected {n_samples}"
        )


def test_full_model_has_severity_process():
    """Prior predictive exposes the latent severity process and its B channel."""
    import jax
    from numpyro.infer import Predictive
    from cassandra_threatcast.model.full import full_model

    K, S, T, r, R = 3, 2, 12, 2, 2
    data   = make_synthetic_data(K, S, T, r, R)
    config = {"model": {"K": K, "S": S, "r": r, "R": R, "r_sigma": 2}}

    rng = jax.random.PRNGKey(7)
    samples = Predictive(full_model, num_samples=3)(rng, data=data, config=config)

    # Severity latent (zeta_t), factor path (h_t), and the B observation channel.
    assert "zeta_t" in samples, f"missing zeta_t; keys={list(samples.keys())}"
    assert "B_obs" in samples, f"missing B_obs; keys={list(samples.keys())}"
    # zeta_t is (n_samples, T, K)
    assert samples["zeta_t"].shape == (3, T, K)
    assert bool(np.all(np.isfinite(np.asarray(samples["zeta_t"]))))


def test_severity_obs_masks_missing_marks():
    """severity_obs must tolerate NaN severity marks (empty topic-months)."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import Predictive
    from cassandra_threatcast.model.full import full_model

    K, S, T, r, R = 3, 2, 10, 2, 2
    data = make_synthetic_data(K, S, T, r, R)
    B = np.asarray(data["B"]).copy()
    B[:, 0] = np.nan            # first month has no CVEs for any topic
    B[1, 3] = np.nan            # scattered gap
    data["B"] = B
    config = {"model": {"K": K, "S": S, "r": r, "R": R, "r_sigma": 2}}

    rng = jax.random.PRNGKey(11)
    # Must not raise despite NaNs in the observed severity marks.
    samples = Predictive(full_model, num_samples=2)(rng, data=data, config=config)
    assert "B_obs" in samples


# ---------------------------------------------------------------------------
# latent_dynamics_model
# ---------------------------------------------------------------------------

def test_latent_dynamics_shapes():
    """latent_dynamics_model produces eta_t of shape (n_samples, T, K)."""
    import jax
    from numpyro.infer import Predictive
    from cassandra_threatcast.model.latent import latent_dynamics_model

    K, r, R, T = 3, 2, 2, 10
    rng  = jax.random.PRNGKey(2)
    pred = Predictive(latent_dynamics_model, num_samples=5)
    samples = pred(rng, T=T, K=K, r=r, R=R)

    assert "eta_t" in samples, (
        f"'eta_t' not found in latent_dynamics_model output. "
        f"Keys: {list(samples.keys())}"
    )
    assert samples["eta_t"].shape == (5, T, K), (
        f"Expected eta_t shape (5, {T}, {K}), got {samples['eta_t'].shape}"
    )


def test_latent_dynamics_finite():
    """Prior draws of eta_t must be finite."""
    import jax
    from numpyro.infer import Predictive
    from cassandra_threatcast.model.latent import latent_dynamics_model

    K, r, R, T = 4, 2, 3, 20
    rng  = jax.random.PRNGKey(3)
    pred = Predictive(latent_dynamics_model, num_samples=10)
    samples = pred(rng, T=T, K=K, r=r, R=R)

    eta = np.array(samples["eta_t"])
    assert np.all(np.isfinite(eta)), "eta_t contains non-finite values"


# ---------------------------------------------------------------------------
# damage_function
# ---------------------------------------------------------------------------

def test_damage_function_monotone():
    """damage_function is non-decreasing in shock_load for each sector."""
    import jax.numpy as jnp
    from cassandra_threatcast.model.economic import damage_function, DamageFunctionParams

    S = 3
    params = DamageFunctionParams(
        shape=np.ones(S) * 5.0,
        scale=np.ones(S) * 0.5,
        max_damage=np.ones(S) * 0.8,
    )
    loads = jnp.linspace(0.0, 1.0, 30)

    for s in range(S):
        damages = np.array([
            float(damage_function(jnp.full(S, float(l)), params)[s])
            for l in loads
        ])
        diffs = np.diff(damages)
        assert np.all(diffs >= -1e-6), (
            f"damage_function is not monotone for sector {s}: "
            f"min diff = {diffs.min():.6f}"
        )


def test_damage_function_zero_load():
    """damage_function returns 0 when shock_load is 0."""
    import jax.numpy as jnp
    from cassandra_threatcast.model.economic import damage_function, DamageFunctionParams

    S = 2
    params = DamageFunctionParams(
        shape=np.ones(S) * 3.0,
        scale=np.ones(S) * 1.0,
        max_damage=np.ones(S) * 0.9,
    )
    d = damage_function(jnp.zeros(S), params)
    np.testing.assert_allclose(np.array(d), np.zeros(S), atol=1e-6)


def test_damage_function_bounded():
    """damage_function output <= max_damage for all loads."""
    import jax.numpy as jnp
    from cassandra_threatcast.model.economic import damage_function, DamageFunctionParams

    S = 4
    max_damage = np.array([0.5, 0.7, 0.6, 0.9])
    params = DamageFunctionParams(
        shape=np.ones(S) * 2.0,
        scale=np.ones(S) * 0.3,
        max_damage=max_damage,
    )
    for load in [0.0, 0.5, 1.0, 5.0, 100.0]:
        d = np.array(damage_function(jnp.full(S, load), params))
        assert np.all(d <= max_damage + 1e-6), (
            f"damage_function exceeded max_damage at load={load}"
        )


# ---------------------------------------------------------------------------
# leontief_propagation
# ---------------------------------------------------------------------------

def test_leontief_propagation_identity():
    """leontief_propagation with identity Lambda_L returns d_s = g_s * x_s."""
    import jax.numpy as jnp
    from cassandra_threatcast.model.economic import leontief_propagation

    S = 4
    g_s = jnp.array([0.10, 0.20, 0.05, 0.15])
    x_s = jnp.array([1e9,  2e9,  5e8,  3e9])
    Lambda_L = jnp.eye(S)

    d_s, ell = leontief_propagation(g_s, x_s, Lambda_L)
    expected = g_s * x_s
    np.testing.assert_allclose(np.array(d_s), np.array(expected), rtol=1e-5)
    np.testing.assert_allclose(np.array(ell), np.array(expected), rtol=1e-5)


def test_leontief_propagation_amplification():
    """With off-diagonal Lambda_L > 0, aggregate loss >= sum of direct losses."""
    import jax.numpy as jnp
    from cassandra_threatcast.model.economic import leontief_propagation

    S = 3
    g_s = jnp.array([0.1, 0.1, 0.1])
    x_s = jnp.array([1e9, 1e9, 1e9])
    # Introduce positive inter-sector linkages
    Lambda_L = jnp.array([[1.0, 0.2, 0.1],
                           [0.0, 1.0, 0.3],
                           [0.0, 0.0, 1.0]])

    d_s, ell = leontief_propagation(g_s, x_s, Lambda_L)
    direct = float(jnp.sum(g_s * x_s))
    total  = float(jnp.sum(ell))
    assert total >= direct - 1e-6, (
        f"Leontief total {total:.2e} should be >= direct {direct:.2e}"
    )
