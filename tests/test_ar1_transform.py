import numpy as np
import pytensor
import pytensor.tensor as pt
from scipy.linalg import solve_triangular

from cyber_threat_modelling.models.ar1_transform import ar1_transform, transform


def test_stationary_transform_matches_triangular_solution():
    rng = np.random.default_rng(43)
    innovations = rng.normal(size=(4, 15))
    phis = np.array([-0.9, 0.0, 0.5, 0.97])
    states = transform(innovations, phis)
    for k, phi in enumerate(phis):
        matrix = np.eye(15) - phi * np.eye(15, k=-1)
        noise = innovations[k].copy()
        noise[0] /= np.sqrt(1 - phi**2)
        assert np.allclose(states[k], solve_triangular(matrix, noise, lower=True))


def test_reverse_derivative_including_stationary_initial_state():
    rng = np.random.default_rng(44)
    e = rng.normal(size=(4, 7))
    phi = np.array([-0.9, 0.0, 0.5, 0.97])
    weight = rng.normal(size=e.shape)
    symbolic_e, symbolic_phi = pt.dmatrix(), pt.dvector()
    objective = pt.sum(ar1_transform(symbolic_e, symbolic_phi) * weight)
    fn = pytensor.function(
        [symbolic_e, symbolic_phi], pt.grad(objective, [symbolic_e, symbolic_phi])
    )
    grad_e, grad_phi = fn(e, phi)
    eps = 1e-6
    for index in np.ndindex(e.shape):
        high, low = e.copy(), e.copy()
        high[index] += eps
        low[index] -= eps
        numerical = np.sum((transform(high, phi) - transform(low, phi)) * weight) / (2 * eps)
        assert np.isclose(numerical, grad_e[index], rtol=1e-6, atol=1e-7)
    for k in range(len(phi)):
        high, low = phi.copy(), phi.copy()
        high[k] += eps
        low[k] -= eps
        numerical = np.sum((transform(e, high) - transform(e, low)) * weight) / (2 * eps)
        assert np.isclose(numerical, grad_phi[k], rtol=1e-6, atol=1e-7)


def test_simulated_prior_has_stationary_covariance():
    rng = np.random.default_rng(45)
    innovations = rng.normal(size=(60000, 5))
    phi = 0.7
    states = transform(innovations, np.full(60000, phi))
    expected = phi ** np.abs(np.arange(5)[:, None] - np.arange(5)) / (1 - phi**2)
    assert np.max(np.abs(np.cov(states, rowvar=False) - expected)) < 0.04
