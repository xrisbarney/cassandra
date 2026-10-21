import numpy as np
import pytest
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logit

from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.stage7_priors import CURRENT_PRIOR_ID
from cyber_threat_modelling.models.synthetic_state_space import build_model


@pytest.mark.parametrize('solver', ['dense', 'block'])
def test_conditional_severity_full_density_and_compiled_gradient_with_endpoints(solver):
    data, truth, options = generate('severity_missing', 7101)
    data = {key: value[:4] if key != 'M' else value for key, value in data.items()}
    data['B'] = np.array([[.05, 6., 7., 4.], [3., np.nan, 9.95, 6.],
                          [4., 7., 3., 7.], [5., 6., 8., 9.]])
    data['mask_B'] = np.isfinite(data['B'])
    options.update(months=4, prior_id=CURRENT_PRIOR_ID, factor_parameterization='anchored',
                   severity_factors=2, known_auxiliary_coefficients=False)
    anchored = build_model(data, options, truth)
    conditional = build_model(data, dict(options, severity_parameterization='conditional',
                                         gaussian_solver=solver), truth)
    old, new = anchored.initial_point(), conditional.initial_point()
    phi, noise = np.array([.3, .6]), np.array([.2, .3, .4, .5])
    bound = np.sqrt(2)
    for point in (old, new):
        point['h_phi_interval__'] = logit((phi+1)/2)
        point['h_free_interval__'][:] = logit((.1+bound)/(2*bound))
        point['kappa_log__'] = np.log(noise)
    loading = np.array([[.3, 0], [.1, .3], [.1, .1], [.1, .1]])
    covariance = np.zeros((10, 10))
    for j in range(2):
        indices = np.arange(5)*2+j
        covariance[np.ix_(indices, indices)] = phi[j]**np.abs(
            np.arange(5)[:, None]-np.arange(5))
    precision = np.linalg.inv(covariance)
    natural = np.zeros(10)
    for t in range(4):
        mask = data['mask_B'][t]
        h, weight = loading[mask], 1/noise[mask]**2
        indices = slice((t+1)*2, (t+2)*2)
        precision[indices, indices] += h.T @ (weight[:, None]*h)
        natural[indices] = h.T @ (weight*(logit(data['B'][t, mask]/10)-1))
    chol = cholesky(precision, lower=True)
    u = np.linspace(-.2, .2, 10)
    states = (np.linalg.solve(precision, natural)
              + solve_triangular(chol.T, u, lower=False)).reshape(5, 2)
    new['h_whitened_states'] = u.reshape(5, 2)
    old['h_scaled_states'] = states*.3
    expected_ratio = 5*2*np.log(.3)-np.log(np.diag(chol)).sum()
    difference = conditional.compile_logp(jacobian=False)(new)-anchored.compile_logp(
        jacobian=False)(old)
    assert np.allclose(difference, expected_ratio, atol=1e-8)
    native = conditional.compile_dlogp()(new)
    assert np.isfinite(native).all()
    compiled = conditional.compile_dlogp(mode='NUMBA')(new)
    assert np.allclose(compiled, native, rtol=1e-7, atol=1e-7)
    combined = conditional.compile_fn([conditional.logp(), conditional.dlogp()],
                                      inputs=conditional.value_vars, mode='NUMBA')(new)
    assert np.allclose(combined[0], conditional.compile_logp()(new), atol=1e-8)
    assert np.allclose(combined[1], native, rtol=1e-7, atol=1e-7)
    # Independent finite differences for phi/noise/level, whose appearance in
    # the coordinate proposal makes omission of the Jacobian consequential.
    function = conditional.compile_logp()
    names = [value.name for value in conditional.value_vars]
    offset = 0
    for name in names:
        count = new[name].size
        if name in ['h_phi_interval__', 'kappa_log__', 'n']:
            delta = np.zeros_like(new[name])
            delta.flat[0] = 1e-5
            plus, minus = dict(new), dict(new)
            plus[name], minus[name] = new[name]+delta, new[name]-delta
            numerical = (function(plus)-function(minus))/(2e-5)
            assert np.allclose(native[offset], numerical, rtol=2e-5, atol=2e-5)
        offset += count
