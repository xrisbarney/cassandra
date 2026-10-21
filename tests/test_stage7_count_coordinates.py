import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logit

from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.stage7_priors import CURRENT_PRIOR_ID
from cyber_threat_modelling.models.synthetic_state_space import build_model


def test_conditional_count_target_and_gradients_with_zeros_and_missingness():
    data, truth, options = generate('representative', 7101)
    data = {key: value[:4] if key != 'M' else value for key, value in data.items()}
    data['N'][0, -1] = 0
    data['mask_N'][1, -1] = False
    data['effort'] = np.array([-.2, 0, .1, .3])
    options.update(months=4, prior_id=CURRENT_PRIOR_ID, factor_parameterization='anchored')
    old_model = build_model(data, options, truth)
    new_model = build_model(data, dict(options, count_parameterization='conditional'), truth)
    old, new = old_model.initial_point(), new_model.initial_point()
    phi, dispersion = np.array([.3, .7]), np.linspace(4, 12, 10)
    bound = np.sqrt(2)
    for point in (old, new):
        point['f_phi_interval__'] = logit((phi+1)/2)
        point['f_free_interval__'][:] = logit((.1+bound)/(2*bound))
        point['psi_log__'] = np.log(dispersion)
    loading = np.full((10, 2), .1)
    loading[0] = [.3, 0]
    loading[1, 1] = .3
    covariance = np.zeros((10, 10))
    for j in range(2):
        indices = np.arange(5)*2+j
        covariance[np.ix_(indices, indices)] = phi[j]**np.abs(
            np.arange(5)[:, None]-np.arange(5))
    precision = np.linalg.inv(covariance)
    natural = np.zeros(10)
    for t in range(4):
        mask = data['mask_N'][t]
        adjusted = data['N'][t, mask]+.5
        h = loading[mask]
        weight = 1/(1/adjusted+1/dispersion[mask])
        indices = slice((t+1)*2, (t+2)*2)
        precision[indices, indices] += h.T @ (weight[:, None]*h)
        residual = np.log(adjusted)-new['m_known'][mask]-data['effort'][t]
        natural[indices] = h.T @ (weight*residual)
    chol = cholesky(precision, lower=True)
    u = np.linspace(-.2, .2, 10)
    states = (np.linalg.solve(precision, natural)
              + solve_triangular(chol.T, u, lower=False)).reshape(5, 2)
    new['f_whitened_states'] = u.reshape(5, 2)
    old['f_scaled_states'] = states*.3
    expected = 10*np.log(.3)-np.log(np.diag(chol)).sum()
    difference = new_model.compile_logp(jacobian=False)(new)-old_model.compile_logp(
        jacobian=False)(old)
    assert np.allclose(difference, expected, atol=1e-8)
    native = new_model.compile_dlogp()(new)
    compiled = new_model.compile_dlogp(mode='NUMBA')(new)
    assert np.isfinite(native).all()
    assert np.allclose(compiled, native, rtol=1e-7, atol=1e-7)
    function = new_model.compile_logp()
    offset = 0
    for variable in new_model.value_vars:
        name = variable.name
        if name in ['f_phi_interval__', 'psi_log__', 'm_known', 'f_diag_interval__']:
            delta = np.zeros_like(new[name])
            delta.flat[0] = 1e-5
            plus, minus = dict(new), dict(new)
            plus[name], minus[name] = new[name]+delta, new[name]-delta
            numerical = (function(plus)-function(minus))/(2e-5)
            assert np.allclose(native[offset], numerical, rtol=2e-5, atol=2e-5)
        offset += new[name].size
