import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logit

from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.joint_coordinates import joint_count_states
from cyber_threat_modelling.models.stage7_priors import CURRENT_PRIOR_ID, level_settings
from cyber_threat_modelling.models.synthetic_state_space import build_model, loading_matrix


@pytest.mark.parametrize('scenario', ['representative', 'fallback_zero'])
def test_joint_gaussian_density_gradients_and_canonical_reconstruction(scenario):
    data, truth, options = generate(scenario, 7101)
    data = {key: value[:4] for key, value in data.items()}
    data['N'][0, -1] = 0
    data['mask_N'][1, -1] = False
    options.update(months=4, prior_id=CURRENT_PRIOR_ID, factor_parameterization='anchored',
                   count_parameterization='conditional', gaussian_solver='block')
    settings = level_settings(data, options)
    known, fallback = settings['known_indices'], settings['fallback_index']
    center = np.full(10, settings['anchor'])
    if fallback is not None:
        center[fallback] = settings['fallback_anchor']
    old_model = build_model(data, options, truth)
    with pm.Model() as model:
        sd = pm.TruncatedNormal('m_sd', 0, settings['scale_sd'], lower=0,
                                 upper=settings['scale_upper'], initval=.7)
        cov = pt.zeros((10, 10))
        c_known = sd**2*np.eye(len(known))+settings['hyper_sd']**2*np.ones((len(known), len(known)))
        cov = pt.set_subtensor(cov[np.ix_(known, known)], c_known)
        if fallback is not None:
            cov = pt.set_subtensor(cov[fallback, fallback], 1.)
        psi = pm.LogNormal('psi', np.log(10), 1, shape=10, initval=np.full(10, 10))
        phi = pm.Uniform('f_phi', -1, 1, shape=2, initval=np.full(2, .4))
        loading = loading_matrix('f', 10, 2, True, 1.)
        m, contribution = joint_count_states(4, 2, phi, loading, center, cov, psi,
                                              data['N'], data['mask_N'], data['effort'],
                                              known, fallback)
        times, cats = np.where(data['mask_N'])
        eta = m+contribution+data['effort'][:, None]
        pm.Potential('observations', pm.logp(pm.NegativeBinomial.dist(
            mu=pt.exp(eta[times, cats]), alpha=psi[cats]), data['N'][times, cats]).sum())
    point, old = model.initial_point(), old_model.initial_point()
    phi_values = np.array([.35, .9999])
    for p in [point, old]:
        p['f_phi_interval__'] = logit((phi_values+1)/2)
        p['f_free_interval__'][:] = logit((.1+np.sqrt(2))/(2*np.sqrt(2)))
    point['joint_level_whitened'] = np.linspace(-.2, .2, 10)
    point['f_whitened_states'] = np.linspace(-.2, .2, 10).reshape(5, 2)
    canonical = model.compile_fn(model.replace_rvs_by_values([model['m'], model['f_states']]),
                                 inputs=model.value_vars, on_unused_input='ignore')(point)
    loading_values = np.full((10, 2), .1)
    loading_values[0] = [.3, 0]
    loading_values[1, 1] = .3
    c = np.zeros((10, 10))
    c[np.ix_(known, known)] = .7**2*np.eye(len(known))+settings['hyper_sd']**2
    if fallback is not None:
        c[fallback, fallback] = 1
    p_f = np.zeros((10, 10))
    for j, persistence in enumerate(phi_values):
        indices = np.arange(5)*2+j
        block = np.diag(np.r_[1, np.full(3, 1+persistence**2), 1])
        block += np.diag(np.full(4, -persistence), 1)+np.diag(np.full(4, -persistence), -1)
        p_f[np.ix_(indices, indices)] = block/(1-persistence**2)
    cross, d, b_m, b_f = np.zeros((10, 10)), np.zeros(10), np.zeros(10), np.zeros(10)
    for t in range(4):
        mask = data['mask_N'][t]
        adjusted = np.where(mask, data['N'][t], 0)+.5
        w = np.where(mask, 1/(1/adjusted+.1), 0)
        y = np.log(adjusted)-data['effort'][t]-center
        ix = slice((t+1)*2, (t+2)*2)
        p_f[ix, ix] += loading_values.T @ (w[:, None]*loading_values)
        cross[:, ix] = w[:, None]*loading_values
        d += w
        b_m += w*y
        b_f[ix] = loading_values.T @ (w*y)
    s = np.linalg.inv(c)+np.diag(d)-cross @ np.linalg.solve(p_f, cross.T)
    chol_m, chol_f = cholesky(s, lower=True), cholesky(p_f, lower=True)
    centered_m = np.linalg.solve(s, b_m-cross @ np.linalg.solve(p_f, b_f))
    centered_m += solve_triangular(chol_m.T, point['joint_level_whitened'], lower=False)
    expected_f = np.linalg.solve(p_f, b_f-cross.T @ centered_m)
    expected_f += solve_triangular(chol_f.T, point['f_whitened_states'].ravel(), lower=False)
    assert np.allclose(canonical[0], center+centered_m, rtol=1e-7, atol=1e-7)
    assert np.allclose(canonical[1], expected_f.reshape(5, 2), rtol=1e-7, atol=1e-7)
    old['m_known'] = canonical[0][known]
    if fallback is not None:
        old['m_fallback'] = np.array(canonical[0][fallback])
    old['f_whitened_states'] = point['f_whitened_states']
    difference = (model.compile_logp(jacobian=False)(point)
                  - old_model.compile_logp(jacobian=False)(old))
    assert np.allclose(difference, -np.log(np.diag(chol_m)).sum(), atol=1e-7)
    native = model.compile_dlogp()(point)
    integrated = build_model(data, dict(options, level_parameterization='joint_conditional'), truth)
    assert [v.name for v in integrated.value_vars] == [v.name for v in model.value_vars]
    assert np.allclose(integrated.compile_logp()(point), model.compile_logp()(point), atol=1e-7)
    assert np.allclose(integrated.compile_dlogp()(point), native, rtol=1e-7, atol=1e-7)
    fused = model.compile_fn([model.logp(), model.dlogp()],
                             inputs=model.value_vars, mode='NUMBA')(point)
    function = model.compile_logp()
    assert np.allclose(fused[0], function(point), atol=1e-7)
    assert np.allclose(fused[1], native, rtol=1e-6, atol=1e-6)
    offset = 0
    for variable in model.value_vars:
        name = variable.name
        delta = np.zeros_like(point[name])
        delta.flat[0] = 1e-5
        plus, minus = dict(point), dict(point)
        plus[name], minus[name] = point[name]+delta, point[name]-delta
        fd = (function(plus)-function(minus))/2e-5
        assert np.allclose(fd, native[offset], rtol=5e-5, atol=5e-5), name
        offset += point[name].size


@pytest.mark.parametrize('change', [dict(factors=1), dict(regimes=2),
                                    dict(residual='iid'), dict(gaussian_solver='dense')])
def test_joint_coordinates_reject_unsupported_models(change):
    data, truth, options = generate('representative', 7101)
    options.update(prior_id=CURRENT_PRIOR_ID, count_parameterization='conditional',
                   gaussian_solver='block', level_parameterization='joint_conditional')
    options.update(change)
    with pytest.raises(ValueError, match='Joint level coordinates'):
        build_model(data, options, truth)
