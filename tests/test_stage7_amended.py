import runpy
from pathlib import Path

import numpy as np
import pymc as pm
import pytest
from scipy.special import logit
from scipy.stats import multivariate_normal, truncnorm

from cyber_threat_modelling.evaluation.synthetic import SCENARIOS, generate
from cyber_threat_modelling.models.stage7_priors import PRIOR_ID, stationary_truth
from cyber_threat_modelling.models.synthetic_state_space import build_model


def test_truth_translation_preserves_contributions_and_all_fixtures_inside_support():
    for scenario in SCENARIOS:
        _, old, options = generate(scenario, 7101)
        new = stationary_truth(old, options)
        assert np.allclose(old['f'] @ old['loading'].T, new['f'] @ new['loading'].T)
        assert np.array_equal(old['eta'], new['eta'])
        assert np.array_equal(old['future_counts'], new['future_counts'])
        if options['factors']:
            assert np.max(np.abs(new['loading'])) < 2 / np.sqrt(options['factors'])
        if options['residual'] == 'ar':
            assert np.allclose(new['tau']**2, old['tau']**2 / (1-old['phi_u']**2))
            assert np.max(new['tau']) < 1
        if options['severity_factors']:
            assert np.allclose(new['loading_h']*np.sqrt(1-old['phi_h']**2),
                               old['loading_h'])
        assert not np.shares_memory(new['f'], old['f'])


@pytest.mark.parametrize('scenario', ['factors_sparse', 'kernel_present', 'severity_missing'])
def test_amended_compiled_gradients_and_stationary_density(scenario):
    data, truth, options = generate(scenario, 7101)
    for name in data:
        if name != 'M':
            data[name] = data[name][:4]
    options.update(months=4, prior_id=PRIOR_ID, factor_parameterization='centered')
    model = build_model(data, options, stationary_truth(truth, options))
    point = model.initial_point()
    point['f_states'] = np.array([[.2], [-.3], [.1], [.7], [.5]])
    phi = .4
    covariance = phi ** np.abs(np.arange(5)[:, None] - np.arange(5))
    expected = multivariate_normal.logpdf(point['f_states'][:, 0], np.zeros(5), covariance)
    assert np.allclose(model.compile_logp(vars=[model['f_state_prior']])(point), expected)
    native = model.compile_dlogp()(point)
    assert np.isfinite(native).all()
    assert np.allclose(model.compile_dlogp(mode='NUMBA')(point), native, atol=1e-8)
    if scenario == 'kernel_present':
        point['a_first_interval__'] = logit(.2)
        point['a_remaining_interval__'] = logit((np.array([-.3, .1, .4])+1)/2)
        for scale in (.1, .5, 2.):
            point['a_scale_log__'] = np.log(scale)
            actual = model.compile_logp(vars=[model['a_first'], model['a_remaining']],
                                        jacobian=False)(point)
            expected = truncnorm.logpdf(.2, 0, 1/scale, scale=scale)
            expected += truncnorm.logpdf([-.3, .1, .4], -1/scale, 1/scale,
                                         scale=scale).sum()
            assert np.allclose(actual, expected, atol=1e-9)
        full_logp = model.compile_logp()
        derivative = model.compile_dlogp(vars=[model['a_scale']])
        left, right = dict(point), dict(point)
        left['a_scale_log__'] -= 1e-5
        right['a_scale_log__'] += 1e-5
        finite_difference = (full_logp(right)-full_logp(left))/2e-5
        assert np.allclose(derivative(point), finite_difference, rtol=1e-5, atol=1e-5)


def test_normalized_loading_and_level_scale_laws():
    data, truth, options = generate('representative', 7101)
    for name in data:
        data[name] = data[name][:4]
    options.update(months=4, prior_id=PRIOR_ID)
    model = build_model(data, options, stationary_truth(truth, options))
    point = model.initial_point()
    rank = options['factors']
    sd, upper = 1/np.sqrt(rank), 2/np.sqrt(rank)
    diag = np.full(rank, .3)
    expected = truncnorm.logpdf(diag, 0, upper/sd, scale=sd).sum()
    actual = model.compile_logp(vars=[model['f_diag']], jacobian=False)(point)
    assert np.allclose(actual, expected)
    # Standalone oracle also checks the truncated half-normal's normalization.
    with pm.Model():
        value = pm.TruncatedNormal.dist(0, 2, lower=0, upper=3)
        actual = pm.logp(value, np.array([.01, 1., 2.9])).eval()
    assert np.allclose(actual, truncnorm.logpdf([.01, 1., 2.9], 0, 1.5, scale=2))


def test_amended_covariance_matches_original_generating_process():
    script = Path(__file__).resolve().parents[1]/'scripts/analyze_stage7.py'
    covariance = runpy.run_path(str(script))['covariance']
    for scenario in ('representative', 'latent_ar', 'kernel_present', 'dispersion_dense'):
        _, original, options = generate(scenario, 7101)
        revised = stationary_truth(original, options)
        for lag in (0, 1, 6, 12):
            results = []
            for truth, flag in ((original, False), (revised, True)):
                results.append(covariance(
                    truth['loading'][None], truth['phi'][None],
                    truth['a'][None] if options['kernel'] else None,
                    truth['width'][None] if options['kernel'] else None,
                    truth['tau'][None], truth['phi_u'][None], options['residual'], lag, flag))
            assert np.allclose(*results, atol=1e-12)
