import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm, truncnorm

from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.stage7_priors import (
    CANDIDATES,
    CURRENT_PRIOR_ID,
    level_settings,
    stationary_truth,
    verify_activation,
)
from cyber_threat_modelling.models.synthetic_state_space import build_model


def test_anchor_excludes_declared_fallback_future_and_masked_counts():
    counts = np.full((120, 3), 20., dtype=object)
    counts[:, 2] = 0
    counts[108:] = 1000000
    mask = np.ones((120, 3), dtype=bool)
    counts[0, 0], mask[0, 0] = np.nan, False
    data = {'N': counts, 'mask_N': mask}
    config = {'fallback_index': 2}
    settings = level_settings(data, config)
    assert settings['anchor'] == np.log(20.5)
    assert settings['fallback_anchor'] == np.log(.5)
    # A zero known series is still pooled; exclusion is by identity only.
    counts[:108, 1] = 0
    assert level_settings(data, config)['known_indices'] == [0, 1]
    counts[:108, :2] = np.nan
    with pytest.raises(ValueError, match='finite observed'):
        level_settings(data, config)


@pytest.mark.parametrize('candidate', list(CANDIDATES))
def test_level_density_and_fallback_factorization(candidate):
    data, truth, options = generate('fallback_zero', 7101)
    data = {key: value[:4] for key, value in data.items()}
    options.update(months=4, prior_id=CURRENT_PRIOR_ID, prior_candidate=candidate,
                   factor_parameterization='anchored')
    settings = level_settings(data, options)
    model = build_model(data, options, stationary_truth(truth, options))
    point = model.initial_point()
    point['m_known'] = np.linspace(1, 4, 9)
    point['m_fallback'] = np.array(-3.)
    prior = model.compile_logp(vars=[model['m_known'], model['m_fallback']], jacobian=False)
    expected = multivariate_normal.logpdf(
        point['m_known'], np.full(9, settings['anchor']),
        .7**2*np.eye(9)+settings['hyper_sd']**2*np.ones((9, 9)))
    expected += norm.logpdf(-3, settings['fallback_anchor'], 1)
    assert np.allclose(prior(point), expected)
    actual = model.compile_logp(vars=[model['m_sd']], jacobian=False)(point)
    expected = truncnorm.logpdf(.7, 0, settings['scale_upper']/settings['scale_sd'],
                                scale=settings['scale_sd'])
    assert np.allclose(actual, expected)
    known_density = model.compile_logp(vars=[model['m_known']], jacobian=False)
    before = known_density(point)
    point['m_fallback'] = np.array(-12.)
    assert known_density(point) == before
    assert np.isfinite(model.compile_dlogp(mode='NUMBA')(point)).all()


def test_second_activation_and_fallback_generator():
    root = Path(__file__).resolve().parents[1]
    assert verify_activation(root, CURRENT_PRIOR_ID)['prior_id'] == CURRENT_PRIOR_ID
    for seed in [7101, 7102, 7103]:
        data, truth, options = generate('fallback_zero', seed)
        assert options['fallback_index'] == 9
        assert truth['m'][-1] == np.log(.0001)
        assert np.isfinite(data['N']).all()
        assert np.all(truth['mu'][:, -1] > 0)


def test_histogram_trajectory_bootstrap_matches_explicit_resampling():
    root = Path(__file__).resolve().parents[1]
    bootstrap = runpy.run_path(str(root/'scripts/check_stage7_level_prior.py'))['uncertainty']
    counts = np.array([[[0, 3], [1, 4]], [[0, 8], [1, 9]], [[5, 4], [9, 12]]])
    result = bootstrap(counts, np.random.default_rng(42))
    rng = np.random.default_rng(42)
    direct = []
    for _ in range(200):
        values = counts[rng.integers(3, size=3)]
        direct.append([np.median(values), np.quantile(values, .999), np.mean(values == 0)])
    assert np.allclose(list(result.values()), np.quantile(direct, [.025, .975], axis=0).T)
