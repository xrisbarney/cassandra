import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nbinom

from cyber_threat_modelling.evaluation import stage7_supplement as supplement
from cyber_threat_modelling.models.stage7_priors import level_settings
from cyber_threat_modelling.models.synthetic_state_space import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('case', [c['id'] for c in supplement.declaration(ROOT)['cases']])
def test_frozen_fixture_reproduces_declared_intervention(case):
    loaded = supplement.load_fixture(ROOT, case)
    made = supplement.make_fixture(ROOT, case)
    for a, b in zip(loaded[:3], made[:3]):
        assert set(a) == set(b)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert loaded[3] == made[3]
    data, truth, reference, options = loaded
    assert data['N'].shape == (108, 10)
    assert options['level_settings'] == level_settings(reference, options)
    assert options['regimes'] == 1 and not options['kernel']
    if case.startswith('reduced'):
        assert options['factors'] == 1 and truth['loading'].shape == (10, 2)
        np.testing.assert_array_equal(data['N'], reference['N'])
    if case == 'masked_fallback':
        np.testing.assert_array_equal(data['N'], reference['N'])
        assert data['mask_N'][:, :9].all() and not data['mask_N'][:, 9].any()
        assert reference['mask_N'].all()


def test_matched_fixture_uses_one_factor_truth_and_fixed_count_rng():
    data, truth, _, _ = supplement.load_fixture(ROOT, 'matched_one_factor_fallback')
    np.testing.assert_allclose(truth['eta'], truth['m'] + truth['f'][1:] @ truth['loading'].T)
    assert truth['phi'].shape == (1,) and truth['f'].shape == (109, 1)
    rng = np.random.default_rng(78101)
    expected = rng.negative_binomial(truth['psi'], truth['psi']/(truth['psi']+truth['mu']))
    np.testing.assert_array_equal(data['N'], expected)
    expected_future = rng.negative_binomial(
        truth['psi'], truth['psi']/(truth['psi']+truth['future_mu']))
    np.testing.assert_array_equal(truth['future_counts'], expected_future)


def test_undeclared_case_and_changed_fixture_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match='Undeclared'):
        supplement.load_fixture(ROOT, 'extra_retry')
    original = supplement.file_hash
    monkeypatch.setattr(supplement, 'file_hash',
                        lambda p: 'changed' if p.name == 'data.npz' else original(p))
    with pytest.raises(ValueError, match='Changed'):
        supplement.load_fixture(ROOT, 'reduced_fallback')


def test_masked_likelihood_matches_known_category_nb_without_changing_anchor():
    data, truth, reference, options = supplement.load_fixture(ROOT, 'masked_fallback')
    data = {k: v[:4].copy() for k, v in data.items()}
    reference = {k: v[:4].copy() for k, v in reference.items()}
    options = dict(options, months=4)
    model = build_model(data, options, truth, level_reference=reference)
    function = model.compile_fn(model.replace_rvs_by_values([
        model['eta_by_regime'], model['psi'], model['observations']]),
        inputs=model.value_vars, mode='FAST_COMPILE', on_unused_input='ignore')
    eta, psi, observed = function(model.initial_point())
    mu = np.exp(eta[:, 0, :9])
    expected = nbinom.logpmf(data['N'][:, :9], psi[:9], psi[:9]/(psi[:9]+mu)).sum()
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)
    assert level_settings(reference, options)['fallback_anchor'] == np.log(.5)


def test_prior_reference_cannot_change_counts_or_mask_known_categories():
    data, truth, reference, options = supplement.load_fixture(ROOT, 'masked_fallback')
    data['N'][0, 0] += 1
    with pytest.raises(ValueError, match='preserve the original counts'):
        build_model(data, options, truth, level_reference=reference)
    data['N'][0, 0] -= 1
    data['mask_N'][0, 0] = False
    with pytest.raises(ValueError, match='Only the declared fallback'):
        build_model(data, options, truth, level_reference=reference)


def test_single_attempt_budget_cannot_be_escalated():
    sys.path.insert(0, str(ROOT/'scripts'))
    try:
        spec = importlib.util.spec_from_file_location('supplement_runner',
                                                     ROOT/'scripts/run_stage7_supplement.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record = supplement.declaration(ROOT)
        assert module.budget(record) == (4, 2000, 2000, .99, 15, 7103, 7301)
        with pytest.raises(ValueError, match='budget or seeds'):
            module.budget(dict(record, retained_draws=4000))
        with pytest.raises(ValueError, match='budget or seeds'):
            module.budget(dict(record, attempts_per_case=2))
    finally:
        sys.path.pop(0)


def test_runner_refuses_existing_case_and_hosted_rerun(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT/'scripts'))
    try:
        spec = importlib.util.spec_from_file_location('supplement_runner_guard',
                                                     ROOT/'scripts/run_stage7_supplement.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '2')
        with pytest.raises(ValueError, match='Hosted reruns'):
            module.fit(tmp_path, 'reduced_fallback', tmp_path/'unused')
        monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '1')
        monkeypatch.setattr(module, 'verify_fit_authority', lambda *args: {})
        record = supplement.declaration(ROOT)
        monkeypatch.setattr(module, 'declaration', lambda *args: record)
        output = tmp_path/'artifacts/stage7/supplement-v1/reduced_fallback'
        output.mkdir(parents=True)
        marker = output/'preserved.txt'
        marker.write_text('original evidence')
        with pytest.raises(FileExistsError):
            module.fit(tmp_path, 'reduced_fallback', output)
        assert marker.read_text() == 'original evidence'
    finally:
        sys.path.pop(0)
