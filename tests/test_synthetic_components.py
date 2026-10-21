import itertools

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, nbinom, norm

from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.inference.hmm import forward, forward_backward, hmm_logp, recover
from cyber_threat_modelling.models.components import (
    censored_mark_logp,
    kernel_covariance,
    kernel_weights,
    symbolic_mark_logp,
)
from cyber_threat_modelling.models.synthetic_state_space import build_model


def enumerate_hmm(emission, transition):
    months, regimes = emission.shape
    paths = list(itertools.product(range(regimes), repeat=months))
    values = np.array([
        -np.log(regimes) + sum(emission[t, z] for t, z in enumerate(path))
        + sum(transition[a, b] for a, b in zip(path[:-1], path[1:], strict=True))
        for path in paths
    ])
    total = logsumexp(values)
    weights = np.exp(values - total)
    marginal = np.zeros_like(emission)
    for weight, path in zip(weights, paths, strict=True):
        for t, z in enumerate(path):
            marginal[t, z] += weight
    return total, marginal


@pytest.mark.parametrize("regimes", [1, 2, 3])
@pytest.mark.parametrize("months", [1, 3, 6])
def test_hmm_enumeration_and_smoothed_probabilities(regimes, months):
    rng = np.random.default_rng(710)
    transition = np.log(rng.dirichlet(np.ones(regimes), size=regimes))
    emission = rng.normal(-2, 4, size=(months, regimes))
    total, marginal = enumerate_hmm(emission, transition)
    actual, _ = forward(emission, transition)
    smoothed, expected_transitions = forward_backward(emission, transition)
    assert np.allclose(actual, total, atol=1e-12, rtol=0)
    assert np.allclose(smoothed, marginal, atol=1e-12, rtol=0)
    assert np.allclose(expected_transitions.sum(), months - 1)


def test_hmm_gradient_emission_and_transition_and_numba():
    e, q = pt.dmatrix("emission"), pt.dmatrix("log_transition")
    ll = hmm_logp(e, q)
    compiled = pytensor.function([e, q], [ll, *pt.grad(ll, [e, q])])
    numba = pytensor.function([e, q], [ll, *pt.grad(ll, [e, q])], mode="NUMBA")
    rng = np.random.default_rng(710)
    values = [rng.normal(size=(6, 3)), np.log(rng.dirichlet(np.ones(3), size=3))]
    result = compiled(*values)
    for actual, expected in zip(numba(*values), result, strict=True):
        assert np.allclose(actual, expected, atol=1e-10)
    for component in range(2):
        numerical = np.empty_like(values[component])
        for index in np.ndindex(numerical.shape):
            plus, minus = [v.copy() for v in values], [v.copy() for v in values]
            plus[component][index] += 1e-5
            minus[component][index] -= 1e-5
            numerical[index] = (compiled(*plus)[0] - compiled(*minus)[0]) / 2e-5
        assert np.allclose(numerical, result[component + 1], atol=1e-8)


def test_hmm_missing_months_extreme_emissions_and_ffbs():
    p = np.array([[0.8, 0.2], [0.1, 0.9]])
    missing = np.zeros((6, 2))
    assert abs(forward(missing, np.log(p))[0]) < 1e-12
    emission = np.array([[-1000., -1001.], [0., 0.], [-999., -1002.]])
    reference, marginal = enumerate_hmm(emission, np.log(p))
    assert np.allclose(forward(emission, np.log(p))[0], reference)
    rng = np.random.default_rng(710)
    paths = np.array([recover(emission, p, rng)["path"] for _ in range(10000)])
    assert np.allclose((paths == 1).mean(axis=0), marginal[:, 1], atol=0.02)


@pytest.mark.parametrize("upper,epsilon", [(1., 1e-4), (10., .005)])
def test_censored_marks_integrate_to_one_and_symbolic_gradients(upper, epsilon):
    loc, sd = -0.7, 3.0
    lo, hi = upper * epsilon, upper * (1 - epsilon)
    mass = np.exp(censored_mark_logp(lo, loc, sd, upper, epsilon))
    mass += np.exp(censored_mark_logp(hi, loc, sd, upper, epsilon))
    mass += quad(lambda x: np.exp(censored_mark_logp(x, loc, sd, upper, epsilon)), lo, hi)[0]
    assert np.allclose(mass, 1, atol=1e-8)
    location = pt.dscalar("location")
    points = np.array([lo, upper * .3, hi])
    density = symbolic_mark_logp(points, location, sd, upper, epsilon)
    fn = pytensor.function([location], [density, pt.grad(density.sum(), location)])
    actual, derivative = fn(loc)
    assert np.allclose(actual, censored_mark_logp(points, loc, sd, upper, epsilon))
    expected = (fn(loc + 1e-5)[0].sum() - fn(loc - 1e-5)[0].sum()) / 2e-5
    assert np.allclose(derivative, expected, atol=1e-8)


def test_kernel_covariance_normalization_psd_and_causality():
    for width in [.25, 6, 12, 18]:
        cov = kernel_covariance(width, np.arange(90))
        matrix = cov[np.abs(np.arange(80)[:, None] - np.arange(80))]
        assert abs(cov[0] - 1) < 1e-14
        assert np.all(cov[61:] == 0)
        assert np.linalg.eigvalsh(matrix).min() > -1e-12
        xi = np.arange(110, dtype=float)
        original = np.convolve(xi, kernel_weights(width), mode="valid")
        xi[90:] += 999
        changed = np.convolve(xi, kernel_weights(width), mode="valid")
        assert np.array_equal(changed[:30], original[:30])


def test_count_likelihood_independent_oracle_and_no_double_counted_prior():
    data, truth, options = generate("independent", 7101)
    model = build_model(data, options, truth)
    point = model.initial_point()
    m = point['m']
    psi = np.exp(point['psi_log__'])
    expected = nbinom.logpmf(data['N'], psi, psi / (psi + np.exp(m))).sum()
    actual = model.compile_logp(vars=model.potentials)(point)
    assert np.allclose(actual, expected, atol=1e-8)
    full = model.compile_logp()(point)
    prior = model.compile_logp(vars=model.free_RVs)(point)
    assert np.allclose(full, prior + expected, atol=1e-8)


def test_collapsed_hierarchical_level_matches_integrated_original_prior():
    levels, sd = np.array([0.3, 1.2, -0.8, 0.5]), 1.6
    integrated = quad(
        lambda mean: norm.pdf(mean, 0, 3) * np.prod(norm.pdf(levels, mean, sd)),
        -30, 30, epsabs=1e-13,
    )[0]
    covariance = sd**2 * np.eye(4) + 9 * np.ones((4, 4))
    expected = multivariate_normal.logpdf(levels, np.zeros(4), covariance)
    assert np.allclose(np.log(integrated), expected, atol=1e-10)
    data, truth, options = generate('independent', 7101)
    model = build_model(data, options, truth)
    point = model.initial_point()
    point['m'] = levels
    point['m_sd_log__'] = np.log(sd)
    actual = model.compile_logp(vars=[model['m']])(point)
    assert np.allclose(actual, expected, atol=1e-10)


def test_masked_invalid_values_never_enter_likelihood_or_gradient():
    data, truth, options = generate("severity_missing", 7101)
    assert not (data['mask_B'] & ~data['mask_N']).any()
    assert not (data['mask_E'] & ~data['mask_N']).any()
    # Small computational fixture while preserving all channel types.
    for key, value in list(data.items()):
        if key != 'M':
            data[key] = value[:16]
    options['months'] = 16
    data['mask_N'][3:5] = False
    data['N'][3:5] = np.nan
    model = build_model(data, options, truth)
    point = model.initial_point()
    val = model.compile_logp()(point)
    grad = model.compile_dlogp()(point)
    assert np.isfinite(val) and np.isfinite(grad).all()
    for name in ['N', 'B', 'E', 'V', 'D']:
        data[name][~data['mask_' + name]] = -np.inf
    altered = build_model(data, options, truth)
    assert np.allclose(altered.compile_logp()(point), val)
    assert np.allclose(altered.compile_dlogp()(point), grad)


def test_reporting_and_threat_sum_not_separately_identified_by_counts():
    data, truth, _ = generate('effort_none', 7101)
    report_mu = np.exp(truth['eta'] + truth['true_effort'][:, None])
    threat_eta = truth['eta'] + truth['true_effort'][:, None]
    assert np.array_equal(report_mu, np.exp(threat_eta))
    assert np.array_equal(generate('effort_oracle', 7101)[0]['N'], data['N'])


def test_regime_model_native_numba_gradient_and_independent_emission_enumeration():
    data, truth, options = generate('regimes', 7101)
    for name in data:
        data[name] = data[name][:4]
    options['months'] = 4
    model = build_model(data, options, truth)
    point = model.initial_point()
    native = model.compile_dlogp()(point)
    assert np.isfinite(native).all()
    assert np.allclose(model.compile_dlogp(mode='NUMBA')(point), native, atol=1e-8)
    emissions, transition = model.replace_rvs_by_values(
        [model.synthetic_emissions, model.synthetic_transition]
    )
    fn = model.compile_fn([emissions, transition], inputs=model.value_vars,
                          on_unused_input='ignore')
    e, p = fn(point)
    exact, _ = enumerate_hmm(e, np.log(p))
    assert np.allclose(model.compile_logp(vars=model.potentials)(point), exact)
