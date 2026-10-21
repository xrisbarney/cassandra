import arviz as az
import numpy as np

from cyber_threat_modelling.evaluation.synthetic_recovery import future_counts
from cyber_threat_modelling.models.components import kernel_weights


def test_factor_forecast_means_against_analytic_lognormal_nb_mixture():
    draws, k, phi, load, state = 60000, 2, .7, .4, .8
    posterior = az.from_dict(posterior={
        'm': np.full((1, 1, k), np.log(10)), 'psi': np.full((1, 1, k), 8.),
        'f_phi': np.full((1, 1, 1), phi), 'f_states': np.full((1, 1, 2, 1), state),
        'f_diag': np.full((1, 1, 1), load), 'f_free': np.full((1, 1, 1), -load),
    }).posterior
    options = {'categories': k, 'factors': 1, 'conditional_level': False,
               'residual': 'none', 'kernel': False, 'regimes': 1}
    paths = future_counts(posterior, {'effort': np.zeros(5)}, {}, options,
                          np.zeros(draws, int), np.zeros(draws, int), np.random.default_rng(710))
    for h in (1, 3, 6, 12):
        variance = sum(phi**(2*j) for j in range(h))
        mean_state = phi**h * state
        expected = np.exp(np.log(10) + np.array([load, -load])*mean_state
                          + .5*load**2*variance)
        assert np.allclose(paths[:, h-1].mean(axis=0), expected, rtol=.015)
    # Opposite loadings imply negative cross-category dependence on the same state draw.
    assert np.corrcoef(paths[:, 0].T)[0, 1] < -.2


def test_kernel_forecast_carries_sampled_history_and_appends_only_new_innovations():
    draws, width, loading = 60000, 6., .5
    history = np.linspace(-1, 1, 60)
    posterior = az.from_dict(posterior={
        'm': np.full((1, 1, 1), np.log(10)), 'psi': np.full((1, 1, 1), 8.),
        'width': np.full((1, 1), width), 'a': np.full((1, 1, 1), loading),
        'xi': history[None, None],
    }).posterior
    options = {'categories': 1, 'factors': 0, 'conditional_level': False,
               'residual': 'none', 'kernel': True, 'regimes': 1}
    paths = future_counts(posterior, {'effort': np.zeros(5)}, {}, options,
                          np.zeros(draws, int), np.zeros(draws, int), np.random.default_rng(710))
    w = kernel_weights(width)
    for h in (1, 3, 6, 12):
        # At h, lags 0..h-1 refer to fresh Gaussian innovations; higher lags
        # refer to the same posterior history at the origin.
        conditional_mean = np.dot(w[h:], history[-(61-h):][::-1])
        conditional_variance = np.sum(w[:h]**2)
        expected = np.exp(np.log(10) + loading*conditional_mean
                          + .5*loading**2*conditional_variance)
        assert np.allclose(paths[:, h-1, 0].mean(), expected, rtol=.015)
