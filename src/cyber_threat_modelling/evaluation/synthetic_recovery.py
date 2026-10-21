"""Posterior recovery and conditional forecasts on generated data only."""

import numpy as np
from scipy.special import expit
from scipy.stats import nbinom

from ..inference.hmm import recover
from ..models.stage7_priors import amended


def posterior_array(posterior, name):
    value = posterior[name].stack(sample=("chain", "draw"))
    return value.transpose("sample", *[d for d in value.dims if d != "sample"]).values


def interval_metrics(samples, truth):
    truth = np.asarray(truth)
    result = {"rmse": float(np.sqrt(np.mean((samples.mean(axis=0) - truth)**2)))}
    for level in (.5, .9):
        tail = (1 - level) / 2
        lo, hi = np.quantile(samples, [tail, 1 - tail], axis=0)
        result[f"coverage_{int(100 * level)}"] = float(np.mean((truth >= lo) & (truth <= hi)))
        result[f"width_{int(100 * level)}"] = float(np.mean(hi - lo))
    return result


def reconstructed_loading(posterior, prefix, k, rank, indices):
    diag = posterior_array(posterior, prefix + '_diag')[indices]
    free = posterior_array(posterior, prefix + '_free')[indices]
    matrix = np.zeros((len(indices), k, rank))
    matrix[:, np.arange(rank), np.arange(rank)] = diag
    rows, cols = np.where(np.tril(np.ones((k, rank)), -1))
    matrix[:, rows, cols] = free
    return matrix


def summarize(posterior, data, truth, options, seed):
    rng = np.random.default_rng(seed)
    eta_regime = posterior_array(posterior, 'eta_by_regime')
    n, t, regimes, k = eta_regime.shape
    psi = posterior_array(posterior, 'psi')
    terminal_z = np.zeros(n, dtype=int)
    regime_report = None
    if regimes == 1:
        eta = eta_regime[:, :, 0]
    else:
        eta = np.empty((n, t, k))
        smooth, filtered = np.zeros((t, regimes)), np.zeros((t, regimes))
        transitions = posterior_array(posterior, 'transition')
        for draw in range(n):
            means = np.exp(eta_regime[draw] + data['effort'][:, None, None])
            emission = nbinom.logpmf(
                data['N'][:, None, :], psi[draw], psi[draw] / (psi[draw] + means)
            )
            emission = np.where(data['mask_N'][:, None, :], emission, 0).sum(axis=2)
            rec = recover(emission, transitions[draw], rng)
            eta[draw] = eta_regime[draw, np.arange(t), rec['path']]
            terminal_z[draw] = rec['path'][-1]
            smooth += rec['smoothed'] / n
            filtered += rec['filtered'] / n
        regime_report = {
            'smoothed_accuracy': float((smooth.argmax(axis=1) == truth['regime']).mean()),
            'filtered_accuracy_retrospective': float(
                (filtered.argmax(axis=1) == truth['regime']).mean()
            ),
            'smoothed_probabilities': smooth.tolist(),
            'within_fit_filtered_probabilities': filtered.tolist(),
            'known_transitions': (np.flatnonzero(np.diff(truth['regime'])) + 1).tolist(),
            'interpretation': 'fixed-path stress case; within-fit filtering is retrospective',
        }
    mu = np.exp(eta + data['effort'][None, :, None])
    # Reference intensity changes with the working effort proxy.
    target_eta = np.log(truth['mu']) - data['effort'][:, None]
    predictive = rng.negative_binomial(psi[:, None, :], psi[:, None, :] / (psi[:, None, :] + mu))
    result = {
        'eta_operational': interval_metrics(eta, target_eta),
        'eta_generating_interpretation': interval_metrics(eta, truth['eta']),
        'expected_observed_counts': interval_metrics(mu, truth['mu']),
        'training_posterior_predictive': interval_metrics(
            predictive[:, data['mask_N']], data['N'][data['mask_N']]
        ),
        'parameters': {}, 'regimes': regime_report,
    }
    mapping = {'m': 'm', 'psi': 'psi', 'tau': 'tau', 'phi_u': 'phi_u', 'f_phi': 'phi',
               'h_phi': 'phi_h', 'n': 'n', 'kappa': 'kappa', 'width': 'width', 'a': 'a'}
    for name, target in mapping.items():
        if name not in posterior or target not in truth:
            continue
        samples = posterior_array(posterior, name)
        result['parameters'][name] = {
            'truth': truth[target].tolist(), 'mean': samples.mean(axis=0).tolist(),
            'sd': samples.std(axis=0).tolist(),
            'q05': np.quantile(samples, .05, axis=0).tolist(),
            'q95': np.quantile(samples, .95, axis=0).tolist(),
            **interval_metrics(samples, truth[target]),
        }
    if 'tau' in posterior:
        tau = posterior_array(posterior, 'tau')
        result['tau_logpsi_correlation'] = [
            float(np.corrcoef(tau[:, j], np.log(psi[:, j]))[0, 1]) for j in range(k)
        ]
        # Marginal relative overdispersion after integrating iid Gaussian noise.
        if options['residual'] == 'iid':
            combination = np.exp(tau**2) * (1 + 1 / psi) - 1
            true_combination = np.exp(truth['tau']**2) * (1 + 1 / truth['psi']) - 1
            result['combined_relative_overdispersion'] = interval_metrics(
                combination, true_combination
            )
    if options['kernel']:
        result['kernel_loading_norm'] = {
            **interval_metrics(posterior_array(posterior, 'a_norm'), np.linalg.norm(truth['a'])),
            'mean': float(posterior_array(posterior, 'a_norm').mean()),
            'q90': float(np.quantile(posterior_array(posterior, 'a_norm'), .9)),
        }
    if options['severity_factors']:
        sigma = expit(posterior_array(posterior, 'severity_location'))
        result['severity_location'] = interval_metrics(sigma, truth['sigma'])
    # Every sample keeps its parameter tuple and terminal state throughout the future path.
    indices = rng.integers(n, size=6000)
    paths = future_counts(posterior, data, truth, options, indices, terminal_z[indices], rng)
    result['future_predictive'] = {
        str(h): interval_metrics(paths[:, h-1], truth['future_counts'][h-1])
        for h in (1, 3, 6, 12)
    }
    return result, paths


def future_counts(posterior, data, truth, options, indices, z, rng):
    revised = amended(options)
    size, k = len(indices), options['categories']
    get = lambda name: posterior_array(posterior, name)[indices]  # noqa: E731
    m = np.broadcast_to(truth['m'], (size, k)) if options['conditional_level'] else get('m')
    psi = get('psi')
    rank = options['factors']
    if rank:
        state = get('f_states')[:, -1].copy()
        phi = get('f_phi')
        loading = reconstructed_loading(posterior, 'f', k, rank, indices)
    if options['residual'] != 'none':
        tau = get('tau')
        u = get('u')[:, -1].copy()
        if options['residual'] == 'ar':
            phi_u = get('phi_u')
    if options['kernel']:
        xi = get('xi')[:, -60:].copy()
        width, a = get('width'), get('a')
        raw = expit((width[:, None] - np.abs(np.arange(61) - 30)) / 2)
        weights = raw / np.linalg.norm(raw, axis=1)[:, None]
    if options['regimes'] > 1:
        transition = get('transition')
        shifts = np.concatenate(
            [np.zeros((size, 1)), np.cumsum(get('regime_increments'), axis=1)], 1
        )
    paths = np.empty((size, 12, k), dtype=np.int64)
    for h in range(12):
        eta = m.copy()
        if rank:
            state = phi * state + rng.normal(size=state.shape) * (
                np.sqrt(1 - phi**2) if revised else 1)
            eta += np.einsum('skr,sr->sk', loading, state)
        if options['residual'] != 'none':
            noise = tau * rng.normal(size=(size, k))
            if revised and options['residual'] == 'ar':
                noise *= np.sqrt(1 - phi_u**2)
            u = phi_u * u + noise if options['residual'] == 'ar' else noise
            eta += u
        if options['kernel']:
            extended = np.column_stack([xi, rng.normal(size=size)])
            g = (extended[:, ::-1] * weights).sum(axis=1)
            xi = extended[:, 1:]
            eta += g[:, None] * a
        if options['regimes'] > 1:
            cdf = np.cumsum(transition[np.arange(size), z], axis=1)
            z = (rng.random(size)[:, None] > cdf).sum(axis=1)
            eta += shifts[np.arange(size), z, None]
        means = np.exp(eta + data['effort'][-1])
        paths[:, h] = rng.negative_binomial(psi, psi / (psi + means))
    return paths
