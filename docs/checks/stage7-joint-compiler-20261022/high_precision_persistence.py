"""Independent 60-digit target derivative at the diagnosed saved state.

Reconstruct joint Gaussian coordinates using Decimal block Cholesky algebra,
then evaluate all persistence-dependent original-prior, likelihood and Jacobian
terms. Terms constant in persistence are omitted. No model implementation is
imported. This checks a visited state, not the rejected leapfrog proposal.
"""
import json
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

import numpy as np

from cyber_threat_modelling.provenance import file_hash


def decimal_array(values):
    return np.array([D(float(x)) for x in np.asarray(values).ravel()], dtype=object).reshape(
        np.shape(values))


def zeros(shape):
    return np.full(shape, D(0), dtype=object)


def lower_cholesky(matrix):
    out = zeros(matrix.shape)
    for i in range(len(matrix)):
        for j in range(i + 1):
            value = matrix[i, j] - sum(out[i, p] * out[j, p] for p in range(j))
            out[i, j] = value.sqrt() if i == j else value / out[j, j]
    return out


def triangular(matrix, rhs, transpose=False):
    out = zeros(rhs.shape)
    indices = range(len(matrix) - 1, -1, -1) if transpose else range(len(matrix))
    for i in indices:
        previous = range(i + 1, len(matrix)) if transpose else range(i)
        value = rhs[i] if rhs.ndim == 1 else rhs[i].copy()
        for j in previous:
            value -= (matrix[j, i] if transpose else matrix[i, j]) * out[j]
        out[i] = value / matrix[i, i]
    return out


def sigmoid(value):
    return D(1) / (D(1) + (-value).exp())


def main():
    folder = Path('artifacts/stage7/rank-two-joint-v1/primary/fallback_zero-7102')
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    options, settings = run['options'], run['options']['level_settings']
    assert options['regimes'] == 1 and options['factors'] == 2
    assert options['residual'] == 'none' and not options['kernel']
    with np.load(folder / 'data.npz') as f:
        data = dict(f)
    with np.load(folder / 'failed_state_0.npz') as f:
        point = {key: decimal_array(f[key]) for key in f.files}
    t, k = data['N'].shape
    center = decimal_array(np.full(k, settings['anchor']))
    center[settings['fallback_index']] = D(settings['fallback_anchor'])
    sd = D(settings['scale_upper']) * sigmoid(point['m_sd_interval__'].item())
    known, fallback = settings['known_indices'], settings['fallback_index']
    inverse_cov = zeros((k, k))
    variance, hyper = sd * sd, D(settings['hyper_sd']) ** 2
    for i in known:
        for j in known:
            inverse_cov[i, j] = (D(i == j) / variance
                                 - hyper / (variance * (variance + len(known) * hyper)))
    inverse_cov[fallback, fallback] = D(1)
    bound = D(float(2 * settings['amplitude_multiplier'] / np.sqrt(2)))
    loading = zeros((k, 2))
    for j in range(2):
        loading[j, j] = bound * sigmoid(point['f_diag_interval__'][j])
    rows, columns = np.where(np.tril(np.ones((k, 2)), -1))
    for index, (i, j) in enumerate(zip(rows, columns, strict=True)):
        loading[i, j] = bound * (2 * sigmoid(point['f_free_interval__'][index]) - 1)
    psi = np.array([x.exp() for x in point['psi_log__']], dtype=object)
    weights, residual = zeros((t, k)), zeros((t, k))
    for month in range(t):
        for category in range(k):
            if data['mask_N'][month, category]:
                adjusted = D(int(data['N'][month, category])) + D('.5')
                weights[month, category] = 1 / (1 / adjusted + 1 / psi[category])
                residual[month, category] = (adjusted.ln() - D(float(data['effort'][month]))
                                             - center[category])
    cross, natural = zeros((t + 1, 2, k)), zeros((t + 1, 2))
    observation = zeros((t + 1, 2, 2))
    for month in range(t):
        cross[month + 1] = loading.T * weights[month]
        natural[month + 1] = cross[month + 1] @ residual[month]
        observation[month + 1] = cross[month + 1] @ loading
    level_natural = (weights * residual).sum(axis=0)
    level_precision = inverse_cov + np.diag(weights.sum(axis=0))

    def target(transformed_phi):
        phi = np.array([2 * sigmoid(x) - 1 for x in transformed_phi], dtype=object)
        q = 1 - phi * phi
        lower, below = zeros((t + 1, 2, 2)), zeros((t + 1, 2, 2))
        off = np.diag(-phi / q)
        logdet_lower = D(0)
        for month in range(t + 1):
            diagonal = observation[month] + np.diag(
                1 / q if month in (0, t) else (1 + phi * phi) / q)
            if month:
                below[month] = triangular(lower[month - 1], off.T).T
                diagonal -= below[month] @ below[month].T
            lower[month] = lower_cholesky(diagonal)
            logdet_lower += sum(x.ln() for x in np.diag(lower[month]))

        def solve(rhs, whitened=None):
            forward, result = zeros(rhs.shape), zeros(rhs.shape)
            for month in range(t + 1):
                value = rhs[month].copy()
                if month:
                    value -= below[month] @ forward[month - 1]
                forward[month] = triangular(lower[month], value)
            for month in range(t, -1, -1):
                value = forward[month].copy()
                if whitened is not None:
                    value += whitened[month]
                if month < t:
                    value -= below[month + 1].T @ result[month + 1]
                result[month] = triangular(lower[month], value, transpose=True)
            return result

        solved_cross, solved_natural = solve(cross), solve(natural)
        schur, rhs = level_precision.copy(), level_natural.copy()
        for month in range(t + 1):
            schur -= cross[month].T @ solved_cross[month]
            rhs -= cross[month].T @ solved_natural[month]
        chol_level = lower_cholesky((schur + schur.T) / 2)
        level = triangular(chol_level, triangular(chol_level, rhs)
                           + point['joint_level_whitened'], transpose=True)
        conditional = natural.copy()
        for month in range(t + 1):
            conditional[month] -= cross[month] @ level
        states = solve(conditional, point['f_whitened_states'])
        result = -(level @ inverse_cov @ level) / 2 - sum(states[0] ** 2) / 2
        result -= D(t) * sum(x.ln() for x in q) / 2
        for month in range(1, t + 1):
            result -= sum((states[month] - phi * states[month - 1]) ** 2 / q) / 2
            eta = center + level + loading @ states[month] + D(float(data['effort'][month - 1]))
            for category in range(k):
                if data['mask_N'][month - 1, category]:
                    y = D(int(data['N'][month - 1, category]))
                    result += y * eta[category] - (y + psi[category]) * (
                        psi[category] + eta[category].exp()).ln()
        result -= logdet_lower + sum(x.ln() for x in np.diag(chol_level))
        result += sum(x.ln() for x in q)  # Uniform-to-unconstrained Jacobians, up to constants.
        return result

    reference = Path('docs/checks/stage7-saved-state-numerics-20261022/difference_steps.json')
    previous = json.loads(reference.read_text(encoding='utf-8'))
    records = []
    for index in range(2):
        estimates = []
        for step in [D('0.001'), D('0.00001'), D('0.0000001')]:
            plus, minus = point['f_phi_interval__'].copy(), point['f_phi_interval__'].copy()
            plus[index] += step
            minus[index] -= step
            estimate = (target(plus) - target(minus)) / (2 * step)
            estimates.append({'step': str(step), 'derivative': str(estimate)})
        final = float(estimates[-1]['derivative'])
        analytic = previous['records'][index]['analytic']
        np.testing.assert_allclose(final, analytic, rtol=5e-5, atol=5e-5)
        records.append({'index': index, 'analytic_float64': analytic, 'differences': estimates,
                        'absolute_error': abs(final - analytic), 'original_tolerance_passed': True})
        print(json.dumps(records[-1]), flush=True)
    output = reference.with_name('high_precision.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'source_sha256': file_hash(Path(__file__)), 'decimal_precision': 60,
                   'checkpoint_sha256': file_hash(folder / 'failed_state_0.npz'),
                   'reference_sha256': file_hash(reference), 'records': records,
                   'passed': True, 'interpretation': __doc__}, stream, indent=2)


if __name__ == '__main__':
    with localcontext() as context:
        context.prec = 60
        main()
