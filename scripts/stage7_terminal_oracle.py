"""Independent dense SciPy density for the declared factor-only count targets."""

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import expit, log_expit
from scipy.stats import multivariate_normal, nbinom, norm, truncnorm


def dense_density(point, data, options, *, fixed_phi=None, restricted_bound=None):
    t, k = data['N'].shape
    rank = options['factors']
    settings = options['level_settings']
    logp = 0.

    def bounded(name, lower, upper):
        nonlocal logp
        z = point[name+'_interval__']
        logp += np.sum(np.log(upper-lower)+log_expit(z)+log_expit(-z))
        return lower+(upper-lower)*expit(z)

    sd = bounded('m_sd', 0., settings['scale_upper'])
    logp += truncnorm.logpdf(sd, 0., settings['scale_upper']/settings['scale_sd'],
                            scale=settings['scale_sd'])
    known = settings['known_indices']
    fallback = settings['fallback_index']
    cov = np.zeros((k, k))
    cov[np.ix_(known, known)] = sd**2*np.eye(len(known))+settings['hyper_sd']**2
    center = np.full(k, settings['anchor'])
    if fallback is not None:
        cov[fallback, fallback] = 1.
        center[fallback] = settings['fallback_anchor']
    psi = np.exp(point['psi_log__'])
    logp += norm.logpdf(point['psi_log__'], np.log(10), 1).sum()
    if fixed_phi is not None:
        phi = np.asarray(fixed_phi)
    elif restricted_bound is not None:
        from scipy.stats import beta

        z = point['persistence_unit_logodds__']
        q = expit(z)
        phi = restricted_bound*(2*q-1)
        logp += (beta.logpdf(q, 2, 2)+log_expit(z)+log_expit(-z)).sum()
    else:
        phi = bounded('f_phi', -1., 1.)
        logp -= rank*np.log(2.)
    scale = settings['amplitude_multiplier']/np.sqrt(rank)
    bound = 2*scale
    diag = bounded('f_diag', 0., bound)
    free = bounded('f_free', -bound, bound)
    logp += truncnorm.logpdf(diag, 0., 2., scale=scale).sum()
    logp += truncnorm.logpdf(free, -2., 2., scale=scale).sum()
    loading = np.zeros((k, rank))
    loading[np.arange(rank), np.arange(rank)] = diag
    rows, cols = np.where(np.tril(np.ones((k, rank)), -1))
    loading[rows, cols] = free

    size = (t+1)*rank
    precision = np.zeros((size, size))
    for j in range(rank):
        indices = np.arange(t+1)*rank+j
        block = np.diag(np.r_[1., np.full(t-1, 1+phi[j]**2), 1.])
        block += np.diag(np.full(t, -phi[j]), 1)+np.diag(np.full(t, -phi[j]), -1)
        precision[np.ix_(indices, indices)] = block/(1-phi[j]**2)
    adjusted = np.where(data['mask_N'], data['N'], 0)+.5
    weights = np.where(data['mask_N'], 1/(1/adjusted+1/psi), 0.)
    proxy = np.log(adjusted)-data['effort'][:, None]
    joint = 'joint_level_whitened' in point
    if not joint:
        m = np.empty(k)
        m[known] = point['m_known']
        if fallback is not None:
            m[fallback] = point['m_fallback']
    baseline = center if joint else m
    natural = np.zeros(size)
    cross = np.zeros((k, size))
    for month in range(t):
        ix = slice((month+1)*rank, (month+2)*rank)
        precision[ix, ix] += loading.T @ (weights[month, :, None]*loading)
        natural[ix] = loading.T @ (weights[month]*(proxy[month]-baseline))
        cross[:, ix] = weights[month, :, None]*loading
    chol = cholesky(precision, lower=True)
    if joint:
        schur = np.linalg.inv(cov)+np.diag(weights.sum(axis=0))
        schur -= cross @ cho_solve((chol, True), cross.T)
        schur = (schur+schur.T)/2
        cm = cholesky(schur, lower=True)
        rhs = (weights*(proxy-center)).sum(axis=0)-cross @ cho_solve((chol, True), natural)
        delta = cho_solve((cm, True), rhs)
        delta += solve_triangular(cm.T, point['joint_level_whitened'], lower=False)
        m = center+delta
        natural -= cross.T @ delta
        logp -= np.log(np.diag(cm)).sum()
    states = cho_solve((chol, True), natural)
    states += solve_triangular(chol.T, point['f_whitened_states'].ravel(), lower=False)
    states = states.reshape((t+1, rank))
    logp -= np.log(np.diag(chol)).sum()
    logp += multivariate_normal.logpdf(m, mean=center, cov=cov)
    logp += norm.logpdf(states[0]).sum()
    logp += norm.logpdf(states[1:], phi*states[:-1], np.sqrt(1-phi**2)).sum()
    eta = m+states[1:] @ loading.T+data['effort'][:, None]
    mask = data['mask_N']
    logp += nbinom.logpmf(data['N'], psi, psi/(psi+np.exp(eta)))[mask].sum()
    return float(logp), {'m': m, 'f_states': states, 'phi': phi, 'loading': loading,
                         'shift_prior_curvature': np.array([
                             loading[:, j] @ np.linalg.solve(cov, loading[:, j])
                             +1+t*(1-phi[j])/(1+phi[j]) for j in range(rank)])}
