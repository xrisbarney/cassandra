"""Exact data-informed coordinates for stationary Gaussian factor processes.

The Gaussian observation model is only a coordinate proposal. The actual
count or censored-mark likelihood is still evaluated by the caller; neither
count observations nor censored endpoints become Gaussian-density observations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logit


def gaussian_states(prefix, months, rank, phi, loading, level, variance, proxy, mask):
    """Whiten a working Gaussian conditional, retaining the exact prior/Jacobian."""
    dimension = (months+1)*rank
    innovation_var = 1-phi**2
    diagonal = pt.concatenate([
        1/innovation_var,
        pt.tile((1+phi**2)/innovation_var, months-1),
        1/innovation_var,
    ])
    precision = pt.diag(diagonal)
    indices = np.arange(dimension-rank)
    off_diagonal = pt.tile(-phi/innovation_var, months)
    precision = pt.set_subtensor(precision[indices, indices+rank], off_diagonal)
    precision = pt.set_subtensor(precision[indices+rank, indices], off_diagonal)
    weights = pt.as_tensor_variable(mask.astype(float))/variance
    blocks = pt.sum(weights[:, :, None, None]*loading[None, :, :, None]
                    *loading[None, :, None, :], axis=1)
    locations = np.arange(1, months+1)[:, None]*rank+np.arange(rank)[None, :]
    precision = pt.inc_subtensor(precision[locations[:, :, None], locations[:, None, :]], blocks)
    natural = pt.concatenate([
        pt.zeros(rank),
        pt.sum(weights[:, :, None]*loading[None]*(proxy-level)[..., None], axis=1).ravel(),
    ])
    cholesky = pt.linalg.cholesky(precision)
    whitened = pm.Flat(prefix+'_whitened_states', shape=(months+1, rank),
                       initval=np.zeros((months+1, rank)))
    right = pt.slinalg.solve_triangular(cholesky, natural, lower=True)+whitened.ravel()
    states = pt.slinalg.solve_triangular(cholesky.T, right, lower=False).reshape((months+1, rank))
    states = pm.Deterministic(prefix+'_states', states)
    log_jacobian = -pt.sum(pt.log(pt.diag(cholesky)))
    log_prior = pm.logp(pm.Normal.dist(0, 1), states[0]).sum()
    log_prior += pm.logp(pm.Normal.dist(phi*states[:-1], pt.sqrt(innovation_var)),
                         states[1:]).sum()
    pm.Potential(prefix+'_state_prior', log_prior+log_jacobian)
    return pm.Deterministic(prefix+'_contribution', states[1:] @ loading.T)


def severity_states(months, rank, phi, loading, level, noise, marks, mask):
    # Mask before transformation; endpoints have finite coarsened logits.
    proxy = logit(np.where(mask, marks/10, .5))
    return gaussian_states('h', months, rank, phi, loading, level, noise[None, :]**2,
                           proxy, mask)


def count_states(months, rank, phi, loading, level, dispersion, counts, mask, effort):
    # A variance-stabilizing working approximation defines coordinates only.
    # The caller still uses the exact NB likelihood, including observed zeros.
    adjusted = np.where(mask, counts, 0).astype(float)+.5
    proxy = np.log(adjusted)
    variance = 1/adjusted+1/dispersion[None, :]
    return gaussian_states('f', months, rank, phi, loading, level+effort[:, None],
                           variance, proxy, mask)
