"""Exact data-informed coordinates for a stationary Gaussian severity process.

The Gaussian observation model is only a coordinate proposal. The actual
censored likelihood is still evaluated by the caller, so endpoint observations
do not turn into Gaussian-density observations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logit


def severity_states(months, rank, phi, loading, level, noise, marks, mask):
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
    # Mask before transformation; endpoints have finite coarsened logits.
    proxy = logit(np.where(mask, marks/10, .5))
    weights = pt.as_tensor_variable(mask.astype(float))/noise[None, :]**2
    blocks = pt.sum(weights[:, :, None, None]*loading[None, :, :, None]
                    *loading[None, :, None, :], axis=1)
    locations = np.arange(1, months+1)[:, None]*rank+np.arange(rank)[None, :]
    precision = pt.inc_subtensor(precision[locations[:, :, None], locations[:, None, :]], blocks)
    natural = pt.concatenate([
        pt.zeros(rank),
        pt.sum(weights[:, :, None]*loading[None]*(proxy-level)[..., None], axis=1).ravel(),
    ])
    cholesky = pt.linalg.cholesky(precision)
    whitened = pm.Flat('h_whitened_states', shape=(months+1, rank),
                       initval=np.zeros((months+1, rank)))
    right = pt.slinalg.solve_triangular(cholesky, natural, lower=True)+whitened.ravel()
    states = pt.slinalg.solve_triangular(cholesky.T, right, lower=False).reshape((months+1, rank))
    states = pm.Deterministic('h_states', states)
    log_jacobian = -pt.sum(pt.log(pt.diag(cholesky)))
    log_prior = pm.logp(pm.Normal.dist(0, 1), states[0]).sum()
    log_prior += pm.logp(pm.Normal.dist(phi*states[:-1], pt.sqrt(innovation_var)),
                         states[1:]).sum()
    pm.Potential('h_state_prior', log_prior+log_jacobian)
    return pm.Deterministic('h_contribution', states[1:] @ loading.T)
