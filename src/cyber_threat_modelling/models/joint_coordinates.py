"""Exact joint level/factor coordinates with the original priors and likelihood."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .block_gaussian import block_gaussian


def joint_count_states(months, rank, phi, loading, center, covariance, dispersion,
                       counts, mask, effort, known, fallback):
    categories = counts.shape[1]
    adjusted = np.where(mask, counts, 0).astype(float)+.5
    weights = pt.where(mask, 1/(1/adjusted+1/dispersion[None, :]), 0.)
    residual = np.log(adjusted)-np.asarray(effort)[:, None]-center[None, :]
    innovation_var = 1-phi**2
    prior_diagonal = pt.concatenate([
        1/innovation_var,
        pt.tile((1+phi**2)/innovation_var, months-1),
        1/innovation_var,
    ]).reshape((months+1, rank))[:, :, None]*np.eye(rank)
    observation_blocks = pt.sum(weights[:, :, None, None]*loading[None, :, :, None]
                                *loading[None, :, None, :], axis=1)
    diagonal = pt.inc_subtensor(prior_diagonal[1:], observation_blocks)
    off = pt.diag(-phi/innovation_var)
    cross = pt.concatenate([pt.zeros((1, rank, categories)),
                            weights[:, None, :]*loading.T[None]], axis=0)
    natural = pt.concatenate([pt.zeros((1, rank)),
                              pt.sum(weights[:, :, None]*loading[None]
                                     *residual[:, :, None], axis=1)])
    zeros = pt.zeros((months+1, rank))
    solved_cross = pt.stack([block_gaussian(diagonal, off, cross[:, :, j], zeros)[0]
                             for j in range(categories)], axis=2)
    solved_natural = block_gaussian(diagonal, off, natural, zeros)[0]
    schur = (pt.linalg.inv(covariance)+pt.diag(weights.sum(axis=0))
             - pt.tensordot(cross, solved_cross, axes=[[0, 1], [0, 1]]))
    schur = .5*(schur+schur.T)
    level_natural = ((weights*residual).sum(axis=0)
                     - pt.tensordot(cross, solved_natural, axes=[[0, 1], [0, 1]]))
    chol = pt.linalg.cholesky(schur)
    z_level = pm.Flat('joint_level_whitened', shape=categories,
                      initval=np.zeros(categories))
    right = pt.slinalg.solve_triangular(chol, level_natural, lower=True)+z_level
    centered_level = pt.slinalg.solve_triangular(chol.T, right, lower=False)
    level = pm.Deterministic('m', center+centered_level)
    pm.Deterministic('m_known', level[known])
    if fallback is not None:
        pm.Deterministic('m_fallback', level[fallback])
    z_state = pm.Flat('f_whitened_states', shape=(months+1, rank),
                      initval=np.zeros((months+1, rank)))
    states, state_jacobian = block_gaussian(
        diagonal, off, natural-pt.tensordot(cross, centered_level, axes=[2, 0]), z_state)
    states = pm.Deterministic('f_states', states)
    log_prior = pm.logp(pm.MvNormal.dist(mu=center, cov=covariance), level)
    log_prior += pm.logp(pm.Normal.dist(0, 1), states[0]).sum()
    log_prior += pm.logp(pm.Normal.dist(phi*states[:-1], pt.sqrt(innovation_var)),
                         states[1:]).sum()
    pm.Potential('joint_gaussian_prior', log_prior+state_jacobian-pt.log(pt.diag(chol)).sum())
    return level, pm.Deterministic('f_contribution', states[1:] @ loading.T)
