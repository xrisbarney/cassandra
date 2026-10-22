"""Exact data-informed coordinates for stationary Gaussian factor processes.

The Gaussian observation model is only a coordinate proposal. The actual
count or censored-mark likelihood is still evaluated by the caller; neither
count observations nor censored endpoints become Gaussian-density observations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logit


def iid_count_states(tau, psi, base, counts, mask, effort):
    """Exact working-conditional transform of iid standard-Normal innovations."""
    observed = np.where(mask, counts, 0).astype(float)
    adjusted = observed + .5
    weight = pt.where(mask, adjusted*psi/(adjusted+psi), 0.)
    variance = 1/(1+tau**2*weight)
    location = tau*weight*(np.log(adjusted)-np.asarray(effort)[:, None]-base)*variance
    z = pm.Flat('u_whitened', shape=observed.shape, initval=np.zeros_like(observed))
    innovation = location+pt.sqrt(variance)*z
    pm.Potential('u_state_prior', pm.logp(pm.Normal.dist(0, 1), innovation).sum()
                 + .5*pt.log(variance).sum())
    return tau*innovation


def _small_cholesky(matrix, rank):
    if rank == 1:
        return pt.sqrt(matrix)
    if rank == 2:
        first = pt.sqrt(matrix[0, 0])
        below = matrix[1, 0]/first
        return pt.stack([pt.stack([first, pt.zeros_like(first)]),
                         pt.stack([below, pt.sqrt(matrix[1, 1]-below**2)])])
    return pt.linalg.cholesky(matrix)


def _small_solve(lower, right, rank, transpose=False):
    if rank == 1:
        return right/lower[0, 0]
    if rank == 2:
        if transpose:
            second = right[1]/lower[1, 1]
            first = (right[0]-lower[1, 0]*second)/lower[0, 0]
        else:
            first = right[0]/lower[0, 0]
            second = (right[1]-lower[1, 0]*first)/lower[1, 1]
        return pt.stack([first, second])
    return pt.slinalg.solve_triangular(lower.T if transpose else lower, right,
                                       lower=not transpose)


def symbolic_block_conditional(diagonal, off_diagonal, natural, whitened, rank):
    """Block Cholesky and two triangular solves in time order, without dense P."""
    import pytensor

    first = _small_cholesky(diagonal[0], rank)
    first_right = _small_solve(first, natural[0], rank)

    def forward(diag, rhs, previous_chol, previous_right, off):
        below = _small_solve(previous_chol, off.T, rank).T
        chol = _small_cholesky(diag-below @ below.T, rank)
        right = _small_solve(chol, rhs-below @ previous_right, rank)
        return chol, right, below

    (chol, right, below), _ = pytensor.scan(
        forward, sequences=[diagonal[1:], natural[1:]],
        outputs_info=[first, first_right, None], non_sequences=[off_diagonal], strict=True,
    )
    all_chol = pt.concatenate([first[None], chol])
    all_right = pt.concatenate([first_right[None], right])+whitened
    last_state = _small_solve(all_chol[-1], all_right[-1], rank, transpose=True)

    def backward(chol, right, next_below, next_state):
        return _small_solve(chol, right-next_below.T @ next_state, rank, transpose=True)

    reversed_states, _ = pytensor.scan(
        backward, sequences=[all_chol[:-1][::-1], all_right[:-1][::-1], below[::-1]],
        outputs_info=[last_state], strict=True,
    )
    states = pt.concatenate([reversed_states[::-1], last_state[None]])
    log_jacobian = -pt.log(all_chol[:, np.arange(rank), np.arange(rank)]).sum()
    return states, log_jacobian


def block_conditional(diagonal, off_diagonal, natural, whitened, rank):
    from .block_gaussian import block_gaussian

    return block_gaussian(diagonal, off_diagonal, natural, whitened)


def gaussian_states(prefix, months, rank, phi, loading, level, variance, proxy, mask,
                    solver='dense'):
    """Whiten a working Gaussian conditional, retaining the exact prior/Jacobian."""
    dimension = (months+1)*rank
    innovation_var = 1-phi**2
    diagonal = pt.concatenate([
        1/innovation_var,
        pt.tile((1+phi**2)/innovation_var, months-1),
        1/innovation_var,
    ])
    weights = pt.as_tensor_variable(mask.astype(float))/variance
    blocks = pt.sum(weights[:, :, None, None]*loading[None, :, :, None]
                    *loading[None, :, None, :], axis=1)
    natural = pt.concatenate([
        pt.zeros((1, rank)),
        pt.sum(weights[:, :, None]*loading[None]*(proxy-level)[..., None], axis=1),
    ])
    whitened = pm.Flat(prefix+'_whitened_states', shape=(months+1, rank),
                       initval=np.zeros((months+1, rank)))
    if solver == 'block':
        diagonal_blocks = diagonal.reshape((months+1, rank))[:, :, None]*np.eye(rank)
        diagonal_blocks = pt.inc_subtensor(diagonal_blocks[1:], blocks)
        states, log_jacobian = block_conditional(
            diagonal_blocks, pt.diag(-phi/innovation_var), natural, whitened, rank,
        )
    elif solver == 'dense':
        precision = pt.diag(diagonal)
        indices = np.arange(dimension-rank)
        off_diagonal = pt.tile(-phi/innovation_var, months)
        precision = pt.set_subtensor(precision[indices, indices+rank], off_diagonal)
        precision = pt.set_subtensor(precision[indices+rank, indices], off_diagonal)
        locations = np.arange(1, months+1)[:, None]*rank+np.arange(rank)[None, :]
        precision = pt.inc_subtensor(
            precision[locations[:, :, None], locations[:, None, :]], blocks)
        cholesky = pt.linalg.cholesky(precision)
        right = pt.slinalg.solve_triangular(cholesky, natural.ravel(), lower=True)+whitened.ravel()
        states = pt.slinalg.solve_triangular(cholesky.T, right, lower=False).reshape(
            (months+1, rank))
        log_jacobian = -pt.sum(pt.log(pt.diag(cholesky)))
    else:
        raise ValueError('Unknown conditional Gaussian solver.')
    states = pm.Deterministic(prefix+'_states', states)
    log_prior = pm.logp(pm.Normal.dist(0, 1), states[0]).sum()
    log_prior += pm.logp(pm.Normal.dist(phi*states[:-1], pt.sqrt(innovation_var)),
                         states[1:]).sum()
    pm.Potential(prefix+'_state_prior', log_prior+log_jacobian)
    return pm.Deterministic(prefix+'_contribution', states[1:] @ loading.T)


def severity_states(months, rank, phi, loading, level, noise, marks, mask, solver='dense'):
    # Mask before transformation; endpoints have finite coarsened logits.
    proxy = logit(np.where(mask, marks/10, .5))
    return gaussian_states('h', months, rank, phi, loading, level, noise[None, :]**2,
                           proxy, mask, solver)


def count_states(months, rank, phi, loading, level, dispersion, counts, mask, effort,
                 solver='dense'):
    # A variance-stabilizing working approximation defines coordinates only.
    # The caller still uses the exact NB likelihood, including observed zeros.
    adjusted = np.where(mask, counts, 0).astype(float)+.5
    proxy = np.log(adjusted)
    variance = 1/adjusted+1/dispersion[None, :]
    return gaussian_states('f', months, rank, phi, loading, level+effort[:, None],
                           variance, proxy, mask, solver)
