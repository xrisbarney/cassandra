import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.linalg import cholesky, solve_triangular

from cyber_threat_modelling.models.gaussian_coordinates import block_conditional


@pytest.mark.parametrize('rank', [1, 2])
def test_block_conditional_matches_dense_states_jacobian_and_gradients(rank):
    rng = np.random.default_rng(82+rank)
    blocks = np.broadcast_to(3*np.eye(rank), (5, rank, rank)).copy()
    if rank == 2:
        blocks[:, 1, 0] = .2
    off = rng.normal(0, .1, (rank, rank))
    rhs, white = rng.normal(size=(5, rank)), rng.normal(size=(5, rank))

    def reference(diagonal, off_diagonal, natural, whitened):
        precision = np.zeros((5*rank, 5*rank))
        for t in range(5):
            index = slice(t*rank, (t+1)*rank)
            precision[index, index] = np.tril(diagonal[t])+np.tril(diagonal[t], -1).T
            if t:
                previous = slice((t-1)*rank, t*rank)
                precision[index, previous] = off_diagonal
                precision[previous, index] = off_diagonal.T
        lower = cholesky(precision, lower=True)
        states = (np.linalg.solve(precision, natural.ravel())
                  + solve_triangular(lower.T, whitened.ravel(), lower=False))
        log_jacobian = -np.log(np.diag(lower)).sum()
        return states.reshape(5, rank), log_jacobian

    symbolic = [pt.tensor3('diagonal'), pt.matrix('off'), pt.matrix('rhs'), pt.matrix('white')]
    states, jacobian = block_conditional(*symbolic, rank)
    objective = (states**2).sum()+.3*jacobian
    outputs = [states, jacobian, *pt.grad(objective, symbolic)]
    native = pytensor.function(symbolic, outputs)
    compiled = pytensor.function(symbolic, outputs, mode='NUMBA')
    values = [blocks, off, rhs, white]
    expected = reference(*values)
    actual = native(*values)
    for x, y in zip(actual[:2], expected, strict=True):
        assert np.allclose(x, y, atol=1e-10)
    for x, y in zip(actual, compiled(*values), strict=True):
        assert np.allclose(x, y, rtol=1e-9, atol=1e-9)
    indices = [(2, rank-1, 0), (0, rank-1), (1, 0), (4, rank-1)]
    for j, index in enumerate(indices):
        plus, minus = [x.copy() for x in values], [x.copy() for x in values]
        plus[j][index] += 1e-5
        minus[j][index] -= 1e-5
        hp, jp = reference(*plus)
        hm, jm = reference(*minus)
        derivative = ((hp**2).sum()+.3*jp-(hm**2).sum()-.3*jm)/(2e-5)
        assert np.allclose(actual[j+2][index], derivative, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('rank', [1, 2])
def test_cached_block_near_unit_persistence_and_fortran_inputs(rank):
    from cyber_threat_modelling.models.block_gaussian import reverse, transform

    length = 6
    rng = np.random.default_rng(431+rank)
    phi = np.array([.9998, -.995])[:rank]
    rhs = np.asfortranarray(rng.normal(size=(length, rank)))
    white = np.asfortranarray(rng.normal(size=(length, rank)))

    def precision_at(persistence):
        q = 1-persistence**2
        blocks = np.broadcast_to(np.diag((1+persistence**2)/q),
                                 (length, rank, rank)).copy()
        blocks[0] = blocks[-1] = np.diag(1/q)
        blocks[1:] += .3*np.eye(rank)+.1*np.ones((rank, rank))
        return np.asfortranarray(blocks), np.asfortranarray(np.diag(-persistence/q))

    def dense_reference(persistence):
        blocks, off = precision_at(persistence)
        matrix = np.zeros((length*rank, length*rank))
        for t in range(length):
            index = slice(t*rank, (t+1)*rank)
            matrix[index, index] = blocks[t]
            if t:
                previous = slice((t-1)*rank, t*rank)
                matrix[index, previous] = off
                matrix[previous, index] = off.T
        lower = cholesky(matrix, lower=True)
        states = np.linalg.solve(matrix, rhs.ravel())+solve_triangular(
            lower.T, white.ravel(), lower=False)
        return states.reshape(length, rank), -np.log(np.diag(lower)).sum()

    blocks, off = precision_at(phi)
    states, jacobian = transform(blocks, off, rhs, white)
    expected, expected_jacobian = dense_reference(phi)
    assert np.allclose(states, expected, rtol=1e-9, atol=1e-9)
    assert np.allclose(jacobian, expected_jacobian, rtol=1e-10, atol=1e-10)
    gd, go, _, _ = reverse(blocks, off, rhs, white, np.asfortranarray(2*states), .3)
    q = 1-phi**2
    analytical = (gd[0, 0, 0]+gd[-1, 0, 0])*2*phi[0]/q[0]**2
    analytical += gd[1:-1, 0, 0].sum()*4*phi[0]/q[0]**2
    analytical -= go[0, 0]*(1+phi[0]**2)/q[0]**2
    plus, minus = phi.copy(), phi.copy()
    plus[0] += 1e-7
    minus[0] -= 1e-7
    hp, jp = dense_reference(plus)
    hm, jm = dense_reference(minus)
    numerical = ((hp**2).sum()+.3*jp-(hm**2).sum()-.3*jm)/(2e-7)
    assert np.allclose(analytical, numerical, rtol=2e-5, atol=2e-5)
