"""Linear-time block Gaussian transform with an exact reverse derivative.

Only deterministic numerical kernels are cached. NUTS needs first derivatives;
this Op intentionally does not supply Hessians. Diagonal blocks are symmetric
and represented by their lower triangle, as in a lower Cholesky factorization.
"""

import numpy as np
import pytensor.tensor as pt
from numba import njit
from pytensor.gradient import DisconnectedType
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch import numba_funcify


@njit(cache=True, error_model='numpy')
def _chol(matrix):
    size = matrix.shape[0]
    result = np.zeros_like(matrix)
    for i in range(size):
        for j in range(i+1):
            value = matrix[i, j]
            for k in range(j):
                value -= result[i, k]*result[j, k]
            result[i, j] = np.sqrt(value) if i == j else value/result[j, j]
    return result


@njit(cache=True, error_model='numpy')
def _solve(lower, rhs, transpose=False):
    result = np.empty_like(rhs)
    size = lower.shape[0]
    for step in range(size):
        i = size-1-step if transpose else step
        for col in range(rhs.shape[1]):
            value = rhs[i, col]
            for previous in range(step):
                j = size-1-previous if transpose else previous
                value -= (lower[j, i] if transpose else lower[i, j])*result[j, col]
            result[i, col] = value/lower[i, i]
    return result


@njit(cache=True, error_model='numpy')
def _dot(left, right):
    result = np.zeros((left.shape[0], right.shape[1]))
    for i in range(left.shape[0]):
        for j in range(right.shape[1]):
            for k in range(left.shape[1]):
                result[i, j] += left[i, k]*right[k, j]
    return result


@njit(cache=True, error_model='numpy')
def forward(diagonal, off, natural, whitened):
    length, rank = natural.shape
    if (length < 1 or rank < 1 or diagonal.shape != (length, rank, rank)
            or off.shape != (rank, rank) or whitened.shape != natural.shape):
        raise ValueError('Incompatible block Gaussian dimensions.')
    lower = np.empty_like(diagonal)
    below = np.zeros_like(diagonal)
    right = np.empty_like(natural)
    states = np.empty_like(natural)
    lower[0] = _chol(diagonal[0])
    right[0] = _solve(lower[0], natural[0].copy().reshape(rank, 1))[:, 0]
    for t in range(1, length):
        below[t] = _solve(lower[t-1], off.T).T
        lower[t] = _chol(diagonal[t]-_dot(below[t], below[t].T))
        rhs = natural[t].copy().reshape(rank, 1)-_dot(
            below[t], right[t-1].copy().reshape(rank, 1))
        right[t] = _solve(lower[t], rhs)[:, 0]
    for t in range(length-1, -1, -1):
        rhs = (right[t]+whitened[t]).reshape(rank, 1)
        if t < length-1:
            rhs = rhs-_dot(below[t+1].T, states[t+1].copy().reshape(rank, 1))
        states[t] = _solve(lower[t], rhs, True)[:, 0]
    jacobian = 0.
    for t in range(length):
        for j in range(rank):
            jacobian -= np.log(lower[t, j, j])
    return states, jacobian, lower, below, right


@njit(cache=True, error_model='numpy')
def _chol_reverse(lower, adjoint):
    product = _dot(lower.T, np.tril(adjoint))
    projected = np.tril(product)
    for i in range(lower.shape[0]):
        projected[i, i] *= .5
    left = _solve(lower, projected, True)
    value = _solve(lower, left.T, True).T
    return .5*(value+value.T)


@njit(cache=True, error_model='numpy')
def reverse(diagonal, off, natural, whitened, grad_states, grad_jacobian):
    states, _, lower, below, right = forward(diagonal, off, natural, whitened)
    length, rank = natural.shape
    adj_states = grad_states.copy()
    adj_lower, adj_below = np.zeros_like(lower), np.zeros_like(below)
    adj_right, adj_white = np.zeros_like(natural), np.zeros_like(whitened)
    adj_diagonal, adj_off = np.zeros_like(diagonal), np.zeros_like(off)
    adj_natural = np.zeros_like(natural)
    # Reverse the backward triangular solve, then the forward solve/Cholesky.
    for t in range(length):
        adj_rhs = _solve(lower[t], adj_states[t].copy().reshape(rank, 1))
        adj_right[t] += adj_rhs[:, 0]
        adj_white[t] += adj_rhs[:, 0]
        adj_lower[t] -= _dot(states[t].copy().reshape(rank, 1), adj_rhs.T)
        if t < length-1:
            adj_below[t+1] -= _dot(states[t+1].copy().reshape(rank, 1), adj_rhs.T)
            adj_states[t+1] -= _dot(below[t+1], adj_rhs)[:, 0]
        for j in range(rank):
            adj_lower[t, j, j] -= grad_jacobian/lower[t, j, j]
    for t in range(length-1, -1, -1):
        adj_rhs = _solve(lower[t], adj_right[t].copy().reshape(rank, 1), True)
        adj_natural[t] += adj_rhs[:, 0]
        adj_lower[t] -= _dot(adj_rhs, right[t].copy().reshape(1, rank))
        if t:
            adj_below[t] -= _dot(adj_rhs, right[t-1].copy().reshape(1, rank))
            adj_right[t-1] -= _dot(below[t].T, adj_rhs)[:, 0]
        symmetric = _chol_reverse(lower[t], adj_lower[t])
        # Only the lower triangle of each diagonal input block is read.
        for i in range(rank):
            for j in range(i+1):
                adj_diagonal[t, i, j] = symmetric[i, j]*(1 if i == j else 2)
        if t:
            adj_below[t] -= 2*_dot(symmetric, below[t])
            contribution = _solve(lower[t-1], adj_below[t].T, True).T
            adj_off += contribution
            adj_lower[t-1] -= _dot(contribution.T, below[t])
    return adj_diagonal, adj_off, adj_natural, adj_white


@njit(cache=True, error_model='numpy')
def transform(diagonal, off, natural, whitened):
    states, jacobian, _, _, _ = forward(diagonal, off, natural, whitened)
    return states, jacobian


class BlockGaussianReverse(Op):
    __props__ = ()
    itypes = [pt.dtensor3, pt.dmatrix, pt.dmatrix, pt.dmatrix, pt.dmatrix, pt.dscalar]
    otypes = [pt.dtensor3, pt.dmatrix, pt.dmatrix, pt.dmatrix]

    def perform(self, node, inputs, outputs):
        for slot, value in zip(outputs, reverse(*inputs), strict=True):
            slot[0] = value

    def infer_shape(self, fgraph, node, shapes):
        return shapes[:4]


class BlockGaussian(Op):
    __props__ = ()
    itypes = [pt.dtensor3, pt.dmatrix, pt.dmatrix, pt.dmatrix]
    otypes = [pt.dmatrix, pt.dscalar]

    def perform(self, node, inputs, outputs):
        states, jacobian = transform(*inputs)
        outputs[0][0], outputs[1][0] = states, np.asarray(jacobian)

    def grad(self, inputs, gradients):
        gh, gj = gradients
        if isinstance(gh.type, DisconnectedType):
            gh = pt.zeros_like(inputs[2])
        if isinstance(gj.type, DisconnectedType):
            gj = pt.constant(0., dtype='float64')
        return BlockGaussianReverse()(*inputs, gh, gj)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[2], ()]


block_gaussian = BlockGaussian()


@numba_funcify.register(BlockGaussian)
def numba_block_gaussian(op, node, **kwargs):
    return transform


@numba_funcify.register(BlockGaussianReverse)
def numba_block_reverse(op, node, **kwargs):
    return reverse
