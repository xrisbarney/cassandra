"""Linear-time stationary AR(1) innovation transform and exact reverse derivative.

Numba caches only these deterministic numerical kernels, never data or fits.
The PyTensor Op provides first derivatives for NUTS; no Hessian is implemented.
"""

import numpy as np
import pytensor.tensor as pt
from numba import njit
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch import numba_funcify


@njit(cache=True, error_model="numpy")
def transform(innovations, phi):
    result = np.empty_like(innovations)
    for k in range(len(phi)):
        result[k, 0] = innovations[k, 0] / np.sqrt(1 - phi[k] ** 2)
        for t in range(1, innovations.shape[1]):
            result[k, t] = phi[k] * result[k, t - 1] + innovations[k, t]
    return result


@njit(cache=True, error_model="numpy")
def reverse(innovations, phi, states, upstream):
    grad_innovations = np.empty_like(innovations)
    grad_phi = np.zeros_like(phi)
    for k in range(len(phi)):
        adjoint = 0.0
        for t in range(innovations.shape[1] - 1, 0, -1):
            adjoint = upstream[k, t] + phi[k] * adjoint
            grad_innovations[k, t] = adjoint
            grad_phi[k] += adjoint * states[k, t - 1]
        adjoint = upstream[k, 0] + phi[k] * adjoint
        grad_innovations[k, 0] = adjoint / np.sqrt(1 - phi[k] ** 2)
        grad_phi[k] += adjoint * innovations[k, 0] * phi[k] / (1 - phi[k] ** 2) ** 1.5
    return grad_innovations, grad_phi


class AR1Reverse(Op):
    __props__ = ()
    itypes = [pt.dmatrix, pt.dvector, pt.dmatrix, pt.dmatrix]
    otypes = [pt.dmatrix, pt.dvector]

    def perform(self, node, inputs, outputs):
        outputs[0][0], outputs[1][0] = reverse(*inputs)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0], shapes[1]]


class AR1Transform(Op):
    __props__ = ()
    itypes = [pt.dmatrix, pt.dvector]
    otypes = [pt.dmatrix]

    def perform(self, node, inputs, outputs):
        innovations, phi = inputs
        if innovations.shape[0] != len(phi) or innovations.shape[1] < 1:
            raise ValueError("Invalid AR innovation dimensions.")
        outputs[0][0] = transform(innovations, phi)

    def grad(self, inputs, gradients):
        return AR1Reverse()(*inputs, self(*inputs), gradients[0])

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


ar1_transform = AR1Transform()


@numba_funcify.register(AR1Transform)
def numba_ar1_transform(op, node, **kwargs):
    return transform


@numba_funcify.register(AR1Reverse)
def numba_ar1_reverse(op, node, **kwargs):
    return reverse
