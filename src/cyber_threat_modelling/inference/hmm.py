"""Emission-only HMM marginalization, exact first derivatives, and state recovery.

Inputs are (month, regime) log emissions and (previous, next) log transitions.
Continuous latent priors belong outside this recursion. No probability floor.
"""

import numpy as np
import pytensor.tensor as pt
from numba import njit
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch import numba_funcify


@njit(cache=True)
def _lse(x):
    top = np.max(x)
    if top == -np.inf:
        return top
    return top + np.log(np.exp(x - top).sum())


@njit(cache=True)
def forward(emission, log_transition):
    months, regimes = emission.shape
    alpha = np.empty_like(emission)
    alpha[0] = emission[0] - np.log(regimes)
    for t in range(1, months):
        for r in range(regimes):
            alpha[t, r] = emission[t, r] + _lse(alpha[t - 1] + log_transition[:, r])
    return _lse(alpha[-1]), alpha


@njit(cache=True)
def forward_backward(emission, log_transition):
    logp, alpha = forward(emission, log_transition)
    months, regimes = emission.shape
    beta = np.zeros_like(emission)
    for t in range(months - 2, -1, -1):
        for r in range(regimes):
            beta[t, r] = _lse(log_transition[r] + emission[t + 1] + beta[t + 1])
    gamma = np.exp(alpha + beta - logp)
    expected_transitions = np.zeros_like(log_transition)
    for t in range(months - 1):
        for r in range(regimes):
            expected_transitions[r] += np.exp(
                alpha[t, r] + log_transition[r] + emission[t + 1] + beta[t + 1] - logp
            )
    return gamma, expected_transitions


def recover(emission, transition, rng=None):
    """Conditional filtered/smoothed probabilities; optionally one FFBS path."""
    log_transition = np.log(np.asarray(transition, dtype=float))
    _, alpha = forward(np.asarray(emission, dtype=float), log_transition)
    filtered = np.exp(alpha - np.logaddexp.reduce(alpha, axis=1)[:, None])
    smoothed, _ = forward_backward(emission, log_transition)
    result = {"filtered": filtered, "smoothed": smoothed}
    if rng is not None:
        path = np.empty(len(emission), dtype=int)
        path[-1] = rng.choice(len(transition), p=filtered[-1] / filtered[-1].sum())
        for t in range(len(path) - 2, -1, -1):
            weights = filtered[t] * transition[:, path[t + 1]]
            path[t] = rng.choice(len(transition), p=weights / weights.sum())
        result["path"] = path
    return result


class HMMGradient(Op):
    __props__ = ()
    itypes = [pt.dmatrix, pt.dmatrix]
    otypes = [pt.dmatrix, pt.dmatrix]

    def perform(self, node, inputs, outputs):
        outputs[0][0], outputs[1][0] = forward_backward(*inputs)

    def infer_shape(self, fgraph, node, shapes):
        return shapes


class HMMLogLikelihood(Op):
    __props__ = ()
    itypes = [pt.dmatrix, pt.dmatrix]
    otypes = [pt.dscalar]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.asarray(forward(*inputs)[0])

    def grad(self, inputs, gradients):
        gamma, transitions = HMMGradient()(*inputs)
        return [gradients[0] * gamma, gradients[0] * transitions]

    def infer_shape(self, fgraph, node, shapes):
        return [()]


hmm_logp = HMMLogLikelihood()


@numba_funcify.register(HMMLogLikelihood)
def numba_hmm(op, node, **kwargs):
    @njit
    def evaluate(emission, log_transition):
        return np.asarray(forward(emission, log_transition)[0])

    return evaluate


@numba_funcify.register(HMMGradient)
def numba_hmm_gradient(op, node, **kwargs):
    return forward_backward
