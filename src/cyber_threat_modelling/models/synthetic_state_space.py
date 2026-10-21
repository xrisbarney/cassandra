"""Stage 7 research fixtures using frozen component laws, not a real-data adapter.

Array convention is month x category. Nuisance values conditioned on in a fixture
are explicit in its options. The production Stage 8 interface remains separate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..inference.hmm import hmm_logp
from .ar1_transform import ar1_transform
from .components import add_masked, symbolic_mark_logp


def loading_matrix(prefix, categories, rank):
    diag = pm.HalfNormal(prefix + "_diag", 1, shape=rank, initval=np.full(rank, 0.3))
    rows, cols = np.where(np.tril(np.ones((categories, rank)), -1))
    free = pm.Normal(prefix + "_free", 0, 1, shape=len(rows), initval=np.zeros(len(rows)))
    matrix = pt.zeros((categories, rank))
    matrix = pt.set_subtensor(matrix[np.arange(rank), np.arange(rank)], diag)
    return pt.set_subtensor(matrix[rows, cols], free)


def factors(prefix, categories, rank, months):
    phi = pm.Uniform(prefix + "_phi", -1, 1, shape=rank, initval=np.full(rank, 0.4))
    innovations = pm.Normal(prefix + "_innovations", 0, 1, shape=(rank, months + 1))
    states = pm.Deterministic(prefix + "_states", ar1_transform(innovations, phi).T)
    matrix = loading_matrix(prefix, categories, rank)
    contribution = pm.Deterministic(prefix + "_contribution", states[1:] @ matrix.T)
    return contribution


def hierarchical_level(prefix, categories, center, hyper_sd, scale_sd, init):
    mean = pm.Normal(prefix + "_mean", center, hyper_sd, initval=float(np.mean(init)))
    sd = pm.HalfNormal(prefix + "_sd", scale_sd, initval=1)
    raw = pm.Normal(prefix + "_raw", 0, 1, shape=categories, initval=init - np.mean(init))
    return pm.Deterministic(prefix, mean + sd * raw)


def build_model(data, options, conditional=None):
    t, k = data["N"].shape
    r, rank = options["regimes"], options["factors"]
    with pm.Model() as model:
        if options["conditional_level"]:
            m = pt.as_tensor_variable(np.asarray(conditional["m"]))
        else:
            m = hierarchical_level("m", k, 0, 3, 2, np.log(data["N"].mean(axis=0) + 0.5))
        psi = pm.LogNormal("psi", np.log(10), 1, shape=k, initval=np.full(k, 10))
        base = pt.broadcast_to(m, (t, k))
        if rank:
            base = base + factors("f", k, rank, t)
        if options["residual"] != "none":
            tau = pm.Deterministic("tau", 0.01 + pm.HalfNormal("tau_excess", 1, shape=k))
            if options["residual"] == "iid":
                u = pm.Normal("u_raw", 0, 1, shape=(t, k)) * tau
            else:
                phi_u = pm.Uniform("phi_u", -1, 1, shape=k, initval=np.full(k, 0.4))
                innovations = pm.Normal("u_innovations", 0, 1, shape=(k, t + 1))
                u = ar1_transform(innovations, phi_u).T[1:] * tau
            base = base + pm.Deterministic("u", u)
        if options["kernel"]:
            width = pm.Truncated("width", pm.LogNormal.dist(np.log(12), 0.5), lower=0, upper=18)
            raw = pm.math.sigmoid((width - np.abs(np.arange(61) - 30)) / 2)
            weights = raw / pt.sqrt(pt.sum(raw**2))
            xi = pm.Normal("xi", 0, 1, shape=t + 60)
            indices = np.arange(t)[:, None] + 60 - np.arange(61)[None, :]
            g = pm.Deterministic("g", pt.sum(xi[indices] * weights, axis=1))
            scale = pm.HalfNormal("a_scale", 0.5)
            first = pm.HalfNormal("a_first_raw", 1)
            remaining = pm.Normal("a_remaining_raw", 0, 1, shape=k - 1)
            a = pm.Deterministic("a", scale * pt.concatenate([first[None], remaining]))
            pm.Deterministic("a_norm", pt.sqrt(pt.sum(a**2)))
            base = base + g[:, None] * a
        if r > 1:
            increments = pm.HalfNormal("regime_increments", 1, shape=r - 1)
            shifts = pt.concatenate([pt.zeros(1), pt.cumsum(increments)])
            transition = pm.Dirichlet("transition", 1 + 9 * np.eye(r), shape=(r, r))
        else:
            shifts, transition = pt.zeros(1), pt.ones((1, 1))
        eta = pm.Deterministic("eta_by_regime", base[:, None, :] + shifts[None, :, None])
        emissions = pt.zeros((t, r))
        months, cats = np.where(data["mask_N"])
        nterm = pm.logp(
            pm.NegativeBinomial.dist(
                mu=pt.exp(eta[months, :, cats] + data["effort"][months, None]),
                alpha=psi[cats, None],
            ), data["N"][months, cats, None],
        )
        emissions = add_masked(emissions, nterm, months)
        if options["severity_factors"]:
            n = hierarchical_level("n", k, 1, 1, 1, np.ones(k))
            loc = n + factors("h", k, options["severity_factors"], t)
            kappa = pm.HalfNormal("kappa", 0.5, shape=k, initval=np.full(k, 0.3))
            # This fixture has R=1. Severity-regime shifts get separate component tests.
            loc = pm.Deterministic("severity_location", loc)
            months, cats = np.where(data["mask_B"])
            term = symbolic_mark_logp(
                data["B"][months, cats, None], loc[months, cats, None],
                kappa[cats, None], upper=10, epsilon=0.005,
            )
            emissions = add_masked(emissions, term, months)
        if options["known_auxiliary_coefficients"]:
            months, cats = np.where(data["mask_E"])
            emissions = add_masked(emissions, symbolic_mark_logp(
                data["E"][months, cats, None], -3 + 0.4 * eta[months, :, cats], 0.5,
            ), months)
            months, cats = np.where(data["mask_V"])
            emissions = add_masked(emissions, pm.logp(pm.NegativeBinomial.dist(
                mu=pt.exp(-3 + 0.5 * eta[months, :, cats]), alpha=2,
            ), data["V"][months, cats, None]), months)
            months, sectors = np.where(data["mask_D"])
            rate = 0.02 * pt.sum(
                pt.exp(eta[months]) * data["M"][sectors, None, :], axis=2
            ) + 0.1
            emissions = add_masked(emissions, pm.logp(
                pm.Poisson.dist(rate), data["D"][months, sectors, None]
            ), months)
        # R=1 shortcut is algebraically identical and avoids a redundant Op.
        pm.Potential("observations", pt.sum(emissions) if r == 1 else hmm_logp(
            emissions, pt.log(transition)
        ))
        model.synthetic_emissions = emissions
        model.synthetic_transition = transition
    return model
