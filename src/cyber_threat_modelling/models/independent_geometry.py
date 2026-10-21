"""Exact Gaussian conditioning of the independent AR(1) initial level.

This changes coordinates only. The original stationary law and all priors remain.
"""

import numpy as np

from ..forecasting.count_laws import integer_counts


def initial_conditioning(tau, phi):
    stationary_variance = tau**2 / (1 - phi**2)
    marginal_variance = 4 + stationary_variance
    return marginal_variance, 4 / marginal_variance, 4 * stationary_variance / marginal_variance


def build_conditioned_model(
    train, initial_counts, *, center_states=False, mutable=False, stationary_scale_sd=None
):
    import pymc as pm
    import pytensor.tensor as pt

    from .ar1_transform import ar1_transform

    y = integer_counts(train)
    initial = integer_counts(initial_counts)
    t, k = y.shape
    m0_value = np.log(initial.mean(axis=0) + 0.5)
    centered = (
        np.arange(k)
        if center_states is True
        else np.array([], dtype=int)
        if center_states is False
        else np.asarray(center_states, dtype=int)
    )
    if len(set(centered)) != len(centered) or np.any(centered < 0) or np.any(centered >= k):
        raise ValueError("Centered category positions must be unique and in range.")
    n_centered = len(centered)
    noncentered = np.array([j for j in range(k) if j not in centered], dtype=int)

    def rows(value, indices):
        # Basic slices keep both values and reverse derivatives in compiled mode.
        return pt.concatenate([value[j : j + 1] for j in indices], axis=0)

    with pm.Model(coords={"category": np.arange(k), "month": np.arange(t)}) as model:
        m0 = pm.Data("prior_mean", m0_value) if mutable else m0_value
        observations = (
            pm.Data("training_counts", y.T, dims=("category", "month")) if mutable else y.T
        )
        log_psi = pm.Normal("log_psi", 0, 1, dims="category")
        psi = pm.Deterministic("psi", pt.exp(log_psi), dims="category")
        phi = pm.Uniform("phi", -1, 1, dims="category")
        if stationary_scale_sd is None:
            tau = pm.HalfNormal("tau", 0.3, dims="category")
            variance, weight, conditional_variance = initial_conditioning(tau, phi)
        else:
            stationary_scale = pm.HalfNormal(
                "stationary_scale", stationary_scale_sd, dims="category"
            )
            tau = pm.Deterministic("tau", stationary_scale * pt.sqrt(1 - phi**2), dims="category")
            variance = 4 + stationary_scale**2
            weight = 4 / variance
            conditional_variance = 4 * stationary_scale**2 / variance
        initial_eta = pm.Normal("initial_eta", m0, pt.sqrt(variance), dims="category")
        m_residual = pm.Normal("m_residual", 0, 1, dims="category")
        m = pm.Deterministic(
            "m",
            m0 + weight * (initial_eta - m0) + pt.sqrt(conditional_variance) * m_residual,
            dims="category",
        )
        state_blocks = []
        if n_centered:
            future_eta = pm.Flat(
                "future_eta",
                shape=(n_centered, t - 1),
                initval=np.broadcast_to(m0_value[centered, None], (n_centered, t - 1)).copy(),
            )
            center_m = rows(m, centered)[:, None]
            center_phi = rows(phi, centered)[:, None]
            center_tau = rows(tau, centered)[:, None]
            eta = pt.concatenate([rows(initial_eta, centered)[:, None], future_eta], axis=1)
            pm.Potential(
                "state_transitions",
                pm.logp(
                    pm.Normal.dist(
                        center_m + center_phi * (eta[:, :-1] - center_m),
                        center_tau,
                    ),
                    future_eta,
                ).sum(),
            )
            state_blocks.append(eta - center_m)
        if n_centered < k:
            innovations = pm.Normal("innovations", 0, 1, shape=(k - n_centered, t - 1))
            ar_input = pt.concatenate(
                [
                    rows((initial_eta - m) * pt.sqrt(1 - phi**2), noncentered)[:, None],
                    rows(tau, noncentered)[:, None] * innovations,
                ],
                axis=1,
            )
            state_blocks.append(ar1_transform(ar_input, rows(phi, noncentered)))
        grouped = np.r_[centered, noncentered]
        u = rows(pt.concatenate(state_blocks, axis=0), np.argsort(grouped))
        pm.Deterministic("unit_state", u / tau[:, None], dims=("category", "month"))
        pm.Deterministic("terminal_u", u[:, -1], dims="category")
        pm.NegativeBinomial(
            "counts",
            mu=pt.exp(m[:, None] + u),
            alpha=psi[:, None],
            observed=observations,
            dims=("category", "month"),
        )
    return model
