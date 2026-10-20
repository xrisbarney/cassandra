"""Prior/posterior predictive checks for the Stage 6 count controls."""

import numpy as np

from ..models.count_baselines import stream_seed


def predictive_checks(posterior, method, counts, initial_counts, origin, draws=1000):
    """Report discrepancy distributions, including zeros; these are not fit-selection scores."""
    y = np.asarray(counts, dtype=int)
    t, k = y.shape
    rng = np.random.default_rng(stream_seed(20261020, method, origin, "predictive_checks"))
    post = posterior.stack(sample=("chain", "draw"))
    indices = rng.integers(post.sizes["sample"], size=draws)
    psi = post.psi.transpose("sample", "category").values[indices]
    if method == "pooled_nb":
        posterior_eta = (
            post.b0.values[indices, None, None]
            + post.b.transpose("sample", "category").values[indices, None, :]
            + post.beta.values[indices, None, None] * np.arange(t)[None, :, None] / 12
        )
        prior_eta = (
            rng.normal(np.log(np.mean(initial_counts) + 0.5), 2, (draws, 1, 1))
            + rng.normal(size=(draws, 1, k))
            + rng.normal(0, 0.5, (draws, 1, 1)) * np.arange(t)[None, :, None] / 12
        )
    else:
        posterior_eta = (
            post.m.transpose("sample", "category").values[indices, None, :]
            + post.tau.transpose("sample", "category").values[indices, None, :]
            * post.unit_state.transpose("sample", "month", "category").values[indices]
        )
        m = rng.normal(np.log(np.mean(initial_counts, axis=0) + 0.5), 2, (draws, k))
        tau = abs(rng.normal(0, 0.3, (draws, k)))
        phi = rng.uniform(-1, 1, (draws, k))
        u = tau * rng.normal(size=(draws, k)) / np.sqrt(1 - phi**2)
        prior_eta = np.empty((draws, t, k))
        for month in range(t):
            if month:
                u = phi * u + tau * rng.normal(size=(draws, k))
            prior_eta[:, month, :] = m + u
    result = {
        "draws": draws,
        "origin": origin,
        "method": method,
        "interpretation": "in-sample predictive checks, not out-of-sample calibration",
        "categories": [],
    }
    prior_psi = np.exp(rng.normal(size=(draws, k)))
    with np.errstate(over="raise", under="raise"):
        prior_mu, posterior_mu = np.exp(prior_eta), np.exp(posterior_eta)
    prior_y = rng.negative_binomial(
        prior_psi[:, None, :], prior_psi[:, None, :] / (prior_psi[:, None, :] + prior_mu)
    )
    post_y = rng.negative_binomial(
        psi[:, None, :], psi[:, None, :] / (psi[:, None, :] + posterior_mu)
    )
    for j in range(k):
        row = {"category_position": j, "statistics": {}}
        for name, statistic in (
            ("mean", lambda a: np.mean(a, axis=-1)),
            ("variance", lambda a: np.var(a, axis=-1)),
            ("zero_fraction", lambda a: np.mean(a == 0, axis=-1)),
        ):
            observed = float(statistic(y[:, j]))
            before, after = statistic(prior_y[:, :, j]), statistic(post_y[:, :, j])
            row["statistics"][name] = {
                "observed": observed,
                "prior_05_50_95": np.quantile(before, [0.05, 0.5, 0.95]).tolist(),
                "posterior_05_50_95": np.quantile(after, [0.05, 0.5, 0.95]).tolist(),
                "posterior_p_greater_equal": float(np.mean(after >= observed)),
            }
        result["categories"].append(row)
    return result
