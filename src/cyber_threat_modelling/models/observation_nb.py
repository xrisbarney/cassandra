"""Independent Bayesian log-linear NB autoregression, conditional on the first count."""

import numpy as np

from ..forecasting.count_laws import CountForecast, integer_counts
from .control_priors import OBSERVATION_METHOD as METHOD
from .control_priors import OBSERVATION_RULE as RULE
from .count_baselines import stream_seed


def build_model(train, initial_counts):
    import pymc as pm
    import pytensor.tensor as pt

    y, initial = integer_counts(train), integer_counts(initial_counts)
    if y.ndim != 2 or len(y) < 2 or initial.ndim != 2 or y.shape[1] != initial.shape[1]:
        raise ValueError("Incompatible conditional count arrays.")
    with pm.Model(coords={"category": np.arange(y.shape[1])}) as model:
        m = pm.Normal("m", np.log(initial.mean(axis=0) + 0.5), 2, dims="category")
        phi = pm.Uniform("phi", -1, 1, dims="category")
        log_psi = pm.Normal("log_psi", 0, 1, dims="category")
        psi = pm.Deterministic("psi", pt.exp(log_psi), dims="category")
        eta = (1 - phi) * m + phi * np.log1p(y[:-1])
        pm.NegativeBinomial("counts", mu=pt.exp(eta), alpha=psi, observed=y[1:])
    return model


def simulate(m, phi, psi, first_counts, months, rng):
    """Keep parameters fixed per path and feed simulated counts into every next step."""
    m, phi, psi = np.broadcast_arrays(m, phi, psi)
    if m.ndim != 2 or np.any(abs(phi) >= 1) or np.any(psi <= 0):
        raise ValueError("Invalid autoregression parameters.")
    previous = np.broadcast_to(integer_counts(first_counts), m.shape).copy()
    means = np.empty((len(m), months, m.shape[1]))
    paths = np.empty(means.shape, dtype=np.int64)
    for h in range(months):
        with np.errstate(over="raise", invalid="raise"):
            mu = np.exp((1 - phi) * m + phi * np.log1p(previous))
        paths[:, h] = rng.negative_binomial(psi, psi / (psi + mu))
        means[:, h] = mu
        previous = paths[:, h]
    concentration = np.broadcast_to(psi[:, None, :], means.shape).copy()
    return CountForecast("nb2_mixture", paths, {"mean": means, "concentration": concentration})


def posterior_parameters(posterior, draws, rng):
    post = posterior.stack(sample=("chain", "draw"))
    parameters = {key: np.empty((draws, post.sizes["category"])) for key in ("m", "phi", "psi")}
    for j in range(post.sizes["category"]):
        # The joint likelihood and priors factorize. Do not align posterior indices
        # across independent categories, but preserve each category's parameter tuple.
        indices = rng.integers(post.sizes["sample"], size=draws)
        for key in parameters:
            parameters[key][:, j] = post[key].isel(category=j).values[indices]
    return parameters


def forecast(posterior, last_counts, draws, seed):
    if last_counts is None:
        raise ValueError("The forecast requires the last observed training counts.")
    rng = np.random.default_rng(seed)
    p = posterior_parameters(posterior, draws, rng)
    return simulate(**p, first_counts=last_counts, months=12, rng=rng).validate()


def predictive_checks(posterior, counts, initial_counts, origin, draws=1000):
    y = integer_counts(counts)
    rng = np.random.default_rng(stream_seed(20261020, METHOD, origin, "predictive_checks"))
    shape = (draws, y.shape[1])
    prior = {
        "m": rng.normal(np.log(np.mean(initial_counts, axis=0) + 0.5), 2, shape),
        "phi": rng.uniform(-1, 1, shape),
        "psi": np.exp(rng.normal(size=shape)),
    }
    post = posterior_parameters(posterior, draws, rng)
    # Conditional replications recursively generate the training interval after
    # its first count. They never plug actual intervening counts into simulations.
    before = simulate(**prior, first_counts=y[0], months=len(y) - 1, rng=rng).validate().paths
    after = simulate(**post, first_counts=y[0], months=len(y) - 1, rng=rng).validate().paths
    result = {
        "method": METHOD,
        "rule": RULE,
        "draws": draws,
        "origin": origin,
        "conditioned_on_first_count": y[0].tolist(),
        "interpretation": "recursive in-sample conditional checks, not forecast calibration",
        "categories": [],
    }
    for j in range(y.shape[1]):
        row = {"category_position": j, "statistics": {}}
        for name, statistic in (
            ("mean", lambda a: a.mean(axis=-1)),
            ("variance", lambda a: a.var(axis=-1)),
            ("zero_fraction", lambda a: (a == 0).mean(axis=-1)),
        ):
            observed = float(statistic(y[1:, j]))
            prior_values, post_values = statistic(before[:, :, j]), statistic(after[:, :, j])
            row["statistics"][name] = {
                "observed": observed,
                "prior_05_50_95": np.quantile(prior_values, [0.05, 0.5, 0.95]).tolist(),
                "posterior_05_50_95": np.quantile(post_values, [0.05, 0.5, 0.95]).tolist(),
                "posterior_p_greater_equal": float(np.mean(post_values >= observed)),
            }
        result["categories"].append(row)
    return result
