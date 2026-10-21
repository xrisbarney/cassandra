"""Shared mathematical components for synthetic checks and later model variants."""

import numpy as np
from scipy.special import expit, log_ndtr, logit
from scipy.stats import norm


def kernel_weights(width, centre=30):
    raw = expit((width - np.abs(np.arange(2 * centre + 1) - centre)) / 2)
    return raw / np.linalg.norm(raw)


def kernel_covariance(width, lags, centre=30):
    weights = kernel_weights(width, centre)
    return np.array(
        [np.dot(weights[: len(weights) - d], weights[d:]) if d < len(weights) else 0
         for d in lags]
    )


def censored_mark_logp(value, location, scale, upper=1.0, epsilon=1e-4):
    """NumPy/SciPy oracle on the original coarsened mark scale, including Jacobian."""
    value, location, scale = np.broadcast_arrays(value, location, scale)
    if not np.all(np.isfinite(value)) or np.any(scale <= 0):
        raise ValueError("Select observed marks before calling their density.")
    x = value / upper
    if np.any((value < upper * epsilon) | (value > upper * (1 - epsilon))):
        raise ValueError("Mark is outside its coarsened support.")
    z = logit(x)
    interior = norm.logpdf(z, location, scale) - np.log(upper * x * (1 - x))
    low = log_ndtr((logit(epsilon) - location) / scale)
    high = log_ndtr((location - logit(1 - epsilon)) / scale)
    return np.where(value <= upper * epsilon, low,
                    np.where(value >= upper * (1 - epsilon), high, interior))


def symbolic_mark_logp(value, location, scale, upper=1.0, epsilon=1e-4):
    import pymc as pm
    import pytensor.tensor as pt

    x = value / upper
    z = pt.log(x) - pt.log1p(-x)
    interior = pm.logp(pm.Normal.dist(location, scale), z) - pt.log(upper * x * (1 - x))
    low = pm.logcdf(pm.Normal.dist(location, scale), logit(epsilon))
    high = pm.logcdf(pm.Normal.dist(-location, scale), -logit(1 - epsilon))
    return pt.switch(value <= upper * epsilon, low,
                     pt.switch(value >= upper * (1 - epsilon), high, interior))


def add_masked(emissions, term, months):
    """Scatter gathered cells into T x R emissions; repeated months sum correctly."""
    import pytensor.tensor as pt

    return pt.inc_subtensor(emissions[months], term)


def fitted_effort(counts, observed=None):
    """Origin-specific filtered local-linear-trend proxy; never a full-panel smoother."""
    import warnings

    from statsmodels.tsa.statespace.structural import UnobservedComponents

    counts = np.asarray(counts, dtype=float)
    y = np.log1p(counts.sum(axis=1))
    if observed is not None:
        y[~np.asarray(observed, dtype=bool)] = np.nan
    if np.isfinite(y).sum() < 24:
        raise ValueError("Insufficient audited months for synthetic effort preprocessing.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = UnobservedComponents(y, level="local linear trend").fit(
            disp=False, maxiter=2000
        )
    if not result.mle_retvals.get("converged", False):
        raise ValueError("Effort fit did not converge.")
    level = result.filtered_state[0]
    sd = level.std()
    effort = (level - level.mean()) / sd if sd > 0 else np.zeros_like(level)
    return effort, {
        "converged": True, "parameters": result.params.tolist(),
        "warnings": [str(w.message) for w in caught],
        "filtered_level_mean": float(level.mean()), "filtered_level_sd": float(sd),
    }
