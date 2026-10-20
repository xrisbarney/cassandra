"""Protocol scores with an O(M log M) empirical CRPS and exact count log mass."""

import numpy as np

SCORING_VERSION = "count-v1"


def crps(draws, observation):
    x = np.sort(np.asarray(draws, dtype=float))
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("CRPS requires a finite nonempty ensemble.")
    n = len(x)
    return float(
        np.mean(np.abs(x - observation)) - np.dot(2 * np.arange(1, n + 1) - n - 1, x) / n**2
    )


def interval_score(low, high, y, alpha):
    return float(high - low + 2 / alpha * max(low - y, 0) + 2 / alpha * max(y - high, 0))


def score_cell(forecast, y, h, k, rng):
    samples = forecast.paths[:, h, k]
    median = forecast.quantile(0.5, h, k)
    result = {
        "crps": crps(samples, y),
        "negative_log_mass": -forecast.logpmf(y, h, k),
        "absolute_error": float(abs(median - y)),
        "squared_error": float((samples.mean() - y) ** 2),
        "predictive_mean_mc": float(samples.mean()),
        "predictive_median": median,
    }
    for level, alpha in ((50, 0.5), (90, 0.1)):
        low, high = forecast.quantile(alpha / 2, h, k), forecast.quantile(1 - alpha / 2, h, k)
        result.update(
            {
                f"coverage_{level}": float(low <= y <= high),
                f"width_{level}": float(high - low),
                f"interval_score_{level}": interval_score(low, high, y, alpha),
            }
        )
    lower, upper = forecast.cdf(y - 1, h, k), forecast.cdf(y, h, k)
    if not 0 <= lower <= upper <= 1 or np.isnan(result["negative_log_mass"]):
        raise ValueError("Invalid predictive probability calculation.")
    for i, v in enumerate(rng.uniform(size=10)):
        result[f"pit_{i}"] = lower + v * (upper - lower)
    return result


def score_aggregate(samples, y):
    samples = np.asarray(samples)
    result = {
        "crps": crps(samples, y),
        "negative_log_mass": np.nan,
        "absolute_error": float(abs(np.quantile(samples, 0.5, method="inverted_cdf") - y)),
        "squared_error": float((samples.mean() - y) ** 2),
        "predictive_mean_mc": float(samples.mean()),
        "predictive_median": float(np.quantile(samples, 0.5, method="inverted_cdf")),
    }
    for level, alpha in ((50, 0.5), (90, 0.1)):
        low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2], method="inverted_cdf")
        result.update(
            {
                f"coverage_{level}": float(low <= y <= high),
                f"width_{level}": float(high - low),
                f"interval_score_{level}": interval_score(low, high, y, alpha),
            }
        )
    return result
