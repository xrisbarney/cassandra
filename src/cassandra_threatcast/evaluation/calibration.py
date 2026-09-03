"Calibration diagnostics: PIT, coverage, KS test."
import numpy as np
import pandas as pd
from scipy import stats


def pit_values(obs: np.ndarray, samples: np.ndarray) -> np.ndarray:
    "Probability Integral Transform values."
    # Fraction of samples strictly below obs
    below = np.mean(samples < obs[:, np.newaxis], axis=1)
    # Probability mass at obs
    at = np.mean(samples == obs[:, np.newaxis], axis=1)
    # Randomized PIT for discrete distributions
    u = np.random.uniform(size=len(obs))
    return below + u * at


def coverage(obs: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    "Empirical coverage rate: fraction of obs in [lower, upper]."
    return float(np.mean((obs >= lower) & (obs <= upper)))


def interval_score(
    obs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    "Winkler interval score for a (1 - alpha) prediction interval."
    width = upper - lower
    below_penalty = (2.0 / alpha) * np.maximum(lower - obs, 0.0)
    above_penalty = (2.0 / alpha) * np.maximum(obs - upper, 0.0)
    return float(np.mean(width + below_penalty + above_penalty))


def pit_ks_test(pit_vals: np.ndarray) -> tuple[float, float]:
    "KS test for PIT uniformity."
    stat, pval = stats.kstest(pit_vals, "uniform")
    return float(stat), float(pval)


def calibration_report(
    obs_dict: dict,
    predictive_dict: dict,
    alpha_levels: list[float] = [0.1, 0.5],
) -> pd.DataFrame:
    "Compute calibration diagnostics for each channel and alpha level."
    records = []
    for channel, obs in obs_dict.items():
        if channel not in predictive_dict:
            continue
        pred = predictive_dict[channel]  # (n_samples, ...)

        obs_flat = obs.ravel()
        pred_flat = pred.reshape(pred.shape[0], -1).T  # (N, n_samples)
        valid = np.isfinite(obs_flat) & np.all(np.isfinite(pred_flat), axis=1)
        obs_flat = obs_flat[valid]
        pred_flat = pred_flat[valid]
        if obs_flat.size == 0:
            continue

        # PIT values (computed once per channel)
        pit_vals = pit_values(obs_flat, pred_flat)
        ks_stat, ks_pval = pit_ks_test(pit_vals)

        for alpha in alpha_levels:
            lo = np.quantile(pred_flat, alpha / 2.0, axis=1)
            hi = np.quantile(pred_flat, 1.0 - alpha / 2.0, axis=1)
            cov = coverage(obs_flat, lo, hi)
            is_val = interval_score(obs_flat, lo, hi, alpha)
            records.append({
                "channel": channel,
                "alpha": alpha,
                "coverage_nominal": 1.0 - alpha,
                "coverage_empirical": cov,
                "interval_score": is_val,
                "pit_ks_stat": ks_stat,
                "pit_ks_pval": ks_pval,
            })
    return pd.DataFrame(records)
