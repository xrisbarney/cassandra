"""Proper scoring rules and point metrics for forecast evaluation."""
import numpy as np
import pandas as pd


def crps_ensemble(obs: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    CRPS computed from ensemble samples using the energy score identity.

    CRPS(F, y) = E_F|X - y| - 0.5 * E_F|X - X'|

    The second term is computed via the sorted-weights identity:
        E|X - X'| = (2 / n^2) * sum_i (2i - n - 1) * x_{(i)}
    where x_{(i)} are the order statistics (1-indexed).

    obs     : (...) array of observations
    samples : (..., n_samples) array of predictive samples
    Returns : CRPS array of shape (...)
    """
    n = samples.shape[-1]

    # Term 1: E_F |X - y|
    term1 = np.mean(np.abs(samples - obs[..., np.newaxis]), axis=-1)

    # Term 2: 0.5 * E_F |X - X'| via the sorted-weights identity
    sorted_s = np.sort(samples, axis=-1)                     # (..., n)
    i = np.arange(1, n + 1)                                  # 1-indexed
    weights = (2 * i - n - 1).astype(float)                  # (n,)
    # Broadcast weights to (..., n)
    term2 = np.sum(weights * sorted_s, axis=-1) / (n * n)   # E|X-X'| / 2

    # CRPS = term1 - 0.5 * E|X-X'| = term1 - term2: (term2 already equals sum(w * x) / n^2 = 0.5 * E|X-X'|)
    return term1 - term2


def log_score_negbin(obs: np.ndarray, mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Log score for NegBin predictive: log p(obs | mu, phi).
    NB parameterization: p = phi/(phi+mu), r = phi.
    obs, mu: (...) arrays; phi: (K,) or scalar broadcastable to obs.
    Returns log-score array of same shape as obs.
    """
    from scipy.stats import nbinom
    p = phi / (phi + mu)
    return nbinom.logpmf(obs, n=phi, p=p)


def log_score_ensemble(obs: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Logarithmic score for count observations from ensemble samples, reported
    NEGATIVELY oriented (lower is better, matching CRPS): -log p(obs).

    The predictive PMF is obtained by moment-matching a NegBin2 to the
    samples per element: mu = sample mean, phi = mu^2 / (var - mu) when the
    samples are overdispersed, else a near-Poisson phi. This keeps the score
    comparable across the full model and residual-bootstrap baselines, which
    only provide samples, not parametric forms.

    Moments are computed on samples winsorised at the 1st/99th percentiles:
    the state-space model's log-scale latent admits rare draws up to e^30,
    which make the raw ensemble variance (hence the matched PMF) meaningless
    while leaving the predictive bulk unchanged.

    obs     : (...) count observations
    samples : (..., n_samples) predictive samples (sample axis LAST)
    Returns : (...) array of -log p(obs).
    """
    lo = np.quantile(samples, 0.01, axis=-1, keepdims=True)
    hi = np.quantile(samples, 0.99, axis=-1, keepdims=True)
    samples = np.clip(samples, lo, hi)
    m = np.maximum(samples.mean(axis=-1), 1e-8)
    v = samples.var(axis=-1)
    overdispersed = v > m * (1.0 + 1e-6)
    phi = np.where(overdispersed, m**2 / np.maximum(v - m, 1e-8), 1e8)
    obs_r = np.maximum(np.round(obs), 0.0)
    return -log_score_negbin(obs_r, m, phi)


def mae(obs: np.ndarray, pred_median: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(obs - pred_median)))


def rmse(obs: np.ndarray, pred_median: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((obs - pred_median) ** 2)))


def mape(obs: np.ndarray, pred_median: np.ndarray, eps: float = 1.0) -> float:
    """Mean absolute percentage error with epsilon floor to avoid division by zero."""
    return float(np.mean(np.abs(obs - pred_median) / (np.abs(obs) + eps)))


def evaluate_forecasts(
    obs_dict: dict,        # {"N": (K,T), "D": (S,T), ...}
    predictive_dict: dict, # {"N": (n_samples,K,T), "D": (n_samples,S,T), ...}
    horizons: list[int],
) -> pd.DataFrame:
    """
    Computes CRPS, LogS, MAE, RMSE for each channel and horizon.
    Returns a tidy DataFrame with columns: channel, horizon, metric, value.
    """
    records = []
    for channel, obs in obs_dict.items():
        if channel not in predictive_dict:
            continue
        pred = predictive_dict[channel]  # (n_samples, ..., T)
        for h in horizons:
            if h > obs.shape[-1]:
                continue
            obs_h = obs[..., :h]               # (..., h)
            pred_h = pred[..., :h]             # (n_samples, ..., h)

            # Move sample axis to last: (..., h, n_samples)
            pred_h_t = np.moveaxis(pred_h, 0, -1)
            # Reshape so CRPS operates on (...*h, n_samples)
            obs_flat = obs_h.ravel()
            pred_flat = pred_h_t.reshape(-1, pred_h_t.shape[-1])
            crps_val = float(np.mean(crps_ensemble(obs_flat, pred_flat)))
            logs_val = float(np.mean(log_score_ensemble(obs_flat, pred_flat)))

            pred_median = np.median(pred_h, axis=0)  # (..., h)
            records.append({"channel": channel, "horizon": h, "metric": "CRPS", "value": crps_val})
            records.append({"channel": channel, "horizon": h, "metric": "LogS", "value": logs_val})
            records.append({"channel": channel, "horizon": h, "metric": "MAE",
                            "value": mae(obs_h.ravel(), pred_median.ravel())})
            records.append({"channel": channel, "horizon": h, "metric": "RMSE",
                            "value": rmse(obs_h.ravel(), pred_median.ravel())})
    return pd.DataFrame(records)
