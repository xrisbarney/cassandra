"Estimates latent reporting effort e_t from raw monthly CVE counts."
from __future__ import annotations

import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.structural import UnobservedComponents


def estimate_effort(
    N_raw: np.ndarray,
    method: str = "hp_filter",
    hp_lambda: float = 1600,
) -> np.ndarray:
    "Estimate latent reporting effort e_t from raw monthly CVE counts."
    N_raw = np.asarray(N_raw, dtype=np.float64)
    T = len(N_raw)

    # Take log for positivity, guarding against zeros
    log_N = np.log(np.clip(N_raw, 1e-6, None))

    if method == "hp_filter":
        cycle, trend = hpfilter(log_N, lamb=hp_lambda)
        e_raw = trend  # log-transformed trend component
    elif method == "state_space":
        model = UnobservedComponents(log_N, level="local linear trend")
        result = model.fit(disp=False, method="lbfgs")
        e_raw = result.smoothed_state[0]  # smoothed level
        e_raw = np.asarray(e_raw, dtype=np.float64)
        if e_raw.shape[0] != T:
            # Some versions return (n_states, T)
            e_raw = e_raw[:T]
    elif method == "moving_avg":
        # Centered moving-average trend of log counts (window ≈ 1 year).
        window = min(12, T) if T > 0 else 1
        kernel = np.ones(window) / window
        e_raw = np.convolve(log_N, kernel, mode="same")
    elif method == "log_diff":
        # Month-over-month log growth as an effort proxy; prepend to keep...
        e_raw = np.diff(log_N, prepend=log_N[0])
    else:
        raise ValueError(
            "method must be one of 'hp_filter', 'state_space', 'moving_avg', "
            f"'log_diff', got {method!r}"
        )

    # Normalise to mean 0, std 1
    mu = e_raw.mean()
    sigma = e_raw.std()
    if sigma < 1e-12:
        sigma = 1.0
    e_t = (e_raw - mu) / sigma
    return e_t.astype(np.float64)


def detrend_counts(N_raw: np.ndarray, e_t: np.ndarray, scale: float = 1.0) -> np.ndarray:
    "Remove reporting-effort trend from raw counts."
    N_raw = np.asarray(N_raw, dtype=np.float64)
    e_t = np.asarray(e_t, dtype=np.float64)
    return N_raw / np.exp(e_t * scale)


def effort_summary(N_raw: np.ndarray, e_t: np.ndarray) -> dict:
    "Compute summary statistics of the effort covariate."
    N_raw = np.asarray(N_raw, dtype=np.float64)
    e_t = np.asarray(e_t, dtype=np.float64)

    log_N = np.log(np.clip(N_raw, 1e-6, None))
    total_var = np.var(log_N)
    trend_var = np.var(e_t * log_N.std())  # rescale e_t back to log-N units
    trend_variance_fraction = float(trend_var / total_var) if total_var > 0 else 0.0

    # Autocorrelation at lag 1
    if len(e_t) > 1:
        ac = float(np.corrcoef(e_t[:-1], e_t[1:])[0, 1])
    else:
        ac = float("nan")

    return {
        "trend_variance_fraction": trend_variance_fraction,
        "min_effort": float(e_t.min()),
        "max_effort": float(e_t.max()),
        "autocorr_lag1": ac,
    }
