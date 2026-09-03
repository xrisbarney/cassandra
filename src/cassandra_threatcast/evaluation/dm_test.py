"""Diebold-Mariano test with Harvey et al. small-sample correction."""
import numpy as np
import pandas as pd
from scipy import stats


def dm_test(
    loss_a: np.ndarray,  # (T,) loss series for model A
    loss_b: np.ndarray,  # (T,) loss series for model B
    h: int = 1,          # forecast horizon
) -> tuple[float, float]:
    """
    Diebold-Mariano test (1995) with Harvey, Leybourne & Newbold (1997)
    small-sample correction.

    H0: E[loss_a] = E[loss_b]
    H1: E[loss_a] != E[loss_b]  (two-sided)

    The variance of the loss differential is estimated via a Newey-West
    long-run variance with (h - 1) autocovariance lags, matching the
    autocorrelation structure introduced by h-step-ahead forecasts.

    The Harvey et al. correction multiplies the DM statistic by
        sqrt((T + 1 - 2h + h*(h-1)/T) / T)
    and compares to a t(T-1) distribution instead of N(0,1).

    Parameters
    ----------
    loss_a, loss_b : (T,) arrays of per-period loss values
    h              : forecast horizon (determines number of NW lags = h - 1)

    Returns
    -------
    (dm_stat_corrected, p_value) : two-sided test
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    T = len(loss_a)
    if len(loss_b) != T:
        raise ValueError("loss_a and loss_b must have the same length")

    d = loss_a - loss_b          # loss differential
    d_bar = np.mean(d)

    # Newey-West long-run variance estimate with (h - 1) lags
    gamma_0 = np.mean((d - d_bar) ** 2)

    if h > 1:
        n_lags = h - 1
        gamma_lags = np.array([
            np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
            for lag in range(1, n_lags + 1)
        ])
        # Newey-West weights (Bartlett): w_j = 1 - j/(h)
        nw_weights = 1.0 - np.arange(1, n_lags + 1) / h
        long_run_var = gamma_0 + 2.0 * np.sum(nw_weights * gamma_lags)
    else:
        long_run_var = gamma_0

    # Variance of d_bar
    var_d_bar = max(long_run_var / T, 1e-10)

    # Raw DM statistic
    dm_stat = d_bar / np.sqrt(var_d_bar)

    # Harvey et al. (1997) small-sample correction factor
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat_corrected = dm_stat * correction

    # Compare to t-distribution with T - 1 degrees of freedom
    pval = 2.0 * stats.t.sf(np.abs(dm_stat_corrected), df=T - 1)
    return float(dm_stat_corrected), float(pval)


def dm_table(
    scores_dict: dict,      # {model_name: (T,) loss array}
    reference_model: str,   # key in scores_dict
    h: int = 1,
) -> pd.DataFrame:
    """
    Run DM tests of all models against a reference model.

    Returns DataFrame indexed by model with columns:
        dm_stat, p_value, significant_5pct, note.
    """
    if reference_model not in scores_dict:
        raise KeyError(f"reference_model '{reference_model}' not found in scores_dict")

    ref_losses = np.asarray(scores_dict[reference_model], dtype=float)
    rows = []

    for model_name, losses in scores_dict.items():
        losses = np.asarray(losses, dtype=float)
        if model_name == reference_model:
            rows.append({
                "model": model_name,
                "dm_stat": 0.0,
                "p_value": 1.0,
                "significant_5pct": False,
                "note": "reference",
            })
            continue

        stat, pval = dm_test(losses, ref_losses, h=h)

        if pval < 0.05:
            # stat < 0 means model A (this model) has lower losses than B (reference)
            note = "better" if stat < 0 else "worse"
        else:
            note = "n.s."

        rows.append({
            "model": model_name,
            "dm_stat": stat,
            "p_value": pval,
            "significant_5pct": pval < 0.05,
            "note": note,
        })

    return pd.DataFrame(rows).set_index("model")
