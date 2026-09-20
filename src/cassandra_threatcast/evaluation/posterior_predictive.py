"Posterior predictive checks for paper Step 37."
from __future__ import annotations

import numpy as np
import pandas as pd


def posterior_predictive_checks(obs_dict: dict, predictive_dict: dict) -> pd.DataFrame:
    "Compare observed totals, dispersion, and zero rates with replications."
    rows = []
    statistics = {
        "mean": lambda x: np.nanmean(x, axis=-1),
        "variance": lambda x: np.nanvar(x, axis=-1),
        "zero_rate": lambda x: np.nanmean(x == 0, axis=-1),
    }
    for channel, observed in obs_dict.items():
        if channel not in predictive_dict:
            continue
        obs = np.asarray(observed, dtype=float).ravel()
        valid = np.isfinite(obs)
        if not valid.any():
            continue
        rep = np.asarray(predictive_dict[channel], dtype=float).reshape(
            predictive_dict[channel].shape[0], -1
        )[:, valid]
        obs = obs[valid]
        for name, statistic in statistics.items():
            obs_stat = float(statistic(obs[None, :])[0])
            rep_stats = statistic(rep)
            rows.append({
                "channel": channel,
                "statistic": name,
                "observed": obs_stat,
                "predictive_mean": float(np.mean(rep_stats)),
                "bayesian_p_value": float(np.mean(rep_stats >= obs_stat)),
            })
    return pd.DataFrame(rows)
