"Visualization utilities for the cassandra_threatcast paper figures."
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional

# Design tokens
PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#7C3AED",
    "warn":      "#D97706",
    "danger":    "#DC2626",
}


# Fan chart
def plot_threat_forecast(
    topic_k: int,
    obs: np.ndarray,           # (T_obs,) historical observations
    predictive: np.ndarray,    # (n_samples, T_pred) predictive samples
    dates: list,               # combined list of length T_obs + T_pred, OR
    obs_dates: Optional[list] = None,   # supply separately if dates is None
    pred_dates: Optional[list] = None,
    topic_label: str = "",
    ax: Optional[plt.Axes] = None,
    actual: Optional[np.ndarray] = None,  # (T_pred,) real values for the forecast period, if since become known
) -> plt.Figure:
    "Fan chart with 50 % and 90 % posterior predictive bands."
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if obs_dates is None:
        obs_dates = dates[: len(obs)]
    if pred_dates is None:
        pred_dates = dates[len(obs) :]

    # matplotlib has no native support for pandas Period objects (only
    obs_dates = [d.to_timestamp() if isinstance(d, pd.Period) else d for d in obs_dates]
    pred_dates = [d.to_timestamp() if isinstance(d, pd.Period) else d for d in pred_dates]

    T_pred = predictive.shape[-1]
    if len(pred_dates) != T_pred:
        # Silently trim to the shorter of the two
        n = min(len(pred_dates), T_pred)
        pred_dates = pred_dates[:n]
        predictive = predictive[:, :n]

    # --- Historical ---
    ax.plot(obs_dates, obs, color=PALETTE["primary"], lw=1.8,
            label="Observed", zorder=3)

    # --- Predictive quantiles ---
    p05 = np.percentile(predictive, 5,  axis=0)
    p25 = np.percentile(predictive, 25, axis=0)
    p50 = np.percentile(predictive, 50, axis=0)
    p75 = np.percentile(predictive, 75, axis=0)
    p95 = np.percentile(predictive, 95, axis=0)

    ax.fill_between(pred_dates, p05, p95, alpha=0.20,
                    color=PALETTE["primary"], label="90 % CI")
    ax.fill_between(pred_dates, p25, p75, alpha=0.38,
                    color=PALETTE["primary"], label="50 % CI")
    ax.plot(pred_dates, p50, color=PALETTE["primary"], lw=1.6,
            ls="--", label="Median forecast")

    if actual is not None:
        n = min(len(pred_dates), len(actual))
        ax.plot(pred_dates[:n], actual[:n], color=PALETTE["danger"], lw=2.0,
                marker="o", markersize=4, label="Actual", zorder=4)

    # Vertical separator at forecast origin
    ax.axvline(obs_dates[-1], color="grey", lw=0.8, ls=":", alpha=0.7)

    ax.set_title(
        f"Topic {topic_k}: {topic_label} - CVE Count Forecast", fontsize=11
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("CVE Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# PIT histogram
def plot_pit_histogram(
    pit_vals: np.ndarray,
    channel_name: str = "",
    n_bins: int = 20,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    "PIT histogram with a uniform reference line and KS-test annotation."
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.hist(pit_vals, bins=n_bins, density=True,
            color=PALETTE["primary"], alpha=0.70,
            edgecolor="white", label="PIT")
    ax.axhline(1.0, color=PALETTE["danger"], lw=1.5,
               ls="--", label="Uniform reference")

    # KS test annotation
    from scipy.stats import kstest
    stat, pval = kstest(pit_vals, "uniform")
    ax.text(
        0.97, 0.95,
        f"KS stat={stat:.3f}, p={pval:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT Histogram - {channel_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# Regime probability chart
def plot_regime_probs(
    regime_probs: np.ndarray,         # (T, R) filtered / smoothed probabilities
    dates: list,
    event_labels: Optional[dict] = None,   # {date_idx: label}
    regime_names: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    "Stacked area chart of P(z_t = r | F_T)."
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    T, R = regime_probs.shape
    if regime_names is None:
        regime_names = [f"Regime {r + 1}" for r in range(R)]

    colors = sns.color_palette("Set2", R)
    cumulative = np.zeros(T)

    for r in range(R):
        ax.fill_between(
            dates,
            cumulative,
            cumulative + regime_probs[:, r],
            alpha=0.75,
            color=colors[r],
            label=regime_names[r],
        )
        cumulative += regime_probs[:, r]

    if event_labels:
        for idx, label in event_labels.items():
            if 0 <= idx < len(dates):
                ax.axvline(dates[idx], color="black", lw=1.0, ls=":", alpha=0.7)
                ax.text(
                    dates[idx], 1.02, label,
                    fontsize=7, ha="center",
                    transform=ax.get_xaxis_transform(),
                    rotation=45,
                )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Regime Probability")
    ax.set_title("Filtered Regime Probabilities P(z_t | F_t)")
    ax.legend(loc="lower left", fontsize=8, ncol=max(1, R // 2))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# Loss distribution
def plot_loss_distribution(
    loss_samples: np.ndarray,   # (n_samples,) or (n_samples, T)
    var_alpha: float = 0.05,
    es_alpha: float = 0.05,
    units: str = "$ billions",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    "Histogram of aggregate posterior predictive losses with VaR and ES."
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    # Aggregate over time / sectors if multi-dimensional
    if loss_samples.ndim > 1:
        loss_samples = loss_samples.sum(axis=-1)

    var_level = 1.0 - var_alpha
    es_level  = 1.0 - es_alpha
    var_val = float(np.quantile(loss_samples, var_level))
    tail_mask = loss_samples >= var_val
    es_val = float(np.mean(loss_samples[tail_mask])) if tail_mask.any() else var_val

    ax.hist(
        loss_samples, bins=80, density=True,
        color=PALETTE["secondary"], alpha=0.65, edgecolor="white",
    )

    ymax = ax.get_ylim()[1]
    if ymax == 0:
        ymax = 1.0

    ax.axvline(var_val, color=PALETTE["warn"], lw=2.0, ls="--",
               label=f"VaR({int(var_level * 100)} %) = {var_val:.2f} {units}")
    ax.axvline(es_val, color=PALETTE["danger"], lw=2.0, ls="-",
               label=f"ES({int(es_level * 100)} %) = {es_val:.2f} {units}")

    # Shade the tail region
    ax.fill_betweenx(
        [0, ymax],
        var_val, float(loss_samples.max()),
        alpha=0.15, color=PALETTE["danger"],
    )

    ax.set_xlabel(f"Aggregate Loss ({units})")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Loss Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# Sector exposure
def plot_sector_exposure(
    sector_means: np.ndarray,    # (S,) posterior mean sector losses
    sector_cis: np.ndarray,      # (S, 2) credible interval [lo, hi]
    sector_names: list[str],
    title: str = "Sector Cyber-Induced Loss Exposure",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    "Horizontal bar chart with 90 % credible interval error bars."
    S = len(sector_names)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, S * 0.5)))
    else:
        fig = ax.figure

    y_pos = np.arange(S)
    xerr_lo = np.maximum(sector_means - sector_cis[:, 0], 0.0)
    xerr_hi = np.maximum(sector_cis[:, 1] - sector_means, 0.0)

    ax.barh(
        y_pos, sector_means,
        xerr=[xerr_lo, xerr_hi],
        color=PALETTE["primary"], alpha=0.75, height=0.6,
        error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "black"},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_names, fontsize=9)
    ax.set_xlabel("Expected Annual Loss ($ billions)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# Utility
def save_figure(fig: plt.Figure, path: str, dpi: int = 300) -> None:
    "Save *fig* to *path*, creating parent directories as needed."
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
