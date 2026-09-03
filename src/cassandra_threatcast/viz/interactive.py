"""
interactive.py
==============
Interactive (plotly) chart builders shared by the CLI scripts and the
Streamlit dashboard.  Every figure uses one design system:

  blue #2a78d6 = model output (predictions, bands, data series)
  ink  #0b0b0b = observed/actual values
  hairline grid, unified hover, px-sized system-ui typography

Figures are written two ways by ``save_interactive``:
  <name>.html  — standalone, opens in any browser (plotly.min.js written
                 once per directory via include_plotlyjs='directory')
  <name>.json  — consumed by the dashboard with plotly.io.from_json +
                 st.plotly_chart, so charts stay interactive inline.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Design tokens (dataviz reference palette)
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# One-hue sequential ramp (light -> dark) for magnitude encodings
BLUES_RAMP = [
    [0.0, "#f3f8fe"], [0.15, "#cde2fb"], [0.3, "#9ec5f4"], [0.45, "#6da7ec"],
    [0.6, "#3987e5"], [0.75, "#256abf"], [0.9, "#184f95"], [1.0, "#0d366b"],
]


def _base_layout(fig: go.Figure, title: str | None = None, height: int = 440) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=SURFACE, paper_bgcolor=PAGE,
        font=dict(family=FONT, color=INK_2, size=12),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=70 if title else 30, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11)),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color=INK, size=15)))
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False,
                     tickfont=dict(color=MUTED))
    return fig


def fan_chart(
    obs: np.ndarray,             # (T_obs,) historical observations
    obs_dates: list,             # pandas Periods or timestamps, length T_obs
    pred_samples: np.ndarray,    # (n_samples, H) predictive draws
    pred_dates: list,            # length H
    topic_label: str = "",
    y_title: str = "CVEs per month",
) -> go.Figure:
    """History line + forecast median with 50%/90% credible bands."""
    obs_x = [p.to_timestamp() if hasattr(p, "to_timestamp") else pd.Timestamp(p)
             for p in obs_dates]
    pred_x = [p.to_timestamp() if hasattr(p, "to_timestamp") else pd.Timestamp(p)
              for p in pred_dates]

    q05, q25, q50, q75, q95 = np.percentile(pred_samples, [5, 25, 50, 75, 95], axis=0)

    fig = go.Figure()
    # 90% band
    fig.add_trace(go.Scatter(x=pred_x, y=q95, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=pred_x, y=q05, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(42,120,214,0.10)",
                             name="90% credible band", hoverinfo="skip"))
    # 50% band
    fig.add_trace(go.Scatter(x=pred_x, y=q75, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=pred_x, y=q25, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(42,120,214,0.18)",
                             name="50% credible band", hoverinfo="skip"))
    # Median forecast
    fig.add_trace(go.Scatter(x=pred_x, y=q50, mode="lines",
                             line=dict(color=BLUE, width=2),
                             name="Forecast (median)",
                             hovertemplate="forecast %{y:,.0f}<extra></extra>"))
    # Observed history
    if len(obs_x):
        fig.add_trace(go.Scatter(x=obs_x, y=np.asarray(obs, dtype=float),
                                 mode="lines", line=dict(color=INK, width=2),
                                 name="Observed",
                                 hovertemplate="observed %{y:,.0f}<extra></extra>"))
        fig.add_vline(x=obs_x[-1], line_width=1, line_color=BASELINE)
        fig.add_annotation(x=obs_x[-1], y=1.02, yref="paper", showarrow=False,
                           text="forecast →", xanchor="left",
                           font=dict(size=11, color=MUTED))

    _base_layout(fig, topic_label, height=460)
    fig.update_yaxes(title=y_title, rangemode="tozero")
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.08),
        rangeselector=dict(
            buttons=[
                dict(count=24, label="2y", step="month", stepmode="backward"),
                dict(count=60, label="5y", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            font=dict(size=11, color=INK_2), bgcolor=SURFACE,
            activecolor=GRID, bordercolor=BASELINE, borderwidth=1),
    )
    return fig


def loss_distribution(
    samples: np.ndarray,   # (n_samples,) aggregate losses (already scaled)
    units: str = "$ billions",
    title: str = "Predictive distribution of aggregate economic loss",
) -> go.Figure:
    """Histogram of the predictive loss distribution with VaR/ES markers."""
    samples = np.asarray(samples, dtype=float)
    var95 = float(np.quantile(samples, 0.95))
    es95 = float(samples[samples >= var95].mean()) if np.any(samples >= var95) else var95
    median = float(np.median(samples))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=60, marker=dict(color=BLUE, line=dict(width=0)),
        opacity=0.85, name="Predictive draws",
        hovertemplate="%{y} draws<extra></extra>"))
    for x, name, color in [(median, "median", INK_2),
                           (var95, "VaR 95%", INK),
                           (es95, "ES 95%", INK)]:
        fig.add_vline(x=x, line_width=1.5, line_color=color,
                      line_dash="dot" if name != "median" else "solid")
        fig.add_annotation(x=x, y=1.04, yref="paper", showarrow=False,
                           text=f"{name}: {x:,.3f}", font=dict(size=11, color=color))
    _base_layout(fig, title, height=420)
    fig.update_layout(hovermode="x")
    fig.update_xaxes(title=units)
    fig.update_yaxes(title="Number of draws")
    return fig


def activity_line(total: np.ndarray, dates: list,
                  y_title: str = "New vulnerabilities") -> go.Figure:
    """Monthly total activity as a line with a light area wash."""
    x = [pd.Timestamp(d) for d in dates]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=np.asarray(total, dtype=float), mode="lines",
        line=dict(color=BLUE, width=2), fill="tozeroy",
        fillcolor="rgba(42,120,214,0.10)", name=y_title,
        hovertemplate="%{y:,.0f}<extra></extra>"))
    _base_layout(fig, None, height=320)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=y_title, rangemode="tozero")
    return fig


def topic_heatmap(N: np.ndarray, topic_labels: list, dates: list) -> go.Figure:
    """Topics x months heatmap on a single-hue sequential ramp."""
    x = [pd.Timestamp(d) for d in dates]
    labels = [f"{lbl} — topic {k:02d}" for k, lbl in enumerate(topic_labels)]
    fig = go.Figure(go.Heatmap(
        z=np.asarray(N, dtype=float), x=x, y=labels,
        colorscale=BLUES_RAMP,
        colorbar=dict(title=dict(text="CVEs", font=dict(size=11)),
                      tickfont=dict(color=MUTED, size=10)),
        hovertemplate="%{y}<br>%{x|%b %Y}: %{z:,.0f} CVEs<extra></extra>"))
    _base_layout(fig, None, height=380)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig


def sector_bar(values: np.ndarray, sector_names: list,
               y_title: str = "Incidents") -> go.Figure:
    """Totals by sector as a thin-bar chart."""
    fig = go.Figure(go.Bar(
        x=list(sector_names), y=np.asarray(values, dtype=float),
        marker=dict(color=BLUE, cornerradius=4), width=0.55,
        hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    _base_layout(fig, None, height=340)
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.35)
    fig.update_yaxes(title=y_title, rangemode="tozero")
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=10))
    return fig


def regime_area(
    dates: list,
    probs: np.ndarray,            # (T, R) regime probabilities, rows sum to 1
    regime_names: list,
    forecast_start: int | None = None,   # index where pure forecast begins
    title: str = "Regime probabilities — the model's early-warning signal",
) -> go.Figure:
    """Stacked-area chart of filtered/forecast regime probabilities."""
    x = [p.to_timestamp() if hasattr(p, "to_timestamp") else pd.Timestamp(p)
         for p in dates]
    # Categorical slots 1/3/6 from the reference palette: calm blue,
    colors = ["#2a78d6", "#eda100", "#e34948", "#4a3aa7", "#1baf7a"]

    fig = go.Figure()
    for r in range(probs.shape[1]):
        fig.add_trace(go.Scatter(
            x=x, y=probs[:, r], mode="lines", stackgroup="regimes",
            line=dict(width=0.5, color=colors[r % len(colors)]),
            fillcolor=colors[r % len(colors)],
            name=regime_names[r] if r < len(regime_names) else f"Regime {r+1}",
            hovertemplate="%{y:.0%}<extra>" + (
                regime_names[r] if r < len(regime_names) else f"Regime {r+1}") + "</extra>",
        ))
    if forecast_start is not None and 0 < forecast_start < len(x):
        fig.add_vline(x=x[forecast_start - 1], line_width=1, line_color=BASELINE)
        fig.add_annotation(x=x[forecast_start - 1], y=1.06, yref="paper",
                           showarrow=False, text="forecast →", xanchor="left",
                           font=dict(size=11, color=MUTED))
    _base_layout(fig, title, height=380)
    fig.update_yaxes(title="Probability", range=[0, 1], tickformat=".0%")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08))
    return fig


# Fixed categorical order for model identity — color follows the model,
MODEL_COLORS = {
    "FullModel": "#2a78d6",   # blue      (the proposed model)
    "BSTS-U":    "#1baf7a",   # aqua
    "RF":        "#eda100",   # yellow
    "ARIMA":     "#008300",   # green
    "ETS":       "#4a3aa7",   # violet
    "Naive":     "#e34948",   # red
}


def baseline_lines(scores_df: pd.DataFrame, metric: str = "CRPS") -> go.Figure:
    """
    Model-vs-baselines comparison: one line per model across forecast
    horizons for the chosen metric (lower is better for all four).
    Expects the tidy results/evaluation/scores.csv (model, horizon, metric,
    value; already aggregated over folds by evaluate.py).
    """
    df = scores_df[scores_df["metric"] == metric]
    # evaluate.py saves fold-aggregated scores as value_mean/value_std;
    value_col = "value_mean" if "value_mean" in df.columns else "value"
    agg = df.groupby(["model", "horizon"])[value_col].mean().reset_index()
    agg = agg.rename(columns={value_col: "value"})

    fig = go.Figure()
    for model in MODEL_COLORS:
        sub = agg[agg["model"] == model].sort_values("horizon")
        if not len(sub):
            continue
        emphasis = model == "FullModel"
        fig.add_trace(go.Scatter(
            x=sub["horizon"], y=sub["value"], mode="lines+markers",
            line=dict(color=MODEL_COLORS[model], width=3 if emphasis else 2),
            marker=dict(size=9 if emphasis else 7,
                        line=dict(color=SURFACE, width=2)),
            name=model,
            hovertemplate=f"{model}: %{{y:,.1f}}<extra></extra>"))
    _base_layout(fig, None, height=380)
    fig.update_xaxes(title="Forecast horizon (months)", tickvals=sorted(agg["horizon"].unique()))
    fig.update_yaxes(title=f"{metric} (lower is better)", rangemode="tozero")
    return fig


def incident_sector_chart(sector_df: pd.DataFrame, sector_name: str) -> go.Figure:
    """
    Incident channel, predicted vs actual for one sector, from the
    backtest's incident_comparison.csv rows (date, actual_incidents,
    predicted_q05/q50/q95_incidents).
    """
    d = sector_df.sort_values("date")
    x = [pd.Period(v, freq="M").to_timestamp() for v in d["date"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d["predicted_q95_incidents"], mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=d["predicted_q05_incidents"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.12)",
                             name="90% credible band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=d["predicted_q50_incidents"], mode="lines",
                             line=dict(color=BLUE, width=2),
                             name="Predicted (median)",
                             hovertemplate="predicted %{y:,.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=d["actual_incidents"], mode="lines+markers",
                             line=dict(color=INK, width=2),
                             marker=dict(size=7, line=dict(color=SURFACE, width=2)),
                             name="Actual",
                             hovertemplate="actual %{y:,.0f}<extra></extra>"))
    _base_layout(fig, f"{sector_name} — disclosed incidents", height=320)
    fig.update_yaxes(title="Incidents per month", rangemode="tozero")
    return fig


def kernel_covariance_curve(lags: np.ndarray, cov_mean: np.ndarray,
                            cov_lo: np.ndarray, cov_hi: np.ndarray,
                            halfwidth_mean: float) -> go.Figure:
    """Learned temporal covariance Cov(g_t, g_{t+d}) vs lag d, with 90% band."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags, y=cov_hi, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=lags, y=cov_lo, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(42,120,214,0.12)",
                             name="90% credible band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lags, y=cov_mean, mode="lines",
                             line=dict(color=BLUE, width=2),
                             name="Posterior mean covariance",
                             hovertemplate="lag %{x} mo: %{y:.4f}<extra></extra>"))
    fig.add_vline(x=halfwidth_mean, line_width=1, line_color=BASELINE)
    fig.add_annotation(x=halfwidth_mean, y=1.04, yref="paper", showarrow=False,
                       text=f"learned half-width ≈ {halfwidth_mean:.0f} mo",
                       font=dict(size=11, color=MUTED), xanchor="left")
    _base_layout(fig, "How far apart do months still co-move?", height=360)
    fig.update_xaxes(title="Separation between months (lag, months)")
    fig.update_yaxes(title="Cov(g_t, g_t+lag)", rangemode="tozero")
    return fig


def kernel_loadings_bar(topic_labels: list, a_mean: np.ndarray,
                        a_sd: np.ndarray) -> go.Figure:
    """Per-topic kernel loadings a_k with +/-1 sd error bars."""
    labels = [f"{lbl} — {k:02d}" for k, lbl in enumerate(topic_labels)]
    fig = go.Figure(go.Bar(
        x=labels, y=a_mean, width=0.55,
        marker=dict(color=BLUE, cornerradius=4),
        error_y=dict(type="data", array=a_sd, color=INK_2, thickness=1.5),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    _base_layout(fig, "Which topics feel the local covariance? (loadings a_k)",
                 height=340)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=10))
    fig.update_yaxes(title="Loading a_k", zeroline=True, zerolinecolor=BASELINE)
    return fig


def save_interactive(fig: go.Figure, path_base: str) -> None:
    """Write <path_base>.html (standalone) and <path_base>.json (dashboard)."""
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    fig.write_html(path_base + ".html", include_plotlyjs="directory", full_html=True)
    fig.write_json(path_base + ".json")
