#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Backtest: sequential one-step-ahead filtered prediction over the FULL history.

The model is a Bayesian state-space system, so the correct picture of its
performance is sequential: starting from the very first month of data, the
model predicts the next month BEFORE seeing it, is scored against what
actually happened, then updates its beliefs with that month's actual data,
and repeats.  By the time it reaches the test window it has absorbed every
month of history in order, exactly the way the Bayesian machinery prescribes.

Concretely, for each of the 8,000 posterior draws:

  1. The Hamilton forward filter runs over the entire panel (2010-01 ...)
     starting from the uniform regime prior at t=0.  loglik_regime_t --
     computed inside the NumPyro model during training -- gives
     log p(y_t | z_t = r) for ALL FOUR observation channels (CVE counts,
     EPSS, incidents, severity), so each month's belief update incorporates
     everything the model observed that month.
  2. At every month t the one-step-ahead PREDICTED regime distribution
     P(z_t | y_{1:t-1}) (data through t-1 only -- never month t itself) is
     combined with the factor state propagated from t-1 to sample the
     predictive distribution of that month's CVE counts and incidents,
     using the exact fitted observation model:
         eta_t   = mu_r[z] + Gamma f_t + tau_k eps
         N_kt    ~ NegBin2(mean = exp(softclip(eta + e)), conc = psi_k)
         D_st    ~ Poisson(clip(rho_s (M exp(softclip(eta))) + pi_s))
     The covariates e and M enter the prediction LAGGED (their t-1 values):
     month t's own e_t / M_skt are month-t data and must not inform its own
     prediction.  The belief update then uses month t's actual values.
  3. After the panel ends, prediction continues with NO more belief updates
     (pure forecast): the regime chain evolves via Pi alone, the factor AR
     compounds its own noise, and e_t / M_skt are held at their last
     observed values (annotated on the charts).

Honest caveats: (a) regime beliefs are strictly filtered (past data only),
but the continuous factor state f_{t-1} comes from the MCMC posterior, which
conditioned on the full sample -- the standard one-step-ahead
posterior-predictive check for MCMC-fitted state-space models.  (b) Static
parameters are likewise estimated from the full sample; for a fully
out-of-sample test of the 2023-2024 window, retrain on data through 2022.
(c) e_t itself is produced by a two-sided HP filter over the full sample
(scripts/build_features.py), so even its lagged value carries some future
information by construction; a fully causal evaluation would rebuild it
with a one-sided estimator (see --effort-method in build_features.py).

Outputs (results/backtest/):
  backtest_scores.csv      CRPS / MAE / RMSE / coverage, test window & full history
  threat_comparison.csv    per topic-month: actual, predicted quantiles, coverage
  incident_comparison.csv  per sector-month over the test window
  backtest_summary.txt     plain-language summary
  figures/backtest_topic_XX.html   interactive chart (open in a browser)
  figures/backtest_topic_XX.json   same figure for the dashboard (st.plotly_chart)
  figures/backtest_overview.html   all topics on one page

Usage:
    python scripts/backtest.py                      # defaults: test window 2023-01..2024-12
    python scripts/backtest.py --test-start 2023-01 --test-end 2024-12 --tail-months 12
"""
import argparse
import json
import os
import pickle
import sys
import yaml
import numpy as np
import pandas as pd

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.features.topic_map import load_topic_labels
from cassandra_threatcast.data.bea_io import get_default_sector_labels
from cassandra_threatcast.evaluation.scoring import crps_ensemble, mae, rmse
from cassandra_threatcast.evaluation.sequential import sequential_one_step_predict


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential one-step-ahead filtered backtest over the full history."
    )
    parser.add_argument("--idata", default="results/idata.nc")
    parser.add_argument("--data-dir", default="data/processed/")
    parser.add_argument("--output-dir", default="results/backtest/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--test-start", default="2023-01",
        help="Start of the scored test window, YYYY-MM.  Default: 2023-01.",
    )
    parser.add_argument(
        "--test-end", default="2024-12",
        help="End of the scored test window, YYYY-MM.  Default: 2024-12.",
    )
    parser.add_argument(
        "--tail-months", type=int, default=12,
        help="Months of pure forecast to append after the data ends.  Default: 12.",
    )
    parser.add_argument(
        "--max-draws", type=int, default=0,
        help="Thin the posterior to at most this many draws (0 = use all).",
    )
    parser.add_argument(
        "--extend-to", default="now",
        help="Continue the filter past the training panel on ACTUAL CVE data "
             "(fetched with the frozen topic mapper, no retraining) up to this "
             "month: 'now' (default; the last complete calendar month), a "
             "YYYY-MM, or 'none' to stop at the training panel.",
    )
    parser.add_argument("--cache-dir", default="data/cache/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhanced-mode", action="store_true")
    return parser.parse_args()


def _fetch_extension_counts(panel_end: pd.Period, extend_to: str,
                            cache_dir: str, data_dir: str, K: int):
    """
    Actual monthly CVE counts for the months AFTER the training panel,
    classified with the FROZEN topic mapper (never refit: refitting would
    shuffle topic indices and break correspondence with the trained model).

    Returns (N_ext (K, T_ext) or None, ext_periods or None).
    """
    if str(extend_to).lower() == "none":
        return None, None
    if str(extend_to).lower() == "now":
        # Last complete calendar month: the present month's data is partial,
        end_p = pd.Timestamp.now().to_period("M") - 1
    else:
        end_p = pd.Period(extend_to, freq="M")
    if end_p <= panel_end:
        return None, None

    ext_periods = pd.period_range(panel_end + 1, end_p, freq="M")
    print(f"      Extension window: {ext_periods[0]} .. {ext_periods[-1]} "
          f"({len(ext_periods)} months of actual data)")
    try:
        from cassandra_threatcast.data import nvd
        start_dt = str(ext_periods[0].to_timestamp().date())
        end_dt = str((ext_periods[-1].to_timestamp() + pd.offsets.MonthEnd(0)).date())
        cve_df = nvd.fetch_cves(start_dt, end_dt, cache_dir)
        if not len(cve_df):
            print("      No CVEs returned for the extension window; skipping.")
            return None, None
        with open(os.path.join(data_dir, "topic_mapper.pkl"), "rb") as fh:
            mapper = pickle.load(fh)
        assign = mapper.assign_hard(cve_df["description"].fillna("").tolist())
        months = pd.PeriodIndex(pd.to_datetime(cve_df["published_date"]), freq="M")
        month_idx = ext_periods.get_indexer(months)
        valid = month_idx >= 0
        N_ext = np.zeros((K, len(ext_periods)), dtype=np.int64)
        np.add.at(N_ext, (assign[valid], month_idx[valid]), 1)
        print(f"      {int(valid.sum()):,} CVEs classified into the extension panel.")
        return N_ext.astype(float), ext_periods
    except Exception as exc:
        print(f"      Extension fetch failed ({exc}); continuing without it.")
        return None, None


# The sequential filter / one-step-ahead predictive machinery lives in


# Interactive charts: Palette: dataviz reference instance (categorical slot 1 + chart chrome).
_BLUE      = "#2a78d6"   # predicted median + bands
_INK       = "#0b0b0b"   # actual line / primary ink
_INK_2     = "#52514e"   # secondary ink
_MUTED     = "#898781"   # axis labels
_GRID      = "#e1e0d9"   # hairline grid
_BASELINE  = "#c3c2b7"   # axis / reference lines
_SURFACE   = "#fcfcfb"
_PAGE      = "#f9f9f7"
_FONT      = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def build_topic_figure(
    dates: pd.DatetimeIndex,      # (T_all,)
    actual: np.ndarray,           # (T,) NaN-padded to T_all by caller
    q: dict,                      # {"q05","q25","q50","q75","q95"} each (T_all,)
    topic_label: str,
    data_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    train_end: pd.Timestamp | None = None,   # marks where TRAINING data ends
                                             # when the filter continues on: post-training actuals
):
    import plotly.graph_objects as go

    x = list(dates)
    fig = go.Figure()

    # 90% band (q05..q95) -- one legend entry, boundaries silent
    fig.add_trace(go.Scatter(
        x=x, y=q["q95"], mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=q["q05"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(42,120,214,0.10)",
        name="90% credible band", hoverinfo="skip"))
    # 50% band (q25..q75) layered on top
    fig.add_trace(go.Scatter(
        x=x, y=q["q75"], mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=q["q25"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(42,120,214,0.18)",
        name="50% credible band", hoverinfo="skip"))

    fig.add_trace(go.Scatter(
        x=x, y=q["q50"], mode="lines",
        line=dict(color=_BLUE, width=2),
        name="Predicted (1-month-ahead median)",
        hovertemplate="predicted %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=actual, mode="lines",
        line=dict(color=_INK, width=2),
        name="Actual",
        hovertemplate="actual %{y:,.0f}<extra></extra>"))

    # Test-window shading + end-of-data marker
    fig.add_vrect(
        x0=test_start, x1=test_end,
        fillcolor="rgba(137,135,129,0.08)", line_width=0,
        annotation_text="test window", annotation_position="top left",
        annotation_font=dict(size=11, color=_MUTED))
    if train_end is not None:
        fig.add_vline(x=train_end, line_width=1, line_color=_BASELINE)
        fig.add_annotation(
            x=train_end, y=1.10, yref="paper", showarrow=False,
            text="training data ends (filter continues on actuals)",
            font=dict(size=10, color=_MUTED), xanchor="left")
    fig.add_vline(x=data_end, line_width=1, line_color=_BASELINE)
    fig.add_annotation(
        x=data_end, y=1.02, yref="paper", showarrow=False,
        text="data ends → pure forecast (effort index held at last value)",
        font=dict(size=11, color=_MUTED), xanchor="left")

    fig.update_layout(
        title=dict(text=topic_label, font=dict(color=_INK, size=15)),
        plot_bgcolor=_SURFACE, paper_bgcolor=_PAGE,
        font=dict(family=_FONT, color=_INK_2, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0,
                    font=dict(size=11)),
        margin=dict(l=60, r=20, t=90, b=10),
        height=460,
        yaxis=dict(title="CVEs per month", gridcolor=_GRID, zeroline=False,
                   tickfont=dict(color=_MUTED), rangemode="tozero",
                   linecolor=_BASELINE),
        xaxis=dict(
            gridcolor=_GRID, linecolor=_BASELINE, tickfont=dict(color=_MUTED),
            rangeslider=dict(visible=True, thickness=0.08),
            rangeselector=dict(
                buttons=[
                    dict(count=24, label="2y", step="month", stepmode="backward"),
                    dict(count=60, label="5y", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                font=dict(size=11, color=_INK_2), bgcolor=_SURFACE,
                activecolor=_GRID, bordercolor=_BASELINE, borderwidth=1),
        ),
    )
    return fig


# Main
def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    K = config["model"]["K"]
    S = config["model"]["S"]

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load the panel (the "actual" line comes straight from it)
    print("[1/5] Loading processed panel ...")
    N_actual = np.load(os.path.join(args.data_dir, "N.npy")).astype(float)   # (K, T)
    D_actual = np.load(os.path.join(args.data_dir, "D.npy")).astype(float)   # (S, T)
    e_t      = np.load(os.path.join(args.data_dir, "e_t.npy"))               # (T,)
    M_skt    = np.load(os.path.join(args.data_dir, "M_skt.npy"))             # (S, K, T)
    with open(os.path.join(args.data_dir, "panel_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    date_strs = meta.get("dates") or meta.get("metadata", {}).get("dates", [])
    periods = pd.PeriodIndex(date_strs, freq="M")
    T = N_actual.shape[1]
    assert len(periods) == T, f"panel_meta dates ({len(periods)}) != N columns ({T})"
    print(f"      Panel: {periods[0]} .. {periods[-1]}  (T={T}, K={K}, S={S})")

    test_start_p = pd.Period(args.test_start, freq="M")
    test_end_p   = pd.Period(args.test_end, freq="M")
    test_mask    = (periods >= test_start_p) & (periods <= test_end_p)
    n_test       = int(test_mask.sum())
    if n_test == 0:
        print(f"      Test window {args.test_start}..{args.test_end} not in panel -- aborting.")
        sys.exit(1)
    print(f"      Test window: {args.test_start} .. {args.test_end}  ({n_test} months)")

    # 2. Load the posterior
    print("[2/5] Loading trained posterior ...")
    try:
        import arviz as az
        pkl_path = os.path.splitext(args.idata)[0] + ".pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as fh:
                idata = pickle.load(fh)
        else:
            idata = az.from_netcdf(args.idata)
    except Exception as exc:
        print(f"      Error loading idata: {exc}")
        sys.exit(1)

    post = {k: np.asarray(v) for k, v in idata.posterior.items()}
    post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items()}
    n_draws = post["Pi"].shape[0]
    if args.max_draws and n_draws > args.max_draws:
        keep = np.linspace(0, n_draws - 1, args.max_draws).astype(int)
        post = {k: v[keep] for k, v in post.items()}
        n_draws = args.max_draws
    print(f"      {n_draws} posterior draws.")

    enhanced = args.enhanced_mode or bool(config.get("enhanced", {}).get("enabled", False))
    student_t_df = float(config.get("enhanced", {}).get("student_t_df", 4.0))

    # 3. Sequential predict -> update, month 1 through the present + tail
    N_ext, ext_periods = _fetch_extension_counts(
        periods[-1], args.extend_to, args.cache_dir, args.data_dir, K)
    T_ext = 0 if N_ext is None else N_ext.shape[1]

    last_obs = ext_periods[-1] if T_ext else periods[-1]
    print(f"[3/5] Sequential 1-step-ahead prediction: {periods[0]} .. {last_obs}"
          f" + {args.tail_months}-month forecast tail ...")
    out = sequential_one_step_predict(
        post, e_t, M_skt, args.tail_months, enhanced, student_t_df, args.seed,
        N_ext=N_ext)
    N_pred = out["N_pred"]   # (n, K, T + T_ext + tail)
    D_pred = out["D_pred"]   # (n, S, T + T_ext + tail)
    T_all = N_pred.shape[2]
    T_obs_end = T + T_ext    # months with actual data

    # Actuals covering training panel + extension, for scoring and charts.
    N_obs_full = np.concatenate([N_actual, N_ext], axis=1) if T_ext else N_actual

    all_periods = pd.period_range(periods[0], periods=T_all, freq="M")
    dates_ts = all_periods.to_timestamp()

    # 4. Score
    print("[4/5] Scoring (every prediction is 1-step-ahead) ...")
    q_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    N_q = np.quantile(N_pred, q_levels, axis=0)   # (5, K, T_all)
    D_q = np.quantile(D_pred, q_levels, axis=0)   # (5, S, T_all)

    def _trimmed_mean(pred):
        # The raw ensemble mean is dominated by rare extreme draws (the
        thr = np.quantile(pred, 0.99, axis=0, keepdims=True)
        return np.nanmean(np.where(pred <= thr, pred, np.nan), axis=0)

    N_mean = _trimmed_mean(N_pred)                # (K, T_all)
    D_mean = _trimmed_mean(D_pred)                # (S, T_all)

    def _channel_scores(obs, pred, q, mask):
        """obs (C,T), pred (n,C,T_all), q (5,C,T_all), mask (T,) over panel months."""
        obs_w = obs[:, mask]
        samp_w = np.moveaxis(pred[:, :, :obs.shape[1]][:, :, mask], 0, -1)  # (C,Tw,n)
        med_w = q[2][:, :obs.shape[1]][:, mask]
        cov90 = float(np.mean((q[0][:, :obs.shape[1]][:, mask] <= obs_w)
                              & (obs_w <= q[4][:, :obs.shape[1]][:, mask])))
        cov50 = float(np.mean((q[1][:, :obs.shape[1]][:, mask] <= obs_w)
                              & (obs_w <= q[3][:, :obs.shape[1]][:, mask])))
        return {
            "CRPS": float(crps_ensemble(obs_w, samp_w).mean()),
            "MAE": mae(obs_w, med_w),
            "RMSE": rmse(obs_w, med_w),
            "coverage_50": cov50,
            "coverage_90": cov90,
        }

    full_mask = np.ones(T, dtype=bool)
    records = []
    for channel, obs, pred, q in [("N", N_actual, N_pred, N_q),
                                  ("D", D_actual, D_pred, D_q)]:
        for window, mask in [("test", test_mask), ("full_history", full_mask)]:
            for metric, value in _channel_scores(obs, pred, q, mask).items():
                records.append({"channel": channel, "window": window,
                                "metric": metric, "value": value})
    if T_ext:
        # Extension window (post-training actuals): genuinely out of sample
        ext_mask = np.zeros(T_obs_end, dtype=bool)
        ext_mask[T:] = True
        for metric, value in _channel_scores(
                N_obs_full, N_pred, N_q, ext_mask).items():
            records.append({"channel": "N", "window": "extension",
                            "metric": metric, "value": value})
    scores_df = pd.DataFrame(records)
    scores_path = os.path.join(args.output_dir, "backtest_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"      Scores -> {scores_path}")
    print(scores_df.pivot_table(index=["channel", "window"], columns="metric",
                                values="value").round(3).to_string())

    # Per topic-month detail across the WHOLE history + forecast tail
    topic_labels = load_topic_labels(args.data_dir, K)
    detail_records = []
    for k in range(K):
        for t in range(T_all):
            has_actual = t < T_obs_end
            actual = float(N_obs_full[k, t]) if has_actual else np.nan
            q05, q25, q50, q75, q95 = (float(N_q[j, k, t]) for j in range(5))
            if t < T:
                period = "test" if test_mask[t] else "history"
                inside = float(q05 <= actual <= q95)
            elif has_actual:
                period = "extension"
                inside = float(q05 <= actual <= q95)
            else:
                period, inside = "forecast", np.nan
            detail_records.append({
                "topic": k, "topic_label": topic_labels[k],
                "date": str(all_periods[t]), "period": period,
                "actual_cves": actual,
                "predicted_mean_cves": float(N_mean[k, t]),
                "predicted_q05_cves": q05, "predicted_q25_cves": q25,
                "predicted_q50_cves": q50, "predicted_q75_cves": q75,
                "predicted_q95_cves": q95,
                "within_90pct_interval": inside,
            })
    detail_df = pd.DataFrame(detail_records)
    detail_path = os.path.join(args.output_dir, "threat_comparison.csv")
    detail_df.to_csv(detail_path, index=False)

    hist_cov = detail_df.loc[detail_df.period.isin(["history", "test"]),
                             "within_90pct_interval"].mean()
    test_cov = detail_df.loc[detail_df.period == "test", "within_90pct_interval"].mean()
    print(f"      Threat comparison (per topic-month, full history) -> {detail_path}")
    print(f"      90% coverage -- full history: {hist_cov:.1%}   test window: {test_cov:.1%}   (target ~90%)")
    if T_ext:
        ext_cov = detail_df.loc[detail_df.period == "extension",
                                "within_90pct_interval"].mean()
        print(f"      90% coverage -- extension ({ext_periods[0]}..{ext_periods[-1]}, "
              f"fully out of sample): {ext_cov:.1%}")

    # Per sector-month detail over the test window (incident channel)
    sector_names = get_default_sector_labels()[:S]
    test_idx = np.where(test_mask)[0]
    sector_records = []
    for s in range(S):
        for t in test_idx:
            q05, q25, q50, q75, q95 = (float(D_q[j, s, t]) for j in range(5))
            actual = float(D_actual[s, t])
            sector_records.append({
                "sector": sector_names[s], "date": str(all_periods[t]),
                "actual_incidents": actual,
                "predicted_mean_incidents": float(D_mean[s, t]),
                "predicted_q05_incidents": q05, "predicted_q50_incidents": q50,
                "predicted_q95_incidents": q95,
                "within_90pct_interval": float(q05 <= actual <= q95),
            })
    sector_df = pd.DataFrame(sector_records)
    sector_path = os.path.join(args.output_dir, "incident_comparison.csv")
    sector_df.to_csv(sector_path, index=False)
    d_cov = sector_df["within_90pct_interval"].mean()
    print(f"      Incident comparison (test window) -> {sector_path}  (90% coverage: {d_cov:.1%})")

    # 5. Interactive figures
    print("[5/5] Building interactive charts ...")
    import plotly.io as pio

    # Remove stale static PNGs so the dashboard doesn't show both eras.
    for stale in sorted(os.listdir(fig_dir)):
        if stale.startswith("backtest_topic_") and stale.endswith(".png"):
            os.remove(os.path.join(fig_dir, stale))

    overview_parts = []
    for k in range(K):
        actual_padded = np.full(T_all, np.nan)
        actual_padded[:T_obs_end] = N_obs_full[k]
        q = {name: N_q[j, k, :] for j, name in
             enumerate(["q05", "q25", "q50", "q75", "q95"])}
        fig = build_topic_figure(
            dates_ts, actual_padded, q, f"{topic_labels[k]} — topic {k:02d}",
            data_end=last_obs.to_timestamp(),
            test_start=test_start_p.to_timestamp(),
            test_end=(test_end_p + 1).to_timestamp(),
            train_end=periods[-1].to_timestamp() if T_ext else None,
        )
        html_path = os.path.join(fig_dir, f"backtest_topic_{k:02d}.html")
        fig.write_html(html_path, include_plotlyjs="directory", full_html=True)
        fig.write_json(os.path.join(fig_dir, f"backtest_topic_{k:02d}.json"))
        overview_parts.append(pio.to_html(
            fig, include_plotlyjs=False, full_html=False))

    overview_path = os.path.join(fig_dir, "backtest_overview.html")
    with open(overview_path, "w", encoding="utf-8") as fh:
        fh.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Backtest — sequential 1-step-ahead predictions</title>"
            "<script src='plotly.min.js'></script>"
            f"<style>body{{background:{_PAGE};font-family:{_FONT};"
            "margin:24px auto;max-width:1100px}}</style></head><body>"
            "<h2 style='color:#0b0b0b'>Sequential 1-step-ahead backtest</h2>"
            "<p style='color:#52514e'>Each month the model predicts the next month "
            "before seeing it, then updates its beliefs with the actual data. "
            "Blue: predicted median with 50%/90% credible bands. Black: actual.</p>"
            + "\n".join(overview_parts) + "</body></html>")
    print(f"      {K} interactive charts + overview -> {fig_dir}/")

    # Summary
    test_crps = scores_df.query("channel=='N' and window=='test' and metric=='CRPS'")["value"].iloc[0]
    ext_line = ""
    if T_ext:
        ext_cov_v = detail_df.loc[detail_df.period == "extension",
                                  "within_90pct_interval"].mean()
        ext_line = (
            f"Extension {ext_periods[0]}..{ext_periods[-1]} ({T_ext} months "
            f"beyond training, filter updated on actual CVE counts, fully "
            f"out of sample): 90% coverage {ext_cov_v:.0%}.\n")
    summary = (
        f"Sequential 1-step-ahead backtest ({periods[0]} .. {last_obs}, "
        f"scored on {args.test_start}..{args.test_end}).\n"
        + ext_line +
        f"Each month the model predicted the next month BEFORE seeing it, was "
        f"scored, then updated its regime beliefs with that month's actual data "
        f"(all four observation channels).\n\n"
        f"CVE-count channel, test window: CRPS {test_crps:,.1f}, "
        f"90% interval coverage {test_cov:.0%} (target ~90%), "
        f"50% coverage {detail_df.loc[detail_df.period == 'test'].pipe(lambda d: ((d.predicted_q25_cves <= d.actual_cves) & (d.actual_cves <= d.predicted_q75_cves)).mean()):.0%} (target ~50%).\n"
        f"Full-history 90% coverage: {hist_cov:.0%}.\n"
        f"Incident channel, test window: 90% coverage {d_cov:.0%}.\n\n"
        f"Caveats: static parameters were estimated on the full sample, so the "
        f"test window is out-of-sample only for the sequential state updates; "
        f"prediction-step covariates (effort index, exposure map) enter lagged "
        f"by one month, though the effort index itself is a two-sided full-"
        f"sample estimate; the forecast tail beyond {last_obs} holds both "
        f"at their last observed values."
    )
    summary_path = os.path.join(args.output_dir, "backtest_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(f"\n{summary}\n")

    # Optional AI-written accuracy note appended to the summary file.
    try:
        from cassandra_threatcast.llm.deepseek import explain_backtest
        ai_note = explain_backtest(
            detail_df[detail_df.period != "forecast"], scores_df, float(test_cov))
        if ai_note:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write("\n\n--- AI summary ---\n" + ai_note)
            print("      (AI summary appended.)")
    except Exception as exc:
        print(f"      (AI summary skipped: {exc})")

    print("Done.")


if __name__ == "__main__":
    main()
