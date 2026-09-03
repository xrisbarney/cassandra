#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Kernel experiment report (docs/PAPER_NOTES.md §9).

Distills the trained kernel-variant posterior (results/ablation/kernel/)
into lightweight artifacts for the dashboard's Kernel lab tab, so the app
never has to load the multi-hundred-MB posterior pickle:

  results/kernel/kernel_params.json       learned h, sigma_g posteriors +
                                          CRPS comparison vs the full model
  results/kernel/covariance_curve.csv     Cov(g_t, g_{t+d}) vs lag d
  results/kernel/g_t.csv                  the kernel factor through time
  results/kernel/loadings.csv             per-topic loadings a_k
  results/kernel/figures/*.html/.json     interactive charts

Usage:  python scripts/kernel_report.py       (after
        python scripts/ablation.py --variants kernel)
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.features.topic_map import load_topic_labels
from cassandra_threatcast.viz.interactive import (
    BLUE, INK, SURFACE, kernel_covariance_curve, kernel_loadings_bar,
    save_interactive, _base_layout,
)

IDATA = os.path.join("results", "ablation", "kernel", "idata.pkl")
ABLATION = os.path.join("results", "ablation", "ablation.csv")
OUT = os.path.join("results", "kernel")
DATA_DIR = os.path.join("data", "processed")


def main() -> None:
    if not os.path.exists(IDATA):
        print(f"No kernel posterior at {IDATA} -- run "
              "'python scripts/ablation.py --variants kernel' first.")
        sys.exit(1)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    print("[1/3] Loading kernel posterior ...")
    with open(IDATA, "rb") as fh:
        idata = pickle.load(fh)
    post = {k: np.asarray(v) for k, v in idata.posterior.items()}
    post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items() if v.size > 0}

    hw = post["g_halfwidth"].reshape(-1)          # (n,)
    sg = post["sigma_g"].reshape(-1)              # (n,)
    a_g = post["a_g"]                             # (n, K)
    w_all = post["g_kernel_weights"]              # (n, 2D+1)
    g_t = post["g_t"]                             # (n, T)
    n, K = a_g.shape
    D = (w_all.shape[1] - 1) // 2

    # Covariance curve: Cov(g_t, g_{t+d}) = sigma^2 * sum_j w_j w_{j+d}
    print("[2/3] Computing learned covariance curve ...")
    lags = np.arange(0, D + 1)
    cov = np.zeros((n, D + 1))
    for i in range(n):
        acf = np.correlate(w_all[i], w_all[i], mode="full")   # (4D+1,)
        cov[i] = (sg[i] ** 2) * acf[2 * D: 2 * D + D + 1]
    cov_mean = cov.mean(axis=0)
    cov_lo, cov_hi = np.quantile(cov, [0.05, 0.95], axis=0)

    pd.DataFrame({"lag_months": lags, "cov_mean": cov_mean,
                  "cov_q05": cov_lo, "cov_q95": cov_hi}).to_csv(
        os.path.join(OUT, "covariance_curve.csv"), index=False)

    # g_t through time
    with open(os.path.join(DATA_DIR, "panel_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    dates = (meta.get("dates") or meta.get("metadata", {}).get("dates", []))[: g_t.shape[1]]
    g_mean = g_t.mean(axis=0)
    g_lo, g_hi = np.quantile(g_t, [0.05, 0.95], axis=0)
    pd.DataFrame({"date": dates, "g_mean": g_mean,
                  "g_q05": g_lo, "g_q95": g_hi}).to_csv(
        os.path.join(OUT, "g_t.csv"), index=False)

    # Loadings
    topic_labels = load_topic_labels(DATA_DIR, K)
    a_mean, a_sd = a_g.mean(axis=0), a_g.std(axis=0)
    pd.DataFrame({"topic": range(K), "topic_label": topic_labels[:K],
                  "a_mean": a_mean, "a_sd": a_sd}).to_csv(
        os.path.join(OUT, "loadings.csv"), index=False)

    # Comparison vs full (both at the same reduced MCMC settings)
    comparison = {}
    if os.path.exists(ABLATION):
        ab = pd.read_csv(ABLATION)
        for name in ("full", "kernel"):
            row = ab[ab["variant"] == name]
            if len(row):
                comparison[name] = {
                    k: float(row.iloc[0][k])
                    for k in ("CRPS_h1", "CRPS_h6", "cov90_h1")
                    if k in row.columns and pd.notna(row.iloc[0][k])
                }

    params = {
        "halfwidth_months": {"mean": float(hw.mean()), "sd": float(hw.std()),
                             "q05": float(np.quantile(hw, 0.05)),
                             "q95": float(np.quantile(hw, 0.95))},
        "sigma_g": {"mean": float(sg.mean()), "sd": float(sg.std()),
                    "q05": float(np.quantile(sg, 0.05)),
                    "q95": float(np.quantile(sg, 0.95))},
        "comparison": comparison,
        "n_draws": int(n),
    }
    with open(os.path.join(OUT, "kernel_params.json"), "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)

    # Interactive figures
    print("[3/3] Building interactive figures ...")
    import plotly.graph_objects as go

    fig_cov = kernel_covariance_curve(lags, cov_mean, cov_lo, cov_hi,
                                      float(hw.mean()))
    save_interactive(fig_cov, os.path.join(OUT, "figures", "kernel_covariance"))

    x = [pd.Period(d, freq="M").to_timestamp() for d in dates]
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=x, y=g_hi, mode="lines", line=dict(width=0),
                               hoverinfo="skip", showlegend=False))
    fig_g.add_trace(go.Scatter(x=x, y=g_lo, mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor="rgba(42,120,214,0.12)",
                               name="90% credible band", hoverinfo="skip"))
    fig_g.add_trace(go.Scatter(x=x, y=g_mean, mode="lines",
                               line=dict(color=BLUE, width=2),
                               name="Kernel factor g_t (posterior mean)",
                               hovertemplate="%{y:.2f}<extra></extra>"))
    fig_g.add_hline(y=0.0, line_width=1, line_color=INK, opacity=0.25)
    _base_layout(fig_g, "The moving-window factor through time", height=340)
    fig_g.update_yaxes(title="g_t (log-intensity units)")
    fig_g.update_xaxes(rangeslider=dict(visible=True, thickness=0.08))
    save_interactive(fig_g, os.path.join(OUT, "figures", "kernel_g_t"))

    fig_a = kernel_loadings_bar(topic_labels[:K], a_mean, a_sd)
    save_interactive(fig_a, os.path.join(OUT, "figures", "kernel_loadings"))

    print(f"\nKernel report -> {OUT}/")
    print(f"  learned half-width: {hw.mean():.1f} months "
          f"(90% CI {np.quantile(hw, 0.05):.1f}-{np.quantile(hw, 0.95):.1f})")
    print(f"  sigma_g:            {sg.mean():.3f} "
          f"(90% CI {np.quantile(sg, 0.05):.3f}-{np.quantile(sg, 0.95):.3f})")
    if comparison.get("kernel") and comparison.get("full"):
        d = 100 * (comparison["kernel"]["CRPS_h1"] - comparison["full"]["CRPS_h1"]) / comparison["full"]["CRPS_h1"]
        print(f"  CRPS h=1 vs full:   {comparison['kernel']['CRPS_h1']:.2f} vs "
              f"{comparison['full']['CRPS_h1']:.2f}  ({d:+.1f}%)")


if __name__ == "__main__":
    main()
