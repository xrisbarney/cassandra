#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Generate 12-month-ahead forecasts from a fitted model.

Steps:
  1. Load the fitted InferenceData (NetCDF) and processed panel.
  2. Generate posterior predictive samples via predict().
  3. Save predictive quantile CSV and loss distribution summary CSV.
  4. Save fan-chart figures for each CVE topic.

Usage:
    python scripts/forecast.py \\
        --idata      results/idata.nc \\
        --data-dir   data/processed/ \\
        --output-dir results/forecasts/ \\
        --config     configs/default.yaml \\
        --horizon    12
"""
import argparse
import os
import sys
import warnings
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.data import pipeline
from cassandra_threatcast.viz.plots import plot_threat_forecast, plot_loss_distribution, save_figure


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 12-month-ahead forecasts from a fitted model."
    )
    parser.add_argument(
        "--idata", default="results/idata.nc",
        help="Path to ArviZ InferenceData NetCDF file.",
    )
    parser.add_argument(
        "--data-dir", default="data/processed/",
        help="Directory with processed panel arrays.",
    )
    parser.add_argument(
        "--output-dir", default="results/forecasts/",
        help="Directory for forecast outputs.",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="YAML config file.",
    )
    parser.add_argument(
        "--horizon", type=int, default=12,
        help="Forecast horizon in months.  Default: 12",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for posterior predictive draws.  Default: 0",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of posterior samples to draw (default: all).",
    )
    parser.add_argument(
        "--enhanced-mode", action="store_true",
        help="Match a model fitted with --enhanced-mode (Student-t innovations, "
             "Negative-Binomial incidents) when generating forecasts.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quantile_df(
    pred_samples: np.ndarray,  # (n_samples, K, horizon)
    dates_pred: list,
    topic_labels: list[str],
    quantiles: list[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
) -> pd.DataFrame:
    """Convert predictive samples to a tidy quantile DataFrame."""
    records = []
    K, H = pred_samples.shape[1], pred_samples.shape[2]
    for k in range(K):
        for hi, date in enumerate(dates_pred):
            row = {
                "topic": k,
                "topic_label": topic_labels[k] if k < len(topic_labels) else f"Topic {k}",
                "date": str(date),
                "mean": float(np.mean(pred_samples[:, k, hi])),
            }
            for q in quantiles:
                row[f"q{int(q * 100):02d}"] = float(np.quantile(pred_samples[:, k, hi], q))
            records.append(row)
    return pd.DataFrame(records)


def _loss_summary_df(
    loss_samples: np.ndarray,  # (n_samples,) or (n_samples, S, T_pred)
    dates_pred: list,
    sector_names: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarise the posterior loss distribution."""
    if loss_samples.ndim == 1:
        # Aggregate across all sectors/time already done
        agg = loss_samples
        records = [{
            "scope": "aggregate",
            "mean":  float(np.mean(agg)),
            "sd":    float(np.std(agg)),
            f"VaR{int((1-alpha)*100)}":  float(np.quantile(agg, 1 - alpha)),
            f"ES{int((1-alpha)*100)}":   float(np.mean(agg[agg >= np.quantile(agg, 1 - alpha)])),
            "q05": float(np.quantile(agg, 0.05)),
            "q50": float(np.quantile(agg, 0.50)),
            "q95": float(np.quantile(agg, 0.95)),
        }]
    elif loss_samples.ndim == 3:
        # (n_samples, S, T_pred)  -> aggregate over time, then per-sector + total
        loss_agg_time = loss_samples.sum(axis=-1)   # (n_samples, S)
        loss_total    = loss_agg_time.sum(axis=-1)  # (n_samples,)
        records = []
        for s, name in enumerate(sector_names):
            col = loss_agg_time[:, s]
            records.append({
                "scope": name,
                "mean":  float(np.mean(col)),
                "sd":    float(np.std(col)),
                f"VaR{int((1-alpha)*100)}":  float(np.quantile(col, 1 - alpha)),
                f"ES{int((1-alpha)*100)}":   float(np.mean(col[col >= np.quantile(col, 1 - alpha)])),
                "q05": float(np.quantile(col, 0.05)),
                "q50": float(np.quantile(col, 0.50)),
                "q95": float(np.quantile(col, 0.95)),
            })
        records.append({
            "scope": "aggregate",
            "mean":  float(np.mean(loss_total)),
            "sd":    float(np.std(loss_total)),
            f"VaR{int((1-alpha)*100)}":  float(np.quantile(loss_total, 1 - alpha)),
            f"ES{int((1-alpha)*100)}":   float(np.mean(loss_total[loss_total >= np.quantile(loss_total, 1 - alpha)])),
            "q05": float(np.quantile(loss_total, 0.05)),
            "q50": float(np.quantile(loss_total, 0.50)),
            "q95": float(np.quantile(loss_total, 0.95)),
        })
    else:
        raise ValueError(f"Unexpected loss_samples shape: {loss_samples.shape}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    K = config["model"]["K"]
    S = config["model"]["S"]
    topic_labels = config.get("topic_labels", [f"Topic {k}" for k in range(K)])
    sector_names = config.get("sector_names", [f"Sector {s}" for s in range(S)])

    # --- Load panel ---------------------------------------------------------
    print("[1/4] Loading panel ...")
    panel = pipeline.load_panel(args.data_dir)
    N_kt  = panel["N"].astype(float)   # (K, T)
    D_st  = panel["D"].astype(float)   # (S, T)
    T     = N_kt.shape[1]

    def _load(fname, fallback):
        p = os.path.join(args.data_dir, fname)
        return np.load(p) if os.path.exists(p) else fallback

    M_skt    = _load("M_skt.npy",    np.ones((S, K, T)) / S)
    e_t      = _load("e_t.npy",      np.zeros(T))
    Lambda_L = _load("Lambda_L.npy", np.eye(S))
    x_s      = _load("x_s.npy",      np.ones(S) * 1e12)

    data_dict = {
        "N": N_kt, "D": D_st,
        "B": panel.get("B", np.zeros_like(N_kt)),
        "E": panel.get("E", np.zeros_like(N_kt)),
        "M_skt": M_skt, "e_t": e_t,
        "x_s": x_s, "Lambda_L": Lambda_L,
    }

    # Reconstruct dates
    dates_all = panel.get("dates", list(range(T + args.horizon)))
    obs_dates  = dates_all[:T]
    pred_dates = dates_all[T : T + args.horizon]
    if len(pred_dates) < args.horizon:
        # Extend with synthetic indices if dates array is too short
        last = dates_all[-1] if dates_all else T - 1
        pred_dates = list(pred_dates) + list(range(int(last) + 1, int(last) + 1 + args.horizon - len(pred_dates)))

    # --- Load InferenceData -------------------------------------------------
    print("[2/4] Loading InferenceData and generating posterior predictive ...")
    try:
        import arviz as az
        idata = az.from_netcdf(args.idata)
    except Exception as exc:
        print(f"Error loading idata: {exc}")
        sys.exit(1)

    # --- Generate forecasts -------------------------------------------------
    enhanced = args.enhanced_mode or bool(config.get("enhanced", {}).get("enabled", False))
    student_t_df = float(config.get("enhanced", {}).get("student_t_df", 4.0))
    print(f"      Model variant: {'ENHANCED' if enhanced else 'paper-native'}")
    out = None
    try:
        from cassandra_threatcast.model import full as full_module
        from cassandra_threatcast.model.economic import DamageFunctionParams

        # Flatten posterior (chain, draw, ...) -> (sample, ...)
        post = {k: np.asarray(v) for k, v in idata.posterior.items()}
        post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items()}

        # Future exposure: hold the last observed month constant over the horizon.
        M_future = np.repeat(M_skt[:, :, -1:], args.horizon, axis=2)  # (S, K, horizon)

        # Damage-function parameters (calibrated params if provided, else defaults).
        dmg = config.get("damage_params", {})
        damage_params = DamageFunctionParams(
            shape=np.full(S, float(dmg.get("shape", 2.0))),
            scale=np.full(S, float(dmg.get("scale", 1.0))),
            max_damage=np.full(S, float(dmg.get("max_damage", 0.5))),
        )

        out = full_module.predict(
            post, data_dict, args.horizon,
            Lambda_L, x_s, M_future, damage_params,
            enhanced=enhanced, student_t_df=student_t_df,
        )
        pred_N = out["N_pred"]   # (n_samples, K, horizon) forecast CVE counts
    except Exception as exc:
        warnings.warn(f"full_module.predict failed: {exc}. "
                      "Attempting to use posterior_predictive from idata.")
        if hasattr(idata, "posterior_predictive") and "N_obs" in idata.posterior_predictive:
            N_pp = np.array(idata.posterior_predictive["N_obs"])
            N_pp = N_pp.reshape(-1, *N_pp.shape[2:])  # (n_samples, K, T)
            pred_N = N_pp[:, :, -args.horizon :]
        else:
            print("Cannot obtain posterior predictive samples. Exiting.")
            sys.exit(1)

    if args.n_samples is not None:
        pred_N = pred_N[: args.n_samples]

    n_samples, _K, _H = pred_N.shape
    print(f"      pred_N shape: {pred_N.shape}  (n_samples={n_samples}, K={_K}, H={_H})")

    # Economic loss samples: prefer the losses computed inside predict()
    # (damage function + Leontief propagation + severity), else fall back.
    if out is not None:
        loss_samples = out["ell_pred"]                       # (n_samples, S, horizon)
        if args.n_samples is not None:
            loss_samples = loss_samples[: args.n_samples]
    else:
        warnings.warn("predict() unavailable; using proxy losses.")
        proxy = pred_N.sum(axis=1, keepdims=True) * (x_s.mean() * 1e-5)
        loss_samples = np.broadcast_to(proxy, (n_samples, S, args.horizon)).copy()

    # --- Save quantile CSV --------------------------------------------------
    print("[3/4] Saving forecast outputs ...")
    q_df = _quantile_df(pred_N, pred_dates, topic_labels)
    q_path = os.path.join(args.output_dir, "forecast_quantiles.csv")
    q_df.to_csv(q_path, index=False)
    print(f"      Predictive quantiles -> {q_path}")

    loss_df = _loss_summary_df(loss_samples, pred_dates, sector_names)
    loss_path = os.path.join(args.output_dir, "loss_distribution.csv")
    loss_df.to_csv(loss_path, index=False)
    print(f"      Loss distribution   -> {loss_path}")
    print(loss_df.to_string(index=False))

    # --- Fan-chart figures --------------------------------------------------
    print("[4/4] Saving fan-chart figures ...")
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for scripts

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for k in range(K):
        label = topic_labels[k] if k < len(topic_labels) else f"Topic {k}"
        fig = plot_threat_forecast(
            topic_k=k,
            obs=N_kt[k],
            predictive=pred_N[:, k, :],   # (n_samples, horizon)
            dates=[],
            obs_dates=obs_dates,
            pred_dates=pred_dates[: args.horizon],
            topic_label=label,
        )
        fig_path = os.path.join(fig_dir, f"fan_chart_topic_{k:02d}.png")
        save_figure(fig, fig_path, dpi=200)

    print(f"      {K} fan charts saved to {fig_dir}/")

    # Loss distribution figure
    agg_loss = loss_samples.sum(axis=(1, 2)) if loss_samples.ndim == 3 else loss_samples
    fig_loss = plot_loss_distribution(agg_loss, units="$ billions")
    loss_fig_path = os.path.join(fig_dir, "loss_distribution.png")
    save_figure(fig_loss, loss_fig_path, dpi=200)
    print(f"      Loss distribution figure -> {loss_fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
