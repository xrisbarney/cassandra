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
import json
import os
import sys
import warnings
import yaml
import numpy as np
import pandas as pd

# Force UTF-8 stdout/stderr: Windows' default console codepage cannot encode
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.data import pipeline
from cassandra_threatcast.data.bea_io import get_default_sector_labels
from cassandra_threatcast.features.topic_map import load_topic_labels
from cassandra_threatcast.viz.interactive import fan_chart, loss_distribution, save_interactive


# Argument parsing
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


# Helpers
def _quantile_df(
    pred_samples: np.ndarray,  # (n_samples, K, horizon)
    dates_pred: list,
    topic_labels: list[str],
    quantiles: list[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
) -> pd.DataFrame:
    """Convert predictive samples to a tidy quantile DataFrame.

    Value columns (``mean_cves_per_month``, ``q05_cves_per_month``, ...) are
    counts of new CVEs published that month for that topic -- a MONTHLY
    figure (unlike the loss table's horizon-cumulative totals), made explicit
    via the column name suffix.

    The mean is a 99%-trimmed mean (top 1% of draws excluded): the log-scale
    latent allows rare draws up to e^30, which blow the raw ensemble mean
    orders of magnitude past the same row's q95 at long horizons (see
    docs/PAPER_NOTES.md §7).  Quantiles are the headline summaries.
    """
    records = []
    K, H = pred_samples.shape[1], pred_samples.shape[2]
    for k in range(K):
        for hi, date in enumerate(dates_pred):
            samp = pred_samples[:, k, hi]
            thr = np.quantile(samp, 0.99)
            row = {
                "topic": k,
                "topic_label": topic_labels[k] if k < len(topic_labels) else f"Topic {k}",
                "date": str(date),
                "mean_cves_per_month": float(samp[samp <= thr].mean()),
            }
            for q in quantiles:
                row[f"q{int(q * 100):02d}_cves_per_month"] = float(np.quantile(pred_samples[:, k, hi], q))
            records.append(row)
    return pd.DataFrame(records)




def _loss_summary_df(
    loss_samples: np.ndarray,  # (n_samples,) or (n_samples, S, T_pred)
    dates_pred: list,
    sector_names: list[str],
    horizon_months: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarise the posterior loss distribution.

    All value columns are in USD and are TOTALS ACCUMULATED OVER THE WHOLE
    forecast horizon (summed across months), not a monthly figure -- this is
    made explicit in the column names (``_usd_total``) and via the
    ``horizon_months`` column, since a bare "mean"/"VaR95" column is easy to
    misread as a monthly run-rate.
    """
    var_pct = int((1 - alpha) * 100)

    def _row(scope: str, vals: np.ndarray) -> dict:
        return {
            "scope": scope,
            "horizon_months": horizon_months,
            "mean_usd_total": float(np.mean(vals)),
            "sd_usd_total": float(np.std(vals)),
            f"VaR{var_pct}_usd_total": float(np.quantile(vals, 1 - alpha)),
            f"ES{var_pct}_usd_total": float(np.mean(vals[vals >= np.quantile(vals, 1 - alpha)])),
            "q05_usd_total": float(np.quantile(vals, 0.05)),
            "q50_usd_total": float(np.quantile(vals, 0.50)),
            "q95_usd_total": float(np.quantile(vals, 0.95)),
        }

    if loss_samples.ndim == 1:
        # Aggregate across all sectors/time already done
        records = [_row("aggregate", loss_samples)]
    elif loss_samples.ndim == 3:
        # (n_samples, S, T_pred)  -> aggregate over time, then per-sector + total
        loss_agg_time = loss_samples.sum(axis=-1)   # (n_samples, S)
        loss_total    = loss_agg_time.sum(axis=-1)  # (n_samples,)
        records = [_row(name, loss_agg_time[:, s]) for s, name in enumerate(sector_names)]
        records.append(_row("aggregate", loss_total))
    else:
        raise ValueError(f"Unexpected loss_samples shape: {loss_samples.shape}")

    return pd.DataFrame(records)


# Main
def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    K = config["model"]["K"]
    S = config["model"]["S"]
    topic_labels = config.get("topic_labels") or load_topic_labels(args.data_dir, K)
    sector_names = config.get("sector_names") or get_default_sector_labels()[:S]

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
        "mandatory_t": np.array(
            [str(date) >= "2023-12" for date in panel.get("dates", [])], dtype=float
        ) if len(panel.get("dates", [])) == T else np.zeros(T),
    }

    # Reconstruct dates
    dates_all = panel.get("dates", list(range(T + args.horizon)))
    obs_dates  = dates_all[:T]
    pred_dates = dates_all[T : T + args.horizon]
    if len(pred_dates) < args.horizon:
        # The panel's date list only covers the observed T months, so the
        n_missing = args.horizon - len(pred_dates)
        last = dates_all[-1] if dates_all else T - 1
        if isinstance(last, pd.Period):
            future = pd.period_range(start=last + 1, periods=n_missing, freq=last.freq)
            pred_dates = list(pred_dates) + list(future)
        else:
            pred_dates = list(pred_dates) + list(range(int(last) + 1, int(last) + 1 + n_missing))

    # --- Load InferenceData -------------------------------------------------
    print("[2/4] Loading InferenceData and generating posterior predictive ...")
    try:
        import arviz as az
        pkl_path = os.path.splitext(args.idata)[0] + ".pkl"
        if os.path.exists(pkl_path):
            import pickle
            with open(pkl_path, "rb") as fh:
                idata = pickle.load(fh)
        else:
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
        from cassandra_threatcast.model import paper_exact
        from cassandra_threatcast.model.economic import DamageFunctionParams

        # Flatten posterior (chain, draw, ...) -> (sample, ...)
        post = {k: np.asarray(v) for k, v in idata.posterior.items()}
        post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items()}

        # Future exposure: hold the last observed month constant over the horizon.
        M_future = np.repeat(M_skt[:, :, -1:], args.horizon, axis=2)  # (S, K, horizon)

        # Damage-function parameters: the event-calibrated posterior (paper
        cal_path = os.path.join("results", "calibration", "damage_params.json")
        if os.path.exists(cal_path):
            with open(cal_path, encoding="utf-8") as fh:
                dmg_cal = json.load(fh)
            damage_params = DamageFunctionParams(
                shape=np.asarray(dmg_cal["shape"], dtype=float),
                scale=np.asarray(dmg_cal["scale"], dtype=float),
                max_damage=np.asarray(dmg_cal["max_damage"], dtype=float),
            )
            print(f"      Damage functions: event-calibrated ({cal_path})")
        else:
            dmg = config.get("damage_params", {})
            damage_params = DamageFunctionParams(
                shape=np.full(S, float(dmg.get("shape", 2.0))),
                scale=np.full(S, float(dmg.get("scale", 1.0))),
                max_damage=np.full(S, float(dmg.get("max_damage", 0.5))),
            )
            print("      Damage functions: config defaults -- run "
                  "scripts/calibrate_damage.py for the event-calibrated set.")

        # BEA gross output arrives in $ millions; convert so every loss
        x_s_usd = x_s * 1e6

        if "Phi_r" in post and "z_t" in post:
            damage_draws_path = os.path.join("results", "calibration", "damage_posterior.npz")
            damage_draws = None
            if os.path.exists(damage_draws_path):
                with np.load(damage_draws_path) as saved:
                    damage_draws = {name: saved[name] for name in saved.files}
            out = paper_exact.predict(
                post, data_dict, args.horizon, Lambda_L, x_s_usd, M_future,
                damage_params, damage_draws=damage_draws,
                exposure_concentration=float(config.get("forecast", {}).get(
                    "exposure_concentration", 200.0)),
            )
        else:
            out = full_module.predict(
                post, data_dict, args.horizon,
                Lambda_L, x_s_usd, M_future, damage_params,
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
    print(f"      Predictive quantiles -> {q_path}  (CVE counts per month, per topic)")

    loss_df = _loss_summary_df(loss_samples, pred_dates, sector_names, horizon_months=args.horizon)
    loss_path = os.path.join(args.output_dir, "loss_distribution.csv")
    loss_df.to_csv(loss_path, index=False)
    print(f"      Loss distribution   -> {loss_path}")
    print(f"      (All figures in USD, TOTAL accumulated over the "
          f"{args.horizon}-month forecast horizon -- not a monthly rate)")
    print(loss_df.to_string(index=False))

    # --- Systemic-event probabilities P(agg loss > c)  (paper Problem 2) ----
    agg_total = loss_samples.sum(axis=(1, 2)) if loss_samples.ndim == 3 else loss_samples
    thresholds = config.get("evaluation", {}).get(
        "loss_thresholds_usd", [1e9, 2e9, 5e9, 1e10, 5e10, 1e11])
    sys_df = pd.DataFrame([
        {"threshold_usd": float(c),
         "prob_exceed": float(np.mean(agg_total > c)),
         "horizon_months": args.horizon}
        for c in thresholds
    ])
    sys_path = os.path.join(args.output_dir, "systemic_probs.csv")
    sys_df.to_csv(sys_path, index=False)
    print(f"      Systemic-event probabilities -> {sys_path}")
    for _, row in sys_df.iterrows():
        print(f"        P(total loss > ${row.threshold_usd/1e9:,.0f}B) = {row.prob_exceed:.1%}")

    # --- Table 8: ranked sectoral exposure ----------------------------------
    if out is not None and "g_pred" in out:
        g_pred = out["g_pred"]                      # (n, S, H) damage fractions
        sigma_pred = out["sigma_pred"]              # (n, K, H)
        lam_pred = np.exp(np.clip(out["lambda_pred"], -30.0, 30.0))   # (n, K, H)
        ell_by_sector = loss_samples.sum(axis=2)    # (n, S) horizon totals

        g_mean_h = g_pred.mean(axis=2)              # (n, S) mean over horizon
        expo_mean = g_mean_h.mean(axis=0)
        expo_q05, expo_q95 = np.quantile(g_mean_h, [0.05, 0.95], axis=0)
        contrib = ell_by_sector.mean(axis=0)
        contrib_pct = 100.0 * contrib / max(contrib.sum(), 1e-12)

        # Dominant threat topics per sector: mean contribution to the shock
        drive = (lam_pred * sigma_pred).mean(axis=(0, 2))   # (K,)
        M_h_mean = M_future.mean(axis=2)                    # (S, K)
        rows8 = []
        for s in range(S):
            weights = M_h_mean[s] * drive
            top2 = np.argsort(weights)[::-1][:2]
            rows8.append({
                "sector": sector_names[s],
                "exposure_index_mean": float(expo_mean[s]),
                "exposure_index_q05": float(expo_q05[s]),
                "exposure_index_q95": float(expo_q95[s]),
                "loss_contribution_pct": float(contrib_pct[s]),
                "dominant_threat_topics": "; ".join(
                    topic_labels[k] if k < len(topic_labels) else f"Topic {k}"
                    for k in top2),
            })
        table8 = pd.DataFrame(rows8).sort_values(
            "loss_contribution_pct", ascending=False).reset_index(drop=True)
        table8.index += 1
        table8_path = os.path.join(args.output_dir, "sector_exposure.csv")
        table8.to_csv(table8_path, index_label="rank")
        print(f"      Sector exposure (Table 8) -> {table8_path}")

    # --- Regime probabilities: the early-warning signal ---------------------
    try:
        from cassandra_threatcast.evaluation.sequential import batch_forward_filter
        from cassandra_threatcast.viz.interactive import regime_area

        Pi_all = np.asarray(post["Pi"], dtype=float)
        if "z_t" in post:
            z_draws = np.asarray(post["z_t"], dtype=int)
            R = Pi_all.shape[-1]
            hist_probs = np.stack([(z_draws == r).mean(axis=0) for r in range(R)], axis=1)
            p = np.eye(R)[z_draws[:, -1]]
        else:
            loglik_all = np.asarray(post["loglik_regime_t"], dtype=float)
            filtered, _ = batch_forward_filter(loglik_all, Pi_all)
            R = filtered.shape[-1]
            hist_probs = filtered.mean(axis=0)
            p = filtered[:, -1]

        fwd_probs = []
        for _h in range(args.horizon):
            p = np.einsum("nr,nrj->nj", p, Pi_all)
            fwd_probs.append(p.mean(axis=0))
        fwd_probs = np.asarray(fwd_probs)                           # (H, R)

        # Order regime labels by their posterior mean intensity level so
        mu_level = np.asarray(post["mu_r"]).mean(axis=(0, 2))       # (R,)
        order = np.argsort(mu_level)
        names = ["low activity", "elevated", "high activity"][:R]
        regime_names = [""] * R
        for rank, r_idx in enumerate(order):
            regime_names[r_idx] = f"Regime {rank + 1} ({names[min(rank, len(names)-1)]})"

        all_dates = list(obs_dates) + list(pred_dates[: args.horizon])
        probs_all = np.vstack([hist_probs, fwd_probs])              # (T+H, R)
        reg_df = pd.DataFrame(probs_all, columns=regime_names)
        reg_df.insert(0, "date", [str(d) for d in all_dates])
        reg_df["period"] = ["history"] * len(obs_dates) + ["forecast"] * len(fwd_probs)
        reg_path = os.path.join(args.output_dir, "regime_probs.csv")
        reg_df.to_csv(reg_path, index=False)
        print(f"      Regime probabilities (early-warning) -> {reg_path}")
        print("        Current filtered regime: "
              + ", ".join(f"{regime_names[r]} {hist_probs[-1, r]:.0%}" for r in range(R)))

        fig_dir_reg = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir_reg, exist_ok=True)
        fig_reg = regime_area(all_dates, probs_all, regime_names,
                              forecast_start=len(obs_dates))
        save_interactive(fig_reg, os.path.join(fig_dir_reg, "regime_probs"))
        print(f"      Regime chart -> {fig_dir_reg}/regime_probs.html")
    except Exception as exc:
        warnings.warn(f"Regime-probability output failed: {exc}")

    # --- Plain-English summary (optional, needs DEEPSEEK_API_KEY) -----------
    from cassandra_threatcast.llm.deepseek import explain_forecast
    summary = explain_forecast(q_df, loss_df)
    if summary:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(summary)
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(summary)
        print(f"\n      Summary -> {summary_path}")
    else:
        print("\n      (Set DEEPSEEK_API_KEY in .env to get an AI-generated "
              "plain-English summary here and in the dashboard.)")

    # --- Interactive fan-chart figures ---------------------------------------
    print("[4/4] Saving interactive fan-chart figures ...")
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Remove stale static PNGs so the dashboard doesn't show both eras.
    for stale in sorted(os.listdir(fig_dir)):
        if stale.endswith(".png"):
            os.remove(os.path.join(fig_dir, stale))

    for k in range(K):
        label = topic_labels[k] if k < len(topic_labels) else f"Topic {k}"
        fig = fan_chart(
            obs=N_kt[k],
            obs_dates=obs_dates,
            pred_samples=pred_N[:, k, :],   # (n_samples, horizon)
            pred_dates=pred_dates[: args.horizon],
            topic_label=f"{label} — topic {k:02d}",
        )
        save_interactive(fig, os.path.join(fig_dir, f"fan_chart_topic_{k:02d}"))

    print(f"      {K} interactive fan charts (.html + .json) -> {fig_dir}/")

    # Loss distribution figure (values pre-scaled to $ billions; the units
    agg_loss = loss_samples.sum(axis=(1, 2)) if loss_samples.ndim == 3 else loss_samples
    fig_loss = loss_distribution(
        agg_loss / 1e9, units=f"$ billions, total over {args.horizon} months")
    save_interactive(fig_loss, os.path.join(fig_dir, "loss_distribution"))
    print(f"      Loss distribution figure -> {fig_dir}/loss_distribution.html")

    print("\nDone.")


if __name__ == "__main__":
    main()
