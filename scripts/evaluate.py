#!/usr/bin/env python3
"""
Rolling-origin (expanding window) forecast evaluation.

For each fold:
  - Train on all data up to the fold boundary.
  - Generate h-step-ahead forecasts for each horizon in config.evaluation.horizons.
  - Evaluate the full model and all baselines.
  - Run Diebold-Mariano tests against the full model.

Results are saved as CSV files in --output-dir.

Usage:
    python scripts/evaluate.py \\
        --idata       results/idata.nc \\
        --data-dir    data/processed/ \\
        --output-dir  results/evaluation/ \\
        --config      configs/default.yaml
"""
import argparse
import os
import sys
import warnings
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cyberthreats.data import pipeline
from cyberthreats.evaluation.scoring import crps_ensemble, mae, rmse
from cyberthreats.evaluation.calibration import calibration_report
from cyberthreats.evaluation.dm_test import dm_table
from cyberthreats.evaluation.baselines import run_all_baselines


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling-origin forecast evaluation for the full model and baselines."
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
        "--output-dir", default="results/evaluation/",
        help="Directory for evaluation CSV files.",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="YAML config file.",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None,
        help="Cap the number of evaluation folds (for quick testing).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scores_from_predictive(
    obs: np.ndarray,          # (..., T_test)
    pred_samples: np.ndarray, # (n_samples, ..., T_test)
    horizons: list[int],
) -> pd.DataFrame:
    """Compute per-horizon CRPS/MAE/RMSE from posterior predictive samples."""
    records = []
    for h in horizons:
        h = min(h, obs.shape[-1])
        obs_h   = obs[..., :h].ravel()
        pred_h  = pred_samples[..., :h].reshape(pred_samples.shape[0], -1)
        pred_h_T = pred_h.T   # (N, n_samples)

        crps_val = float(np.mean(crps_ensemble(obs_h, pred_h_T)))
        median   = np.median(pred_h, axis=0)
        records.append({"horizon": h, "metric": "CRPS", "value": crps_val})
        records.append({"horizon": h, "metric": "MAE",  "value": mae(obs_h, median)})
        records.append({"horizon": h, "metric": "RMSE", "value": rmse(obs_h, median)})
    return pd.DataFrame(records)


def _crps_series(
    obs: np.ndarray,          # (N,)
    pred_samples: np.ndarray, # (n_samples, N)
) -> np.ndarray:
    """Return per-observation CRPS values (for DM test loss series)."""
    return crps_ensemble(obs, pred_samples.T)   # (N,)


def _get_full_model_predictive(
    idata,
    data: dict,
    config: dict,
    train_T: int,
    horizon: int,
) -> np.ndarray:
    """
    Draw posterior predictive samples for observations in [train_T, train_T+horizon).

    Returns (n_samples, K, horizon) array of CVE-count predictive samples.
    Uses the predict() function from the model module if available;
    otherwise falls back to extracting posterior_predictive samples that
    were already generated during training.
    """
    try:
        from cyberthreats.model import full as full_module
        from cyberthreats.inference import nuts as nuts_module
        pred = nuts_module.predict(
            idata=idata,
            model=full_module.full_model,
            data=data,
            config=config,
            forecast_horizon=horizon,
            seed=0,
        )
        return pred  # (n_samples, K, horizon)
    except Exception as exc:
        warnings.warn(f"nuts_module.predict failed: {exc}. "
                      "Falling back to posterior_predictive if available.")

    # Fallback: slice posterior_predictive stored during training
    if hasattr(idata, "posterior_predictive") and "N_obs" in idata.posterior_predictive:
        N_pred = np.array(idata.posterior_predictive["N_obs"])
        # Shape: (chain, draw, K, T)  -> flatten chains
        N_pred = N_pred.reshape(-1, *N_pred.shape[2:])
        return N_pred[:, :, train_T : train_T + horizon]

    raise RuntimeError("Cannot obtain posterior predictive samples.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    eval_cfg   = config.get("evaluation", {})
    horizons   = eval_cfg.get("horizons", [1, 3, 6, 12])
    test_years = eval_cfg.get("test_years", 2)
    val_years  = eval_cfg.get("val_years",  1)

    # --- Load data ----------------------------------------------------------
    print("[1/4] Loading panel ...")
    panel = pipeline.load_panel(args.data_dir)
    N_kt  = panel["N"].astype(float)   # (K, T)
    D_st  = panel["D"].astype(float)   # (S, T)
    K, T  = N_kt.shape

    def _load(fname, fallback):
        p = os.path.join(args.data_dir, fname)
        return np.load(p) if os.path.exists(p) else fallback

    S = config["model"]["S"]
    M_skt    = _load("M_skt.npy",    np.ones((S, K, T)) / S)
    e_t      = _load("e_t.npy",      np.zeros(T))
    Lambda_L = _load("Lambda_L.npy", np.eye(S))
    x_s      = _load("x_s.npy",      np.ones(S) * 1e12)

    # --- Load InferenceData -------------------------------------------------
    print("[2/4] Loading InferenceData ...")
    try:
        import arviz as az
        idata = az.from_netcdf(args.idata)
        print("      InferenceData loaded.")
    except Exception as exc:
        print(f"      Warning: Could not load idata ({exc}).")
        idata = None

    # --- Define rolling folds -----------------------------------------------
    test_T = test_years * 12
    # Earliest start: need enough history for baselines (at least 2 seasonal periods)
    min_train_T = max(24, T - test_T * 3)
    fold_starts = list(range(min_train_T, T - max(horizons), 6))  # every 6 months

    if args.max_folds is not None:
        fold_starts = fold_starts[: args.max_folds]

    print(f"[3/4] Running {len(fold_starts)} evaluation folds ...")
    print(f"      Horizons: {horizons}  T={T}  test_T={test_T}")

    all_scores: list[pd.DataFrame] = []
    dm_crps_series: dict[str, list[float]] = {}  # model -> flat CRPS list

    for fold_idx, train_T in enumerate(fold_starts):
        avail_h = T - train_T
        fold_horizons = [h for h in horizons if h <= avail_h]
        if not fold_horizons:
            continue

        horizon = max(fold_horizons)
        obs_test_N = N_kt[:, train_T : train_T + horizon]   # (K, horizon)

        print(f"  Fold {fold_idx + 1}/{len(fold_starts)}: "
              f"train_T={train_T}  horizon={horizon}")

        # --- Full model predictive ------------------------------------------
        if idata is not None:
            try:
                data_dict = {
                    "N": N_kt, "D": D_st,
                    "B": panel.get("B", np.zeros_like(N_kt)),
                    "E": panel.get("E", np.zeros_like(N_kt)),
                    "M_skt": M_skt, "e_t": e_t,
                    "x_s": x_s, "Lambda_L": Lambda_L,
                }
                full_pred = _get_full_model_predictive(
                    idata, data_dict, config, train_T, horizon
                )   # (n_samples, K, horizon)

                scores_full = _scores_from_predictive(obs_test_N, full_pred, fold_horizons)
                scores_full["model"] = "FullModel"
                scores_full["fold"] = fold_idx
                all_scores.append(scores_full)

                # Accumulate flat CRPS for DM test
                flat_crps = _crps_series(
                    obs_test_N.ravel(),
                    full_pred.reshape(full_pred.shape[0], -1),
                )
                dm_crps_series.setdefault("FullModel", []).extend(flat_crps.tolist())

            except Exception as exc:
                warnings.warn(f"  Full model evaluation failed at fold {fold_idx}: {exc}")

        # --- Baselines -------------------------------------------------------
        train_panel_N = N_kt[:, :train_T]
        try:
            baseline_results = run_all_baselines(
                panel=train_panel_N,
                horizons=fold_horizons,
                quantiles=[0.1, 0.5, 0.9],
                test_T=horizon,
            )
        except Exception as exc:
            warnings.warn(f"  Baselines failed at fold {fold_idx}: {exc}")
            baseline_results = {}

        for bname, bres in baseline_results.items():
            preds_q = bres["predictions"]   # (K, horizon, 3)
            # Use the median (quantile index 1) as point forecast
            median_pred = preds_q[:, :, 1]   # (K, horizon)
            # Simulate samples by adding Gaussian noise around the median
            # scaled by the IQR from the quantile predictions
            iqr = preds_q[:, :, 2] - preds_q[:, :, 0]   # (K, horizon)
            sigma = iqr / (2 * 0.6745)
            n_sim = 500
            rng = np.random.default_rng(fold_idx)
            sim_samples = (
                median_pred[np.newaxis, :, :]
                + rng.standard_normal((n_sim, K, horizon)) * sigma[np.newaxis, :, :]
            )
            sim_samples = np.maximum(sim_samples, 0.0)

            scores_b = _scores_from_predictive(obs_test_N, sim_samples, fold_horizons)
            scores_b["model"] = bname
            scores_b["fold"]  = fold_idx
            all_scores.append(scores_b)

            flat_crps_b = _crps_series(
                obs_test_N.ravel(),
                sim_samples.reshape(n_sim, -1),
            )
            dm_crps_series.setdefault(bname, []).extend(flat_crps_b.tolist())

    # --- Aggregate scores ---------------------------------------------------
    print("[4/4] Aggregating scores and running DM tests ...")

    if not all_scores:
        print("      No scores collected — exiting.")
        return

    scores_df = pd.concat(all_scores, ignore_index=True)
    agg = (
        scores_df
        .groupby(["model", "horizon", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "value_mean", "std": "value_std", "count": "n_folds"})
    )

    scores_path = os.path.join(args.output_dir, "scores.csv")
    agg.to_csv(scores_path, index=False)
    print(f"      Scores saved to {scores_path}")
    print(agg.pivot_table(index="model", columns=["horizon", "metric"],
                          values="value_mean").to_string())

    # --- DM test table ------------------------------------------------------
    if "FullModel" in dm_crps_series and len(dm_crps_series) > 1:
        # Equalise lengths (trim to the shortest series)
        min_len = min(len(v) for v in dm_crps_series.values())
        dm_losses = {k: np.array(v[:min_len]) for k, v in dm_crps_series.items()}
        dm_df = dm_table(dm_losses, reference_model="FullModel", h=1)
        dm_path = os.path.join(args.output_dir, "dm_test.csv")
        dm_df.to_csv(dm_path)
        print(f"\nDM test results (ref = FullModel):\n{dm_df.to_string()}")
        print(f"      Saved to {dm_path}")

    # --- Calibration report -------------------------------------------------
    # Use the last fold only for calibration diagnostics
    print("\nCalibration report (last fold) ...")
    try:
        from cyberthreats.evaluation.calibration import calibration_report as cal_report
        if idata is not None:
            train_T_last = fold_starts[-1] if fold_starts else T - 12
            horizon_last = min(12, T - train_T_last)
            data_dict = {
                "N": N_kt, "D": D_st,
                "B": panel.get("B", np.zeros_like(N_kt)),
                "E": panel.get("E", np.zeros_like(N_kt)),
                "M_skt": M_skt, "e_t": e_t,
                "x_s": x_s, "Lambda_L": Lambda_L,
            }
            full_pred = _get_full_model_predictive(
                idata, data_dict, config, train_T_last, horizon_last
            )
            obs_dict  = {"N": N_kt[:, train_T_last : train_T_last + horizon_last]}
            pred_dict = {"N": full_pred}
            cal_df = cal_report(obs_dict, pred_dict, alpha_levels=[0.1, 0.5])
            cal_path = os.path.join(args.output_dir, "calibration.csv")
            cal_df.to_csv(cal_path, index=False)
            print(f"      Calibration report saved to {cal_path}")
            print(cal_df.to_string(index=False))
    except Exception as exc:
        warnings.warn(f"Calibration report failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
