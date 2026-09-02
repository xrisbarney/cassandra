#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

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
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import numpy as np
import pandas as pd

# Force UTF-8 stdout/stderr: Windows' default console codepage cannot encode
# many Unicode characters, which raises UnicodeEncodeError and kills the
# process -- especially when output is redirected to a log file.
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.data import pipeline
from cassandra_threatcast.evaluation.scoring import crps_ensemble, log_score_ensemble, mae, rmse
from cassandra_threatcast.evaluation.dm_test import dm_table
from cassandra_threatcast.evaluation.baselines import run_all_baselines


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
    parser.add_argument(
        "--reuse-posterior", action="store_true",
        help="Diagnostic shortcut only; paper Step 33 refits at every origin by default.",
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
    """Compute per-horizon CRPS/LogS/MAE/RMSE from posterior predictive samples."""
    records = []
    for h in horizons:
        h = min(h, obs.shape[-1])
        obs_h   = obs[..., :h].ravel()
        pred_h  = pred_samples[..., :h].reshape(pred_samples.shape[0], -1)
        pred_h_T = pred_h.T   # (N, n_samples)

        crps_val = float(np.mean(crps_ensemble(obs_h, pred_h_T)))
        logs_val = float(np.mean(log_score_ensemble(obs_h, pred_h_T)))
        median   = np.median(pred_h, axis=0)
        records.append({"horizon": h, "metric": "CRPS", "value": crps_val})
        records.append({"horizon": h, "metric": "LogS", "value": logs_val})
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
) -> dict:
    """
    Draw posterior predictive samples for observations in [train_T, train_T+horizon).

    Returns the complete predictive-draw dictionary for all paper channels.
    """
    from cassandra_threatcast.model import full as full_module
    from cassandra_threatcast.model import paper_exact
    from cassandra_threatcast.model.economic import DamageFunctionParams

    S = int(config["model"]["S"])
    # Flatten posterior (chain, draw, ...) -> (sample, ...)
    post = {k: np.asarray(v) for k, v in idata.posterior.items()}
    post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items()}

    # Exposure for the forecast window [train_T, train_T + horizon).
    M = np.asarray(data["M_skt"])
    if train_T + horizon <= M.shape[2]:
        M_future = M[:, :, train_T : train_T + horizon]
    else:  # hold the last observed exposure if we run past the panel
        M_future = np.repeat(M[:, :, -1:], horizon, axis=2)

    dmg = config.get("damage_params", {})
    damage_params = DamageFunctionParams(
        shape=np.full(S, float(dmg.get("shape", 2.0))),
        scale=np.full(S, float(dmg.get("scale", 1.0))),
        max_damage=np.full(S, float(dmg.get("max_damage", 0.5))),
    )
    enhanced = bool(config.get("enhanced", {}).get("enabled", False))
    student_t_df = float(config.get("enhanced", {}).get("student_t_df", 4.0))

    if "Phi_r" in post and "z_t" in post:
        out = paper_exact.predict(
            post, data, horizon,
            np.asarray(data["Lambda_L"]), np.asarray(data["x_s"]) * 1e6,
            M_future, damage_params, start_t=train_T,
        )
    else:
        out = full_module.predict(
            post, data, horizon,
            np.asarray(data["Lambda_L"]), np.asarray(data["x_s"]) * 1e6,
            M_future, damage_params,
            enhanced=enhanced, student_t_df=student_t_df,
            start_t=train_T,
        )
    return out


def _evaluate_fold(
    fold_idx: int,
    train_T: int,
    horizons: list[int],
    N_kt: np.ndarray,
    D_st: np.ndarray,
    panel: dict,
    M_skt: np.ndarray,
    e_t: np.ndarray,
    Lambda_L: np.ndarray,
    x_s: np.ndarray,
    idata,
    config: dict,
) -> tuple[list[pd.DataFrame], dict[str, list[float]]]:
    """Evaluate one rolling-origin fold (full model + all baselines).

    Every fold trains/predicts on its own [0, train_T) / [train_T,
    train_T+horizon) split and seeds its baseline-noise RNG with its own
    fold_idx, so folds share no mutable state -- safe to run in any order or
    in parallel processes with bit-identical results to the sequential loop.
    Kept as a module-level function (not a closure) so it can be pickled and
    sent to worker processes by ProcessPoolExecutor.
    """
    T, K = N_kt.shape[1], N_kt.shape[0]
    avail_h = T - train_T
    fold_horizons = [h for h in horizons if h <= avail_h]
    fold_scores: list[pd.DataFrame] = []
    fold_dm_crps: dict[str, list[float]] = {}
    if not fold_horizons:
        return fold_scores, fold_dm_crps

    horizon = max(fold_horizons)
    obs_test_N = N_kt[:, train_T : train_T + horizon]   # (K, horizon)

    print(f"  Fold {fold_idx + 1}: train_T={train_T}  horizon={horizon}", flush=True)

    # --- Full model predictive ------------------------------------------
    try:
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
            fold_idata = idata
            if fold_idata is None:
                from copy import deepcopy
                from cassandra_threatcast.inference.blocked import run_blocked_nuts_ffbs
                fold_cfg = deepcopy(config)
                eval_mcmc = fold_cfg.get("evaluation", {}).get("mcmc", {})
                fold_cfg["mcmc"] = {**fold_cfg.get("mcmc", {}), **eval_mcmc}
                fit_data = {
                    key: (value[..., :train_T] if isinstance(value, np.ndarray)
                          and value.ndim > 0 and value.shape[-1] == T else value)
                    for key, value in data_dict.items()
                }
                fold_idata = run_blocked_nuts_ffbs(fit_data, fold_cfg)
            full_out = _get_full_model_predictive(
                fold_idata, data_dict, config, train_T, horizon
            )
            full_pred = full_out["N_pred"]

            scores_full = _scores_from_predictive(obs_test_N, full_pred, fold_horizons)
            scores_full["model"] = "FullModel"
            scores_full["fold"] = fold_idx
            fold_scores.append(scores_full)

            flat_crps = _crps_series(
                obs_test_N.ravel(),
                full_pred.reshape(full_pred.shape[0], -1),
            )
            fold_dm_crps.setdefault("FullModel", []).extend(flat_crps.tolist())

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
            seed=fold_idx * 100,  # *100 margin so each baseline's internal +1/+2 offset can't collide across folds
        )
    except Exception as exc:
        warnings.warn(f"  Baselines failed at fold {fold_idx}: {exc}")
        baseline_results = {}

    for bname, bres in baseline_results.items():
        preds_q = bres["predictions"]   # (K, horizon, 3)
        median_pred = preds_q[:, :, 1]   # (K, horizon)
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
        scores_b["fold"] = fold_idx
        fold_scores.append(scores_b)

        flat_crps_b = _crps_series(
            obs_test_N.ravel(),
            sim_samples.reshape(n_sim, -1),
        )
        fold_dm_crps.setdefault(bname, []).extend(flat_crps_b.tolist())

    return fold_scores, fold_dm_crps


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
    # Prefer the pickle (written unconditionally by train.py, guaranteed
    # complete) over the .nc file, which some xarray/arviz version
    # combinations can leave empty/corrupt for a large InferenceData.
    print("[2/4] Loading InferenceData ...")
    try:
        pkl_path = os.path.splitext(args.idata)[0] + ".pkl"
        if os.path.exists(pkl_path):
            import pickle
            with open(pkl_path, "rb") as fh:
                idata = pickle.load(fh)
            print("      InferenceData loaded (pickle).")
        else:
            import arviz as az
            idata = az.from_netcdf(args.idata)
            print("      InferenceData loaded (netCDF).")
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

    # Independent refits are memory-heavy JAX jobs; run sequentially unless a
    # single pre-fitted diagnostic posterior was explicitly requested.
    n_workers = min(len(fold_starts), os.cpu_count() or 1) if args.reuse_posterior else 1
    print(f"      Using {n_workers} worker process(es) (folds are independent)")

    all_scores: list[pd.DataFrame] = []
    dm_crps_series: dict[str, list[float]] = {}  # model -> flat CRPS list

    fold_args = [
        (fold_idx, train_T, horizons, N_kt, D_st, panel, M_skt, e_t, Lambda_L, x_s,
         idata if args.reuse_posterior else None, config)
        for fold_idx, train_T in enumerate(fold_starts)
    ]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_evaluate_fold, *a): a[0] for a in fold_args}
            for future in as_completed(futures):
                fold_idx = futures[future]
                try:
                    fold_scores, fold_dm_crps = future.result()
                except Exception as exc:  # noqa: BLE001 -- one fold's crash shouldn't sink the rest
                    warnings.warn(f"  Fold {fold_idx} failed entirely: {exc}")
                    continue
                all_scores.extend(fold_scores)
                for name, vals in fold_dm_crps.items():
                    dm_crps_series.setdefault(name, []).extend(vals)
    else:
        for a in fold_args:
            fold_scores, fold_dm_crps = _evaluate_fold(*a)
            all_scores.extend(fold_scores)
            for name, vals in fold_dm_crps.items():
                dm_crps_series.setdefault(name, []).extend(vals)

    # --- Aggregate scores ---------------------------------------------------
    print("[4/4] Aggregating scores and running DM tests ...")

    if not all_scores:
        print("      No scores collected -- exiting.")
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
        from cassandra_threatcast.evaluation.calibration import calibration_report as cal_report
        if idata is not None or not args.reuse_posterior:
            train_T_last = fold_starts[-1] if fold_starts else T - 12
            horizon_last = min(12, T - train_T_last)
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
            calibration_idata = idata
            if not args.reuse_posterior:
                from copy import deepcopy
                from cassandra_threatcast.inference.blocked import run_blocked_nuts_ffbs
                calibration_cfg = deepcopy(config)
                eval_mcmc = calibration_cfg.get("evaluation", {}).get("mcmc", {})
                calibration_cfg["mcmc"] = {
                    **calibration_cfg.get("mcmc", {}), **eval_mcmc,
                }
                calibration_data = {
                    key: (value[..., :train_T_last] if isinstance(value, np.ndarray)
                          and value.ndim > 0 and value.shape[-1] == T else value)
                    for key, value in data_dict.items()
                }
                calibration_idata = run_blocked_nuts_ffbs(
                    calibration_data, calibration_cfg
                )
            full_out = _get_full_model_predictive(
                calibration_idata, data_dict, config, train_T_last, horizon_last
            )
            sl = slice(train_T_last, train_T_last + horizon_last)
            obs_dict = {
                "N": N_kt[:, sl],
                "E": panel.get("E", np.full_like(N_kt, np.nan))[:, sl],
                "B": panel.get("B", np.full_like(N_kt, np.nan))[:, sl],
                "D": D_st[:, sl],
            }
            pred_dict = {key: full_out[f"{key}_pred"] for key in obs_dict}
            if "L" in panel:
                obs_dict["L"] = panel["L"][:, sl]
                pred_dict["L"] = full_out["ell_pred"]
            cal_df = cal_report(obs_dict, pred_dict, alpha_levels=[0.1, 0.5])
            cal_path = os.path.join(args.output_dir, "calibration.csv")
            cal_df.to_csv(cal_path, index=False)
            print(f"      Calibration report saved to {cal_path}")
            print(cal_df.to_string(index=False))

            # Step 34: CRPS for every real-valued target; logarithmic score
            # only for the count channels N and D.
            channel_rows = []
            for channel, observed in obs_dict.items():
                samples = np.moveaxis(pred_dict[channel], 0, -1)
                observed_flat = observed.ravel()
                samples_flat = samples.reshape(-1, samples.shape[-1])
                valid = np.isfinite(observed_flat) & np.all(np.isfinite(samples_flat), axis=1)
                if not valid.any():
                    continue
                channel_rows.append({
                    "channel": channel,
                    "metric": "CRPS",
                    "value": float(np.mean(crps_ensemble(observed_flat[valid], samples_flat[valid]))),
                })
                if channel in {"N", "D"}:
                    channel_rows.append({
                        "channel": channel,
                        "metric": "LogS",
                        "value": float(np.mean(log_score_ensemble(observed_flat[valid], samples_flat[valid]))),
                    })
            channel_path = os.path.join(args.output_dir, "channel_scores.csv")
            pd.DataFrame(channel_rows).to_csv(channel_path, index=False)
            print(f"      All-channel scores saved to {channel_path}")
            from cassandra_threatcast.evaluation.posterior_predictive import (
                posterior_predictive_checks,
            )
            ppc_path = os.path.join(args.output_dir, "posterior_predictive_checks.csv")
            posterior_predictive_checks(obs_dict, pred_dict).to_csv(ppc_path, index=False)
            print(f"      Observation-channel checks saved to {ppc_path}")
    except Exception as exc:
        warnings.warn(f"Calibration report failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
