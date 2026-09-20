#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Forecast evaluation following the manuscript's §4.1 protocol.

Design (verbatim commitments from the paper):
  - Rolling-origin over January 2010 .. December 2024 (T = 180 months);
    the final two years (2023-01 .. 2024-12) are the held-out test period;
    horizons h in {1, 3, 6, 12} months.
  - Every forecast of month t is sequential and filtered: it conditions on
    the entire history through t-h and on nothing later, starting from the
    first month of the panel with a uniform prior over regimes; for longer
    horizons the states are propagated h steps forward without intermediate
    updates.  The HEADLINE numbers hold the static parameters fixed (the
    full-panel posterior) and let the states update sequentially; the strict
    variant that re-estimates all parameters on data through 2022 is a
    robustness check (train.py --train-end).
  - Headline comparisons use the ACTIVE-TOPIC SET: threat topics with at
    least 50 CVE assignments and at least 24 non-zero months in the initial
    training window.
  - Baselines: RF, ARIMA, ETS, Naive (seasonal random walk), BSTS-U, each
    refit at every origin on the expanding window; point baselines get
    predictive distributions via residual bootstrap.
  - Metrics: CRPS and LogS (negative binomial moment-matched to the
    winsorized predictive ensemble), MAE and RMSE of the predictive median,
    empirical coverage of the central 50%/90% intervals, PIT uniformity
    (KS p-value), Diebold-Mariano tests versus Full.

Outputs (--output-dir):
  scores.csv          model x horizon x metric (Tables 2-3 source)
  table2.csv          h=1 comparison, wide (Table 2)
  table3_crps.csv     CRPS by horizon + improvement over best baseline (Table 3)
  dm_test.csv         DM tests vs Full per horizon
  calibration.csv     coverage / interval score / PIT KS per channel (h=1)
  channel_scores.csv  CRPS (+ LogS for counts) per observation channel (h=1)
  posterior_predictive_checks.csv  in-sample channel adequacy checks

Usage:
    python scripts/evaluate.py --idata results/idata.nc \\
        --data-dir data/processed/ --output-dir results/evaluation/
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings

import yaml
import numpy as np
import pandas as pd

# Force UTF-8 stdout/stderr: Windows' default console codepage cannot...
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.data import pipeline
from cassandra_threatcast.evaluation.scoring import (
    crps_ensemble, log_score_ensemble, mae, rmse,
)
from cassandra_threatcast.evaluation.dm_test import dm_table
from cassandra_threatcast.evaluation.calibration import (
    calibration_report, pit_values, pit_ks_test,
)
from cassandra_threatcast.evaluation.baselines import (
    RandomForestBaseline, ArimaBaseline, EtsBaseline, NaiveBaseline,
    BstsUnivariate,
)
from cassandra_threatcast.evaluation.sequential import (
    sequential_h_step_predict, sequential_one_step_predict, soft_clip,
)
from cassandra_threatcast.evaluation.posterior_predictive import (
    posterior_predictive_checks,
)

# Active-topic set thresholds (paper §4.1, Evaluation set).
MIN_CVE_ASSIGNMENTS = 50
MIN_NONZERO_MONTHS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper §4.1 evaluation: sequential filtered forecasts vs baselines."
    )
    parser.add_argument("--idata", default="results/idata.nc",
                        help="Path to the trained InferenceData (.nc; .pkl preferred if present).")
    parser.add_argument("--data-dir", default="data/processed/")
    parser.add_argument("--output-dir", default="results/evaluation/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-boot", type=int, default=500,
                        help="Residual-bootstrap samples per point baseline.")
    parser.add_argument("--max-draws", type=int, default=0,
                        help="Thin the posterior to at most this many draws (0 = all).")
    parser.add_argument("--max-origins", type=int, default=0,
                        help="Cap the number of baseline refit origins (0 = all; for smoke tests).")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Score only the full model (for quick checks).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _pooled_scores(obs: np.ndarray, samples: np.ndarray) -> dict:
    """Metrics pooled over topic-months.

    obs     : (N,) observations
    samples : (N, n_draws) predictive ensemble per observation
    """
    crps_series = crps_ensemble(obs, samples)                    # (N,)
    logs_series = log_score_ensemble(obs, samples)               # (N,)
    median = np.median(samples, axis=1)
    q05, q25, q75, q95 = (np.quantile(samples, q, axis=1)
                          for q in (0.05, 0.25, 0.75, 0.95))
    pit = pit_values(obs, samples)
    _, ks_pval = pit_ks_test(pit)
    return {
        "CRPS": float(np.mean(crps_series)),
        "LogS": float(np.mean(logs_series)),
        "MAE": mae(obs, median),
        "RMSE": rmse(obs, median),
        "cov50": float(np.mean((q25 <= obs) & (obs <= q75))),
        "cov90": float(np.mean((q05 <= obs) & (obs <= q95))),
        "pit_ks_pval": float(ks_pval),
        "_crps_series": crps_series,
    }


def _in_sample_ppc(post: dict, data: dict, seed: int, n_rep: int = 500) -> pd.DataFrame:
    "Posterior predictive checks for all four observation channels (§4.1)."
    rng = np.random.default_rng(seed)
    eta = np.asarray(post["eta_t"], dtype=float)     # (n, T, K)
    zeta = np.asarray(post["zeta_t"], dtype=float)   # (n, T, K)
    n = eta.shape[0]
    keep = np.linspace(0, n - 1, min(n_rep, n)).astype(int)
    eta, zeta = eta[keep], zeta[keep]
    psi = np.asarray(post["psi_k"], dtype=float)[keep]        # (m, K)
    alpha = np.asarray(post["alpha_k"], dtype=float)[keep]
    beta = np.asarray(post["beta_k"], dtype=float)[keep]
    varsig = np.asarray(post["varsigma_k"], dtype=float)[keep]
    kappa = np.asarray(post["kappa_k"], dtype=float)[keep]
    rho = np.asarray(post["rho_s"], dtype=float)[keep]        # (m, S)
    pi_s = np.asarray(post["pi_s"], dtype=float)[keep]
    m = eta.shape[0]

    e_t = np.asarray(data["e_t"], dtype=float)                # (T,)
    M = np.asarray(data["M_skt"], dtype=float)                # (S, K, T)

    # N ~ NegBin(e_t lambda, psi)  (Eq. 7)
    mu_N = np.exp(soft_clip(eta + e_t[None, :, None]))        # (m, T, K)
    p_nb = psi[:, None, :] / (psi[:, None, :] + mu_N)
    N_rep = rng.negative_binomial(np.broadcast_to(psi[:, None, :], mu_N.shape), p_nb)

    # logit E ~ N(alpha + beta log lambda, varsigma^2)  (Eq. 8)
    logit_E = (alpha[:, None, :] + beta[:, None, :] * eta
               + varsig[:, None, :] * rng.standard_normal(eta.shape))
    E_rep = 1.0 / (1.0 + np.exp(-np.clip(logit_E, -30.0, 30.0)))

    # D ~ Poisson(rho_s sum_k M lambda + pi_s)  (Eq. 9)
    exp_lam = np.exp(soft_clip(eta))                          # (m, T, K)
    weighted = np.einsum("skt,mtk->mts", M, exp_lam)          # (m, T, S)
    rate = np.clip(rho[:, None, :] * weighted + pi_s[:, None, :], 1e-8, None)
    D_rep = rng.poisson(rate)

    # B ~ LogNormal(log sigma, kappa^2)  (Eq. 10)
    B_rep = np.exp(zeta + kappa[:, None, :] * rng.standard_normal(zeta.shape))

    obs_dict = {
        "N": np.asarray(data["N"], dtype=float),              # (K, T)
        "E": np.asarray(data["E"], dtype=float),
        "B": np.asarray(data["B"], dtype=float),
        "D": np.asarray(data["D"], dtype=float),              # (S, T)
    }
    pred_dict = {
        "N": np.moveaxis(N_rep, 1, 2),                        # (m, K, T)
        "E": np.moveaxis(E_rep, 1, 2),
        "B": np.moveaxis(B_rep, 1, 2),
        "D": np.moveaxis(D_rep, 1, 2),                        # (m, S, T)
    }
    return posterior_predictive_checks(obs_dict, pred_dict)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    eval_cfg = config.get("evaluation", {})
    horizons = [int(h) for h in eval_cfg.get("horizons", [1, 3, 6, 12])]
    test_years = int(eval_cfg.get("test_years", 2))

    # ------------------------------------------------------------------ data
    print("[1/5] Loading panel ...")
    panel = pipeline.load_panel(args.data_dir)
    N_kt = panel["N"].astype(float)   # (K, T)
    D_st = panel["D"].astype(float)   # (S, T)
    K, T = N_kt.shape
    S = int(config["model"]["S"])

    def _load(fname, fallback):
        p = os.path.join(args.data_dir, fname)
        return np.load(p) if os.path.exists(p) else fallback

    M_skt = _load("M_skt.npy", np.ones((S, K, T)) / S)
    e_t = _load("e_t.npy", np.zeros(T))

    dates = panel.get("dates", [str(i) for i in range(T)])
    test_T = test_years * 12
    t0 = T - test_T                    # first test-month index
    test_idx = np.arange(t0, T)
    print(f"      T={T}  test window: {dates[t0]} .. {dates[T-1]}  "
          f"({test_T} months)  horizons={horizons}")

    # Active-topic set (§4.1): computed on the initial training window.
    train_N = N_kt[:, :t0]
    active = ((train_N.sum(axis=1) >= MIN_CVE_ASSIGNMENTS)
              & ((train_N > 0).sum(axis=1) >= MIN_NONZERO_MONTHS))
    act = np.where(active)[0]
    print(f"      Active-topic set: {len(act)}/{K} topics "
          f"(>= {MIN_CVE_ASSIGNMENTS} CVEs and >= {MIN_NONZERO_MONTHS} "
          f"non-zero months in the initial training window)")

    # ------------------------------------------------------------- posterior
    print("[2/5] Loading posterior ...")
    pkl_path = os.path.splitext(args.idata)[0] + ".pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            idata = pickle.load(fh)
    else:
        import arviz as az
        idata = az.from_netcdf(args.idata)
    post = {k: np.asarray(v) for k, v in idata.posterior.items()}
    post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items() if v.size > 0}
    n_draws = post["Pi"].shape[0]
    if args.max_draws and n_draws > args.max_draws:
        keep = np.linspace(0, n_draws - 1, args.max_draws).astype(int)
        post = {k: v[keep] for k, v in post.items()}
        n_draws = args.max_draws
    print(f"      {n_draws} posterior draws.")

    # -------------------------------------------- full model: filtered h-step
    print("[3/5] Full model: sequential filtered h-step predictions ...")
    # For each h: N_pred[:, :, t] is the h-step-ahead predictive of month t
    # issued from the filtered belief at t-h, states propagated h steps with
    # no intermediate updates (paper §4.1).
    full_pred: dict[int, np.ndarray] = {}
    for h in horizons:
        print(f"      h={h} ...", flush=True)
        full_pred[h] = sequential_h_step_predict(
            post, e_t, M_skt, h, enhanced=False, seed=args.seed)

    # samples_by[(model, h)] = (n_obs, n_draws) aligned with obs_by[h]
    obs_by: dict[int, np.ndarray] = {}
    samples_by: dict[tuple[str, int], np.ndarray] = {}
    obs_override: dict[tuple[str, int], np.ndarray] = {}  # models missing months
    for h in horizons:
        obs_by[h] = N_kt[np.ix_(act, test_idx)].ravel()       # topic-major
        pred = full_pred[h][:, :, test_idx][:, act, :]        # (n, K_act, Tw)
        samples_by[("Full", h)] = pred.reshape(pred.shape[0], -1).T

    # ------------------------------------------------------------- baselines
    if not args.skip_baselines:
        print("[4/5] Baselines: per-origin refits on the expanding window ...")
        # Origin o = last training index (data through month o inclusive);
        # its h-step target is month o + h.
        origins = [o for o in range(t0 - max(horizons), T - 1)
                   if any(t0 <= o + h < T for h in horizons)]
        if args.max_origins:
            origins = origins[: args.max_origins]
        bl_names = ["RF", "ARIMA", "ETS", "Naive", "BSTS-U"]
        # store[(name, h)][target t] = (K_act, n_boot)
        store: dict[tuple[str, int], dict[int, np.ndarray]] = {
            (b, h): {} for b in bl_names for h in horizons}

        N_act = N_kt[act]                                     # (K_act, T)
        t_start = time.time()
        for oi, o in enumerate(origins):
            needed = [h for h in horizons if t0 <= o + h < T]
            hmax = max(needed)
            train = N_act[:, : o + 1]
            print(f"      origin {oi + 1}/{len(origins)} "
                  f"({dates[o]}, horizons {needed}) "
                  f"[{(time.time() - t_start) / 60:.1f} min elapsed]", flush=True)

            fits: dict[str, np.ndarray] = {}
            try:
                rf = RandomForestBaseline().fit(train)
                fits["RF"] = rf.predict_samples(train, hmax, args.n_boot,
                                                seed=args.seed + 10 * o)
            except Exception as exc:
                warnings.warn(f"RF failed at origin {o}: {exc}")
            try:
                arima = ArimaBaseline().fit(train)
                fits["ARIMA"] = arima.predict_samples(hmax, args.n_boot,
                                                      seed=args.seed + 10 * o + 1)
            except Exception as exc:
                warnings.warn(f"ARIMA failed at origin {o}: {exc}")
            try:
                ets = EtsBaseline().fit(train)
                fits["ETS"] = ets.predict_samples(hmax, args.n_boot,
                                                  seed=args.seed + 10 * o + 2)
            except Exception as exc:
                warnings.warn(f"ETS failed at origin {o}: {exc}")
            try:
                naive = NaiveBaseline().fit(train)
                fits["Naive"] = naive.predict_samples(hmax, args.n_boot,
                                                      seed=args.seed + 10 * o + 3)
            except Exception as exc:
                warnings.warn(f"Naive failed at origin {o}: {exc}")
            try:
                bsts = np.zeros((len(act), args.n_boot, hmax))
                for j in range(len(act)):
                    model = BstsUnivariate(num_warmup=300, num_samples=400).fit(
                        train[j], seed=args.seed + 100 * o + j)
                    draws = model.predict_samples(hmax, seed=args.seed + 100 * o + j)
                    pick = np.linspace(0, draws.shape[0] - 1, args.n_boot).astype(int)
                    bsts[j] = draws[pick]
                fits["BSTS-U"] = bsts
            except Exception as exc:
                warnings.warn(f"BSTS-U failed at origin {o}: {exc}")

            for name, samp in fits.items():                   # (K_act, n_boot, hmax)
                for h in needed:
                    store[(name, h)][o + h] = samp[:, :, h - 1]

        for name in bl_names:
            for h in horizons:
                per_t = store[(name, h)]
                if len(per_t) != len(test_idx):
                    missing = [t for t in test_idx if t not in per_t]
                    if missing:
                        warnings.warn(f"{name} h={h}: {len(missing)} test months "
                                      f"missing; excluded from scoring.")
                if not per_t:
                    continue
                ts = sorted(per_t)
                samp = np.stack([per_t[t] for t in ts], axis=1)   # (K_act, Tn, n_boot)
                samples_by[(name, h)] = samp.reshape(-1, samp.shape[-1])
                if len(ts) != len(test_idx):
                    # Align the obs vector for this model separately.
                    obs_override[(name, h)] = N_kt[np.ix_(act, ts)].ravel()
    else:
        print("[4/5] Baselines skipped (--skip-baselines).")

    # --------------------------------------------------------------- scoring
    print("[5/5] Scoring, DM tests, calibration, PPC ...")
    score_rows = []
    dm_frames = []
    crps_series: dict[int, dict[str, np.ndarray]] = {h: {} for h in horizons}

    for (key_model, key_h), samples in list(samples_by.items()):
        obs = obs_override.get((key_model, key_h), obs_by[key_h])
        if samples.shape[0] != obs.shape[0]:
            warnings.warn(f"{key_model} h={key_h}: obs/sample mismatch; skipped.")
            continue
        s = _pooled_scores(obs, samples)
        crps_series[key_h][key_model] = s.pop("_crps_series")
        for metric, value in s.items():
            score_rows.append({"model": key_model, "horizon": key_h,
                               "metric": metric, "value": value})

    scores_df = pd.DataFrame(score_rows)
    scores_path = os.path.join(args.output_dir, "scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"      Scores -> {scores_path}")

    # Table 2: h=1 comparison, wide.
    t2 = scores_df[scores_df.horizon == 1].pivot_table(
        index="model", columns="metric", values="value")
    cols = [c for c in ["CRPS", "LogS", "MAE", "RMSE", "cov50", "cov90",
                        "pit_ks_pval"] if c in t2.columns]
    t2 = t2[cols]
    t2.to_csv(os.path.join(args.output_dir, "table2.csv"))
    print("\nTable 2 (h=1, active-topic set, test origins):")
    print(t2.round(3).to_string())

    # Table 3: CRPS by horizon + improvement of Full over the best baseline.
    t3 = scores_df[scores_df.metric == "CRPS"].pivot_table(
        index="model", columns="horizon", values="value")
    if "Full" in t3.index and len(t3.index) > 1:
        best_baseline = t3.drop(index="Full").min(axis=0)
        improvement = 100.0 * (best_baseline - t3.loc["Full"]) / best_baseline
        t3.loc["Improvement over best baseline (%)"] = improvement
    t3.to_csv(os.path.join(args.output_dir, "table3_crps.csv"))
    print("\nTable 3 (CRPS by horizon):")
    print(t3.round(3).to_string())

    # DM tests versus Full, per horizon.
    for h in horizons:
        series = crps_series[h]
        if "Full" not in series or len(series) < 2:
            continue
        min_len = min(len(v) for v in series.values())
        losses = {k: np.asarray(v[:min_len]) for k, v in series.items()}
        dm = dm_table(losses, reference_model="Full", h=h).reset_index()
        dm.insert(1, "horizon", h)
        dm_frames.append(dm)
    if dm_frames:
        dm_df = pd.concat(dm_frames, ignore_index=True)
        dm_path = os.path.join(args.output_dir, "dm_test.csv")
        dm_df.to_csv(dm_path, index=False)
        print(f"\nDM tests (ref = Full) -> {dm_path}")
        print(dm_df.to_string(index=False))

    # Calibration report + channel scores at h=1 (N and D channels).
    try:
        out1 = sequential_one_step_predict(
            post, e_t, M_skt, tail_months=0, enhanced=False,
            student_t_df=4.0, seed=args.seed)
        obs_dict = {"N": N_kt[np.ix_(act, test_idx)],
                    "D": D_st[:, test_idx]}
        pred_dict = {"N": out1["N_pred"][:, :, test_idx][:, act, :],
                     "D": out1["D_pred"][:, :, test_idx]}
        cal_df = calibration_report(obs_dict, pred_dict, alpha_levels=[0.1, 0.5])
        cal_path = os.path.join(args.output_dir, "calibration.csv")
        cal_df.to_csv(cal_path, index=False)
        print(f"\nCalibration (h=1, test window) -> {cal_path}")
        print(cal_df.to_string(index=False))

        channel_rows = []
        for channel, observed in obs_dict.items():
            samp = pred_dict[channel]
            obs_flat = observed.ravel()
            samp_flat = samp.reshape(samp.shape[0], -1).T
            channel_rows.append({"channel": channel, "metric": "CRPS",
                                 "value": float(np.mean(
                                     crps_ensemble(obs_flat, samp_flat)))})
            channel_rows.append({"channel": channel, "metric": "LogS",
                                 "value": float(np.mean(
                                     log_score_ensemble(obs_flat, samp_flat)))})
        channel_path = os.path.join(args.output_dir, "channel_scores.csv")
        pd.DataFrame(channel_rows).to_csv(channel_path, index=False)
        print(f"      Channel scores -> {channel_path}")
    except Exception as exc:
        warnings.warn(f"Calibration/channel scoring failed: {exc}")

    # Posterior predictive checks (in-sample channel adequacy, §4.1).
    try:
        data_full = {
            "N": N_kt, "D": D_st,
            "E": panel.get("E", np.full_like(N_kt, np.nan)).astype(float),
            "B": panel.get("B", np.full_like(N_kt, np.nan)).astype(float),
            "M_skt": M_skt, "e_t": e_t,
        }
        ppc_df = _in_sample_ppc(post, data_full, seed=args.seed)
        ppc_path = os.path.join(args.output_dir, "posterior_predictive_checks.csv")
        ppc_df.to_csv(ppc_path, index=False)
        print(f"      Posterior predictive checks -> {ppc_path}")
        print(ppc_df.to_string(index=False))
    except Exception as exc:
        warnings.warn(f"Posterior predictive checks failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
