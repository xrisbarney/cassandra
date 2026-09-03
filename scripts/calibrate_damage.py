#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Bayesian calibration of the damage functions phi_s against the reference
event set (paper §3 and Table 7).

The paper specifies that the damage functions get weakly informative priors
and are calibrated against well-documented US-relevant events (NotPetya,
Colonial Pipeline, MOVEit, Change Healthcare), replacing scenario-style
economic modelling with Bayesian calibration to observed episodes.

Method
------
1. For each reference event (configs/reference_events.yaml), build the
   PRE-EVENT predictive of the latent states at the event month: regime via
   the Hamilton filter on data through the previous month, factor/severity
   states propagated one AR step from their t-1 posterior values.  This is
   the same conditioning the paper's Table 7 requires ("conditioned on
   pre-event information").
2. The per-draw sectoral shock load s = M_(t-1) (exp(eta) * sigma) feeds a
   3-parameter damage family (global shape, scale, max_damage broadcast to
   all sectors), Leontief propagation, and aggregation to a dollar loss.
3. The documented loss range [low, high] is treated as a 90% interval around
   its log-midpoint; NUTS samples the damage parameters under weakly
   informative priors.

Outputs
-------
results/calibration/damage_params.json    calibrated parameters (posterior
                                          mean + sd) -- consumed by
                                          scripts/forecast.py when present
results/calibration/economic_backtest.csv Table 7: per event, the pre-event
                                          90% predictive loss interval vs
                                          the documented range
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.evaluation.sequential import (
    batch_forward_filter, one_step_state_predictive, soft_clip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate damage functions against the reference event set."
    )
    parser.add_argument("--idata", default="results/idata.nc")
    parser.add_argument("--data-dir", default="data/processed/")
    parser.add_argument("--output-dir", default="results/calibration/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--events", default="configs/reference_events.yaml")
    parser.add_argument("--enhanced-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    with open(args.events) as fh:
        events_cfg = yaml.safe_load(fh)
    events = events_cfg["events"]
    cal_cfg = events_cfg.get("calibration", {})
    n_state_draws = int(cal_cfg.get("n_state_draws", 2000))
    seed = int(cal_cfg.get("seed", 7))

    # 1. Load posterior + panel geometry
    print("[1/4] Loading posterior and panel ...")
    pkl_path = os.path.splitext(args.idata)[0] + ".pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            idata = pickle.load(fh)
    else:
        import arviz as az
        idata = az.from_netcdf(args.idata)
    post = {k: np.asarray(v) for k, v in idata.posterior.items()}
    post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items()}

    n_total = post["Pi"].shape[0]
    if n_total > n_state_draws:
        keep = np.linspace(0, n_total - 1, n_state_draws).astype(int)
        post = {k: v[keep] for k, v in post.items()}
    n = post["Pi"].shape[0]

    M_skt = np.load(os.path.join(args.data_dir, "M_skt.npy"))       # (S, K, T)
    Lambda_L = np.load(os.path.join(args.data_dir, "Lambda_L.npy"))  # (S, S)
    x_s = np.load(os.path.join(args.data_dir, "x_s.npy"))            # (S,)
    with open(os.path.join(args.data_dir, "panel_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    date_strs = meta.get("dates") or meta.get("metadata", {}).get("dates", [])
    periods = pd.PeriodIndex(date_strs, freq="M")
    S = Lambda_L.shape[0]
    # BEA gross output arrives in $ millions -> dollars.
    x_s_usd = x_s * 1e6

    enhanced = args.enhanced_mode or bool(config.get("enhanced", {}).get("enabled", False))
    student_t_df = float(config.get("enhanced", {}).get("student_t_df", 4.0))

    # 2. Pre-event predictive shock loads per event
    print("[2/4] Building pre-event predictive shock loads ...")
    rng = np.random.default_rng(seed)
    Pi = np.asarray(post["Pi"], dtype=float)
    paper_exact = "Phi_r" in post and "z_t" in post
    predicted_probs = None
    if not paper_exact:
        loglik = np.asarray(post["loglik_regime_t"], dtype=float)
        _, predicted_probs = batch_forward_filter(loglik, Pi)

    def state_predictive(t_index: int) -> dict[str, np.ndarray]:
        if not paper_exact:
            return one_step_state_predictive(
                post, t_index, rng, predicted_probs=predicted_probs,
                enhanced=enhanced, student_t_df=student_t_df,
            )
        eta_draws, zeta_draws = [], []
        previous = max(t_index - 1, 0)
        for idx in range(n):
            z_prev = int(post["z_t"][idx, previous])
            z = int(rng.choice(Pi.shape[-1], p=Pi[idx, z_prev]))
            f = post["Phi_r"][idx, z] @ post["f_t"][idx, previous]
            f += post["Q_r"][idx, z] * rng.standard_normal(post["f_init"].shape[-1])
            h = post["A_sigma_r"][idx, z] @ post["h_t"][idx, previous]
            h += post["Q_sigma_r"][idx, z] * rng.standard_normal(post["h_init"].shape[-1])
            excitation = 0.0
            if "hawkes_alpha" in post:
                excitation = np.log1p(
                    post["hawkes_decay"][idx]
                    * (post["hawkes_alpha"][idx]
                       @ np.exp(np.clip(post["eta_t"][idx, previous], -20, 20)))
                )
            eta_draws.append(
                post["mu_r"][idx, z] + post["Gamma"][idx] @ f + excitation
                + post["tau_k"][idx] * rng.standard_normal(post["tau_k"].shape[-1])
            )
            zeta_draws.append(
                post["nu_r"][idx, z] + post["Psi"][idx] @ h
                + post["omega_k"][idx] * rng.standard_normal(post["omega_k"].shape[-1])
            )
        return {"eta": np.asarray(eta_draws), "zeta": np.asarray(zeta_draws)}

    shock_loads = []       # list of (n, S)
    event_rows = []
    for ev in events:
        t_e = int(periods.get_loc(pd.Period(ev["month"], freq="M")))
        states = state_predictive(t_e)
        exp_lam = np.exp(soft_clip(states["eta"]))       # (n, K)
        sigma = np.exp(soft_clip(states["zeta"]))        # (n, K)
        M_prev = M_skt[:, :, max(t_e - 1, 0)]            # (S, K), pre-event
        s_load = np.einsum("sk,nk->ns", M_prev, exp_lam * sigma)   # (n, S)
        shock_loads.append(s_load)
        event_rows.append({"name": ev["name"], "month": ev["month"], "t": t_e,
                           "low": float(ev["low_usd"]), "high": float(ev["high_usd"])})
        print(f"      {ev['name']:28s} t={t_e}  median shock load "
              f"{np.median(s_load.sum(axis=1)):.3f}")

    shock_arr = np.stack(shock_loads)                    # (E, n, S)
    log_mid = np.log(np.array([np.sqrt(r["low"] * r["high"]) for r in event_rows]))
    log_tau = np.array([(np.log(r["high"]) - np.log(r["low"])) / 3.29
                        for r in event_rows])            # 90% interval width -> sd

    # Aggregate annual anchor: typical-month shock loads over recent history.
    anchor = events_cfg.get("aggregate_anchor", {})
    anchor_lo = float(anchor.get("low_usd_per_year", 3.0e10))
    anchor_hi = float(anchor.get("high_usd_per_year", 2.0e11))
    n_typ = int(anchor.get("n_typical_months", 12))
    T = len(periods)
    typ_months = np.linspace(T - 24, T - 1, n_typ).astype(int)
    typ_loads = []
    for t_m in typ_months:
        st = state_predictive(int(t_m))
        lam_sig = np.exp(soft_clip(st["eta"])) * np.exp(soft_clip(st["zeta"]))
        typ_loads.append(np.einsum("sk,nk->ns", M_skt[:, :, max(t_m - 1, 0)], lam_sig))
    typ_arr = np.stack(typ_loads)                        # (M, n, S)
    log_anchor_mid = float(np.log(np.sqrt(anchor_lo * anchor_hi)))
    log_anchor_tau = float((np.log(anchor_hi) - np.log(anchor_lo)) / 3.29)
    print(f"      Aggregate anchor: annual US losses in "
          f"[${anchor_lo/1e9:.0f}B, ${anchor_hi/1e9:.0f}B] "
          f"({n_typ} typical months sampled)")

    # 3. NUTS over the 3 damage parameters
    print("[3/4] Sampling damage parameters (NUTS) ...")
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    shock_j = jnp.asarray(shock_arr)                     # (E, n, S)
    typ_j = jnp.asarray(typ_arr)                         # (M, n, S)
    Lam_j = jnp.asarray(Lambda_L)
    x_j = jnp.asarray(x_s_usd)
    # Per-sector operating points.  Shock loads are wildly heterogeneous
    sector_ref = np.clip(np.median(shock_arr, axis=(0, 1)), 1e-6, None)   # (S,)
    sector_ref_j = jnp.asarray(sector_ref)

    def damage_model():
        # Same functional form as model.economic.damage_function (zero-
        steepness = numpyro.sample("steepness", dist.LogNormal(jnp.log(2.0), 0.5))
        c_scale = numpyro.sample("c_scale", dist.LogNormal(0.0, 0.3))
        max_dmg = numpyro.sample("max_damage", dist.Beta(2.0, 18.0))
        scale_s = numpyro.deterministic("scale_s", c_scale * sector_ref_j)      # (S,)
        shape_s = numpyro.deterministic("shape_s", steepness / sector_ref_j)    # (S,)

        s0 = jax.nn.sigmoid((0.0 - scale_s) * shape_s)                          # (S,)
        g = max_dmg * (jax.nn.sigmoid((shock_j - scale_s) * shape_s) - s0) / jnp.clip(1.0 - s0, 1e-8)
        d = g * x_j                                       # (E, n, S) direct loss
        ell = jnp.einsum("st,ent->ens", Lam_j, d)         # Leontief propagation
        agg = ell.sum(axis=-1)                            # (E, n)
        mean_log_loss = jnp.log(jnp.clip(agg, 1e3)).mean(axis=1)   # (E,)
        numpyro.sample("obs", dist.Normal(mean_log_loss, jnp.asarray(log_tau)),
                       obs=jnp.asarray(log_mid))

        # Aggregate annual anchor: 12 x typical-month expected loss.
        g_t = max_dmg * (jax.nn.sigmoid((typ_j - scale_s) * shape_s) - s0) / jnp.clip(1.0 - s0, 1e-8)
        ell_t = jnp.einsum("st,mnt->mns", Lam_j, g_t * x_j)
        annual = 12.0 * ell_t.sum(axis=-1).mean()
        numpyro.sample("obs_anchor",
                       dist.Normal(jnp.log(jnp.clip(annual, 1e3)), log_anchor_tau),
                       obs=log_anchor_mid)

    mcmc = MCMC(NUTS(damage_model),
                num_warmup=int(cal_cfg.get("num_warmup", 500)),
                num_samples=int(cal_cfg.get("num_samples", 1000)),
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    theta = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    summary = {k: {"mean": float(v.mean()), "sd": float(v.std())}
               for k, v in theta.items()}
    print("      Posterior (global parameters):")
    for k, v in summary.items():
        print(f"        {k:12s} mean={v['mean']:.4g}  sd={v['sd']:.3g}")

    # Per-sector arrays at the posterior mean of the global parameters.
    scale_s_mean = (summary["c_scale"]["mean"] * sector_ref)            # (S,)
    shape_s_mean = (summary["steepness"]["mean"] / sector_ref)          # (S,)
    max_dmg_mean = summary["max_damage"]["mean"]

    params_out = {
        "shape": shape_s_mean.tolist(),
        "scale": scale_s_mean.tolist(),
        "max_damage": [max_dmg_mean] * S,
        "globals": {k: v for k, v in summary.items()},
        "sector_ref": sector_ref.tolist(),
        "events_file": args.events,
        "note": "Calibrated by scripts/calibrate_damage.py against the "
                "reference event set; consumed by forecast.py when present. "
                "shape/scale are per-sector (sector-relative operating "
                "points); see paper Table 7 machinery.",
    }
    params_path = os.path.join(args.output_dir, "damage_params.json")
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(params_out, fh, indent=2)
    print(f"      Calibrated params -> {params_path}")

    # Retain the complete calibration posterior so forecast Step 28 propagates
    damage_draws_path = os.path.join(args.output_dir, "damage_posterior.npz")
    np.savez_compressed(
        damage_draws_path,
        shape=theta["steepness"][:, None] / sector_ref[None, :],
        scale=theta["c_scale"][:, None] * sector_ref[None, :],
        max_damage=np.repeat(theta["max_damage"][:, None], S, axis=1),
    )
    print(f"      Damage posterior draws -> {damage_draws_path}")

    # 4. Table 7: economic-layer backtest, integrating over theta draws
    print("[4/4] Economic-layer backtest (Table 7) ...")
    n_theta = len(theta["steepness"])
    thin = np.linspace(0, n_theta - 1, min(200, n_theta)).astype(int)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    rows = []
    for e_idx, r in enumerate(event_rows):
        s_load = shock_arr[e_idx]                         # (n, S)
        agg_draws = []
        for j in thin:
            sh = theta["steepness"][j] / sector_ref       # (S,)
            sc = theta["c_scale"][j] * sector_ref         # (S,)
            md = theta["max_damage"][j]
            s0 = _sigmoid((0.0 - sc) * sh)                # (S,)
            g = md * (_sigmoid((s_load - sc) * sh) - s0) / np.clip(1.0 - s0, 1e-8, None)
            ell = (g * x_s_usd) @ Lambda_L.T              # (n, S)
            agg_draws.append(ell.sum(axis=1))
        agg = np.concatenate(agg_draws)                   # (n * n_thin,)
        q05, q50, q95 = np.quantile(agg, [0.05, 0.50, 0.95])
        covered = bool(q05 <= r["high"] and q95 >= r["low"])  # intervals overlap
        width_ratio = float((q95 - q05) / max(r["high"] - r["low"], 1.0))
        rows.append({
            "event": r["name"], "month": r["month"],
            "documented_low_usd": r["low"], "documented_high_usd": r["high"],
            "predictive_q05_usd": float(q05), "predictive_q50_usd": float(q50),
            "predictive_q95_usd": float(q95),
            "covered": covered, "width_ratio": width_ratio,
        })
        print(f"      {r['name']:28s} pred 90% [{q05/1e9:8.2f}B, {q95/1e9:8.2f}B]  "
              f"documented [{r['low']/1e9:.2f}B, {r['high']/1e9:.2f}B]  covered={covered}")

    bt_df = pd.DataFrame(rows)
    bt_path = os.path.join(args.output_dir, "economic_backtest.csv")
    bt_df.to_csv(bt_path, index=False)
    print(f"      Table 7 -> {bt_path}")
    print("\nDone.  Re-run scripts/forecast.py to produce forecasts under the "
          "calibrated damage functions.")


if __name__ == "__main__":
    main()
