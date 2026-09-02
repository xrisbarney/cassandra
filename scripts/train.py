#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Infer the full paper model via alternating NUTS and FFBS.

Usage:
    python scripts/train.py \\
        --config   configs/default.yaml \\
        --data-dir data/processed/ \\
        --output-dir results/

The script:
  1. Loads the processed panel and derived features from --data-dir.
  2. Assembles the data dict expected by `model.paper_exact.paper_model`.
  3. Alternates continuous NUTS blocks and discrete FFBS path draws.
  4. Saves the resulting ArviZ InferenceData to <output_dir>/idata.nc.
  5. Prints a posterior summary for key parameters.
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np

# Force UTF-8 stdout/stderr: Windows' default console codepage cannot encode
# many Unicode characters, which raises UnicodeEncodeError and kills the
# process -- especially when output is redirected to a log file.
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Must run before the first jax/numpyro import (including transitively, via
# the cassandra_threatcast imports below) -- JAX locks in its device count
# the moment its backend initializes. Without this, num_chains > 1 with
# chain_method="parallel" silently falls back to running chains SEQUENTIALLY
# on this machine's single visible CPU device (confirmed: NumPyro just prints
# a UserWarning and eats the 4x-plus slowdown). This creates `cpu_count()`
# virtual CPU devices so multiple chains genuinely run in parallel, which
# also means R-hat/ESS convergence diagnostics become meaningful (they are
# NaN with a single chain).
import numpyro
numpyro.set_host_device_count(os.cpu_count() or 1)

from cassandra_threatcast.data import pipeline
from cassandra_threatcast.inference.blocked import (
    run_blocked_nuts_ffbs,
    run_blocked_vi_ffbs,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer the paper model via blocked NUTS/FFBS."
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="YAML config file.  Default: configs/default.yaml",
    )
    parser.add_argument(
        "--data-dir", default="data/processed/",
        help="Directory with processed panel and feature arrays.  Default: data/processed/",
    )
    parser.add_argument(
        "--output-dir", default="results/",
        help="Directory for model output.  Default: results/",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=None,
        help="Override num_warmup from config.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Override num_samples from config.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="JAX random seed.  Default: 42",
    )
    parser.add_argument("--gibbs-blocks", type=int, default=None)
    parser.add_argument("--gibbs-warmup-blocks", type=int, default=None)
    parser.add_argument("--block-warmup", type=int, default=None)
    parser.add_argument(
        "--train-end", type=int, default=None,
        help="Last time index (exclusive) for training.  Default: full series.",
    )
    parser.add_argument(
        "--method", choices=["blocked", "vi"], default="blocked",
        help="Inference method: paper-exact blocked NUTS/FFBS (default), or scalable VI.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_features(data_dir: str, S: int) -> dict:
    """Load derived feature arrays, falling back to sensible defaults."""
    def _try_load(fname: str, fallback: np.ndarray) -> np.ndarray:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            return np.load(fpath)
        print(f"  Warning: {fname} not found, using fallback shape={fallback.shape}")
        return fallback

    # We don't know K/T yet when this is called; use panel to infer dims
    return None   # resolved after panel load


def assemble_data(panel: dict, data_dir: str, config: dict) -> dict:
    """Build the data dict expected by full_model."""
    K = config["model"]["K"]
    S = config["model"]["S"]
    T = panel["N"].shape[1]

    def _load(fname, fallback):
        p = os.path.join(data_dir, fname)
        return np.load(p) if os.path.exists(p) else fallback

    M_skt   = _load("M_skt.npy",   np.ones((S, K, T)) / S)
    e_t     = _load("e_t.npy",     np.zeros(T))
    Lambda_L = _load("Lambda_L.npy", np.eye(S))
    x_s     = _load("x_s.npy",    np.ones(S) * 1e12)

    dates = panel.get("dates", [])
    mandatory_t = np.array(
        [str(date) >= "2023-12" for date in dates], dtype=float
    ) if len(dates) == T else np.zeros(T)
    return {
        "N":        panel["N"].astype(float),        # (K, T)
        "B":        panel.get("B", np.full((K, T), np.nan)).astype(float),  # (K, T); NaN = unobserved
        "E":        panel.get("E", np.zeros((K, T))).astype(float),  # (K, T)
        "D":        panel["D"].astype(float),        # (S, T)
        "M_skt":    M_skt.astype(float),             # (S, K, T)
        "e_t":      e_t.astype(float),               # (T,)
        "x_s":      x_s.astype(float),               # (S,)
        "Lambda_L": Lambda_L.astype(float),          # (S, S)
        "mandatory_t": mandatory_t,                  # SEC rule in force from Dec. 2023
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- Config -------------------------------------------------------------
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # MCMC settings live in config["mcmc"]; CLI flags override them.  We write
    # resolved values back into config because the blocked driver reads them.
    print("      Model variant: paper-exact")

    mcmc_cfg = config.setdefault("mcmc", {})
    if args.num_warmup is not None:
        mcmc_cfg["num_warmup"] = args.num_warmup
    if args.num_samples is not None:
        mcmc_cfg["num_samples"] = args.num_samples
    if args.gibbs_blocks is not None:
        mcmc_cfg["gibbs_blocks"] = args.gibbs_blocks
    if args.gibbs_warmup_blocks is not None:
        mcmc_cfg["gibbs_warmup_blocks"] = args.gibbs_warmup_blocks
    if args.block_warmup is not None:
        mcmc_cfg["block_warmup"] = args.block_warmup
    mcmc_cfg["seed"] = args.seed

    num_warmup  = int(mcmc_cfg.get("num_warmup", 1000))
    num_samples = int(mcmc_cfg.get("num_samples", 1000))

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ----------------------------------------------------------
    print("[1/3] Loading panel and features ...")
    panel = pipeline.load_panel(args.data_dir)

    if args.train_end is not None:
        panel = {
            k: (v[..., : args.train_end] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
            for k, v in panel.items()
        }

    data = assemble_data(panel, args.data_dir, config)

    K  = config["model"]["K"]
    S  = config["model"]["S"]
    T  = data["N"].shape[1]
    r  = config["model"].get("r", 2)
    R  = config["model"].get("R", 2)

    print(f"      K={K}  S={S}  T={T}  r={r}  R={R}")
    print(f"      N total counts: {data['N'].sum():.0f}  "
          f"D total counts: {data['D'].sum():.0f}")

    # --- Run inference ------------------------------------------------------
    t0 = time.time()

    if args.method == "blocked":
        print(f"[2/3] Alternating NUTS and FFBS  (initial warmup={num_warmup}  "
              f"saved Gibbs draws={num_samples}) ...")
        idata = run_blocked_nuts_ffbs(data, config)
    else:
        svi_cfg = config.get("svi", {})
        num_steps = int(svi_cfg.get("num_steps", 30000))
        lr = float(svi_cfg.get("learning_rate", 1e-3))
        print(f"[2/3] Running VI  (steps={num_steps}  lr={lr}) ...")
        idata = run_blocked_vi_ffbs(data, config)

    elapsed = time.time() - t0
    print(f"      {args.method.upper()} completed in {elapsed / 60:.1f} min.")

    # --- Save ---------------------------------------------------------------
    # A pickle is ALWAYS written first: it is the reliable, guaranteed-complete
    # save. netCDF (.nc) is attempted second, best-effort, purely as a portable
    #/inspectable secondary copy — some xarray/arviz version combinations have
    # been observed to silently produce a 0-byte or truncated .nc file for a
    # large, multi-group InferenceData without raising, which would otherwise
    # lose a long (multi-hour) sampling run outright.
    print("[3/3] Saving InferenceData ...")
    out_path = os.path.join(args.output_dir, "idata.nc")
    pkl_path = os.path.join(args.output_dir, "idata.pkl")

    import pickle
    with open(pkl_path, "wb") as fh:
        pickle.dump(idata, fh)
    print(f"      Saved to {pkl_path} (primary)")

    try:
        idata.to_netcdf(out_path)
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            if os.path.exists(out_path):
                os.remove(out_path)
            print("      netCDF write produced an empty/corrupt file; "
                  "skipped (the .pkl above is complete and authoritative).")
        else:
            print(f"      Saved to {out_path} (secondary)")
    except Exception as exc:  # noqa: BLE001 — never fail the run over the secondary copy
        print(f"      netCDF save skipped ({exc}); the .pkl above is complete and authoritative.")

    # --- Summary ------------------------------------------------------------
    try:
        import arviz as az
        key_vars = [v for v in idata.posterior.data_vars
                    if any(tag in v for tag in ["sigma", "phi", "alpha", "rho", "pi_r"])]
        key_vars = key_vars[:10]  # limit output width
        if key_vars:
            print("\nPosterior summary (selected parameters):")
            summary = az.summary(idata, var_names=key_vars, round_to=3)
            print(summary.to_string())
        else:
            print("\nPosterior summary:")
            print(az.summary(idata, round_to=3).head(15).to_string())
    except Exception as exc:
        print(f"\nNote: Could not print ArviZ summary: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
