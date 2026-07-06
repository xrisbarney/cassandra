#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Train the full Bayesian hierarchical state-space model via NUTS.

Usage:
    python scripts/train.py \\
        --config   configs/default.yaml \\
        --data-dir data/processed/ \\
        --output-dir results/

The script:
  1. Loads the processed panel and derived features from --data-dir.
  2. Assembles the data dict expected by cyberthreats.model.full.full_model.
  3. Runs NUTS via cyberthreats.inference.nuts.run_nuts.
  4. Saves the resulting ArviZ InferenceData to <output_dir>/idata.nc.
  5. Prints a posterior summary for key parameters.
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cyberthreats.data import pipeline
from cyberthreats.model import full as full_module
from cyberthreats.inference import nuts as nuts_module


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the full Bayesian hierarchical model via NUTS."
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
        "--num-chains", type=int, default=None,
        help="Override num_chains from config.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="JAX random seed.  Default: 42",
    )
    parser.add_argument(
        "--train-end", type=int, default=None,
        help="Last time index (exclusive) for training.  Default: full series.",
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

    return {
        "N":        panel["N"].astype(float),        # (K, T)
        "B":        panel.get("B", np.zeros((K, T))).astype(float),  # (K, T)
        "E":        panel.get("E", np.zeros((K, T))).astype(float),  # (K, T)
        "D":        panel["D"].astype(float),        # (S, T)
        "M_skt":    M_skt.astype(float),             # (S, K, T)
        "e_t":      e_t.astype(float),               # (T,)
        "x_s":      x_s.astype(float),               # (S,)
        "Lambda_L": Lambda_L.astype(float),          # (S, S)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- Config -------------------------------------------------------------
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    inf_cfg = config.get("inference", {})
    num_warmup  = args.num_warmup  or inf_cfg.get("num_warmup",  1000)
    num_samples = args.num_samples or inf_cfg.get("num_samples", 1000)
    num_chains  = args.num_chains  or inf_cfg.get("num_chains",  4)

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

    # --- Run NUTS -----------------------------------------------------------
    print(f"[2/3] Running NUTS  (warmup={num_warmup}  samples={num_samples}  "
          f"chains={num_chains}) ...")
    t0 = time.time()

    idata = nuts_module.run_nuts(
        model=full_module.full_model,
        data=data,
        config=config,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=args.seed,
    )

    elapsed = time.time() - t0
    print(f"      NUTS completed in {elapsed / 60:.1f} min.")

    # --- Save ---------------------------------------------------------------
    print("[3/3] Saving InferenceData ...")
    out_path = os.path.join(args.output_dir, "idata.nc")
    idata.to_netcdf(out_path)
    print(f"      Saved to {out_path}")

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
