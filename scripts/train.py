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
  2. Assembles the data dict expected by cassandra_threatcast.model.full.full_model.
  3. Runs NUTS via cassandra_threatcast.inference.nuts.run_nuts.
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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.data import pipeline
from cassandra_threatcast.model import full as full_module
from cassandra_threatcast.inference import nuts as nuts_module
from cassandra_threatcast.inference import vi as vi_module


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
    parser.add_argument(
        "--method", choices=["nuts", "vi"], default="nuts",
        help="Inference method: 'nuts' (MCMC, default) or 'vi' (variational).",
    )
    parser.add_argument(
        "--enhanced-mode", action="store_true",
        help="Use Claude's enhanced model (Student-t latent innovations, "
             "Negative-Binomial incident channel) instead of the paper-native "
             "model. See docs/MODEL_VARIANTS.md.",
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
        "B":        panel.get("B", np.full((K, T), np.nan)).astype(float),  # (K, T); NaN = unobserved
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

    # MCMC settings live in config["mcmc"]; CLI flags override them.  We write
    # the resolved values back into config["mcmc"] because run_nuts reads them
    # from there (it takes only (model, data, config)).
    # Enhanced mode: CLI flag overrides the config default.
    if args.enhanced_mode:
        config.setdefault("enhanced", {})["enabled"] = True
    mode = "ENHANCED (Student-t + NegBin)" if config.get("enhanced", {}).get("enabled") \
        else "paper-native"
    print(f"      Model variant: {mode}")

    mcmc_cfg = config.setdefault("mcmc", {})
    if args.num_warmup is not None:
        mcmc_cfg["num_warmup"] = args.num_warmup
    if args.num_samples is not None:
        mcmc_cfg["num_samples"] = args.num_samples
    if args.num_chains is not None:
        mcmc_cfg["num_chains"] = args.num_chains
    mcmc_cfg["seed"] = args.seed

    num_warmup  = int(mcmc_cfg.get("num_warmup", 1000))
    num_samples = int(mcmc_cfg.get("num_samples", 1000))
    num_chains  = int(mcmc_cfg.get("num_chains", 4))

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

    if args.method == "nuts":
        print(f"[2/3] Running NUTS  (warmup={num_warmup}  samples={num_samples}  "
              f"chains={num_chains}) ...")
        idata = nuts_module.run_nuts(full_module.full_model, data, config)
    else:
        svi_cfg = config.get("svi", {})
        num_steps = int(svi_cfg.get("num_steps", 30000))
        lr = float(svi_cfg.get("learning_rate", 1e-3))
        print(f"[2/3] Running VI  (steps={num_steps}  lr={lr}) ...")
        guide, params, losses = vi_module.train_vi(
            full_module.full_model, data, config,
            num_steps=num_steps, learning_rate=lr, seed=args.seed,
        )
        samples = vi_module.vi_predictive_samples(
            guide, params, full_module.full_model, data,
            n_samples=num_samples, seed=args.seed,
        )
        # Wrap the VI posterior draws in an InferenceData (add a chain dim) so
        # downstream scripts consume NUTS and VI output identically.
        import arviz as az
        posterior = {k: np.asarray(v)[np.newaxis, ...] for k, v in samples.items()}
        idata = az.from_dict(posterior=posterior)

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
