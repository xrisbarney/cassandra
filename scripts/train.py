#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Infer the paper model (manuscript §3.5): the discrete regime path is
marginalized analytically inside the likelihood by the hidden-Markov forward
recursion, and every remaining unknown is continuous and sampled directly
with the No-U-Turn sampler.  No categorical variable is ever sampled.

Usage:
    python scripts/train.py \\
        --config   configs/default.yaml \\
        --data-dir data/processed/ \\
        --output-dir results/

The script:
  1. Loads the processed panel and derived features from --data-dir.
  2. Assembles the data dict expected by `model.paper_exact.paper_model`.
  3. Runs NUTS on the marginalized model (or VI with --method vi, the
     scalable variant §4.1 prescribes for K in the thousands).
  4. Saves the resulting ArviZ InferenceData to <output_dir>/idata.pkl/.nc.
  5. Prints a posterior summary for key parameters.
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np

# Force UTF-8 stdout/stderr: Windows' default console codepage cannot...
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Must run before the first jax/numpyro import (including...
import numpyro
numpyro.set_host_device_count(os.cpu_count() or 1)

from cassandra_threatcast.data import pipeline
from cassandra_threatcast.inference.nuts import run_nuts
from cassandra_threatcast.model.paper_exact import paper_model


# Argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer the paper model via marginalized-regime NUTS (§3.5)."
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
        help="Inference: NUTS on the marginalized model (default; §3.5), or "
             "variational inference (the §4.1 population-scale variant).",
    )
    return parser.parse_args()


def assemble_data(panel: dict, data_dir: str, config: dict) -> dict:
    "Build the data dict expected by paper_model."
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


def _run_vi(data: dict, config: dict):
    "VI on the marginalized model; returns an ArviZ InferenceData."
    import arviz as az
    from cassandra_threatcast.inference.vi import train_vi, vi_predictive_samples

    mcfg, scfg = config.get("mcmc", {}), config.get("svi", {})
    n_samples = int(mcfg.get("num_samples", 1000))
    seed = int(scfg.get("seed", mcfg.get("seed", 0)))
    guide, params, _ = train_vi(
        paper_model, data, config,
        num_steps=int(scfg.get("num_steps", 30000)),
        learning_rate=float(scfg.get("learning_rate", 1e-3)),
        seed=seed,
    )
    samples = vi_predictive_samples(
        guide, params, paper_model, data, config=config,
        n_samples=n_samples, seed=seed + 1,
    )
    posterior = {
        name: np.asarray(value)[None, ...]
        for name, value in samples.items()
        if name not in {"N_obs", "E_obs", "B_obs", "D_obs"}
        and np.asarray(value).size > 0
    }
    return az.from_dict({"posterior": posterior})


# Main
def main() -> None:
    args = parse_args()

    # Load configuration.
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # Let CLI flags override MCMC configuration.
    print("      Model variant: paper (marginalized regimes)")

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

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data.
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

    # Run inference.
    t0 = time.time()

    if args.method == "nuts":
        print(f"[2/3] NUTS on the marginalized model  (warmup={num_warmup}  "
              f"samples={num_samples}) ...")
        idata = run_nuts(paper_model, data, config)
    else:
        svi_cfg = config.get("svi", {})
        num_steps = int(svi_cfg.get("num_steps", 30000))
        lr = float(svi_cfg.get("learning_rate", 1e-3))
        print(f"[2/3] Running VI  (steps={num_steps}  lr={lr}) ...")
        idata = _run_vi(data, config)

    elapsed = time.time() - t0
    print(f"      {args.method.upper()} completed in {elapsed / 60:.1f} min.")

    # Save results.
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

    # Print the summary.
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
