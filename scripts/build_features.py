#!/usr/bin/env python3
"""
Build derived features from the processed panel and save them to disk.

Steps:
  1. Load the processed panel (N, D, B, E, dates) from --data-dir.
  2. Compute the sector-topic exposure map M_skt (S, K, T).
  3. Estimate the reporting-effort index e_t (T,).
  4. Load BEA I-O data and compute the Leontief inverse Lambda_L (S, S).
  5. Save all derived arrays to --output-dir.

Usage:
    python scripts/build_features.py \\
        --data-dir  data/processed/ \\
        --output-dir data/processed/ \\
        --config    configs/default.yaml
"""
import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cyberthreats.data import bea_io, pipeline
from cyberthreats.features import exposure_map as em, effort


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute derived features from the processed panel."
    )
    parser.add_argument(
        "--data-dir", default="data/processed/",
        help="Directory containing processed panel arrays.  Default: data/processed/",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/",
        help="Directory to write feature arrays.  Default: data/processed/",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="YAML config file.  Default: configs/default.yaml",
    )
    parser.add_argument(
        "--bea-year", type=int, default=2019,
        help="BEA I-O table year to use.  Default: 2019",
    )
    parser.add_argument(
        "--effort-method", default="hp_filter",
        choices=["hp_filter", "moving_avg", "log_diff"],
        help="Effort estimation method.  Default: hp_filter",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    K = config["model"]["K"]
    S = config["model"]["S"]

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1.  Load panel
    # -------------------------------------------------------------------------
    print("[1/4] Loading processed panel ...")
    panel = pipeline.load_panel(args.data_dir)

    N_kt = panel["N"]   # (K, T)
    D_st = panel["D"]   # (S, T)
    B_kt = panel.get("B", np.zeros_like(N_kt))   # (K, T) severity scores
    T = N_kt.shape[1]

    print(f"      N shape={N_kt.shape}  D shape={D_st.shape}  T={T}")

    # -------------------------------------------------------------------------
    # 2.  Sector-topic exposure map M_skt
    # -------------------------------------------------------------------------
    print("[2/4] Computing exposure map M_skt ...")
    # exposure_map.compute returns (S, K, T) — the fraction of topic-k CVE
    # volume that sector s is exposed to at time t, derived from 8-K filings
    # and NAICS-to-sector mapping stored in the panel.
    if hasattr(em, "compute"):
        M_skt = em.compute(panel, K=K, S=S)
    else:
        # Fallback: uniform exposure across sectors
        print("      Warning: exposure_map.compute not found; using uniform prior.")
        M_raw = np.ones((S, K, T)) / S   # uniform
        # Normalise so columns sum to 1 across sectors
        M_skt = M_raw / M_raw.sum(axis=0, keepdims=True)

    print(f"      M_skt shape={M_skt.shape}  "
          f"row-sum min={M_skt.sum(axis=0).min():.3f}  "
          f"max={M_skt.sum(axis=0).max():.3f}")

    np.save(os.path.join(args.output_dir, "M_skt.npy"), M_skt)

    # -------------------------------------------------------------------------
    # 3.  Reporting-effort index e_t
    # -------------------------------------------------------------------------
    print("[3/4] Estimating reporting-effort index e_t ...")
    # Aggregate CVE counts across topics to get total monthly volume
    N_total = N_kt.sum(axis=0).astype(float)   # (T,)
    e_t = effort.estimate_effort(N_total, method=args.effort_method)

    print(f"      e_t: mean={e_t.mean():.4f}  std={e_t.std():.4f}  "
          f"min={e_t.min():.4f}  max={e_t.max():.4f}")

    np.save(os.path.join(args.output_dir, "e_t.npy"), e_t)

    # -------------------------------------------------------------------------
    # 4.  BEA I-O Leontief inverse
    # -------------------------------------------------------------------------
    print("[4/4] Computing Leontief inverse from BEA I-O table ...")
    try:
        A = bea_io.get_technical_coefficients(year=args.bea_year, n_sectors=S)
        Lambda_L = bea_io.leontief_inverse(A)
        print(f"      Leontief inverse shape={Lambda_L.shape}  "
              f"spectral_radius_A={np.max(np.abs(np.linalg.eigvals(A))):.4f}")
    except Exception as exc:
        print(f"      Warning: BEA I-O computation failed ({exc}).  Using identity.")
        Lambda_L = np.eye(S)

    np.save(os.path.join(args.output_dir, "Lambda_L.npy"), Lambda_L)

    # -------------------------------------------------------------------------
    # Save sector output values x_s  (from BEA or config)
    # -------------------------------------------------------------------------
    if "sector_output" in config.get("model", {}):
        x_s = np.array(config["model"]["sector_output"], dtype=float)
    else:
        try:
            x_s = bea_io.get_sector_output(year=args.bea_year, n_sectors=S)
        except Exception:
            x_s = np.ones(S) * 1e12   # placeholder: $1 trillion each

    np.save(os.path.join(args.output_dir, "x_s.npy"), x_s)
    print(f"      x_s shape={x_s.shape}  "
          f"total output = ${x_s.sum() / 1e12:.2f} trillion")

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    print("\nDone.  Feature arrays written to:", args.output_dir)
    for fname in ["M_skt.npy", "e_t.npy", "Lambda_L.npy", "x_s.npy"]:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            arr = np.load(fpath)
            print(f"  {fname:20s} shape={arr.shape}")


if __name__ == "__main__":
    main()
