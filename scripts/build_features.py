#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

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

# Force UTF-8 stdout/stderr: Windows' default console codepage cannot encode
# many Unicode characters, which raises UnicodeEncodeError and kills the
# process -- especially when output is redirected to a log file.
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.data import bea_io, pipeline
from cassandra_threatcast.features import exposure_map as em, effort


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
    print("[2/4] Loading exposure map M_skt ...")
    # M_skt is built by ingest_data.py from CVE-level CPE data (which is not
    # available here). Load it if present; otherwise fall back to a uniform map.
    m_path = os.path.join(args.data_dir, "M_skt.npy")
    if os.path.exists(m_path):
        M_skt = np.load(m_path)
        print(f"      Loaded M_skt from {m_path}")
    else:
        print("      Warning: M_skt.npy not found (run ingest_data.py first); "
              "using uniform exposure prior.")
        M_skt = np.ones((S, K, T)) / S

    print(f"      M_skt shape={M_skt.shape}  "
          f"col-sum min={M_skt.sum(axis=0).min():.3f}  "
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
        A = bea_io.get_technical_coefficients(year=args.bea_year, n_sectors=S, cache_dir=args.data_dir)
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
            x_s = bea_io.get_sector_output(year=args.bea_year, n_sectors=S, cache_dir=args.data_dir)
        except Exception:
            x_s = np.ones(S) * 1e12   # placeholder: $1 trillion each

    np.save(os.path.join(args.output_dir, "x_s.npy"), x_s)
    # BEA gross output is in millions of dollars; /1e6 -> trillions.
    print(f"      x_s shape={x_s.shape}  "
          f"total output = ${x_s.sum() / 1e6:.2f} trillion")

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
