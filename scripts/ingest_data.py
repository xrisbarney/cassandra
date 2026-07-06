#!/usr/bin/env python3
"""
Ingest all data sources and save processed arrays.

Usage:
    python scripts/ingest_data.py --start 2010-01 --end 2024-12 \\
        --cache-dir data/cache/ --output-dir data/processed/ \\
        --config configs/default.yaml
"""
import argparse

from dotenv import load_dotenv
load_dotenv()
import os
import pickle
import sys
import yaml
import numpy as np
import pandas as pd

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cyberthreats.data import nvd, epss, cisa_kev, sec_8k, bea_io, pipeline
from cyberthreats.features import topic_map as tm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest cyber-threat and economic data and save processed arrays."
    )
    parser.add_argument(
        "--start", default="2010-01",
        help="Start year-month (YYYY-MM).  Default: 2010-01",
    )
    parser.add_argument(
        "--end", default="2024-12",
        help="End year-month (YYYY-MM).  Default: 2024-12",
    )
    parser.add_argument(
        "--cache-dir", default="data/cache/",
        help="Directory for raw API response caches.  Default: data/cache/",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/",
        help="Directory for processed NumPy arrays.  Default: data/processed/",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="YAML config file.  Default: configs/default.yaml",
    )
    parser.add_argument(
        "--n-topics", type=int, default=None,
        help="Override K (number of CVE topics) from config.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- Load config --------------------------------------------------------
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    K = args.n_topics if args.n_topics is not None else config["model"]["K"]
    S = config["model"]["S"]

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    start_dt = args.start + "-01"
    end_dt   = args.end   + "-28"   # safe upper bound for any month

    # --- NVD CVEs -----------------------------------------------------------
    print(f"[1/6] Fetching NVD CVEs from {args.start} to {args.end} ...")
    cve_df = nvd.fetch_cves(start_dt, end_dt, args.cache_dir)
    print(f"      {len(cve_df):,} CVEs fetched.")

    # --- Topic model --------------------------------------------------------
    print(f"[2/6] Fitting topic model (K={K}) ...")
    descriptions = cve_df["description"].fillna("").tolist()
    mapper = tm.WikiTopicMapper(n_topics=K)
    mapper.fit(descriptions)
    topic_assignments = mapper.assign_hard(descriptions)

    mapper_path = os.path.join(args.output_dir, "topic_mapper.pkl")
    with open(mapper_path, "wb") as fh:
        pickle.dump(mapper, fh)
    print(f"      Topic mapper saved to {mapper_path}")

    # --- EPSS ---------------------------------------------------------------
    print("[3/6] Fetching EPSS scores ...")
    try:
        epss_df = epss.fetch_epss_range(start_dt, end_dt, args.cache_dir)
        print(f"      {len(epss_df):,} EPSS records fetched.")
    except Exception as exc:
        print(f"      Warning: EPSS fetch failed ({exc}).  Proceeding without.")
        epss_df = None

    # --- CISA KEV -----------------------------------------------------------
    print("[4/6] Fetching CISA KEV catalog ...")
    try:
        kev_df = cisa_kev.fetch_kev(args.cache_dir)
        print(f"      {len(kev_df):,} KEV entries fetched.")
    except Exception as exc:
        print(f"      Warning: KEV fetch failed ({exc}).  Proceeding without.")
        kev_df = None

    # --- SEC 8-K ------------------------------------------------------------
    print("[5/6] Fetching SEC 8-K cyber-disclosure filings ...")
    try:
        sec_df = sec_8k.fetch_8k_cyber(start_dt, end_dt, args.cache_dir)
        sec_df = sec_8k.enrich_with_naics(sec_df, args.cache_dir)
        print(f"      {len(sec_df):,} 8-K filings fetched.")
    except Exception as exc:
        print(f"      Warning: SEC 8-K fetch failed ({exc}).  Proceeding without.")
        sec_df = None

    # --- Build panel --------------------------------------------------------
    print("[6/6] Building monthly panel arrays ...")
    sector_map = config.get("sector_map", {str(i): i for i in range(S)})

    panel = pipeline.build_panel(
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        topic_assignments=topic_assignments,
        sector_map=sector_map,
        K=K,
        S=S,
    )

    pipeline.save_panel(panel, args.output_dir)

    # --- Summary ------------------------------------------------------------
    print("\nDone.  Panel saved to:", args.output_dir)
    for key, arr in panel.items():
        if isinstance(arr, np.ndarray):
            print(f"  {key:10s} shape = {arr.shape}  "
                  f"mean = {arr.mean():.3f}  max = {arr.max():.3f}")


if __name__ == "__main__":
    main()
