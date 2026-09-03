#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

"""
Ablation study (paper Table 4): remove one model component at a time,
retrain, and score each variant with the sequential (rolling-origin)
machinery.

Variants
--------
  full          all components (the reference row, retrained at the SAME
                reduced MCMC settings as the ablations so the comparison is
                settings-matched)
  no_regimes    R = 1 (no structural breaks)
  no_factors    Gamma = 0 (independent topics)
  flat_priors   all prior scales x10 (no shrinkage)
  no_E          exploitation channel dropped from the likelihood
  no_D          incident channel dropped from the likelihood
  no_effort     e_t = 0 (raw counts treated as truth)

Each variant is trained by invoking scripts/train.py with a variant config
(so JAX memory is isolated per run), then scored: CRPS at h=1 and h=6 and
90%-interval coverage (h=1) over the test window, via
cassandra_threatcast.evaluation.sequential.

Already-trained variants (results/ablation/<name>/idata.pkl present) are
NOT retrained -- an interrupted overnight run resumes where it stopped.
Pass --rescore to recompute scores from existing posteriors only.

Usage:
    python scripts/ablation.py                        # all variants, NUTS 300/300x1
    python scripts/ablation.py --method vi            # faster, VI posteriors
    python scripts/ablation.py --variants full,no_regimes
    python scripts/ablation.py --smoke                # tiny settings, wiring check
"""
import argparse
import copy
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.evaluation.scoring import crps_ensemble
from cassandra_threatcast.evaluation.sequential import (
    sequential_one_step_predict, sequential_h_step_predict,
)

VARIANTS: dict[str, dict] = {
    "full":        {"overrides": {},                                  "note": "all components"},
    "no_regimes":  {"overrides": {"model.R": 1},                      "note": "no structural breaks (R=1)"},
    "no_factors":  {"overrides": {"ablation.no_factors": True},       "note": "independent topics (Gamma=0)"},
    "flat_priors": {"overrides": {"ablation.flat_priors": True},      "note": "no shrinkage (prior scales x10)"},
    "no_E":        {"overrides": {"ablation.drop_E": True},           "note": "CVE + 8-K only"},
    "no_D":        {"overrides": {"ablation.drop_D": True},           "note": "CVE + EPSS only"},
    "no_effort":   {"overrides": {"ablation.no_effort": True},        "note": "raw counts as truth (e_t=0)"},
    # Addition rather than removal: the learned moving-window covariance
    "kernel":      {"overrides": {"kernel.enabled": True},            "note": "+ learned moving-window kernel"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Table 4 ablation study.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed/")
    parser.add_argument("--output-dir", default="results/ablation/")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names, or 'all'.")
    parser.add_argument("--method", choices=["nuts", "vi"], default="nuts")
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--test-start", default="2023-01")
    parser.add_argument("--test-end", default="2024-12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny settings (30/30) purely to validate wiring.")
    parser.add_argument("--rescore", action="store_true",
                        help="Skip all training; only rescore existing posteriors.")
    return parser.parse_args()


def _apply_overrides(config: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(config)
    for dotted, value in overrides.items():
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def _score_variant(idata_path: str, data_dir: str, test_mask: np.ndarray,
                   N_actual: np.ndarray, e_t: np.ndarray, M_skt: np.ndarray,
                   seed: int) -> dict:
    with open(idata_path, "rb") as fh:
        idata = pickle.load(fh)
    post = {k: np.asarray(v) for k, v in idata.posterior.items()}
    # Zero-size sites (numpyro.factor sites in VI posteriors) can't reshape.
    post = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items() if v.size > 0}

    out1 = sequential_one_step_predict(post, e_t, M_skt, tail_months=0,
                                       enhanced=False, student_t_df=4.0, seed=seed)
    N1 = out1["N_pred"]                                       # (n, K, T)
    N6 = sequential_h_step_predict(post, e_t, M_skt, h=6, seed=seed)

    obs_w = N_actual[:, test_mask]                            # (K, Tw)
    s1 = np.moveaxis(N1[:, :, test_mask], 0, -1)              # (K, Tw, n)
    crps1 = float(crps_ensemble(obs_w, s1).mean())

    valid6 = test_mask & ~np.isnan(N6[0, 0])                  # months with h=6 preds
    obs6 = N_actual[:, valid6]
    s6 = np.moveaxis(N6[:, :, valid6], 0, -1)
    crps6 = float(crps_ensemble(obs6, s6).mean())

    q05, q95 = np.quantile(N1[:, :, test_mask], [0.05, 0.95], axis=0)
    cov90 = float(np.mean((q05 <= obs_w) & (obs_w <= q95)))
    return {"CRPS_h1": crps1, "CRPS_h6": crps6, "cov90_h1": cov90}


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as fh:
        base_config = yaml.safe_load(fh)

    names = list(VARIANTS) if args.variants == "all" else [
        v.strip() for v in args.variants.split(",")]
    for v in names:
        if v not in VARIANTS:
            print(f"Unknown variant '{v}'. Known: {', '.join(VARIANTS)}")
            sys.exit(1)

    warmup = 30 if args.smoke else args.num_warmup
    samples = 30 if args.smoke else args.num_samples

    # Panel pieces for scoring
    N_actual = np.load(os.path.join(args.data_dir, "N.npy")).astype(float)
    e_t = np.load(os.path.join(args.data_dir, "e_t.npy"))
    M_skt = np.load(os.path.join(args.data_dir, "M_skt.npy"))
    with open(os.path.join(args.data_dir, "panel_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    periods = pd.PeriodIndex(meta.get("dates") or meta["metadata"]["dates"], freq="M")
    test_mask = np.asarray((periods >= pd.Period(args.test_start, freq="M"))
                           & (periods <= pd.Period(args.test_end, freq="M")))
    print(f"Variants: {', '.join(names)}   method={args.method}  "
          f"warmup={warmup} samples={samples}")
    print(f"Test window: {args.test_start}..{args.test_end} "
          f"({int(test_mask.sum())} months)\n")

    py = sys.executable
    rows = []
    for name in names:
        vdir = os.path.join(args.output_dir, name)
        os.makedirs(vdir, exist_ok=True)
        idata_path = os.path.join(vdir, "idata.pkl")
        cfg_path = os.path.join(vdir, "config.yaml")

        if os.path.exists(idata_path):
            print(f"=== {name}: posterior exists, skipping training ===")
        elif args.rescore:
            print(f"=== {name}: no posterior and --rescore set, skipping ===")
            continue
        else:
            vcfg = _apply_overrides(base_config, VARIANTS[name]["overrides"])
            with open(cfg_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(vcfg, fh)
            print(f"=== {name}: training ({VARIANTS[name]['note']}) ===")
            t0 = time.time()
            result = subprocess.run(
                [py, os.path.join("scripts", "train.py"),
                 "--config", cfg_path, "--data-dir", args.data_dir,
                 "--output-dir", vdir, "--method", args.method,
                 "--num-warmup", str(warmup), "--num-samples", str(samples),
                 "--num-chains", "1", "--seed", str(args.seed)],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(f"    exit={result.returncode}  ({(time.time()-t0)/60:.1f} min)")
            if result.returncode != 0:
                tail = "\n".join((result.stdout or "").splitlines()[-12:])
                print(f"    TRAINING FAILED; last output:\n{tail}\n{result.stderr[-800:]}")
                rows.append({"variant": name, "note": VARIANTS[name]["note"],
                             "status": "train_failed"})
                continue

        print(f"    scoring {name} ...")
        try:
            scores = _score_variant(idata_path, args.data_dir, test_mask,
                                    N_actual, e_t, M_skt, args.seed)
            rows.append({"variant": name, "note": VARIANTS[name]["note"],
                         "status": "ok", **scores})
            print(f"    CRPS h1={scores['CRPS_h1']:.2f}  h6={scores['CRPS_h6']:.2f}  "
                  f"cov90={scores['cov90_h1']:.1%}")
        except Exception as exc:
            print(f"    SCORING FAILED: {exc}")
            rows.append({"variant": name, "note": VARIANTS[name]["note"],
                         "status": f"score_failed: {exc}"})

        # Write progressively so an interrupted run still leaves results.
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, "ablation.csv")
        if os.path.exists(csv_path):
            prev = pd.read_csv(csv_path)
            prev = prev[~prev["variant"].isin(df["variant"])]
            df = pd.concat([prev, df], ignore_index=True)
        full_row = df[df.variant == "full"]
        if len(full_row) and "CRPS_h1" in df.columns:
            ref = float(full_row["CRPS_h1"].iloc[0])
            df["delta_CRPS_h1_pct"] = 100.0 * (df["CRPS_h1"] - ref) / ref
        df.to_csv(csv_path, index=False)

    print(f"\nTable 4 -> {os.path.join(args.output_dir, 'ablation.csv')}")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
