"""Compare original and amended count-scale priors using the initial window only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cyber_threat_modelling.models.count_baselines import stream_seed
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    panel = root / "data/processed/stage5-20261020/topic_month.parquet"
    frame = pd.read_parquet(panel)
    initial = frame.loc[frame.month.le("2018-12")].pivot(
        index="month", columns="category", values="N"
    )
    results = []
    for category in initial.columns:
        m0 = np.log(initial[category].mean() + 0.5)
        for prior in ("original_innovation_scale", "stationary_scale_halfnormal_v1"):
            rng = np.random.default_rng(
                stream_seed(20261020, "independent_nb_ar1", "2018-12", "predictive_checks")
            )
            rng.integers(4000, size=1000)  # Align the documented predictive-check stream.
            m = rng.normal(m0, 2, 1000)
            scale = abs(rng.normal(0, 0.3 if prior == "original_innovation_scale" else 0.75, 1000))
            phi = rng.uniform(-1, 1, 1000)
            if prior == "original_innovation_scale":
                tau = scale
                u = tau * rng.normal(size=1000) / np.sqrt(1 - phi**2)
            else:
                tau = scale * np.sqrt(1 - phi**2)
                u = scale * rng.normal(size=1000)
            eta = np.empty((1000, len(initial)))
            for month in range(len(initial)):
                if month:
                    u = phi * u + tau * rng.normal(size=1000)
                eta[:, month] = m + u
            row = {
                "category": int(category),
                "prior": prior,
                "max_log_rate": float(eta.max()),
                "log_rate_quantiles_05_50_95_999": np.quantile(
                    eta, [0.05, 0.5, 0.95, 0.999]
                ).tolist(),
                "draws_with_log_rate_above_log_int64_max": int(
                    (eta > np.log(np.iinfo(np.int64).max)).any(axis=1).sum()
                ),
            }
            if prior == "stationary_scale_halfnormal_v1":
                psi = np.exp(rng.normal(size=(1000, 1)))
                simulated = rng.negative_binomial(psi, psi / (psi + np.exp(eta)))
                row["count_simulation_completed"] = True
                row["max_simulated_count"] = int(simulated.max())
            results.append(row)
    record = {
        "training_cutoff": "2018-12",
        "prior_draws": 1000,
        "panel_sha256": file_hash(panel),
        "script_sha256": file_hash(Path(__file__)),
        "scope": "Prior diagnostic only; no development or held-out outcomes scored",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(record, indent=2) + "\n").encode())
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
