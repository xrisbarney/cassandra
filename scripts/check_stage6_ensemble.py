"""Development-only path-count sensitivity for the two initial fixed controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, verify_artifacts
from cyber_threat_modelling.evaluation.scores import crps
from cyber_threat_modelling.forecasting.count_laws import CountForecast, round_counts
from cyber_threat_modelling.models.count_baselines import stream_seed
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.run / "forecast_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["settings"]["split"] != "development":
        raise ValueError("Development only.")
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_artifacts(args.run, manifest)
    root = Path(__file__).resolve().parents[1]
    wide, _ = count_matrix(pd.read_parquet(root / manifest["panel_file"]))
    rows = []
    for method in ("historical_mean", "seasonal_naive"):
        records = [r for r in manifest["index"] if r["method"] == method]
        for horizon in (1, 3, 6, 12):
            eligible = [
                r for r in records if str(pd.Period(r["origin"], freq="M") + horizon) <= "2023-12"
            ][-6:]
            if len(eligible) != 6 or any(r["fit_status"] != "accepted" for r in eligible):
                raise ValueError("Six complete eligible origins required.")
            scores = {n: [] for n in (1000, 3000, 6000)}
            for record in eligible:
                f = CountForecast.load(args.run / record["directory"] / "forecast.npz")
                y = wide.loc[str(pd.Period(record["origin"], freq="M") + horizon)].to_numpy(
                    dtype=int
                )
                for j, category in enumerate(manifest["category_ids"]):
                    for size in scores:
                        rng = np.random.default_rng(
                            stream_seed(
                                record["seed"], method, int(category), record["origin"], "forecast"
                            )
                        )
                        if f.family == "poisson":
                            paths = rng.poisson(f.parameters["rate"][:, j], size=(size, 12))
                        elif f.family == "rounded_normal":
                            paths = round_counts(
                                rng.normal(
                                    f.parameters["location"][:, j],
                                    f.parameters["scale"][:, j],
                                    size=(size, 12),
                                )
                            )
                        else:
                            raise ValueError("This check supports the initial fixed controls only.")
                        if size == 3000 and not np.array_equal(paths, f.paths[:, :, j]):
                            raise ValueError(
                                "Resimulation does not reproduce the saved path stream."
                            )
                        scores[size].append(crps(paths[:, horizon - 1], y[j]))
            means = {str(size): float(np.mean(values)) for size, values in scores.items()}
            tolerance = 0.01 * max(1, means["6000"])
            rows.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "origins": [r["origin"] for r in eligible],
                    "mean_crps": means,
                    "tolerance": tolerance,
                    "passed": all(
                        abs(means[str(n)] - means["6000"]) <= tolerance for n in (1000, 3000)
                    ),
                }
            )
    write_json(
        args.output,
        {
            "forecast_manifest_sha256": file_hash(manifest_path),
            "checks": rows,
            "all_passed": all(r["passed"] for r in rows),
            "scope": "two fixed controls; other methods require their own checks",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
