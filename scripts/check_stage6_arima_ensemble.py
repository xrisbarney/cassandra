"""Replay saved ARIMA parameters and check development forecast simulation error."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, verify_artifacts
from cyber_threat_modelling.evaluation.scores import crps
from cyber_threat_modelling.forecasting.count_laws import CountForecast, round_counts
from cyber_threat_modelling.models.count_baselines import method_id, stream_seed
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.run / "forecast_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["settings"]["split"] != "development":
        raise ValueError("Development only.")
    verify_artifacts(args.run, manifest)
    root = Path(__file__).resolve().parents[1]
    wide, _ = count_matrix(pd.read_parquet(root / manifest["panel_file"]))
    checks = []
    for spec in manifest["settings"]["methods"]:
        if spec["method"] != "arima":
            raise ValueError("ARIMA only.")
        name = method_id(spec)
        records = sorted(
            [r for r in manifest["index"] if r["method"] == name], key=lambda r: r["origin"]
        )
        selected = {}
        for horizon in (1, 3, 6, 12):
            eligible = [
                r for r in records if str(pd.Period(r["origin"], freq="M") + horizon) <= "2023-12"
            ][-6:]
            if len(eligible) != 6 or any(r["fit_status"] != "accepted" for r in eligible):
                raise ValueError("Six complete eligible origins required.")
            for record in eligible:
                selected.setdefault(record["origin"], (record, []))[1].append(horizon)
        scores = {h: {n: [] for n in (1000, 3000, 6000)} for h in (1, 3, 6, 12)}
        for origin, (record, horizons) in selected.items():
            folder = args.run / record["directory"]
            fitted = json.loads((folder / "fit.json").read_text())
            issued = CountForecast.load(folder / "forecast.npz")
            for j, category in enumerate(manifest["category_ids"]):
                params = fitted["per_category"][j]
                if params["category"] != category:
                    raise ValueError("Category mismatch.")
                if "boundary_rule" not in params:
                    model = ARIMA(
                        wide.loc[:origin, category].to_numpy(dtype=float),
                        order=tuple(spec["order"]),
                        trend=spec["trend"],
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    ).filter(params["parameters"])
                    forecast = model.get_forecast(12)
                    if not np.allclose(
                        forecast.predicted_mean, issued.parameters["location"][:, j]
                    ):
                        raise ValueError("Filtered forecast mean differs from saved forecast.")
                for size in (1000, 3000, 6000):
                    rng = np.random.default_rng(
                        stream_seed(record["seed"], name, int(category), origin, "forecast")
                    )
                    paths = (
                        np.zeros((size, 12), dtype=np.int64)
                        if "boundary_rule" in params
                        else round_counts(
                            np.asarray(model.simulate(12, anchor="end", repetitions=size, rng=rng))
                            .reshape(12, size)
                            .T
                        )
                    )
                    if size == 3000 and not np.array_equal(paths, issued.paths[:, :, j]):
                        raise ValueError("Saved ARIMA simulation does not reproduce exactly.")
                    for horizon in horizons:
                        target = str(pd.Period(origin, freq="M") + horizon)
                        scores[horizon][size].append(
                            crps(paths[:, horizon - 1], wide.loc[target, category])
                        )
            print(name, origin, flush=True)
        for horizon, values in scores.items():
            means = {str(n): float(np.mean(v)) for n, v in values.items()}
            tolerance = 0.01 * max(1, means["6000"])
            checks.append(
                {
                    "method": name,
                    "horizon": horizon,
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
            "checks": checks,
            "all_passed": all(c["passed"] for c in checks),
            "log_mass": "Analytic rounded-Gaussian mass; no density Monte Carlo error.",
        },
    )


if __name__ == "__main__":
    main()
