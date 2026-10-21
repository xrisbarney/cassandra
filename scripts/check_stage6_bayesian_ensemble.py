"""Development-only ensemble and NB-mixture log-mass sensitivity checks."""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, verify_artifacts
from cyber_threat_modelling.evaluation.scores import crps
from cyber_threat_modelling.forecasting.count_laws import CountForecast
from cyber_threat_modelling.models.bayesian_controls import posterior_forecast
from cyber_threat_modelling.models.count_baselines import stream_seed
from cyber_threat_modelling.models.independent_controls import forecast_from_files
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    checks, log_mass_checks = [], []
    for method in sorted({r["method"] for r in manifest["index"]}):
        if method not in ("pooled_nb", "independent_nb_ar1", "independent_nb_glar"):
            raise ValueError("Use a run containing Bayesian controls only.")
        records = sorted(
            [r for r in manifest["index"] if r["method"] == method], key=lambda r: r["origin"]
        )
        selected = {}
        for horizon in (1, 3, 6, 12):
            eligible = [
                r for r in records if str(pd.Period(r["origin"], freq="M") + horizon) <= "2023-12"
            ][-6:]
            if len(eligible) != 6 or any(r["fit_status"] != "accepted" for r in eligible):
                raise ValueError("Six complete eligible origins required at every horizon.")
            for record in eligible:
                selected.setdefault(record["origin"], (record, []))[1].append(horizon)
        values = {h: {n: [] for n in (1000, 3000, 6000)} for h in (1, 3, 6, 12)}
        for origin, (record, horizons) in selected.items():
            folder = args.run / record["directory"]
            fit = json.loads((folder / "fit.json").read_text())
            months = len(wide.loc[:origin])
            seed = stream_seed(record["seed"], method, origin, "future")
            issued = CountForecast.load(folder / "forecast.npz")
            if fit.get("posterior_layout") == "separate_categories":
                idata = None

                def generate(n, stream):
                    return forecast_from_files(folder, fit, months, n, stream)
            else:
                idata = az.from_netcdf(folder / f"posterior_attempt_{fit['retry_number']}.nc")

                def generate(n, stream):
                    return posterior_forecast(
                        idata.posterior, method, months, n, stream, fit.get("last_counts")
                    )

            forecasts = {n: generate(n, seed) for n in (1000, 3000, 6000)}
            if not np.array_equal(forecasts[3000].paths, issued.paths):
                raise ValueError("Saved 3000-path stream does not reproduce.")
            repeat = generate(
                6000,
                stream_seed(record["seed"], method, origin, "future_mc_repeat"),
            )
            for horizon in horizons:
                y = wide.loc[str(pd.Period(origin, freq="M") + horizon)].to_numpy(dtype=int)
                for j, category in enumerate(manifest["category_ids"]):
                    for size, forecast in forecasts.items():
                        values[horizon][size].append(crps(forecast.paths[:, horizon - 1, j], y[j]))
                    original_log_mass = forecasts[6000].logpmf(y[j], horizon - 1, j)
                    repeated_log_mass = repeat.logpmf(y[j], horizon - 1, j)
                    log_mass_checks.append(
                        {
                            "method": method,
                            "origin": origin,
                            "horizon": horizon,
                            "category": category,
                            "log_mass_6000": original_log_mass,
                            "independent_seed_log_mass_6000": repeated_log_mass,
                            "difference": repeated_log_mass - original_log_mass,
                        }
                    )
            if idata is not None:
                idata.close()
        for horizon, sizes in values.items():
            means = {str(n): float(np.mean(v)) for n, v in sizes.items()}
            tolerance = 0.01 * max(1, means["6000"])
            checks.append(
                {
                    "method": method,
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
            "all_crps_checks_passed": all(c["passed"] for c in checks),
            "log_mass_seed_sensitivity": log_mass_checks,
            "log_mass_interpretation": "Descriptive MC differences; no invented acceptance cutoff.",
        },
    )


if __name__ == "__main__":
    main()
