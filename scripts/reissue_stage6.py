"""Apply the predeclared common 6000-path budget to saved accepted fits."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cyber_threat_modelling.config import ExperimentConfig, canonical_json, load_config
from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import (
    artifact_path,
    count_matrix,
    validate_settings,
    verify_artifacts,
    verify_finalists,
)
from cyber_threat_modelling.evaluation.reissue import replay_factory, verify_budget_record
from cyber_threat_modelling.evaluation.scores import crps
from cyber_threat_modelling.features.build import verify_panels
from cyber_threat_modelling.models.count_baselines import method_id
from cyber_threat_modelling.provenance import file_hash, prepare_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    decision = root / "docs/decisions/ENSEMBLE_BUDGET_20261021.json"
    reference = {"path": decision.relative_to(root).as_posix(), "sha256": file_hash(decision)}
    verify_budget_record(root, reference)
    parent_path = args.source / "forecast_manifest.json"
    parent = json.loads(parent_path.read_text())
    verify_artifacts(args.source, parent)
    panel_manifest = artifact_path(root, parent["panel_manifest"])
    if file_hash(panel_manifest) != parent["panel_manifest_sha256"]:
        raise ValueError("Source panel manifest changed.")
    verify_panels(root, panel_manifest)
    if parent["settings"]["split"] != "development" or parent["settings"]["draws"] != 3000:
        raise ValueError("Expected the original development ensemble.")
    specs = [
        s
        for s in parent["settings"]["methods"]
        if args.methods is None or method_id(s) in args.methods
    ]
    settings = {**parent["settings"], "methods": specs, "draws": 6000, "budget_record": reference}
    names = validate_settings(settings)
    if args.methods is not None and set(args.methods) != set(names):
        raise ValueError("Requested method is absent from the source run.")
    verify_finalists(root, settings)
    config = load_config(root / "configs/pilot.toml")
    config = ExperimentConfig(
        {**config.settings, "forecast": {**config.settings["forecast"], "draws": 6000}}
    )
    prepare_run(root, config, args.output)
    write_json(args.output / "backtest_settings.json", settings)
    wide, _ = count_matrix(pd.read_parquet(root / parent["panel_file"]))
    selected, records, checks, density_sizes = {}, [], [], []
    config_hash = hashlib.sha256(
        canonical_json(
            {
                "settings": settings,
                "amendment_sha256": parent.get("design_amendment_sha256"),
            }
        )
    ).hexdigest()
    for spec, name in zip(specs, names, strict=True):
        schedule = sorted(
            [r for r in parent["index"] if r["method"] == name], key=lambda r: r["origin"]
        )
        if len(schedule) != 60 or any(r["fit_status"] != "accepted" for r in schedule):
            raise ValueError("Every source origin must be accepted.")
        for h in (1, 3, 6, 12):
            eligible = [
                r for r in schedule if str(pd.Period(r["origin"], freq="M") + h) <= "2023-12"
            ][-6:]
            for record in eligible:
                selected.setdefault((name, record["origin"]), []).append(h)
        values = {h: {n: [] for n in (6000, 12000)} for h in (1, 3, 6, 12)}
        for record in schedule:
            started = time.perf_counter()
            source = artifact_path(args.source, record["directory"])
            target = artifact_path(args.output, record["directory"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("forecast.npz", "metadata.json")
            )
            generate = replay_factory(source, record, spec, wide, parent["category_ids"])
            forecast = generate(6000)
            forecast.save(target / "forecast.npz")
            horizons = selected.get((name, record["origin"]), [])
            if horizons:
                larger = generate(12000)
                for h in horizons:
                    y = wide.loc[str(pd.Period(record["origin"], freq="M") + h)].to_numpy(dtype=int)
                    for j in range(len(parent["category_ids"])):
                        values[h][6000].append(crps(forecast.paths[:, h - 1, j], y[j]))
                        values[h][12000].append(crps(larger.paths[:, h - 1, j], y[j]))
                        if forecast.family == "nb2_mixture":
                            first = forecast.logpmf(y[j], h - 1, j)
                            second = larger.logpmf(y[j], h - 1, j)
                            density_sizes.append(
                                {
                                    "method": name,
                                    "origin": record["origin"],
                                    "horizon": h,
                                    "category": parent["category_ids"][j],
                                    "log_mass_6000": first,
                                    "log_mass_12000": second,
                                    "difference": second - first,
                                }
                            )
            meta = {
                **record,
                "configuration_sha256": config_hash,
                "forecast_sha256": file_hash(target / "forecast.npz"),
                "original_fit_total_seconds": record["total_seconds"],
                "total_seconds": time.perf_counter() - started,
                "fit_reused_without_refitting": True,
                "fit_source_manifest_sha256": file_hash(parent_path),
                "simulation_algorithm": "joint Gaussian covariance"
                if spec["method"] == "arima"
                else "original seeded forecast recursion",
            }
            write_json(target / "metadata.json", meta)
            records.append(meta)
            print(name, record["origin"], flush=True)
        for h, sizes in values.items():
            means = {str(n): float(np.mean(v)) for n, v in sizes.items()}
            tolerance = 0.01 * max(1, means["12000"])
            checks.append(
                {
                    "method": name,
                    "horizon": h,
                    "mean_crps": means,
                    "tolerance": tolerance,
                    "passed": abs(means["6000"] - means["12000"]) <= tolerance,
                }
            )
    manifest = {
        **parent,
        "settings": settings,
        "configuration_sha256": config_hash,
        "index": records,
        "execution": {"refitted": False, "reporting_draws": 6000, "check_draws": 12000},
        "fit_source": {"path": parent_path.as_posix(), "sha256": file_hash(parent_path)},
        "artifacts": {
            p.relative_to(args.output).as_posix(): file_hash(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(args.output / "forecast_manifest.json", manifest)
    previous_check_path = args.source / "ensemble_check.json"
    previous_check = json.loads(previous_check_path.read_text())
    write_json(
        args.output / "ensemble_check.json",
        {
            "forecast_manifest_sha256": file_hash(args.output / "forecast_manifest.json"),
            "checks": checks,
            "all_passed": all(c["passed"] for c in checks),
            "budget_record": reference,
            "log_mass_size_sensitivity": density_sizes,
            "log_mass_seed_sensitivity": previous_check.get("log_mass_seed_sensitivity", []),
            "seed_check_source": {
                "path": previous_check_path.as_posix(),
                "sha256": file_hash(previous_check_path),
            },
            "log_mass_interpretation": (
                "Descriptive differences. The 6000-path independent-seed check is retained "
                "from the same posterior/recursion replay in the parent check."
            ),
        },
    )


if __name__ == "__main__":
    main()
