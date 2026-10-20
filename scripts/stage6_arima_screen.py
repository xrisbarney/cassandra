"""Run the frozen 65-candidate, 20-origin ARIMA screen with explicit failures."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cyber_threat_modelling.config import canonical_json, load_config
from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, training_prefix
from cyber_threat_modelling.evaluation.scores import SCORING_VERSION, crps
from cyber_threat_modelling.features.build import verify_panels
from cyber_threat_modelling.models.count_baselines import (
    FitFailure,
    arima_candidates,
    fit_predict,
    method_id,
)
from cyber_threat_modelling.provenance import file_hash, prepare_run, verify_zero_series_amendment


def screen_candidate(task):
    candidate, wide, output, amendment_sha256, panel_sha256 = task
    name = method_id(candidate)
    rows = []
    for origin in pd.period_range("2018-12", "2023-09", freq="M")[::3]:
        folder = output / name / str(origin)
        folder.mkdir(parents=True, exist_ok=False)
        row = {
            "run_id": output.name,
            "method": name,
            "origin": str(origin),
            "training_cutoff": str(origin),
            "target": str(origin + 1),
            "horizon": 1,
            "split": "development",
            "vintage": "revised",
            "target_kind": "publication_count",
            "fit_status": "failed",
            "crps": None,
            "seed": 20261020,
            "amendment_sha256": amendment_sha256,
            "configuration_sha256": hashlib.sha256(
                canonical_json(
                    {
                        "candidate": candidate,
                        "amendment_sha256": amendment_sha256,
                    }
                )
            ).hexdigest(),
            "source_manifest_sha256": panel_sha256,
            "scoring_version": SCORING_VERSION,
            "observed_flag": True,
            "retry_number": None,
            "forecast_sha256": None,
        }
        try:
            forecast, fit = fit_predict(
                training_prefix(wide, origin), wide.columns, candidate, 3000, str(origin)
            )
            forecast.save(folder / "forecast.npz")
            write_json(folder / "fit.json", fit)
            # Persist the issued distribution before accessing its development target.
            target = wide.loc[str(origin + 1)].to_numpy(dtype=int)
            row.update(
                fit_status="accepted",
                crps=float(
                    np.mean([crps(forecast.paths[:, 0, j], target[j]) for j in range(len(target))])
                ),
                forecast_sha256=file_hash(folder / "forecast.npz"),
                retry_number=fit["retry_number"],
                boundary_categories=[
                    r["category"] for r in fit["per_category"] if "boundary_rule" in r
                ],
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
            write_json(
                folder / "failure.json",
                {
                    "error": str(exc),
                    "attempts": exc.attempts if isinstance(exc, FitFailure) else [],
                },
            )
            row["error"] = str(exc)
        row["eligible_for_scoring"] = row["fit_status"] == "accepted"
        row["eligibility_reason"] = (
            "development_count_checks_passed" if row["eligible_for_scoring"] else "fit_failed"
        )
        write_json(folder / "result.json", row)
        rows.append(row)
    write_json(output / name / "completed.json", {"records": rows})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    amendment = verify_zero_series_amendment(root, required=True)
    panel = root / "docs/panels/stage5-20261020/panel_manifest.json"
    verify_panels(root, panel)
    prepare_run(root, load_config(root / "configs/pilot.toml"), args.output)
    wide, _ = count_matrix(
        pd.read_parquet(root / "data/processed/stage5-20261020/topic_month.parquet")
    )
    rows = []
    write_json(
        args.output / "screen_settings.json",
        {
            "split": "development",
            "candidates": arima_candidates(),
            "draws": 3000,
            "origins": [str(o) for o in pd.period_range("2018-12", "2023-09", freq="M")[::3]],
            "workers": args.workers,
            "panel_manifest_sha256": file_hash(panel),
            "amendment": amendment,
            "held_out_scoring_enabled": False,
        },
    )
    tasks = [
        (c, wide, args.output, amendment["sha256"], file_hash(panel)) for c in arima_candidates()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(screen_candidate, task) for task in tasks]
        for future in as_completed(futures):
            completed = future.result()
            rows.extend(completed)
            print(
                completed[0]["method"],
                sum(r["fit_status"] == "accepted" for r in completed),
                "/20",
                flush=True,
            )
    rows.sort(key=lambda row: (row["method"], row["origin"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "screen.csv", index=False)
    summary = (
        frame.groupby("method")
        .agg(
            accepted=("fit_status", lambda x: int((x == "accepted").sum())),
            mean_crps=("crps", "mean"),
        )
        .reset_index()
    )
    grid = {
        method_id(spec): (spec["order"][0] + spec["order"][2] + 1 + int(spec["trend"] != "n"), i)
        for i, spec in enumerate(arima_candidates())
    }
    summary["parameter_count_per_series"] = summary.method.map(lambda name: grid[name][0])
    summary["grid_order"] = summary.method.map(lambda name: grid[name][1])
    valid = summary.loc[summary.accepted == 20].sort_values(
        ["mean_crps", "parameter_count_per_series", "grid_order"]
    )
    finalists = valid.head(2).method.tolist()
    summary.to_csv(args.output / "candidate_summary.csv", index=False)
    write_json(
        args.output / "selection.json",
        {
            "status": "screen_complete" if finalists else "no_valid_configuration",
            "candidates": 65,
            "origins_per_candidate": 20,
            "finalists": finalists,
            "monthly_finalist_evaluation_required": bool(finalists),
            "panel_manifest_sha256": file_hash(panel),
            "amendment_sha256": amendment["sha256"],
            "held_out_scoring_enabled": False,
            "artifacts": {
                p.relative_to(args.output).as_posix(): file_hash(p)
                for p in args.output.rglob("*")
                if p.is_file()
            },
        },
    )
    print(json.dumps({"finalists": finalists}), flush=True)


if __name__ == "__main__":
    main()
