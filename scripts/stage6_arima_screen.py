"""Run the frozen 65-candidate, 20-origin ARIMA screen with explicit failures."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cyber_threat_modelling.config import load_config
from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, training_prefix
from cyber_threat_modelling.evaluation.scores import crps
from cyber_threat_modelling.features.build import verify_panels
from cyber_threat_modelling.models.count_baselines import (
    FitFailure,
    arima_candidates,
    fit_predict,
    method_id,
)
from cyber_threat_modelling.provenance import file_hash, prepare_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise RuntimeError(
        "ARIMA screening is locked pending the dated zero-series decision. "
        "See docs/STAGE6_BOUNDARY_AMENDMENT_DRAFT.md."
    )
    root = Path(__file__).resolve().parents[1]
    panel = root / "docs/panels/stage5-20261020/panel_manifest.json"
    verify_panels(root, panel)
    prepare_run(root, load_config(root / "configs/pilot.toml"), args.output)
    wide, _ = count_matrix(
        pd.read_parquet(root / "data/processed/stage5-20261020/topic_month.parquet")
    )
    rows = []
    for candidate in arima_candidates():
        name = method_id(candidate)
        for origin in pd.period_range("2018-12", "2023-09", freq="M")[::3]:
            folder = args.output / name / str(origin)
            folder.mkdir(parents=True)
            row = {"method": name, "origin": str(origin), "fit_status": "failed", "crps": None}
            try:
                forecast, fit = fit_predict(
                    training_prefix(wide, origin), wide.columns, candidate, 3000, str(origin)
                )
                forecast.save(folder / "forecast.npz")
                write_json(folder / "fit.json", fit)
                # Each issued distribution is persisted before its development score.
                target = wide.loc[str(origin + 1)].to_numpy(dtype=int)
                row.update(
                    fit_status="accepted",
                    crps=float(
                        np.mean(
                            [crps(forecast.paths[:, 0, j], target[j]) for j in range(len(target))]
                        )
                    ),
                    forecast_sha256=file_hash(folder / "forecast.npz"),
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                write_json(
                    folder / "failure.json",
                    {
                        "error": str(exc),
                        "attempts": exc.attempts if isinstance(exc, FitFailure) else [],
                    },
                )
                row["error"] = str(exc)
            write_json(folder / "result.json", row)
            rows.append(row)
        print(
            name,
            sum(r["fit_status"] == "accepted" for r in rows if r["method"] == name),
            "/20",
            flush=True,
        )
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
