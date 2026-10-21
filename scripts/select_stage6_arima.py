"""Finalize the frozen ARIMA selection from complete monthly development fits."""

import argparse
import json
from pathlib import Path

import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import verify_artifacts, verify_finalists
from cyber_threat_modelling.models.count_baselines import arima_candidates, method_id
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.run / "selection.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((args.run / "forecast_manifest.json").read_text())
    verify_finalists(root, manifest["settings"])
    verify_artifacts(args.run, manifest)
    score_path = args.run / "scores/score_manifest.json"
    scores = json.loads(score_path.read_text())
    if scores["forecast_manifest_sha256"] != file_hash(args.run / "forecast_manifest.json"):
        raise ValueError("Scores belong to a different forecast run.")
    verify_artifacts(score_path.parent, scores)
    summary = pd.read_csv(score_path.parent / "summary.csv")
    print(summary.columns.tolist(), flush=True)
    # Selection uses the primary all-category horizon-one development score.
    rows = summary.query("stratum == 'all' and horizon == 1 and window == 'available'").copy()
    grid = {method_id(spec): (spec, i) for i, spec in enumerate(arima_candidates())}
    rows["parameter_count_per_series"] = rows.method.map(
        lambda name: sum(grid[name][0]["order"][::2]) + 1 + int(grid[name][0]["trend"] != "n")
    )
    rows["grid_order"] = rows.method.map(lambda name: grid[name][1])
    valid = rows.dropna(subset=["crps"]).sort_values(
        ["crps", "parameter_count_per_series", "grid_order"]
    )
    if valid.empty:
        raise ValueError("No finalist completed its monthly schedule.")
    selected = valid.iloc[0].method
    write_json(
        output,
        {
            "status": "development_selection_complete",
            "selected_method": selected,
            "selected_specification": grid[selected][0],
            "ranking": valid[
                ["method", "crps", "parameter_count_per_series", "grid_order"]
            ].to_dict("records"),
            "criterion": (
                "All-category mean h=1 CRPS over 60 monthly development origins; "
                "parameter-count then grid-order ties"
            ),
            "screening_record": manifest["settings"]["screening_record"],
            "forecast_manifest_sha256": file_hash(args.run / "forecast_manifest.json"),
            "score_manifest_sha256": file_hash(score_path),
            "held_out_scoring_enabled": False,
        },
    )
    print(selected, flush=True)


if __name__ == "__main__":
    main()
