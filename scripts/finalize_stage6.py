"""Assemble the Stage 6 audit only after all five initial controls pass their gates."""

import argparse
import json
from pathlib import Path

import pandas as pd

from cyber_threat_modelling.evaluation.backtest import verify_artifacts
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    arima = root / "artifacts/backtests/stage6-arima-finalists-20261021"
    selection = json.loads((arima / "selection.json").read_text())
    runs = [
        (root / "artifacts/backtests/stage6-20261020", ["historical_mean", "seasonal_naive"]),
        (root / "artifacts/backtests/stage6-pooled-20261021", ["pooled_nb"]),
        (arima, [selection["selected_method"]]),
        (root / args.independent_run, ["independent_nb_ar1"]),
    ]
    summaries, diagnoses, runtime, records = [], [], [], []
    for folder, methods in runs:
        manifest_path = folder / "forecast_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        verify_artifacts(folder, manifest)
        score_manifest_path = folder / "scores/score_manifest.json"
        score_manifest = json.loads(score_manifest_path.read_text())
        verify_artifacts(score_manifest_path.parent, score_manifest)
        if score_manifest["forecast_manifest_sha256"] != file_hash(manifest_path):
            raise ValueError("Score/forecast mismatch.")
        ensemble_path = folder / "ensemble_check.json"
        ensemble = json.loads(ensemble_path.read_text())
        if ensemble["forecast_manifest_sha256"] != file_hash(manifest_path):
            raise ValueError("Ensemble check/forecast mismatch.")
        checks = [c for c in ensemble["checks"] if c["method"] in methods]
        if len(checks) != 4 * len(methods) or not all(c["passed"] for c in checks):
            raise ValueError("The ensemble gate has not passed for every method and horizon.")
        frame = pd.read_csv(folder / "scores/summary.csv")
        summaries.append(frame.loc[frame.method.isin(methods)])
        for method in methods:
            schedule = [r for r in manifest["index"] if r["method"] == method]
            if len(schedule) != 60 or any(r["fit_status"] != "accepted" for r in schedule):
                raise ValueError(f"Incomplete development schedule: {method}")
            for record in schedule:
                fit = json.loads((folder / record["directory"] / "fit.json").read_text())
                for attempt in fit.get("attempts", []):
                    diagnoses.append(
                        {
                            "method": method,
                            "origin": record["origin"],
                            "retry_number": attempt["retry_number"],
                            **attempt["diagnostics"],
                        }
                    )
                    runtime.append(
                        {
                            "method": method,
                            "origin": record["origin"],
                            "retry_number": attempt["retry_number"],
                            **{
                                key: value
                                for key, value in attempt.items()
                                if key.endswith("seconds")
                            },
                        }
                    )
        records.append(
            {
                "run": folder.relative_to(root).as_posix(),
                "methods": methods,
                "forecast_manifest_sha256": file_hash(manifest_path),
                "score_manifest_sha256": file_hash(score_manifest_path),
                "ensemble_check_sha256": file_hash(ensemble_path),
                "run_provenance_sha256": file_hash(folder / "run.json"),
            }
        )
    summary = pd.concat(summaries, ignore_index=True)
    if not summary.status.eq("complete").all():
        raise ValueError("The final score table includes incomplete rows.")
    args.output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output / "summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(diagnoses).to_csv(
        args.output / "posterior_diagnostics.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(runtime).to_csv(
        args.output / "bayesian_runtime.csv", index=False, lineterminator="\n"
    )
    record = {
        "status": "stage6_initial_controls_complete",
        "runs": records,
        "arima_selection": selection,
        "held_out_scoring_enabled": False,
        "scope": (
            "Initial five controls and development runner; full requested comparator "
            "suite and proposed-model selection remain later gates."
        ),
        "artifacts": {p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()},
    }
    (args.output / "completion.json").write_bytes((json.dumps(record, indent=2) + "\n").encode())
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
