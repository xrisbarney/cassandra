"""Fit a Stage 6 control at the initial development origin; never read test outcomes."""

import argparse
from pathlib import Path

import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, training_prefix
from cyber_threat_modelling.models.bayesian_controls import fit_predict_bayesian
from cyber_threat_modelling.models.count_baselines import FitFailure
from cyber_threat_modelling.provenance import (
    verify_independent_control_amendment,
    verify_observation_control_amendment,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method", choices=["independent_nb_ar1", "independent_nb_glar", "pooled_nb"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    if args.method == "independent_nb_glar":
        write_json(
            args.output / "model_amendment.json",
            verify_observation_control_amendment(root, required=True),
        )
    if args.method == "independent_nb_ar1":
        write_json(
            args.output / "prior_amendment.json",
            verify_independent_control_amendment(root, required=True),
        )
    wide, counts = count_matrix(
        pd.read_parquet(root / "data/processed/stage5-20261020/topic_month.parquet")
    )
    try:
        forecast, result = fit_predict_bayesian(
            training_prefix(wide, "2018-12"),
            counts[:108],
            {"method": args.method},
            3000,
            "2018-12",
            args.output,
        )
        forecast.save(args.output / "forecast.npz")
        write_json(args.output / "result.json", result)
        print(result, flush=True)
    except FitFailure as exc:
        write_json(args.output / "result.json", {"fit_status": "failed", "attempts": exc.attempts})
        print(str(exc), flush=True)


if __name__ == "__main__":
    main()
