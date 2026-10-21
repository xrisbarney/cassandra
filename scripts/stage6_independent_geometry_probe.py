"""Check exact per-category posterior factorization with dense NUTS adaptation."""

import argparse
import time
from pathlib import Path

import pandas as pd
import pymc as pm

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, training_prefix
from cyber_threat_modelling.models.bayesian_controls import build_model, diagnostic_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    wide, counts = count_matrix(
        pd.read_parquet(root / "data/processed/stage5-20261020/topic_month.parquet")
    )
    j = list(wide.columns).index(args.category)
    train = training_prefix(wide, "2018-12")[:, j : j + 1]
    model = build_model(train, counts[:108, j : j + 1], "independent_nb_ar1")
    budgets = ((1000, 1000, 0.95, 12), (2000, 1000, 0.99, 15), (2000, 2000, 0.99, 15))
    for retry, (warmup, draws, accept, depth) in enumerate(budgets):
        started = time.perf_counter()
        with model:
            points, step = pm.init_nuts(
                init="jitter+adapt_full",
                chains=4,
                random_seed=20261021 + retry,
                tune=warmup,
                target_accept=accept,
                max_treedepth=depth,
                compile_kwargs={"mode": "NUMBA"},
            )
            compiled = time.perf_counter() - started
            idata = pm.sample(
                tune=warmup,
                draws=draws,
                chains=4,
                cores=1,
                step=step,
                initvals=points,
                random_seed=20261021 + retry,
                progressbar=False,
                compute_convergence_checks=False,
            )
        idata.to_netcdf(args.output / f"posterior_attempt_{retry}.nc")
        report, summary = diagnostic_report(idata, depth)
        report.update(compile_seconds=compiled, wall_seconds=time.perf_counter() - started)
        write_json(args.output / f"attempt_{retry}.json", report)
        summary.to_csv(args.output / f"summary_{retry}.csv")
        print(report, flush=True)
        if report["accepted"]:
            break


if __name__ == "__main__":
    main()
