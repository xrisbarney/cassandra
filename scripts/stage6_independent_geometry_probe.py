"""Check exact per-category posterior factorization with dense NUTS adaptation."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, training_prefix
from cyber_threat_modelling.models.bayesian_controls import build_model, diagnostic_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition-initial", action="store_true")
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--center-states", action="store_true")
    parser.add_argument("--center-prefix", type=int)
    parser.add_argument("--center-categories", type=int, nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    wide, counts = count_matrix(
        pd.read_parquet(root / "data/processed/stage5-20261020/topic_month.parquet")
    )
    indices = (
        list(range(len(wide.columns)))
        if args.category == 0
        else [list(wide.columns).index(args.category)]
    )
    train = training_prefix(wide, "2018-12")[:, indices]
    initial = counts[:108, indices]
    model = build_model(train, initial, "independent_nb_ar1")
    if args.condition_initial:
        from cyber_threat_modelling.models.independent_geometry import build_conditioned_model

        centers = np.flatnonzero(initial.mean(axis=0) > 0) if args.center_states else False
        if args.center_prefix is not None:
            centers = np.arange(args.center_prefix)
        if args.center_categories is not None:
            centers = np.array(args.center_categories) - 1
        model = build_conditioned_model(train, initial, center_states=centers)
    budgets = ((1000, 1000, 0.95, 12), (2000, 1000, 0.99, 15), (2000, 2000, 0.99, 15))
    for retry, (warmup, draws, accept, depth) in enumerate(budgets):
        started = time.perf_counter()
        with model:
            points, step = pm.init_nuts(
                init="jitter+adapt_diag" if args.diagonal else "jitter+adapt_full",
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
                callback=lambda trace, draw: (
                    print(f"chain {draw.chain} draw {draw.draw_idx + 1}", flush=True)
                    if (draw.draw_idx + 1) % 500 == 0
                    else None
                ),
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
