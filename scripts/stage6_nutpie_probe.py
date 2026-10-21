"""Execution-backend probe; no development scores or model selection."""

import argparse
import time
from pathlib import Path

import numpy as np
import nutpie
import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, training_prefix
from cyber_threat_modelling.models.bayesian_controls import build_model, diagnostic_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["pooled_nb", "independent_nb_ar1"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--center-prefix", type=int)
    parser.add_argument("--low-rank", action="store_true")
    parser.add_argument("--center-categories", type=int, nargs="+")
    parser.add_argument("--category", type=int, default=0)
    parser.add_argument("--retry", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--amended-prior", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    wide, counts = count_matrix(
        pd.read_parquet(root / "data/processed/stage5-20261020/topic_month.parquet")
    )
    train, initial = training_prefix(wide, "2018-12"), counts[:108]
    if args.category:
        j = list(wide.columns).index(args.category)
        train, initial = train[:, j : j + 1], initial[:, j : j + 1]
    model = build_model(train, initial, args.method)
    if args.center_prefix is not None or args.center_categories is not None:
        from cyber_threat_modelling.models.independent_geometry import build_conditioned_model

        model = build_conditioned_model(
            train,
            initial,
            center_states=np.array(args.center_categories) - 1
            if args.center_categories is not None
            else np.arange(args.center_prefix),
            stationary_scale_sd=0.75 if args.amended_prior else None,
        )
    start = time.perf_counter()
    compiled = nutpie.compile_pymc_model(model, backend="numba")
    compile_seconds = time.perf_counter() - start
    print(f"Compiled in {compile_seconds:.1f}s", flush=True)
    start = time.perf_counter()
    warmup, draws, accept, depth = (
        (1000, 1000, 0.95, 12),
        (2000, 1000, 0.99, 15),
        (2000, 2000, 0.99, 15),
    )[args.retry]
    idata = nutpie.sample(
        compiled,
        draws=draws,
        tune=warmup,
        chains=4,
        cores=1,
        seed=20261020 + args.retry,
        target_accept=accept,
        maxdepth=depth,
        progress_bar=False,
        **(
            {"low_rank_modified_mass_matrix": True, "mass_matrix_eigval_cutoff": 3.0}
            if args.low_rank
            else {}
        ),
    )
    sample_seconds = time.perf_counter() - start
    write_json(
        args.output / "timing.json",
        {"compile_seconds": compile_seconds, "sample_seconds": sample_seconds},
    )
    idata.to_netcdf(args.output / "posterior.nc")
    diag, table = diagnostic_report(idata, depth)
    table.to_csv(args.output / "diagnostics.csv")
    result = {
        "method": args.method,
        "prior": "stationary_scale_halfnormal_v1"
        if args.amended_prior
        else "original_innovation_scale",
        "retry_number": args.retry,
        "warmup": warmup,
        "draws": draws,
        "target_accept": accept,
        "max_treedepth": depth,
        "backend": "nutpie 0.15.2, numba",
        "mass_adaptation": "low rank, eigenvalue cutoff 3"
        if args.low_rank
        else "gradient diagonal",
        "compile_seconds": compile_seconds,
        "sample_seconds": sample_seconds,
        "diagnostics": diag,
        "exploratory_only": True,
    }
    write_json(args.output / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
