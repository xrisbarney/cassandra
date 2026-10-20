"""Execution-backend probe; no development scores or model selection."""

import argparse
import time
from pathlib import Path

import nutpie
import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.backtest import count_matrix, training_prefix
from cyber_threat_modelling.models.bayesian_controls import build_model, diagnostic_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["pooled_nb", "independent_nb_ar1"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    wide, counts = count_matrix(
        pd.read_parquet(root / "data/processed/stage5-20261020/topic_month.parquet")
    )
    model = build_model(training_prefix(wide, "2018-12"), counts[:108], args.method)
    start = time.perf_counter()
    compiled = nutpie.compile_pymc_model(model, backend="numba")
    compile_seconds = time.perf_counter() - start
    print(f"Compiled in {compile_seconds:.1f}s", flush=True)
    start = time.perf_counter()
    idata = nutpie.sample(
        compiled,
        draws=1000,
        tune=1000,
        chains=4,
        cores=1,
        seed=20261020,
        target_accept=0.95,
        maxdepth=12,
        progress_bar=False,
    )
    sample_seconds = time.perf_counter() - start
    write_json(
        args.output / "timing.json",
        {"compile_seconds": compile_seconds, "sample_seconds": sample_seconds},
    )
    idata.to_netcdf(args.output / "posterior.nc")
    diag, table = diagnostic_report(idata, 12)
    table.to_csv(args.output / "diagnostics.csv")
    result = {
        "method": args.method,
        "backend": "nutpie 0.15.2, numba, diagonal gradient-based mass adaptation",
        "compile_seconds": compile_seconds,
        "sample_seconds": sample_seconds,
        "diagnostics": diag,
        "exploratory_only": True,
    }
    write_json(args.output / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
