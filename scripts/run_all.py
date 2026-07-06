#!/usr/bin/env python3
"""
Run the entire CASSANDRA pipeline end-to-end, unattended.

Collect data -> Prepare inputs -> Train -> Evaluate -> Forecast, in order,
stopping at the first failure. Optionally wipes all caches and outputs first
(--fresh) for a true from-scratch run. Designed to be launched and left running
(e.g. overnight); it logs clear step markers the dashboard can read.

    python scripts/run_all.py --fresh            # full fresh run
    python scripts/run_all.py --quick            # quick preview training
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def log(msg: str) -> None:
    print(f"[run_all] {msg}", flush=True)


def nuke() -> None:
    """Delete all downloaded data, processed features, and results."""
    for d in ("data/cache", "data/processed", "results"):
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
    log("FRESH START: cleared data/cache, data/processed, and results.")


def run(step: str, cmd: list[str]) -> None:
    log(f"START {step}")
    t0 = time.time()
    result = subprocess.run([PY, *cmd], cwd=str(ROOT))
    mins = (time.time() - t0) / 60.0
    if result.returncode != 0:
        log(f"FAILED {step} (exit code {result.returncode}) after {mins:.1f} min")
        log("Pipeline stopped. Fix the problem above and run again.")
        sys.exit(result.returncode)
    log(f"DONE {step} in {mins:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole pipeline end-to-end.")
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--fresh", action="store_true", help="Delete all caches/outputs first.")
    ap.add_argument("--quick", action="store_true", help="Quick-preview training settings.")
    ap.add_argument("--enhanced", action="store_true", help="Use the enhanced model variant.")
    args = ap.parse_args()

    t0 = time.time()
    log("PIPELINE STARTING")
    if args.fresh:
        nuke()

    run("1/5 Collect data", ["scripts/ingest_data.py", "--start", args.start,
                             "--end", args.end, "--cache-dir", "data/cache/"])
    run("2/5 Prepare inputs", ["scripts/build_features.py"])

    train = ["scripts/train.py"]
    if args.quick:
        train += ["--num-warmup", "150", "--num-samples", "150", "--num-chains", "1"]
    if args.enhanced:
        train += ["--enhanced-mode"]
    run("3/5 Train the model", train)

    run("4/5 Check accuracy", ["scripts/evaluate.py"])

    forecast = ["scripts/forecast.py", "--horizon", "12"]
    if args.enhanced:
        forecast += ["--enhanced-mode"]
    run("5/5 Forecast", forecast)

    log(f"ALL DONE in {(time.time() - t0) / 60.0:.1f} min total.")


if __name__ == "__main__":
    main()
