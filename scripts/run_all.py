#!/usr/bin/env python3
"Run the entire CASSANDRA pipeline end-to-end, unattended."
from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr here (for run_all.py's own prints) AND...
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    os.environ["PYTHONUTF8"] = "1"

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def log(msg: str) -> None:
    print(f"[run_all] {msg}", flush=True)


def nuke() -> None:
    "Delete all downloaded data, processed features, and results."
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
    args = ap.parse_args()

    t0 = time.time()
    log("PIPELINE STARTING")
    if args.fresh:
        nuke()

    run("1/6 Collect data", ["scripts/ingest_data.py", "--start", args.start,
                             "--end", args.end, "--cache-dir", "data/cache/"])
    run("2/6 Prepare inputs", ["scripts/build_features.py"])

    train = ["scripts/train.py"]
    if args.quick:
        train += ["--num-warmup", "50", "--num-samples", "30"]
    run("3/6 Infer the posterior", train)

    run("4/6 Calibrate damage functions", ["scripts/calibrate_damage.py"])
    run("5/6 Check accuracy", ["scripts/evaluate.py"])

    forecast = ["scripts/forecast.py", "--horizon", "12"]
    run("6/6 Forecast", forecast)

    log(f"ALL DONE in {(time.time() - t0) / 60.0:.1f} min total.")


if __name__ == "__main__":
    main()
