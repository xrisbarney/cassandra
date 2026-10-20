"""Verify native PyTensor compilation and a timed toy NUTS fit, without research data."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output file; compiler reports are not overwritten.")
    if not pytensor.config.cxx:
        raise RuntimeError("PyTensor has no C++ compiler configured.")
    # Windows auto-detection quotes the path for PyTensor's shell command builder.
    compiler_executable = pytensor.config.cxx.strip('"')
    version = subprocess.check_output([compiler_executable, "--version"], text=True).splitlines()[0]
    started = time.perf_counter()
    x = pt.dscalar("x")
    expression = x * x + 2 * x
    native = pytensor.function(
        [x], [expression, pt.grad(expression, x)], mode=Mode(linker="c", optimizer="fast_run")
    )
    # A C-only linker must fail instead of silently running Python implementations.
    assert hasattr(native.vm, "cthunk")
    np.testing.assert_allclose(native(3.0), [15.0, 8.0])
    native_compile_seconds = time.perf_counter() - started

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("observed", mu, 1, observed=np.array([1.0, 2.0]))
        started = time.perf_counter()
        mode = Mode(linker="c", optimizer="fast_run")
        logp = model.compile_logp(mode=mode)
        gradient = model.compile_dlogp(mode=mode)
        assert hasattr(logp.f.vm, "cthunk") and hasattr(gradient.f.vm, "cthunk")
        at_zero = {"mu": np.array(0.0)}
        at_one = {"mu": np.array(1.0)}
        assert np.isfinite(logp(at_zero))
        np.testing.assert_allclose(gradient(at_zero), [3.0])
        np.testing.assert_allclose(gradient(at_one), [0.0], atol=1e-12)
        model_compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        draws = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            random_seed=20261020,
            target_accept=0.95,
            nuts={"max_treedepth": 12},
            progressbar=False,
        )
        sampling_wall_seconds = time.perf_counter() - started

    report = {
        "purpose": "compiler and toy sampler verification; not Stage 7 model recovery",
        "compiler": version,
        "cxx": pytensor.config.cxx,
        "pytensor_version": pytensor.__version__,
        "pymc_version": pm.__version__,
        "native_cthunk": True,
        "logp_cthunk": True,
        "gradient_cthunk": True,
        "native_compile_seconds": native_compile_seconds,
        "model_compile_seconds": model_compile_seconds,
        "sampling_wall_seconds_including_initialization": sampling_wall_seconds,
        "chains": 4,
        "cores": 4,
        "warmup_per_chain": 1000,
        "draws_per_chain": 1000,
        "seed": 20261020,
        "target_accept": 0.95,
        "max_treedepth": 12,
        "posterior_mean": float(draws.posterior.mu.mean()),
        "exact_mean": 1.0,
        "posterior_variance": float(draws.posterior.mu.var()),
        "exact_variance": 1 / 3,
        "rhat": float(az.rhat(draws)["mu"]),
        "ess_bulk": float(az.ess(draws, method="bulk")["mu"]),
        "ess_tail": float(az.ess(draws, method="tail")["mu"]),
        "mcse_mean": float(az.mcse(draws, method="mean")["mu"]),
        "bfmi_min": float(np.min(az.bfmi(draws))),
        "divergences": int(draws.sample_stats.diverging.sum()),
        "tree_depth_saturations": int((draws.sample_stats.tree_depth >= 12).sum()),
    }
    checks = [
        report["rhat"] < 1.01,
        report["ess_bulk"] >= 400,
        report["ess_tail"] >= 400,
        report["divergences"] == 0,
        report["tree_depth_saturations"] == 0,
        report["bfmi_min"] > 0.3,
        abs(report["posterior_mean"] - 1) < 5 * report["mcse_mean"],
        abs(report["posterior_variance"] - 1 / 3) < 0.08,
    ]
    report["status"] = "passed" if all(checks) else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(checks):
        raise RuntimeError("Toy compiler/sampler checks failed; inspect the report.")


if __name__ == "__main__":
    main()
