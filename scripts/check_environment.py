"""Small synthetic compatibility probe, not a cyber model or benchmark.

Run with: uv run --locked --extra science python scripts/check_environment.py
No source data or credentials are used. The exact Normal posterior is known.
"""

import importlib.metadata
import json
import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.parquet"
        frame = pd.DataFrame({"count": [0, 1, 2]})
        frame.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path))

    rf = RandomForestRegressor(n_estimators=5, random_state=1)
    rf.fit(np.arange(12).reshape(-1, 1), np.arange(12))
    assert np.isfinite(rf.predict([[12]])).all()
    series = np.random.default_rng(42).normal(size=40)
    arima = ARIMA(series, order=(1, 0, 0)).fit()
    assert np.isfinite(arima.forecast(3)).all()

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        pm.Normal("observed", mu=mu, sigma=1, observed=np.array([1.0, 2.0]))
        point = model.initial_point()
        assert np.isfinite(model.compile_logp()(point))
        assert np.isfinite(model.compile_dlogp()(point)).all()
        draws = pm.sample(
            draws=300,
            tune=300,
            chains=2,
            cores=1,
            random_seed=20261020,
            progressbar=False,
            target_accept=0.95,
        )
    mean = float(draws.posterior["mu"].mean())
    # Exact posterior is Normal(1, 1/3); a wide smoke tolerance avoids a fit-quality claim.
    assert abs(mean - 1.0) < 0.2
    assert int(draws.sample_stats["diverging"].sum()) == 0
    report = {
        "purpose": "synthetic environment compatibility only",
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "pandas",
                "pyarrow",
                "pymc",
                "pytensor",
                "arviz",
                "xarray",
                "scikit-learn",
                "statsmodels",
            )
        },
        "pytensor_cxx": bool(pytensor.config.cxx),
        "normal_posterior_mean": mean,
        "normal_posterior_expected_mean": 1.0,
        "rhat": float(az.rhat(draws)["mu"]),
        "divergences": int(draws.sample_stats["diverging"].sum()),
        "status": "passed",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
