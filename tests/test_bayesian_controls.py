import numpy as np
import pytest
import xarray as xr

from cyber_threat_modelling.models.bayesian_controls import (
    build_model,
    diagnostic_report,
    posterior_forecast,
)


@pytest.mark.parametrize("method", ["pooled_nb", "independent_nb_ar1"])
def test_bayesian_graph_has_finite_logp_and_gradient(method):
    y = np.column_stack([np.arange(8), np.zeros(8)])
    model = build_model(y, y[:4], method)
    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.isfinite(model.compile_dlogp()(point)).all()


def test_pooled_forecasts_keep_shared_parameter_draws():
    posterior = xr.Dataset(
        {
            "psi": (("chain", "draw", "category"), np.full((1, 2, 2), 2.0)),
            "b": (("chain", "draw", "category"), np.zeros((1, 2, 2))),
            "b0": (("chain", "draw"), np.array([[0.0, 3.0]])),
            "beta": (("chain", "draw"), np.zeros((1, 2))),
        }
    )
    forecast = posterior_forecast(posterior, "pooled_nb", 108, 3000, 5)
    assert np.array_equal(
        forecast.parameters["mean"][:, :, 0], forecast.parameters["mean"][:, :, 1]
    )
    assert len(np.unique(forecast.parameters["mean"])) == 2
    assert forecast.paths.shape == (3000, 12, 2)


def test_bad_chain_diagnostics_fail_gate():
    import arviz as az

    rng = np.random.default_rng(19)
    draws = rng.normal(size=(4, 1000)) + np.arange(4)[:, None] * 10
    idata = az.from_dict(
        posterior={"coefficient": draws},
        sample_stats={
            "diverging": np.zeros((4, 1000), dtype=bool),
            "tree_depth": np.ones((4, 1000)),
            "energy": rng.normal(size=(4, 1000)),
        },
    )
    report, _ = diagnostic_report(idata, 12)
    assert report["max_rhat"] > 1.01
    assert not report["accepted"]
