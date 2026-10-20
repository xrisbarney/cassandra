import numpy as np
import pytest
import xarray as xr

from cyber_threat_modelling.models.bayesian_controls import (
    build_model,
    diagnostic_report,
    pooled_prior_covariance,
    posterior_forecast,
)


def test_centered_pooled_parameterization_preserves_frozen_joint_prior():
    k, center, mean = 10, 4.5, 3.0
    rng = np.random.default_rng(44)
    coefficients = rng.multivariate_normal(
        np.r_[np.full(k, mean), 0.0], pooled_prior_covariance(k, center), size=150000
    )
    beta = coefficients[:, -1]
    alpha = coefficients[:, :k] - center * beta[:, None]
    b0 = (
        mean
        + 4 / (1 + 4 * k) * np.sum(alpha - mean, axis=1)
        + np.sqrt(4 / (1 + 4 * k)) * rng.normal(size=len(alpha))
    )
    b = alpha - b0[:, None]
    recovered = np.column_stack([b0 - mean, b, beta])
    assert np.max(np.abs(recovered.mean(axis=0))) < 0.02
    assert np.max(np.abs(np.cov(recovered, rowvar=False) - np.diag([4] + [1] * k + [0.25]))) < 0.04


@pytest.mark.parametrize("method", ["pooled_nb", "independent_nb_ar1"])
def test_bayesian_graph_has_finite_logp_and_gradient(method):
    y = np.column_stack([np.arange(8), np.zeros(8)])
    model = build_model(y, y[:4], method)
    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.isfinite(model.compile_dlogp()(point)).all()


@pytest.mark.parametrize("method", ["pooled_nb", "independent_nb_ar1"])
def test_numba_log_density_and_gradient_agree_with_native_backend(method):
    y = np.column_stack([np.arange(8), np.zeros(8)])
    model = build_model(y, y[:4], method)
    point = model.initial_point()
    rng = np.random.default_rng(947)
    point = {key: value + rng.normal(0, 0.2, size=value.shape) for key, value in point.items()}
    assert np.allclose(
        model.compile_logp(mode="NUMBA")(point), model.compile_logp()(point), rtol=1e-10, atol=1e-10
    )
    assert np.allclose(
        model.compile_dlogp(mode="NUMBA")(point), model.compile_dlogp()(point), rtol=1e-9, atol=1e-9
    )


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


def test_predictive_checks_preserve_observed_zeros_and_separate_prior_posterior():
    from cyber_threat_modelling.evaluation.control_checks import predictive_checks

    posterior = xr.Dataset(
        {
            "psi": (("chain", "draw", "category"), np.full((1, 20, 2), 3.0)),
            "b": (("chain", "draw", "category"), np.tile([1.0, -7.0], (1, 20, 1))),
            "b0": (("chain", "draw"), np.full((1, 20), 2.0)),
            "beta": (("chain", "draw"), np.zeros((1, 20))),
        }
    )
    y = np.column_stack([np.full(60, 20), np.zeros(60)])
    checks = predictive_checks(posterior, "pooled_nb", y, y[:48], "2018-12", draws=300)
    zero = checks["categories"][1]["statistics"]
    assert zero["zero_fraction"]["observed"] == 1
    assert zero["mean"]["observed"] == 0
    assert zero["mean"]["posterior_05_50_95"][1] < 1
    assert zero["mean"]["prior_05_50_95"][1] > 1


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


def test_nutpie_depth_flag_is_not_silently_omitted():
    import arviz as az

    rng = np.random.default_rng(27)
    flags = np.zeros((4, 1000), dtype=bool)
    flags[1, 10] = True
    idata = az.from_dict(
        posterior={"coefficient": rng.normal(size=(4, 1000))},
        sample_stats={
            "diverging": np.zeros((4, 1000), dtype=bool),
            "maxdepth_reached": flags,
            "energy": rng.normal(size=(4, 1000)),
        },
    )
    report, _ = diagnostic_report(idata, 12)
    assert report["tree_depth_saturations"] == 1
    assert not report["accepted"]
    idata.sample_stats = idata.sample_stats.drop_vars("maxdepth_reached")
    with pytest.raises(ValueError, match="maximum-tree-depth"):
        diagnostic_report(idata, 12)
