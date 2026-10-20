import numpy as np
import pytest
from scipy.stats import nbinom, norm, poisson

from cyber_threat_modelling.evaluation.scores import crps, score_aggregate, score_cell
from cyber_threat_modelling.forecasting.count_laws import (
    CountForecast,
    integer_counts,
    round_counts,
)
from cyber_threat_modelling.models.count_baselines import arima_candidates, fit_predict, stream_seed
from cyber_threat_modelling.models.ordering import loading_order


def law(family="rounded_normal", **parameters):
    return CountForecast(
        family,
        np.zeros((100, 1, 1), dtype=int),
        {k: np.array([[v]]) for k, v in parameters.items()},
    ).validate()


def test_crps_matches_pairwise_definition():
    x = np.array([0, 0, 1, 3, 11, 100])
    expected = np.abs(x - 2).mean() - np.abs(x[:, None] - x).mean() / 2
    assert crps(x, 2) == pytest.approx(expected)
    assert crps([2], 2) == 0


@pytest.mark.parametrize("location,scale", [(-3, 0.5), (0, 0.5), (17, 4), (100, 2)])
def test_rounded_count_mass_matches_cdf_and_normalizes(location, scale):
    f = law(location=location, scale=scale)
    probabilities = np.exp([f.logpmf(y, 0, 0) for y in range(200)])
    assert probabilities.sum() == pytest.approx(1, abs=1e-12)
    assert probabilities[0] == pytest.approx(norm.cdf((0.5 - location) / scale))
    assert np.cumsum(probabilities) == pytest.approx([f.cdf(y, 0, 0) for y in range(200)])


def test_extreme_tail_log_mass_is_not_floored_or_underflowed():
    f = law(location=0, scale=1)
    assert np.isfinite(f.logpmf(100, 0, 0))
    assert f.logpmf(100, 0, 0) < -4900
    f = law(location=100, scale=1)
    assert np.isfinite(f.logpmf(1, 0, 0))
    assert f.logpmf(1, 0, 0) < -4800


def test_nb_mixture_is_average_mass_not_moment_match():
    mu = np.array([1.0, 30.0])[:, None, None]
    alpha = np.array([2.0, 3.0])[:, None, None]
    f = CountForecast(
        "nb2_mixture", np.zeros((2, 1, 1)), {"mean": mu, "concentration": alpha}
    ).validate()
    for y in (0, 1, 15, 30, 1000):
        expected = np.mean(nbinom.pmf(y, alpha, alpha / (alpha + mu)))
        assert np.exp(f.logpmf(y, 0, 0)) == pytest.approx(expected)
    assert np.exp(f.logpmf(15, 0, 0)) != pytest.approx(nbinom.pmf(15, 2.5, 2.5 / (2.5 + 15.5)))
    tiny = CountForecast(
        "nb2_mixture",
        np.zeros((1, 1, 1)),
        {"mean": np.full((1, 1, 1), 1e-200), "concentration": np.ones((1, 1, 1))},
    ).validate()
    assert np.isfinite(tiny.logpmf(1, 0, 0))
    assert tiny.logpmf(1, 0, 0) == pytest.approx(np.log(1e-200))


def test_optimizer_retry_sequence_and_failed_fit_record(monkeypatch):
    from types import SimpleNamespace

    import cyber_threat_modelling.models.count_baselines as baselines

    calls = []

    class FakeModel:
        def fit(self, **kwargs):
            calls.append(kwargs["method_kwargs"])
            return SimpleNamespace(
                mle_retvals={"converged": len(calls) == 2}, llf=0.0, params=np.array([1.0])
            )

    monkeypatch.setattr(baselines, "ARIMA", lambda *args, **kwargs: FakeModel())
    with pytest.raises(baselines.FitFailure) as caught:
        baselines.fit_arima(np.ones(108), {"order": [0, 1, 0], "trend": "n"})
    assert [call["method"] for call in calls] == ["lbfgs", "powell", "lbfgs"]
    assert [call["maxiter"] for call in calls] == [1000, 2000, 1000]
    assert len(caught.value.attempts) == 3


def test_discrete_quantiles_pit_and_genuine_zero_mass():
    f = law("poisson", rate=3)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        k = f.quantile(q, 0, 0)
        assert f.cdf(k - 1, 0, 0) < q <= f.cdf(k, 0, 0)
        assert k == poisson.ppf(q, 3)
    metrics = score_cell(f, 2, 0, 0, np.random.default_rng(5))
    assert all(f.cdf(1, 0, 0) <= metrics[f"pit_{i}"] <= f.cdf(2, 0, 0) for i in range(10))
    degenerate = law(location=0, scale=0)
    assert degenerate.logpmf(0, 0, 0) == 0
    assert degenerate.logpmf(1, 0, 0) == -np.inf


def test_rounding_and_serialization(tmp_path):
    assert round_counts([-100, -0.5, 0, 0.49, 0.5, 1.5]).tolist() == [0, 0, 0, 0, 1, 2]
    f = law(location=1, scale=2)
    path = tmp_path / "forecast.npz"
    f.save(path)
    restored = CountForecast.load(path)
    assert restored.logpmf(0, 0, 0) == f.logpmf(0, 0, 0)
    with pytest.raises(FileExistsError):
        f.save(path)
    for bad in ([np.nan], [-1], [0.1], [np.inf], [2**63]):
        with pytest.raises(ValueError):
            integer_counts(bad)


def test_aggregate_preserves_path_dependence():
    samples = np.array([[0, 10], [10, 0]]).sum(axis=1)
    assert score_aggregate(samples, 10)["crps"] == 0
    assert score_aggregate(samples, 10)["width_90"] == 0


def test_initial_category_order_and_inverse():
    totals = [7343, 16043, 3583, 5127, 3517, 9317, 3313, 18685, 4359, 0]
    permutation, inverse = loading_order([totals], range(1, 11))
    ids = np.arange(1, 11)
    assert ids[permutation].tolist() == [8, 2, 6, 1, 4, 9, 3, 5, 7, 10]
    assert np.array_equal(ids[permutation][inverse], ids)


@pytest.mark.parametrize("method", ["historical_mean", "seasonal_naive"])
def test_zero_series_remains_observed_and_non_degenerate(method):
    f, info = fit_predict(np.zeros((108, 1)), [10], {"method": method}, 3000, "2018-12")
    assert info["fit_status"] == "accepted"
    assert 0 < f.cdf(0, 0, 0) < 1
    assert f.logpmf(0, 0, 0) < 0
    assert f.paths.max() > 0


def test_training_only_seasonal_scale_and_reproducible_streams():
    y = np.arange(108)[:, None]
    f, _ = fit_predict(y, [1], {"method": "seasonal_naive"}, 3000, "2018-12")
    assert np.all(f.parameters["scale"] == 12)
    assert np.array_equal(f.parameters["location"][:, 0], y[-12:, 0])
    again, _ = fit_predict(y, [1], {"method": "seasonal_naive"}, 3000, "2018-12")
    assert np.array_equal(f.paths, again.paths)
    assert stream_seed(1, "a", 1) != stream_seed(1, "a", 2)


def test_arima_grid_and_joint_random_walk_paths():
    assert len(arima_candidates()) == 65
    rng = np.random.default_rng(99)
    y = np.maximum(0, 100 + np.cumsum(rng.integers(-3, 4, 108)))[:, None]
    f, info = fit_predict(
        y, [1], {"method": "arima", "order": [0, 1, 0], "trend": "n"}, 6000, "2018-12"
    )
    assert info["fit_status"] == "accepted"
    assert f.paths[:, 0, 0].mean() == pytest.approx(f.parameters["location"][0, 0], abs=0.2)
    assert f.paths[:, 11, 0].std() == pytest.approx(f.parameters["scale"][11, 0], rel=0.06)
    assert np.corrcoef(f.paths[:, 10, 0], f.paths[:, 11, 0])[0, 1] > 0.9


@pytest.mark.parametrize("order,trend", [([0, 0, 0], "c"), ([0, 1, 0], "t")])
def test_arima_deterministic_terms_extend_into_future(order, trend):
    y = (100 + np.arange(108) * 2 + np.random.default_rng(5).integers(-3, 4, 108))[:, None]
    f, _ = fit_predict(y, [1], {"method": "arima", "order": order, "trend": trend}, 3000, "2018-12")
    assert np.all(np.abs(f.paths.mean(axis=0)[:, 0] - f.parameters["location"][:, 0]) < 5)
