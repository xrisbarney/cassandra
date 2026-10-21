import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import nbinom

from cyber_threat_modelling.models.observation_nb import build_model, forecast, simulate
from cyber_threat_modelling.provenance import verify_observation_control_amendment


def test_conditional_likelihood_matches_independent_nb_calculation_and_gradients():
    y = np.array([[0, 10], [2, 5], [0, 11], [6, 2], [1, 9]])
    model = build_model(y, y[:3])
    point = model.initial_point()
    point["m"] = np.array([0.7, 1.8])
    point["phi_interval__"] = np.array([-0.5, 1.0])
    point["log_psi"] = np.log([2.0, 4.0])
    phi = 2 * expit(point["phi_interval__"]) - 1
    mean = np.exp((1 - phi) * point["m"] + phi * np.log1p(y[:-1]))
    size = np.exp(point["log_psi"])
    expected = nbinom.logpmf(y[1:], size, size / (size + mean)).sum()
    assert np.allclose(model.compile_logp(vars=model.observed_RVs)(point), expected, atol=1e-10)
    assert np.allclose(
        model.compile_dlogp()(point), model.compile_dlogp(mode="NUMBA")(point), atol=1e-9
    )
    separate = 0.0
    for j in range(2):
        single = build_model(y[:, j : j + 1], y[:3, j : j + 1])
        separate += single.compile_logp()({key: value[j : j + 1] for key, value in point.items()})
    assert np.allclose(separate, model.compile_logp()(point), atol=1e-10)


def test_recursive_paths_and_two_step_marginal_probability():
    n, m, phi, size, first = 40000, np.log(4), 0.6, 3.0, 2
    f = simulate(
        np.full((n, 1), m),
        np.full((n, 1), phi),
        np.full((n, 1), size),
        [first],
        2,
        np.random.default_rng(831),
    ).validate()
    mu1 = np.exp((1 - phi) * m + phi * np.log1p(first))
    assert np.allclose(f.parameters["mean"][:, 0, 0], mu1)
    assert np.allclose(
        f.parameters["mean"][:, 1, 0],
        np.exp((1 - phi) * m + phi * np.log1p(f.paths[:, 0, 0])),
    )
    previous = np.arange(300)
    mu2 = np.exp((1 - phi) * m + phi * np.log1p(previous))
    mass1 = nbinom.pmf(previous, size, size / (size + mu1))
    assert abs(mass1.sum() - 1) < 1e-12
    for observed in (0, 2, 5):
        exact = (mass1 * nbinom.pmf(observed, size, size / (size + mu2))).sum()
        assert abs(np.exp(f.logpmf(observed, 1, 0)) - exact) < 0.002


def test_zero_training_counts_remain_informative_and_no_zero_forecast_exception():
    y = np.zeros((12, 1), dtype=int)
    model = build_model(y, y[:8])
    likelihood = model.compile_logp(vars=model.observed_RVs)
    point = model.initial_point()
    at_anchor = likelihood(point)
    point["m"] = point["m"] - 3
    assert likelihood(point) > at_anchor
    f = simulate(
        np.full((10000, 1), -2.0),
        np.zeros((10000, 1)),
        np.ones((10000, 1)),
        [0],
        2,
        np.random.default_rng(83),
    ).validate()
    assert (f.paths > 0).any()
    assert np.isfinite(f.logpmf(1, 0, 0))
    assert abs(f.paths[:, 0, 0].mean() - np.exp(-2)) < 0.015


def test_forecasts_replay_and_independent_categories_preserve_parameter_tuples():
    import arviz as az

    shape = (4, 1000, 2)
    m = np.broadcast_to(np.tile([0.0, 2.0], 500)[None, :, None], shape)
    post = az.from_dict(
        posterior={"m": m, "phi": np.zeros(shape), "psi": np.exp(m)},
        dims={key: ["category"] for key in ("m", "phi", "psi")},
        coords={"category": [0, 1]},
    ).posterior
    a, b = forecast(post, [0, 0], 3000, 78), forecast(post, [0, 0], 3000, 78)
    assert np.array_equal(a.paths, b.paths)
    assert np.array_equal(a.parameters["mean"], a.parameters["concentration"])
    assert abs(np.corrcoef(a.parameters["mean"][:, 0, :].T)[0, 1]) < 0.06
    with pytest.raises(ValueError, match="last observed"):
        forecast(post, None, 3000, 78)


def test_observation_amendment_required_and_hash_protected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    record = verify_observation_control_amendment(root, required=True)
    with pytest.raises(ValueError, match="required"):
        verify_observation_control_amendment(tmp_path, required=True)
    for item in record["record"]["files"]:
        dest = tmp_path / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root / item["path"]).read_bytes())
    (tmp_path / "docs/STAGE2_FREEZE.json").write_bytes(
        (root / "docs/STAGE2_FREEZE.json").read_bytes()
    )
    path = tmp_path / record["path"]
    path.write_text(json.dumps(record["record"]))
    assert verify_observation_control_amendment(tmp_path, required=True)
    dest.write_text("changed law")
    with pytest.raises(ValueError, match="changed"):
        verify_observation_control_amendment(tmp_path, required=True)
