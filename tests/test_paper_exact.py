import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.handlers import seed, trace

from cassandra_threatcast.data.incident_losses import load_monthly_loss_marks
from cassandra_threatcast.evaluation.posterior_predictive import posterior_predictive_checks
from cassandra_threatcast.model.paper_exact import paper_model, reporting_propensity


def _tiny_data():
    return {
        "N": np.ones((2, 4)),
        "B": np.full((2, 4), 5.0),
        "E": np.full((2, 4), 0.1),
        "D": np.zeros((2, 4)),
        "M_skt": np.full((2, 2, 4), 0.5),
        "e_t": np.zeros(4),
        "mandatory_t": np.array([0.0, 0.0, 1.0, 1.0]),
    }


def test_paper_model_orders_regimes_and_builds_all_channels():
    cfg = {"model": {"K": 2, "S": 2, "r": 1, "r_sigma": 1, "R": 2}}
    sites = trace(seed(paper_model, 3)).get_trace(
        _tiny_data(), cfg, jnp.array([0, 0, 1, 1])
    )
    means = np.asarray(sites["mu_r"]["value"]).mean(axis=1)
    assert np.all(np.diff(means) >= 0)
    assert np.asarray(sites["lambda_t"]["value"]).shape == (4, 2)
    assert {"N_obs", "E_obs", "B_obs", "D_obs"}.issubset(sites)


def test_reporting_propensity_is_sector_and_time_varying():
    pi = reporting_propensity(
        jnp.array([-3.0, -2.0]), jnp.array([0.0, 0.2]), 1.0, jnp.array([0.0, 1.0])
    )
    assert pi.shape == (2, 2)
    assert np.all(np.asarray(pi[:, 1]) > np.asarray(pi[:, 0]))


def test_curated_loss_marks_are_aggregated(tmp_path):
    source = tmp_path / "losses.csv"
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-15"],
        "sector_idx": [1, 1],
        "loss_usd": [10.0, 15.0],
    }).to_csv(source, index=False)
    values = load_monthly_loss_marks(source, pd.period_range("2024-01", periods=2, freq="M"), 2)
    assert values[1, 0] == 25.0
    assert np.isnan(values[0, 0])


def test_posterior_predictive_checks_cover_each_channel():
    obs = {"N": np.array([[0.0, 1.0]]), "D": np.array([[0.0, 0.0]])}
    pred = {"N": np.array([[[0.0, 1.0]], [[1.0, 1.0]]]),
            "D": np.zeros((2, 1, 2))}
    report = posterior_predictive_checks(obs, pred)
    assert set(report["channel"]) == {"N", "D"}
    assert set(report["statistic"]) == {"mean", "variance", "zero_rate"}
