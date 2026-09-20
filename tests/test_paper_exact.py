import numpy as np
import pandas as pd
from numpyro.handlers import seed, trace

from cassandra_threatcast.data.incident_losses import load_monthly_loss_marks
from cassandra_threatcast.evaluation.posterior_predictive import posterior_predictive_checks
from cassandra_threatcast.model.paper_exact import (
    paper_config, paper_model, recover_regime_paths,
)


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


def test_paper_config_pins_manuscript_model():
    cfg = paper_config({
        "model": {"K": 2},
        "hawkes": {"enabled": True},
        "enhanced": {"enabled": True},
        "kernel": {"enabled": True},
        "ablation": {"drop_E": True},
    })
    assert "hawkes" not in cfg                     # §3.2: only "a natural alternative"
    assert cfg["enhanced"]["enabled"] is False     # §3.2: Gaussian innovations
    assert cfg["kernel"]["enabled"] is True        # §4.3: optional, config-driven
    assert cfg["ablation"]["drop_E"] is True       # §4.3: Table 4 variants pass through


def test_paper_model_marginalizes_regimes_and_builds_all_channels():
    cfg = {"model": {"K": 2, "S": 2, "r": 1, "r_sigma": 1, "R": 2}}
    sites = trace(seed(paper_model, 3)).get_trace(_tiny_data(), cfg)
    # §3.5: no categorical variable is ever sampled.
    assert not any(s.get("type") == "sample" and s["name"].startswith("z")
                   for s in sites.values())
    assert np.asarray(sites["loglik_regime_t"]["value"]).shape == (4, 2)
    assert "marginal_regime_lik" in sites
    assert {"N_obs", "E_obs", "B_obs", "D_obs"}.issubset(sites)


def test_recover_regime_paths_shapes_and_labels():
    rng = np.random.default_rng(0)
    n, T, R = 3, 5, 2
    posterior = {
        "loglik_regime_t": rng.normal(size=(n, T, R)),
        "Pi": np.full((n, R, R), 0.5),
    }
    z = recover_regime_paths(posterior, seed=1)
    assert z.shape == (n, T)
    assert z.dtype.kind == "i"
    assert set(np.unique(z)).issubset({0, 1})


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
