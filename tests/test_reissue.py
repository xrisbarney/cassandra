import json
from pathlib import Path

import numpy as np
import pytest
from statsmodels.tsa.arima.model import ARIMA

from cyber_threat_modelling.evaluation.backtest import validate_settings
from cyber_threat_modelling.evaluation.reissue import (
    gaussian_forecast_covariance,
    verify_budget_record,
)
from cyber_threat_modelling.provenance import file_hash


@pytest.mark.parametrize(
    "order,parameters,phi", [((0, 1, 0), [2.0], 1.0), ((1, 0, 0), [0.6, 2.0], 0.6)]
)
def test_joint_forecast_covariance_matches_closed_form(order, parameters, phi):
    y = np.random.default_rng(4).normal(size=100)
    result = ARIMA(y, order=order, trend="n").filter(parameters)
    actual = gaussian_forecast_covariance(result, 12)
    expected = np.array(
        [
            [
                2 * phi ** abs(i - j) * sum(phi ** (2 * k) for k in range(min(i, j) + 1))
                for j in range(12)
            ]
            for i in range(12)
        ]
    )
    assert np.allclose(actual, expected, atol=1e-8)
    np.linalg.cholesky(actual)


def test_selected_arima_joint_covariance_matches_native_conditional_simulation():
    y = np.cumsum(np.random.default_rng(51).normal(size=108))
    model = ARIMA(y, order=(2, 1, 1), trend="n").filter([0.4, 0.15, -0.3, 2.0])
    covariance = gaussian_forecast_covariance(model)
    n = 10000
    native = (
        np.asarray(model.simulate(12, anchor="end", repetitions=n, rng=np.random.default_rng(73)))
        .reshape(12, n)
        .T
    )
    assert (
        np.linalg.norm(np.cov(native, rowvar=False) - covariance) / np.linalg.norm(covariance)
        < 0.06
    )
    predicted = np.asarray(model.get_forecast(12).predicted_mean)
    assert np.max(abs(native.mean(axis=0) - predicted) / np.sqrt(np.diag(covariance) / n)) < 4


def test_common_budget_requires_a_bound_failure_record(tmp_path):
    settings = {
        "split": "development",
        "draws": 6000,
        "purpose": "initial_controls_not_model_selection",
        "methods": [{"method": "historical_mean"}],
    }
    with pytest.raises(ValueError):
        validate_settings(settings)
    root = Path(__file__).resolve().parents[1]
    decision_path = root / "docs/decisions/ENSEMBLE_BUDGET_20261021.json"
    reference = {
        "path": decision_path.relative_to(root).as_posix(),
        "sha256": file_hash(decision_path),
    }
    settings["budget_record"] = reference
    assert validate_settings(settings) == ["historical_mean"]
    # Exercise verification independently of untracked experiment files.
    (tmp_path / "docs").mkdir()
    freeze = tmp_path / "docs/STAGE2_FREEZE.json"
    freeze.write_text("frozen")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"checks": [{"passed": True}]}))
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "reporting_draws": 6000,
                "check_draws": 12000,
                "parent_freeze_sha256": file_hash(freeze),
                "evidence": {"path": "evidence.json", "sha256": file_hash(evidence)},
            }
        )
    )
    ref = {"path": "decision.json", "sha256": file_hash(decision)}
    with pytest.raises(ValueError, match="No recorded"):
        verify_budget_record(tmp_path, ref)
    evidence.write_text(json.dumps({"checks": [{"passed": False}]}))
    with pytest.raises(ValueError, match="evidence changed"):
        verify_budget_record(tmp_path, ref)
