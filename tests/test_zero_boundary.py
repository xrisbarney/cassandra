import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyber_threat_modelling.config import ConfigurationError
from cyber_threat_modelling.evaluation.backtest import METRICS, development_origins, summarize
from cyber_threat_modelling.evaluation.scores import crps
from cyber_threat_modelling.models.boundaries import ZERO_SERIES_RULE, uses_zero_boundary
from cyber_threat_modelling.models.count_baselines import FitFailure, arima_candidates, fit_predict
from cyber_threat_modelling.provenance import file_hash, verify_zero_series_amendment


def test_all_frozen_arima_candidates_use_declared_boundary_without_optimizer(monkeypatch):
    import cyber_threat_modelling.models.count_baselines as baselines

    def forbidden(*args):
        raise AssertionError("Gaussian optimization must not run on the zero boundary.")

    monkeypatch.setattr(baselines, "fit_arima", forbidden)
    for spec in arima_candidates():
        forecast, fit = fit_predict(np.zeros((108, 2)), [2, 10], spec, 50, "2018-12")
        assert fit["retry_number"] == 0
        assert np.all(forecast.paths == 0)
        for j in (0, 1):
            assert fit["per_category"][j]["boundary_rule"] == ZERO_SERIES_RULE
            assert fit["per_category"][j]["attempts"] == []
            assert forecast.logpmf(0, 0, j) == 0
            assert forecast.logpmf(1, 0, j) == -np.inf
            assert forecast.cdf(0, 0, j) == 1
            assert forecast.quantile(0.95, 0, j) == 0
            assert crps(forecast.paths[:, 0, j], 3) == 3


def test_boundary_does_not_replace_positive_history_failures(monkeypatch):
    import cyber_threat_modelling.models.count_baselines as baselines

    def failed(y, spec):
        assert y.sum() == 1
        raise FitFailure("Deliberate nonzero-series failure", [])

    monkeypatch.setattr(baselines, "fit_arima", failed)
    y = np.zeros((108, 2))
    y[-1, 1] = 1
    with pytest.raises(FitFailure, match="category 7"):
        fit_predict(y, [10, 7], arima_candidates()[0], 50, "2018-12")


def test_boundary_is_training_only_and_restricted_to_approved_methods():
    zeros = np.zeros(108)
    assert uses_zero_boundary(zeros, "arima")
    assert uses_zero_boundary(zeros, "ets")
    assert not uses_zero_boundary(zeros, "seasonal_naive")
    assert not uses_zero_boundary(np.r_[zeros, 1], "arima")
    for invalid in ([], [0, np.nan], [0, -1]):
        with pytest.raises(ValueError):
            uses_zero_boundary(invalid, "arima")


def test_infinite_log_score_survives_category_and_origin_averaging():
    rows = []
    for origin in development_origins():
        for category in (1, 10):
            rows.append(
                {
                    "method": "test",
                    "origin": str(origin),
                    "category": category,
                    "horizon": 1,
                    "fit_status": "accepted",
                    **dict.fromkeys(METRICS, 0.0),
                }
            )
    rows[1]["negative_log_mass"] = np.inf
    rows[1]["crps"] = 3
    result = summarize(pd.DataFrame(rows), ["test"], [1, 10], [1])
    row = result.query("stratum == 'all' and horizon == 1 and window == 'available'").iloc[0]
    assert row.status == "complete"
    assert row.negative_log_mass == np.inf
    assert row.crps == 3 / 120


def test_amendment_requires_matching_text_and_original_freeze(project):
    root = Path(__file__).resolve().parents[1]
    original = root / "docs/amendments/ZERO_SERIES_20261021.json"
    assert verify_zero_series_amendment(root, required=True)["record"]["status"] == "active"
    assert verify_zero_series_amendment(project) is None
    with pytest.raises(ConfigurationError, match="required"):
        verify_zero_series_amendment(project, required=True)
    directory = project / "docs/amendments"
    directory.mkdir()
    record = json.loads(original.read_text())
    record["parent_freeze_sha256"] = file_hash(project / "docs/STAGE2_FREEZE.json")
    text_path = project / record["files"][0]["path"]
    text_path.write_bytes((root / record["files"][0]["path"]).read_bytes())
    (directory / original.name).write_text(json.dumps(record))
    verify_zero_series_amendment(project, required=True)
    text_path.write_text("modified rule")
    with pytest.raises(ConfigurationError, match="changed"):
        verify_zero_series_amendment(project, required=True)
