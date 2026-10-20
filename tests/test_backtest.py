import json

import numpy as np
import pandas as pd
import pytest

from cyber_threat_modelling.evaluation.backtest import (
    METRICS,
    artifact_path,
    count_matrix,
    development_origins,
    summarize,
    training_prefix,
    validate_settings,
    verify_artifacts,
)
from cyber_threat_modelling.models.count_baselines import fit_predict
from cyber_threat_modelling.provenance import file_hash


def test_forecasts_saved_before_scoring_and_failures_retained(tmp_path, monkeypatch):
    from pathlib import Path

    import cyber_threat_modelling.evaluation.backtest as runner
    from cyber_threat_modelling.config import load_config

    real_root = Path(__file__).resolve().parents[1]
    config = load_config(real_root / "configs/pilot.toml")
    data = panel()
    data.to_parquet(tmp_path / "topic_month.parquet", index=False)
    manifest = tmp_path / "panel.json"
    manifest.write_text(
        json.dumps(
            {"artifacts": {"topic_month.parquet": file_hash(tmp_path / "topic_month.parquet")}}
        )
    )
    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(runner, "verify_freeze", lambda *_: None)
    monkeypatch.setattr(runner, "verify_panels", lambda *_: None)

    def prepare(root, config, output):
        output.mkdir(exist_ok=False)
        (output / "run.json").write_text("{}")

    monkeypatch.setattr(runner, "prepare_run", prepare)

    def fitting(train, ids, spec, draws, origin, seed):
        if origin == "2018-12":
            raise ValueError("deliberate test failure")
        return fit_predict(train, ids, spec, draws, origin, seed)

    monkeypatch.setattr(runner, "fit_predict", fitting)
    settings = {
        "split": "development",
        "draws": 3000,
        "methods": [{"method": "historical_mean"}],
        "purpose": "initial_controls_not_model_selection",
    }
    output = tmp_path / "run"
    runner.forecast_development(tmp_path, settings, manifest, output)
    assert not (output / "scores").exists()
    record = json.loads((output / "forecast_manifest.json").read_text())
    assert len(record["index"]) == 60
    assert record["index"][0]["fit_status"] == "failed"
    assert (output / "historical_mean/2019-01/forecast.npz").exists()
    runner.score_development(tmp_path, output)
    cells = pd.read_parquet(output / "scores/cells.parquet")
    assert cells.target.max() <= "2023-12"
    assert len(cells) == (60 + 58 + 55 + 49) * 2
    assert cells.query("origin == '2018-12'").fit_status.eq("failed").all()
    summary = pd.read_csv(output / "scores/summary.csv")
    assert summary.query("stratum == 'all'").crps.isna().all()
    with pytest.raises(FileExistsError):
        runner.score_development(tmp_path, output)


def panel():
    months = pd.period_range("2010-01", "2025-12", freq="M").astype(str)
    return pd.DataFrame({"month": months, "category": 1, "N": np.arange(len(months)), "m_N": True})


def test_calendar_counts_and_training_cutoff():
    origins = development_origins()
    assert [
        sum(o + h <= pd.Period("2023-12", freq="M") for o in origins) for h in (1, 3, 6, 12)
    ] == [60, 58, 55, 49]
    wide, _ = count_matrix(panel())
    assert len(training_prefix(wide, "2018-12")) == 108
    assert len(training_prefix(wide, "2023-11")) == 167
    with pytest.raises(ValueError, match="development"):
        training_prefix(wide, "2023-12")


def test_future_mutation_cannot_change_forecast():
    original = panel()
    changed = original.copy()
    changed.loc[changed.month > "2018-12", "N"] = 999999
    a, _ = count_matrix(original)
    b, _ = count_matrix(changed)
    spec = {"method": "seasonal_naive"}
    first, _ = fit_predict(training_prefix(a, "2018-12"), [1], spec, 100, "2018-12")
    second, _ = fit_predict(training_prefix(b, "2018-12"), [1], spec, 100, "2018-12")
    assert np.array_equal(first.paths, second.paths)
    changed.loc[changed.month >= "2024-01", "N"] = np.nan
    assert len(count_matrix(changed)[0]) == 168


def test_missing_month_and_missing_count_are_not_compressed_or_zeroed():
    data = panel()
    with pytest.raises(ValueError, match="calendar"):
        count_matrix(data.drop(index=30))
    data.loc[30, "m_N"] = False
    with pytest.raises(ValueError, match="observed"):
        count_matrix(data)


def test_heldout_and_unapproved_settings_locked():
    settings = {
        "split": "development",
        "draws": 3000,
        "methods": [{"method": "historical_mean"}],
        "purpose": "initial_controls_not_model_selection",
    }
    assert validate_settings(settings) == ["historical_mean"]
    with pytest.raises(ValueError, match="locked"):
        validate_settings({**settings, "split": "evaluation"})
    with pytest.raises(ValueError):
        validate_settings({**settings, "draws": 20})


def test_artifact_corruption_and_path_escape_refused(tmp_path):
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps({"saved": True}))
    manifest = {"artifacts": {"forecast.json": file_hash(path)}}
    verify_artifacts(tmp_path, manifest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_artifacts(tmp_path, manifest)
    with pytest.raises(ValueError, match="escapes"):
        artifact_path(tmp_path, "../outside")


def test_missing_fit_does_not_get_silently_dropped_from_primary_mean():
    rows = []
    for origin in development_origins():
        for category, value in [(1, 10), (10, 0)]:
            rows.append(
                {
                    "method": "test",
                    "origin": str(origin),
                    "category": category,
                    "horizon": 1,
                    "fit_status": "accepted",
                    **dict.fromkeys(METRICS, value),
                }
            )
    complete = summarize(pd.DataFrame(rows), ["test"], [1, 10], [1])
    row = complete.query("stratum == 'all' and horizon == 1 and window == 'available'").iloc[0]
    assert row.crps == 5
    rows[0]["fit_status"] = "failed"
    incomplete = summarize(pd.DataFrame(rows), ["test"], [1, 10], [1])
    row = incomplete.query("stratum == 'all' and horizon == 1 and window == 'available'").iloc[0]
    assert row.status == "incomplete_no_primary_score"
    assert np.isnan(row.crps)
