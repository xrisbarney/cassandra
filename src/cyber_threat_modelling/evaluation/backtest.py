"""Immutable development-only forecast artifacts and a separate scoring pass."""

import hashlib
import json
import time
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from ..config import canonical_json, load_config, require_keys
from ..data.store import write_json
from ..features.build import verify_panels
from ..forecasting.count_laws import CountForecast, integer_counts
from ..models.count_baselines import FitFailure, fit_predict, method_id, stream_seed
from ..models.ordering import loading_order
from ..provenance import (
    file_hash,
    prepare_run,
    verify_freeze,
    verify_independent_control_amendment,
    verify_observation_control_amendment,
    verify_zero_series_amendment,
)
from .scores import SCORING_VERSION, score_aggregate, score_cell


def development_origins():
    return pd.period_range("2018-12", "2023-11", freq="M")


def validate_settings(settings):
    keys = {"split", "draws", "methods", "purpose"}
    if settings.get("draws") == 6000:
        keys.add("budget_record")
    if settings.get("purpose") == "arima_finalist_selection":
        keys.add("screening_record")
    require_keys(settings, keys, "backtest")
    if settings["split"] != "development":
        raise ValueError(
            "Held-out execution is locked pending development selection and validation."
        )
    if settings["draws"] not in (3000, 6000) or settings["purpose"] not in (
        "initial_controls_not_model_selection",
        "arima_finalist_selection",
    ):
        raise ValueError("Initial controls require the frozen path budget and documented purpose.")
    if settings["draws"] == 6000:
        require_keys(settings["budget_record"], {"path", "sha256"}, "budget_record")
    if not isinstance(settings["methods"], list) or not settings["methods"]:
        raise ValueError("Specify at least one frozen baseline.")
    names = [method_id(spec) for spec in settings["methods"]]
    if settings["purpose"] == "arima_finalist_selection":
        require_keys(settings["screening_record"], {"path", "sha256"}, "screening_record")
        if len(names) != 2 or any(s["method"] != "arima" for s in settings["methods"]):
            raise ValueError("Finalist evaluation requires the two screened ARIMA candidates.")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate baseline settings.")
    return names


def verify_finalists(root, settings):
    if settings["purpose"] != "arima_finalist_selection":
        return
    ref = settings["screening_record"]
    path = artifact_path(root, ref["path"])
    if file_hash(path) != ref["sha256"]:
        raise ValueError("The screening selection record changed.")
    record = json.loads(path.read_text())
    verify_artifacts(path.parent, record)
    if (
        record["status"] != "screen_complete"
        or record["candidates"] != 65
        or record["origins_per_candidate"] != 20
        or record["finalists"] != [method_id(s) for s in settings["methods"]]
        or record["held_out_scoring_enabled"]
    ):
        raise ValueError("Settings do not match the frozen ARIMA screening result.")


def count_matrix(panel):
    # Held-out values are excluded before validating or constructing the model input.
    frame = panel.loc[panel.month.le("2023-12"), ["month", "category", "N", "m_N"]].copy()
    if frame.duplicated(["month", "category"]).any() or not frame.m_N.fillna(False).all():
        raise ValueError("Counts must be unique and observed; missing counts cannot become zeros.")
    wide = frame.pivot(index="month", columns="category", values="N").sort_index()
    expected = pd.period_range("2010-01", "2023-12", freq="M").astype(str)
    if list(wide.index) != list(expected) or wide.isna().any().any():
        raise ValueError("The monthly training/development calendar must be complete.")
    return wide, integer_counts(wide.to_numpy(dtype=float))


def training_prefix(wide, origin):
    if pd.Period(origin, freq="M") not in development_origins():
        raise ValueError("Only frozen development origins are executable.")
    return integer_counts(wide.loc[: str(origin)].to_numpy(dtype=float))


def artifact_path(folder, relative):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Artifact path escapes its run.")
    return path


def verify_artifacts(folder, manifest):
    for relative, digest in manifest["artifacts"].items():
        if file_hash(artifact_path(folder, relative)) != digest:
            raise ValueError("A saved forecast artifact changed.")


def _forecast_origin(task):
    """Fresh model per origin; persistent workers share code caches, never fitted states."""
    folder = task["folder"]
    folder.mkdir(parents=True)
    name, origin = task["name"], task["origin"]
    meta = {
        "method": name,
        "configuration_sha256": task["configuration_hash"],
        "design_amendment_sha256": task["amendment_sha256"],
        "origin": origin,
        "training_cutoff": origin,
        "split": "development",
        "vintage": "revised",
        "target_kind": "publication_count",
        "category_ids": task["categories"],
        "seed": task["seed"],
        "forecast_sha256": None,
        "fit_status": "failed",
        "retry_number": None,
    }
    started = time.perf_counter()
    try:
        if name in ("independent_nb_ar1", "independent_nb_glar", "pooled_nb"):
            from ..models.bayesian_controls import fit_predict_bayesian

            forecast, fit = fit_predict_bayesian(
                task["train"],
                task["initial"],
                task["spec"],
                task["draws"],
                origin,
                folder,
                task["seed"],
            )
        else:
            forecast, fit = fit_predict(
                task["train"], task["categories"], task["spec"], task["draws"], origin, task["seed"]
            )
        forecast.save(folder / "forecast.npz")
        write_json(folder / "fit.json", fit)
        meta.update(
            fit_status="accepted",
            retry_number=fit["retry_number"],
            forecast_sha256=file_hash(folder / "forecast.npz"),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
        write_json(
            folder / "failure.json",
            {
                "error": type(exc).__name__ + ": " + str(exc),
                "attempts": exc.attempts if isinstance(exc, FitFailure) else [],
            },
        )
    meta["total_seconds"] = time.perf_counter() - started
    write_json(folder / "metadata.json", meta)
    return {**meta, "directory": folder.relative_to(task["output"]).as_posix()}


def forecast_development(root, settings, panel_manifest, output, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four independent origin workers.")
    names = validate_settings(settings)
    if settings["draws"] == 6000:
        from .reissue import verify_budget_record

        verify_budget_record(root, settings["budget_record"])
    verify_finalists(root, settings)
    amendments = []
    if any(spec["method"] == "arima" for spec in settings["methods"]):
        amendments.append(verify_zero_series_amendment(root, required=True))
    if any(spec["method"] == "independent_nb_ar1" for spec in settings["methods"]):
        amendments.append(verify_independent_control_amendment(root, required=True))
    if any(spec["method"] == "independent_nb_glar" for spec in settings["methods"]):
        amendments.append(verify_observation_control_amendment(root, required=True))
    config = load_config(root / "configs/pilot.toml")
    config = type(config)(
        {
            **config.settings,
            "forecast": {**config.settings["forecast"], "draws": settings["draws"]},
        }
    )
    verify_freeze(root, config)
    verify_panels(root, panel_manifest)
    panel_record = json.loads(panel_manifest.read_text(encoding="utf-8"))
    panel_relative = next(
        p for p in panel_record["artifacts"] if PurePosixPath(p).name == "topic_month.parquet"
    )
    wide, counts = count_matrix(pd.read_parquet(root / panel_relative))
    categories = wide.columns.astype(int).tolist()
    initial = counts[:108]
    permutation, inverse = loading_order(initial, categories)
    prepare_run(root, config, output)
    amendment_sha256 = (
        amendments[0]["sha256"]
        if len(amendments) == 1
        else hashlib.sha256(canonical_json(amendments)).hexdigest()
        if amendments
        else None
    )
    effective_settings = {"settings": settings, "amendment_sha256": amendment_sha256}
    configuration_hash = hashlib.sha256(canonical_json(effective_settings)).hexdigest()
    write_json(output / "backtest_settings.json", settings)
    tasks = []
    for origin in development_origins():
        for spec, name in zip(settings["methods"], names, strict=True):
            tasks.append(
                {
                    "spec": spec,
                    "name": name,
                    "origin": str(origin),
                    "folder": output / name / str(origin),
                    "output": output,
                    "configuration_hash": configuration_hash,
                    "amendment_sha256": amendment_sha256,
                    "categories": categories,
                    "seed": config.settings["seed"],
                    "draws": settings["draws"],
                    "train": training_prefix(wide, origin),
                    "initial": initial,
                }
            )
    index = []
    if workers == 1:
        results = map(_forecast_origin, tasks)
        for result in results:
            index.append(result)
            print(
                f"Saved {result['method']} {result['origin']}: {result['fit_status']}", flush=True
            )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_forecast_origin, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                index.append(result)
                print(
                    f"Saved {result['method']} {result['origin']}: {result['fit_status']}",
                    flush=True,
                )
    index.sort(key=lambda r: (names.index(r["method"]), r["origin"]))
    manifest = {
        "schema_version": 1,
        "execution": {"origin_workers": workers, "chains_per_bayesian_fit": 4, "cores_per_fit": 1},
        "status": "forecasts_saved_not_scored",
        "settings": settings,
        "panel_manifest": panel_manifest.relative_to(root).as_posix(),
        "panel_manifest_sha256": file_hash(panel_manifest),
        "panel_file": panel_relative,
        "configuration_sha256": configuration_hash,
        "design_amendment_sha256": amendment_sha256,
        "design_amendments": amendments,
        "category_ids": categories,
        "model_category_order": np.asarray(categories)[permutation].tolist(),
        "inverse_model_permutation": inverse.tolist(),
        "active_categories": np.asarray(categories)[
            (initial.sum(axis=0) >= 50) & ((initial > 0).sum(axis=0) >= 24)
        ].tolist(),
        "index": index,
        "artifacts": {
            p.relative_to(output).as_posix(): file_hash(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "held_out_scoring_enabled": False,
    }
    write_json(output / "forecast_manifest.json", manifest)
    return output / "forecast_manifest.json"


METRICS = [
    "crps",
    "negative_log_mass",
    "absolute_error",
    "squared_error",
    "coverage_50",
    "coverage_90",
    "width_50",
    "width_90",
    "interval_score_50",
    "interval_score_90",
]


def summarize(cells, names, categories, active):
    summaries = []
    strata = {
        "all": categories,
        "active": active,
        "sparse": [k for k in categories if k not in active],
        "unknown": [10],
        "aggregate": [0],
    }
    for name in names:
        for horizon in (1, 3, 6, 12):
            for window, expected_origins in (("available", 61 - horizon), ("common_h12", 49)):
                for stratum, selected in strata.items():
                    if not selected:
                        continue
                    part = cells.loc[
                        (cells.method == name)
                        & (cells.horizon == horizon)
                        & cells.category.isin(selected)
                    ]
                    if window == "common_h12":
                        part = part.loc[part.origin <= "2022-12"]
                    expected = expected_origins * len(selected)
                    complete = len(part) == expected and part.fit_status.eq("accepted").all()
                    row = {
                        "method": name,
                        "horizon": horizon,
                        "window": window,
                        "stratum": stratum,
                        "scheduled_cells": expected,
                        "accepted_cells": int(part.fit_status.eq("accepted").sum()),
                        "status": "complete" if complete else "incomplete_no_primary_score",
                    }
                    # Category-equal means at each origin, then equally weighted origins.
                    for metric in METRICS:
                        row[metric] = (
                            float(
                                np.mean(
                                    part.groupby("origin")[metric]
                                    .agg(lambda values: np.mean(values.to_numpy()))
                                    .to_numpy()
                                )
                            )
                            if complete
                            else np.nan
                        )
                    row["rmse"] = np.sqrt(row.pop("squared_error"))
                    summaries.append(row)
    return pd.DataFrame(summaries)


def score_development(root, folder):
    manifest_path = folder / "forecast_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = validate_settings(manifest["settings"])
    if manifest["settings"]["draws"] == 6000:
        from .reissue import verify_budget_record

        verify_budget_record(root, manifest["settings"]["budget_record"])
    panel_manifest = artifact_path(root, manifest["panel_manifest"])
    if file_hash(panel_manifest) != manifest["panel_manifest_sha256"]:
        raise ValueError("The forecast's panel manifest changed.")
    verify_freeze(root, load_config(root / "configs/pilot.toml"))
    verify_panels(root, panel_manifest)
    verify_artifacts(folder, manifest)
    wide, _ = count_matrix(pd.read_parquet(artifact_path(root, manifest["panel_file"])))
    expected = {(name, str(origin)) for name in names for origin in development_origins()}
    actual = [(r["method"], r["origin"]) for r in manifest["index"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Forecast schedule is incomplete or duplicated.")
    score_folder = folder / "scores"
    score_folder.mkdir(exist_ok=False)
    rows = []
    categories = manifest["category_ids"]
    if categories != wide.columns.astype(int).tolist():
        raise ValueError("Forecast category order differs from the scoring panel.")
    for record in manifest["index"]:
        directory = artifact_path(folder, record["directory"])
        forecast = None
        if record["fit_status"] == "accepted":
            if file_hash(directory / "forecast.npz") != record["forecast_sha256"]:
                raise ValueError("Forecast hash mismatch.")
            forecast = CountForecast.load(directory / "forecast.npz")
            if forecast.paths.shape != (manifest["settings"]["draws"], 12, len(categories)):
                raise ValueError("Unexpected forecast dimensions.")
        for h in (1, 3, 6, 12):
            target = str(pd.Period(record["origin"], freq="M") + h)
            if target > "2023-12":
                continue
            observations = wide.loc[target].astype(int).to_numpy()
            base = {
                key: record[key]
                for key in (
                    "method",
                    "origin",
                    "training_cutoff",
                    "split",
                    "vintage",
                    "seed",
                    "configuration_sha256",
                    "fit_status",
                    "retry_number",
                    "forecast_sha256",
                )
            }
            base.update(
                run_id=folder.name,
                target=target,
                horizon=h,
                target_kind="publication_count",
                source_manifest_sha256=manifest["panel_manifest_sha256"],
                observed_flag=True,
                target_eligible=True,
                eligible_for_scoring=forecast is not None,
                eligibility_reason="development_count_checks_passed"
                if forecast is not None
                else "fit_failed",
                scoring_version=SCORING_VERSION,
                design_amendment_sha256=manifest.get("design_amendment_sha256"),
            )
            for j, category in enumerate(categories):
                metrics = dict.fromkeys(METRICS, np.nan)
                if forecast is not None:
                    rng = np.random.default_rng(
                        stream_seed(record["seed"], "pit", int(category), record["origin"], h)
                    )
                    metrics = score_cell(forecast, int(observations[j]), h - 1, j, rng)
                rows.append(
                    {**base, "category": category, "observed": int(observations[j]), **metrics}
                )
            aggregate = dict.fromkeys(METRICS, np.nan)
            if forecast is not None:
                # Sum matched paths, never independently reshuffle marginal ensembles.
                aggregate = score_aggregate(
                    forecast.paths[:, h - 1, :].sum(axis=1), int(observations.sum())
                )
            rows.append({**base, "category": 0, "observed": int(observations.sum()), **aggregate})
    cells = pd.DataFrame(rows)
    cells.to_parquet(score_folder / "cells.parquet", index=False)
    summary = summarize(cells, names, categories, manifest["active_categories"])
    summary.to_csv(score_folder / "summary.csv", index=False)
    write_json(
        score_folder / "score_manifest.json",
        {
            "scoring_version": SCORING_VERSION,
            "forecast_manifest_sha256": file_hash(manifest_path),
            "artifacts": {p.name: file_hash(p) for p in score_folder.iterdir() if p.is_file()},
            "aggregate_log_score": "not reported; no validated aggregate mass evaluator",
            "held_out_scoring_enabled": False,
            "scorer_source_sha256": {
                p.relative_to(root).as_posix(): file_hash(p)
                for p in sorted((root / "src").rglob("*.py"))
            },
        },
    )
    return score_folder / "summary.csv"
