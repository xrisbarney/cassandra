"""Strict, credential-free configuration for the frozen primary experiment."""

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """The configuration or frozen design is inconsistent."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def month_index(value: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise ConfigurationError("Months must be strings in YYYY-MM format.")
    year, month = map(int, value.split("-"))
    if year < 1:
        raise ConfigurationError("Calendar years must be positive.")
    return year * 12 + month - 1


def require_keys(value: Any, keys: set[str], section: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        # Do not echo arbitrary keys/values; a misplaced credential stays out of logs.
        raise ConfigurationError(f"{section} must contain exactly the documented keys.")


def positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated settings; canonical JSON is the reproducible configuration identity."""

    settings: dict[str, Any]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.settings)).hexdigest()

    def origin_counts(self, split: str = "evaluation") -> dict[int, int]:
        p = self.settings["periods"]
        months = month_index(p[f"{split}_end"]) - month_index(p[f"{split}_start"]) + 1
        return {h: months - h + 1 for h in self.settings["forecast"]["reported_horizons"]}


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    require_keys(
        raw,
        {
            "schema_version",
            "protocol_version",
            "seed",
            "vintage",
            "periods",
            "forecast",
            "model",
            "sampling",
        },
        "configuration",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ConfigurationError("Unsupported configuration schema version.")
    if not isinstance(raw["protocol_version"], str):
        raise ConfigurationError("protocol_version must be a string.")
    if type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**32:
        raise ConfigurationError("seed must be an unsigned 32-bit integer.")
    if raw["vintage"] not in ("revised", "point_in_time"):
        raise ConfigurationError("vintage must be revised or point_in_time.")

    p = raw["periods"]
    require_keys(
        p,
        {
            f"{part}_{edge}"
            for part in ("initial", "development", "evaluation")
            for edge in ("start", "end")
        },
        "periods",
    )
    for part in ("initial", "development", "evaluation"):
        if month_index(p[f"{part}_start"]) > month_index(p[f"{part}_end"]):
            raise ConfigurationError("A period starts after it ends.")
    if month_index(p["initial_end"]) + 1 != month_index(p["development_start"]):
        raise ConfigurationError("Initial and development periods must be contiguous.")
    if month_index(p["development_end"]) + 1 != month_index(p["evaluation_start"]):
        raise ConfigurationError("Development must end immediately before held-out evaluation.")

    f = raw["forecast"]
    require_keys(f, {"path_months", "reported_horizons", "draws"}, "forecast")
    if type(f["path_months"]) is not int or f["path_months"] != 12:
        raise ConfigurationError("The frozen path length is 12 months.")
    if f["reported_horizons"] != [1, 3, 6, 12] or any(
        type(h) is not int for h in f["reported_horizons"]
    ):
        raise ConfigurationError("The frozen reported horizons are 1, 3, 6, 12.")
    if type(f["draws"]) is not int or f["draws"] not in (1000, 3000, 6000, 12000):
        raise ConfigurationError("Unsupported predictive simulation budget.")
    config = ExperimentConfig(raw)
    if min(config.origin_counts().values()) < 1:
        raise ConfigurationError("The evaluation period cannot accommodate the forecast horizons.")

    m = raw["model"]
    require_keys(
        m,
        {
            "topics",
            "intensity_factors",
            "severity_factors",
            "regimes",
            "channels",
            "effort",
            "kernel",
        },
        "model",
    )
    for name in ("topics", "intensity_factors", "severity_factors", "regimes"):
        positive_integer(m[name], name)
    if (m["topics"], m["intensity_factors"], m["severity_factors"]) != (10, 2, 2):
        raise ConfigurationError("The primary grid fixes K=10 and factor ranks at 2.")
    if m["regimes"] not in (1, 2, 3) or m["channels"] not in (["N"], ["N", "B"]):
        raise ConfigurationError("Configuration is outside the frozen primary model grid.")
    if type(m["effort"]) is not bool or type(m["kernel"]) is not bool:
        raise ConfigurationError("effort and kernel must be Boolean.")

    s = raw["sampling"]
    require_keys(s, {"chains", "warmup", "draws", "target_accept", "max_treedepth"}, "sampling")
    for name in ("chains", "warmup", "draws", "max_treedepth"):
        positive_integer(s[name], name)
    budgets = {(4, 1000, 1000, 0.95, 12), (4, 2000, 1000, 0.99, 15), (4, 2000, 2000, 0.99, 15)}
    if not isinstance(s["target_accept"], float):
        raise ConfigurationError("target_accept must be a floating-point value.")
    if (
        s["chains"],
        s["warmup"],
        s["draws"],
        s["target_accept"],
        s["max_treedepth"],
    ) not in budgets:
        raise ConfigurationError("Sampling settings must match an approved initial/retry budget.")
    return config
