"""Prepare auditable run records without importing models or reading credentials."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import ConfigurationError, ExperimentConfig, canonical_json


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ConfigurationError("A provenance path escapes the project directory.")
    if path.suffix not in (".md", ".json", ".toml", ".lock", ".py"):
        raise ConfigurationError("Unsupported provenance file type.")
    return path


def verify_freeze(root: Path, config: ExperimentConfig) -> dict:
    path = root / "docs" / "STAGE2_FREEZE.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if config.settings["protocol_version"] != freeze["design_version"]:
        raise ConfigurationError("Configuration and protocol versions differ.")
    if not freeze["files"]:
        raise ConfigurationError("The protocol freeze has no file hashes.")
    for item in freeze["files"]:
        if file_hash(project_file(root, item["path"])) != item["sha256"]:
            raise ConfigurationError("A frozen design file changed; record an amendment first.")
    p = config.settings["periods"]
    for split, key in (
        ("evaluation", "held_out_target_period"),
        ("development", "development_target_period"),
        ("initial", "initial_training_period"),
    ):
        if [p[f"{split}_start"], p[f"{split}_end"]] != freeze[key]:
            raise ConfigurationError(f"{split} dates differ from the frozen design.")
    return {"path": "docs/STAGE2_FREEZE.json", "sha256": file_hash(path), "record": freeze}


def git_revision(root: Path) -> dict:
    """Use Git if present; source hashes also cover uncommitted or untracked code."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None, "status": "unavailable"}
    return {"revision": revision, "dirty": bool(status), "status": "available"}


def verify_zero_series_amendment(root: Path, required=False):
    """Verify the approved supplement without rewriting the original freeze."""
    relative = "docs/amendments/ZERO_SERIES_20261021.json"
    path = root / relative
    if not path.is_file():
        if required:
            raise ConfigurationError("The approved zero-series amendment is required.")
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record["status"] != "active"
        or record["rule"] != "all_zero_training_point_mass_v1"
        or record["methods"] != ["arima", "ets"]
        or record["parent_freeze_sha256"] != file_hash(root / "docs/STAGE2_FREEZE.json")
        or not record["files"]
    ):
        raise ConfigurationError("Invalid zero-series amendment or parent freeze.")
    for item in record["files"]:
        if file_hash(project_file(root, item["path"])) != item["sha256"]:
            raise ConfigurationError("A frozen amendment file changed.")
    return {"path": relative, "sha256": file_hash(path), "record": record}


def verify_independent_control_amendment(root: Path, required=False):
    from .models.control_priors import INDEPENDENT_PRIOR_RULE, INDEPENDENT_STATIONARY_SCALE_SD

    relative = "docs/amendments/INDEPENDENT_CONTROL_20261021.json"
    path = root / relative
    if not path.is_file():
        if required:
            raise ConfigurationError("The independent-control prior amendment is required.")
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record["status"] != "active"
        or record["rule"] != INDEPENDENT_PRIOR_RULE
        or record["stationary_scale_sd"] != INDEPENDENT_STATIONARY_SCALE_SD
        or record["methods"] != ["independent_nb_ar1"]
        or record["parent_freeze_sha256"] != file_hash(root / "docs/STAGE2_FREEZE.json")
        or not record["files"]
    ):
        raise ConfigurationError("Invalid independent-control amendment or parent freeze.")
    for item in record["files"]:
        if file_hash(project_file(root, item["path"])) != item["sha256"]:
            raise ConfigurationError("A frozen amendment file changed.")
    return {"path": relative, "sha256": file_hash(path), "record": record}


def verify_observation_control_amendment(root: Path, required=False):
    from .models.control_priors import OBSERVATION_METHOD, OBSERVATION_RULE

    relative = "docs/amendments/OBSERVATION_CONTROL_20261021.json"
    path = root / relative
    if not path.is_file():
        if required:
            raise ConfigurationError("The observation-driven control amendment is required.")
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record["status"] != "active"
        or record["rule"] != OBSERVATION_RULE
        or record["methods"] != [OBSERVATION_METHOD]
        or record["parent_freeze_sha256"] != file_hash(root / "docs/STAGE2_FREEZE.json")
        or not record["files"]
    ):
        raise ConfigurationError("Invalid observation-driven amendment or parent freeze.")
    for item in record["files"]:
        if file_hash(project_file(root, item["path"])) != item["sha256"]:
            raise ConfigurationError("A frozen amendment file changed.")
    return {"path": relative, "sha256": file_hash(path), "record": record}


def prepare_run(root: Path, config: ExperimentConfig, output: Path, data_manifest=None) -> Path:
    """Write a new preparation record. Never overwrite a prior run directory."""
    root = root.resolve()
    freeze = verify_freeze(root, config)
    amendment = verify_zero_series_amendment(root)
    independent_amendment = verify_independent_control_amendment(root)
    observation_amendment = verify_observation_control_amendment(root)
    data_record = None
    if data_manifest:
        from .data.audit import verify_data_manifest

        data_record = verify_data_manifest(root, data_manifest, config.settings["vintage"])
    sources = [root / "pyproject.toml", root / "uv.lock"]
    runtime_lock = root / "configs" / "nuts-runtime.lock"
    if runtime_lock.is_file():
        sources.append(runtime_lock)
    sources.extend(sorted((root / "src").rglob("*.py")))
    sources.extend(sorted((root / "scripts").rglob("*.py")))
    if not sources[1].is_file():
        raise ConfigurationError("Resolve and lock dependencies before preparing a run.")
    source_hashes = {str(p.relative_to(root).as_posix()): file_hash(p) for p in sources}
    installed = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    record = {
        "manifest_schema_version": 1,
        "status": "prepared_not_executed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "configuration": config.settings,
        "configuration_sha256": config.sha256,
        "protocol": freeze,
        "design_amendments": [
            a for a in (amendment, independent_amendment, observation_amendment) if a
        ],
        "seed": config.settings["seed"],
        "python": sys.version,
        "platform": platform.platform(),
        "installed_packages": dict(sorted(installed.items())),
        "source_control": git_revision(root),
        "source_sha256": source_hashes,
        "source_bundle_sha256": hashlib.sha256(canonical_json(source_hashes)).hexdigest(),
        "data_manifest": data_record,
        "data_status": "source_audited_panel_pending" if data_record else "not_ingested",
    }
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # All reads and validation finish before creating the new output directory.
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "run.json"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return destination
