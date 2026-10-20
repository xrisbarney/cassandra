import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def project(tmp_path):
    """A tiny frozen project, independent of mutable workspace documentation."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "src").mkdir()
    spec = tmp_path / "docs" / "spec.md"
    spec.write_text("frozen design\n", encoding="utf-8")
    freeze = {
        "design_version": "1.0.1",
        "initial_training_period": ["2010-01", "2018-12"],
        "development_target_period": ["2019-01", "2023-12"],
        "held_out_target_period": ["2024-01", "2025-12"],
        "files": [
            {"path": "docs/spec.md", "sha256": hashlib.sha256(spec.read_bytes()).hexdigest()}
        ],
    }
    (tmp_path / "docs" / "STAGE2_FREEZE.json").write_text(json.dumps(freeze), encoding="utf-8")
    sample = Path(__file__).parents[1] / "configs" / "pilot.toml"
    (tmp_path / "configs" / "pilot.toml").write_bytes(sample.read_bytes())
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test-project"\n', encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return tmp_path
