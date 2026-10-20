import json

import pytest

from cyber_threat_modelling.cli import main
from cyber_threat_modelling.config import ConfigurationError, load_config
from cyber_threat_modelling.provenance import prepare_run, verify_freeze


def change_config(project, old, new):
    path = project / "configs" / "pilot.toml"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_frozen_origin_counts(project):
    config = load_config(project / "configs" / "pilot.toml")
    verify_freeze(project, config)
    assert config.origin_counts() == {1: 24, 3: 22, 6: 19, 12: 13}
    assert config.origin_counts("development") == {1: 60, 3: 58, 6: 55, 12: 49}


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('development_end = "2023-12"', 'development_end = "2024-01"'),
        ('evaluation_start = "2024-01"', 'evaluation_start = "2024-13"'),
        ("regimes = 1", "regimes = 4"),
        ("chains = 4", "chains = true"),
        ("kernel = false", 'kernel = "false"'),
        ("seed = 20261020", "seed = -1"),
        ("reported_horizons = [1, 3, 6, 12]", "reported_horizons = [1, 3, 6, 24]"),
        ("target_accept = 0.95", "target_accept = nan"),
    ],
)
def test_invalid_configuration_rejected(project, old, new):
    path = change_config(project, old, new)
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_calendar_shift_requires_protocol_amendment(project):
    path = change_config(project, 'evaluation_end = "2025-12"', 'evaluation_end = "2026-12"')
    config = load_config(path)
    with pytest.raises(ConfigurationError, match="dates differ"):
        verify_freeze(project, config)


def test_changed_design_blocks_run_before_any_output(project):
    config = load_config(project / "configs" / "pilot.toml")
    (project / "docs" / "spec.md").write_text("changed design", encoding="utf-8")
    output = project / "artifacts" / "run"
    with pytest.raises(ConfigurationError, match="amendment"):
        prepare_run(project, config, output)
    assert not output.exists()


def test_preparation_records_provenance_without_credentials(project, monkeypatch):
    secret = "not-a-real-secret-do-not-record"
    monkeypatch.setenv("NVD_API_KEY", secret)
    (project / ".env").write_text(f"NVD_API_KEY={secret}", encoding="utf-8")
    config = load_config(project / "configs" / "pilot.toml")
    result = prepare_run(project, config, project / "artifacts" / "run")
    text = result.read_text(encoding="utf-8")
    record = json.loads(text)
    assert secret not in text
    assert record["configuration_sha256"] == config.sha256
    assert record["status"] == "prepared_not_executed"
    assert record["data_manifest"] is None
    assert "uv.lock" in record["source_sha256"]
    assert "installed_packages" in record
    original = result.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_run(project, config, result.parent)
    assert result.read_bytes() == original


def test_unknown_secret_field_is_rejected_without_echoing_value(project):
    path = change_config(
        project, "schema_version = 1", 'api_key = "private-value"\nschema_version = 1'
    )
    with pytest.raises(ConfigurationError) as caught:
        load_config(path)
    assert "private-value" not in str(caught.value)


def test_configuration_hash_ignores_toml_comments(project):
    path = project / "configs" / "pilot.toml"
    before = load_config(path).sha256
    path.write_text("# additional comment\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_config(path).sha256 == before


def test_freeze_cannot_reference_outside_project(project):
    path = project / "docs" / "STAGE2_FREEZE.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    freeze["files"][0]["path"] = "../outside.md"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="escapes"):
        verify_freeze(project, load_config(project / "configs" / "pilot.toml"))


def test_cli_validate_and_prepare(project, capsys):
    common = ["--project-root", str(project)]
    assert main([*common, "validate", "configs/pilot.toml"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert (
        main([*common, "prepare-run", "configs/pilot.toml", "--output", "artifacts/example"]) == 0
    )
    assert "no model fitted" in capsys.readouterr().out


def test_cli_missing_config_has_nonzero_exit(project):
    with pytest.raises(SystemExit) as caught:
        main(["--project-root", str(project), "validate", "configs/absent.toml"])
    assert caught.value.code == 2
