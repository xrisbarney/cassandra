"""Safety gates that must fail before any model is constructed or sampled."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def terminal():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('terminal_guard_target',
                                                     scripts/'run_stage7_terminal.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


def test_existing_attempt_never_builds_model(tmp_path, monkeypatch, terminal):
    (tmp_path/'artifacts/stage7/terminal-v1/fits/joint_free').mkdir(parents=True)
    monkeypatch.setattr(terminal, 'authority', lambda root: {})
    monkeypatch.setattr(terminal, 'build_joint', lambda *a, **k: pytest.fail('model built'))
    with pytest.raises(FileExistsError):
        terminal.fit(tmp_path, 'joint_free')


def test_expired_authority_stops_before_source_or_model_access(tmp_path, terminal, monkeypatch):
    from datetime import timedelta

    from cyber_threat_modelling.data.store import write_json

    path = tmp_path/terminal.RECORD
    path.parent.mkdir(parents=True)
    write_json(path, {'status': 'diagnostics_active', 'attempt_limit': 5})
    monkeypatch.setattr(terminal, 'DEADLINE', terminal.datetime.now(terminal.UTC)-timedelta(days=1))
    monkeypatch.setattr(terminal, 'verify_fit_authority',
                        lambda *a: pytest.fail('source accessed after deadline'))
    with pytest.raises(TimeoutError):
        terminal.authority(tmp_path)


def test_closed_authority_cannot_start_new_round(tmp_path, terminal):
    from cyber_threat_modelling.data.store import write_json

    path = tmp_path/terminal.RECORD
    path.parent.mkdir(parents=True)
    write_json(path, {'status': 'closed', 'attempt_limit': 5})
    with pytest.raises(ValueError, match='No active'):
        terminal.authority(tmp_path)
