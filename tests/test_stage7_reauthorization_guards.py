"""The new authority cannot reopen old attempts or relax the declared gates."""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cyber_threat_modelling.data.store import write_json


@pytest.fixture
def runner():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('reauthorized_guard',
                                                     scripts/'run_stage7_reauthorized.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


def record():
    return {'status': 'active', 'cases': ['joint_free', 'joint_fixed'],
            'attempt_limit': 2, 'attempts_per_case': 1, 'per_attempt_wall_seconds': 5400,
            'chains': 4, 'warmup': 2000, 'draws': 2000, 'target_accept': .99,
            'max_treedepth': 15, 'sampler_seed': 7103, 'initialization_seed': 7301,
            'stage8_enabled': False, 'held_out_scoring_enabled': False,
            'global_deadline_utc': (datetime.now(UTC)+timedelta(hours=2)).isoformat()}


def test_existing_attempt_is_never_overwritten(tmp_path, runner, monkeypatch):
    (tmp_path/runner.BASE/'fits/joint_free').mkdir(parents=True)
    monkeypatch.setattr(runner, 'authority', lambda root: record())
    monkeypatch.setattr(runner, 'load_fixture', lambda *a: pytest.fail('fixture accessed'))
    with pytest.raises(FileExistsError):
        runner.fit(tmp_path, 'joint_free')


def test_closure_blocks_relaunch_before_source_access(tmp_path, runner):
    path = tmp_path/runner.CLOSURE
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    with pytest.raises(ValueError, match='already closed'):
        runner.authority(tmp_path)


def test_expired_deadline_blocks_source_access(tmp_path, runner, monkeypatch):
    path = tmp_path/runner.RECORD
    path.parent.mkdir(parents=True)
    data = record()
    data['global_deadline_utc'] = (datetime.now(UTC)-timedelta(seconds=1)).isoformat()
    write_json(path, data)
    monkeypatch.setattr(runner, 'verify_fit_authority', lambda *a: pytest.fail('source accessed'))
    with pytest.raises(TimeoutError):
        runner.authority(tmp_path)


def test_budget_change_is_refused(tmp_path, runner):
    path = tmp_path/runner.RECORD
    path.parent.mkdir(parents=True)
    data = record()
    data['draws'] = 4000
    write_json(path, data)
    with pytest.raises(ValueError, match='budget or scope'):
        runner.authority(tmp_path)
