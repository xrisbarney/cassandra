import importlib.util
import json
from pathlib import Path

import pytest

from cyber_threat_modelling.provenance import file_hash

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ci_stage7', ROOT / 'scripts/ci_stage7.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plan_file(root, **updates):
    record = {'status': 'prepared_verification_only', 'real_data_read': False,
              'held_out_scoring_enabled': False, 'files_sha256': {}, 'cases': []}
    record.update(updates)
    path = root / MODULE.PLAN
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding='utf-8')


def test_hosted_fitting_requires_explicit_activation(tmp_path):
    plan_file(tmp_path)
    with pytest.raises(ValueError, match='not been activated'):
        MODULE.checked_plan(tmp_path, fitting=True)


def test_hosted_source_change_is_rejected(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('original', encoding='utf-8')
    plan_file(tmp_path, files_sha256={'source.py': file_hash(source)})
    source.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed or escaping'):
        MODULE.checked_plan(tmp_path)


def test_hosted_source_escape_is_rejected_before_read(tmp_path):
    plan_file(tmp_path, files_sha256={'../outside.py': 'irrelevant'})
    with pytest.raises(ValueError, match='Changed or escaping'):
        MODULE.checked_plan(tmp_path)


@pytest.mark.parametrize('flag', ['real_data_read', 'held_out_scoring_enabled'])
def test_hosted_real_data_or_scoring_is_rejected(tmp_path, flag):
    plan_file(tmp_path, **{flag: True})
    with pytest.raises(ValueError, match='must remain synthetic'):
        MODULE.checked_plan(tmp_path)


def test_hosted_fixtures_match_declared_local_bytes():
    plan = MODULE.checked_plan(ROOT)
    assert len(plan['cases']) == 8
    for case in plan['cases']:
        _, _, options = MODULE.fixture(case)
        assert options['level_parameterization'] == 'joint_conditional'
        assert options['prior_candidate'] == case['candidate']
