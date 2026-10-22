"""Synthetic scheduling must not admit unverified models or real-data fixtures."""

import json
import runpy
import sys
from pathlib import Path

import pytest

from cyber_threat_modelling.provenance import file_hash

GRID = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/stage7_rank_two.py'))


@pytest.mark.parametrize('violation', ['verification', 'changed_model', 'real_data'])
def test_synthetic_grid_refuses_invalid_entry(tmp_path, monkeypatch, violation):
    monkeypatch.chdir(tmp_path)
    hashes = {}
    for name in ['joint_coordinates.py', 'gaussian_coordinates.py', 'block_gaussian.py',
                 'synthetic_state_space.py', 'stage7_priors.py']:
        path = Path('src/cyber_threat_modelling/models')/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
        hashes[path.as_posix()] = file_hash(path)
    path = Path('src/cyber_threat_modelling/evaluation/synthetic.py')
    path.parent.mkdir(parents=True)
    path.write_text('fixture')
    hashes[path.as_posix()] = file_hash(path)
    audit = Path('docs/checks/stage7-full-tests-v5-20261022/audit.json')
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({'all_passed': violation != 'verification',
                                'source_unchanged': True, 'source_sha256': hashes}))
    if violation == 'changed_model':
        Path('src/cyber_threat_modelling/models/joint_coordinates.py').write_text('changed')
    parent = Path('representative')
    parent.mkdir()
    (parent/'run.json').write_text(json.dumps({'real_data_read': violation == 'real_data',
                                             'held_out_scoring_enabled': False}))
    monkeypatch.setattr(sys, 'argv', ['grid', '--synthetic-grid', '--representative',
                                    str(parent), '--root', 'must_not_exist'])
    message = {'verification': 'verification did not pass',
               'changed_model': 'Untested model change', 'real_data': 'synthetic only'}
    with pytest.raises(ValueError, match=message[violation]):
        GRID['main']()
    assert not Path('must_not_exist').exists()
