import json

import pytest

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation import synthetic_checkpoint as MODULE
from cyber_threat_modelling.provenance import file_hash


def checkpoint(tmp_path):
    folder = tmp_path / 'case'
    folder.mkdir()
    source = tmp_path / 'source.py'
    source.write_text('unchanged', encoding='utf-8')
    write_json(folder / 'run.json', {'source_sha256': {'source.py': file_hash(source)}})
    attempt = {'retry': 0, 'accepted': False, 'warmup': 1000, 'draws': 1000,
               'chains': 4, 'target_accept': .95, 'max_treedepth': 12,
               'early_failure': {'divergences': 1}}
    write_json(folder / 'attempt_0.json', attempt)
    MODULE.seal(folder, [attempt])
    return folder, attempt


def test_checkpoint_restores_completed_failed_attempt(tmp_path):
    folder, attempt = checkpoint(tmp_path)
    _, attempts, digest = MODULE.restore(folder, tmp_path)
    assert attempts == [attempt]
    assert digest == file_hash(folder / 'checkpoint_0.json')


@pytest.mark.parametrize('change', ['extra', 'modified', 'missing'])
def test_checkpoint_rejects_partial_or_changed_files(tmp_path, change):
    folder, _ = checkpoint(tmp_path)
    if change == 'extra':
        (folder / 'heartbeat.json').write_text('partial next attempt', encoding='utf-8')
    elif change == 'modified':
        (folder / 'attempt_0.json').write_text('{}', encoding='utf-8')
    else:
        (folder / 'attempt_0.json').unlink()
    with pytest.raises(ValueError, match='Changed, extra or missing'):
        MODULE.restore(folder, tmp_path)


def test_checkpoint_rejects_changed_source(tmp_path):
    folder, _ = checkpoint(tmp_path)
    (tmp_path / 'source.py').write_text('different', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed checkpoint source'):
        MODULE.restore(folder, tmp_path)


def test_checkpoint_rejects_terminal_case(tmp_path):
    folder, _ = checkpoint(tmp_path)
    write_json(folder / 'result.json', {'status': 'accepted'})
    with pytest.raises(ValueError, match='terminal case'):
        MODULE.restore(folder, tmp_path)


def test_checkpoint_cannot_pause_admitted_or_final_attempt(tmp_path):
    for attempts in ([{'accepted': True}], [{'accepted': False}] * 3):
        with pytest.raises(ValueError, match='Only a completed unsuccessful'):
            MODULE.seal(tmp_path, attempts)


def test_checkpoint_retains_first_seal_across_second_attempt(tmp_path):
    folder, attempt = checkpoint(tmp_path)
    first = file_hash(folder / 'checkpoint_0.json')
    second = {**attempt, 'retry': 1, 'warmup': 2000, 'target_accept': .99,
              'max_treedepth': 15}
    write_json(folder / 'attempt_1.json', second)
    MODULE.seal(folder, [attempt, second])
    _, attempts, _ = MODULE.restore(folder, tmp_path)
    assert attempts == [attempt, second]
    assert file_hash(folder / 'checkpoint_0.json') == first
    record = json.loads((folder / 'checkpoint_1.json').read_text(encoding='utf-8'))
    assert record['artifacts']['checkpoint_0.json'] == first
