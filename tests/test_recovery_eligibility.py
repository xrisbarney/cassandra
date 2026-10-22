from pathlib import Path

import pytest

from cyber_threat_modelling.evaluation.recovery_eligibility import recovery_eligibility
from cyber_threat_modelling.provenance import file_hash


def test_mode_exclusion_preserves_numerical_status_and_binds_exact_evidence(tmp_path):
    folder = tmp_path/'fit'
    folder.mkdir()
    (folder/'result.json').write_text('{"status":"accepted"}')
    evidence = tmp_path/'audit.json'
    evidence.write_text('documented retained orientations')
    item = dict(directory=str(folder), evidence=str(evidence),
                evidence_sha256=file_hash(evidence), result_sha256=file_hash(folder/'result.json'),
                reason='Known mode not explored')
    routing = dict(recovery_exclusions=[item])
    assert recovery_eligibility(folder, 'accepted', {}, tmp_path)['recovery_eligible']
    assert not recovery_eligibility(folder, 'failed_diagnostic_gate', {}, tmp_path)[
        'recovery_eligible']
    result = recovery_eligibility(folder, 'accepted', routing, tmp_path)
    assert not result['recovery_eligible'] and result['recovery_exclusion'] == item
    assert (folder/'result.json').read_text() == '{"status":"accepted"}'
    with pytest.raises(ValueError, match='Duplicate'):
        recovery_eligibility(folder, 'accepted', dict(recovery_exclusions=[item, item]), tmp_path)
    evidence.write_text('changed')
    with pytest.raises(ValueError, match='unchanged evidence'):
        recovery_eligibility(folder, 'accepted', routing, tmp_path)


def test_exclusion_cannot_escape_project_or_apply_to_replaced_fit(tmp_path):
    (tmp_path/'result.json').write_text('one')
    evidence = tmp_path/'audit.json'
    evidence.write_text('evidence')
    item = dict(directory=str(tmp_path), evidence=str(evidence), reason='mode',
                evidence_sha256=file_hash(evidence),
                result_sha256=file_hash(tmp_path/'result.json'))
    (tmp_path/'result.json').write_text('two')
    with pytest.raises(ValueError, match='changed result'):
        recovery_eligibility(tmp_path, 'accepted', dict(recovery_exclusions=[item]), tmp_path)
    item['evidence'] = str(Path(tmp_path).parent/'outside')
    with pytest.raises(ValueError, match='escapes'):
        recovery_eligibility(tmp_path, 'accepted', dict(recovery_exclusions=[item]), tmp_path)
