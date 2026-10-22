"""A completed folder cannot substitute another dataset during batch resume."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(('scenario', 'seed'), [('latent_ar', 7101), ('independent', 7102)])
def test_resume_rejects_wrong_dataset_identity(tmp_path, scenario, seed):
    folder = tmp_path/'independent-7101'
    folder.mkdir()
    (folder/'result.json').write_text(json.dumps({
        'options': {'scenario': scenario, 'seed': seed}, 'status': 'accepted',
    }), encoding='utf-8')
    script = Path(__file__).resolve().parents[1]/'scripts/stage7_batch.py'
    result = subprocess.run([
        sys.executable, str(script), '--root', str(tmp_path), '--include', 'independent',
        '--seeds', '7101', '--workers', '1',
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'different scenario or seed' in result.stderr
    assert not (folder/'run.json').exists()
