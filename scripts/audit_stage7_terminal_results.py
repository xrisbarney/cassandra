"""Independently audit the two terminal diagnostics, including incomplete attempts."""

import json
from contextlib import closing
from pathlib import Path

import arviz as az
import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.bayesian_controls import diagnostic_report
from cyber_threat_modelling.provenance import file_hash

root = Path(__file__).resolve().parents[1]
base = root/'artifacts/stage7/terminal-v1'
activation_path = root/'docs/decisions/STAGE7_TERMINAL_DIAGNOSTICS_20261022.json'
activation = json.loads(activation_path.read_text())
for name, digest in activation['files_sha256'].items():
    assert file_hash(root/name) == digest, name
rows = []
for case in ['joint_free', 'joint_fixed']:
    folder = base/'fits'/case
    supervisor_path = base/(case+'-supervisor.json')
    supervisor = json.loads(supervisor_path.read_text())
    run = json.loads((folder/'run.json').read_text())
    assert run['activation_sha256'] == file_hash(activation_path)
    assert run['attempt_count'] == 1 and run['chains'] == 4
    assert run['draws'] == run['warmup'] == 2000
    assert run['target_accept'] == .99 and run['max_treedepth'] == 15
    assert run['sampler_seed'] == 7103 and run['initialization_seed'] == 7301
    assert not run['stage8_enabled'] and not run['held_out_scoring_enabled']
    row = {'case': case, 'supervisor': supervisor, 'run': run,
           'supervisor_sha256': file_hash(supervisor_path),
           'log_sha256': file_hash(base/(case+'-fit.log')),
           'diagnostics_independently_recomputed': False}
    if (folder/'result.json').exists():
        result = json.loads((folder/'result.json').read_text())
        for name, digest in result['artifacts'].items():
            assert file_hash(folder/name) == digest
        assert set(p.name for p in folder.iterdir()) == set(result['artifacts']) | {'result.json'}
        if result['diagnostics'] is not None:
            with closing(az.from_netcdf(folder/'posterior.nc')) as idata:
                assert idata.posterior.sizes['chain'] == 4
                assert idata.posterior.sizes['draw'] == 2000
                computed, table = diagnostic_report(idata, 15)
            for key, value in result['diagnostics'].items():
                if value is None:
                    assert not np.isfinite(computed[key])
                else:
                    np.testing.assert_allclose(computed[key], value, rtol=1e-10, atol=1e-10)
            row['diagnostics_independently_recomputed'] = True
            row['worst_rhat'] = table.sort_values('r_hat', ascending=False).head(5).to_dict('index')
            accepted = computed['accepted'] and not result['unresolved_warnings']
            assert (result['status'] == 'accepted') == accepted
        else:
            assert result['status'] != 'accepted'
        row['result'] = result
        row['result_sha256'] = file_hash(folder/'result.json')
        row['status'] = result['status']
    else:
        assert supervisor['timed_out'] or supervisor['returncode'] != 0
        row['status'] = ('time_budget_exhausted' if supervisor['timed_out']
                         else 'execution_failure')
        row['partial_artifacts'] = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
    rows.append(row)
output = root/'docs/simulations/stage7-terminal-20261022'
output.mkdir(parents=True, exist_ok=True)
assert not (output/'diagnostics.json').exists()
write_json(output/'diagnostics.json', {
    'cases': rows, 'attempts': 2, 'activation_sha256': file_hash(activation_path),
    'source_sha256': file_hash(Path(__file__)), 'stage8_enabled': False,
    'held_out_scoring_enabled': False,
    'fixed_fit_is_conditional_evidence_only': True})
print(json.dumps([{'case': r['case'], 'status': r['status']} for r in rows]))
