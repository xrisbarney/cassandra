"""Close the four-case round from verified imports without admitting failed fits."""

import json
from pathlib import Path

import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import verify_fit_authority
from cyber_threat_modelling.provenance import file_hash

root = Path.cwd()
output = Path(__file__).resolve().parent
execution_path = root/'docs/decisions/STAGE7_REPAIR_EXECUTION_20261022.json'
execution = json.loads(execution_path.read_text())
rows = []
for case, directory in execution['case_directories'].items():
    verify_fit_authority(root, case)
    folder = root/directory
    result = json.loads((folder/'result.json').read_text())
    imported = json.loads((output/'imports'/f'{case}.json').read_text())
    assert imported['result_sha256'] == file_hash(folder/'result.json')
    assert imported['github_sha'] == execution['head_sha']
    assert int(imported['github_run_id']) == execution['github_run_id']
    assert result['attempt_count'] == 1
    for name, digest in result['artifacts'].items():
        assert file_hash(folder/name) == digest
    preflight = json.loads((folder/'preflight.json').read_text())
    markers = list(result['chain_wall_markers'].values())
    timing = {key: value for key, value in result.items() if key.endswith('_seconds')}
    timing['preflight_compile_seconds'] = preflight['explicit_compile_seconds']
    for phase in ('warmup', 'retained'):
        timing[f'{phase}_step_seconds_sum'] = sum(
            m[f'{phase}_perf_counter_diff_sum'] for m in markers)
    row = {'case': case, 'status': result['status'], 'diagnostics': result['diagnostics'],
           'early_failure': result['early_failure'], 'timing': timing,
           'warmup_draws_completed': sum(m['warmup_draws_completed'] for m in markers),
           'retained_draws_completed': sum(m['retained_draws_completed'] for m in markers),
           'complete_posterior_independently_recomputed':
           imported['complete_posterior_recomputed'],
           'result_sha256': imported['result_sha256'],
           'artifact_sha256': result['artifacts'],
           'preflight_passed': preflight['passed'],
           'unresolved_warnings': result['unresolved_warnings']}
    if imported['complete_posterior_recomputed']:
        diagnostic = pd.read_csv(folder/'diagnostics.csv', index_col=0)
        row['lowest_tail_ess_rows'] = diagnostic.sort_values('ess_tail').head(12).to_dict(
            orient='index')
        row['highest_rhat_rows'] = diagnostic.sort_values('r_hat', ascending=False).head(
            12).to_dict(orient='index')
        row['lowest_bulk_ess_rows'] = diagnostic.sort_values('ess_bulk').head(12).to_dict(
            orient='index')
        row['diagnostic_row_count'] = len(diagnostic)
        row['tail_ess_failing_rows_including_derived_duplicates'] = int(
            (diagnostic.ess_tail <= 400).sum())
    rows.append(row)
accepted = [r['case'] for r in rows if r['status'] == 'accepted']
write_json(output/'audit.json', {
    'status': 'four_terminal_outcomes', 'execution_sha256': file_hash(execution_path),
    'script_sha256': file_hash(Path(__file__)), 'cases': rows,
    'numerical_passes': accepted, 'recovery_eligibility_requires_separate_review': accepted,
    'attempts': 4, 'stage8_enabled': False, 'held_out_scoring_enabled': False,
    'timing_caution': 'Sampling calls include warmup, retained transitions and overhead. '
                      'Step times are nested in those calls, not additive costs. '
                      'Overlapping worker costs are not campaign elapsed time. '
                      'Preflight compilation is distinct from NUTS step construction.'})
print(json.dumps({'terminal_cases': len(rows), 'numerical_passes': accepted}))
