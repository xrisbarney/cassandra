"""Verify the complete v2 campaign without treating failed fits as recovery evidence."""

import argparse
import json
from collections import Counter
from pathlib import Path

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic import SCENARIOS
from cyber_threat_modelling.models.stage7_priors import (
    CANDIDATES,
    CURRENT_PRIOR_ID,
    verify_activation,
)
from cyber_threat_modelling.provenance import file_hash


def checked_run(folder, candidate, scenario, seed, activation):
    result = json.loads((folder/'result.json').read_text())
    run = json.loads((folder/'run.json').read_text())
    options = result['options']
    for key, expected in [('prior_id', CURRENT_PRIOR_ID), ('prior_candidate', candidate),
                          ('scenario', scenario), ('seed', seed)]:
        if options[key] != expected or run['options'][key] != expected:
            raise ValueError(f'Wrong {key}: {folder}')
    if run['real_data_read'] or run['held_out_scoring_enabled']:
        raise ValueError(f'Unexpected real-data access: {folder}')
    if run['prior_amendment']['sha256'] != activation['sha256']:
        raise ValueError(f'Wrong prior activation: {folder}')
    for relative, expected in result['artifacts'].items():
        if file_hash(folder/relative) != expected:
            raise ValueError(f'Changed artifact: {folder/relative}')
    for filename, key in [('data.npz', 'data_sha256'), ('truth.npz', 'truth_sha256'),
                          ('truth_original_coordinates.npz', 'truth_original_coordinates_sha256')]:
        if file_hash(folder/filename) != run[key]:
            raise ValueError(f'Changed generating record: {folder/filename}')
    admitted = [attempt for attempt in result['attempts'] if attempt['accepted']]
    if result['status'] == 'accepted':
        if len(admitted) != 1 or not admitted[0]['diagnostics']['accepted']:
            raise ValueError(f'Inconsistent admission: {folder}')
        if admitted[0]['unresolved_warnings']:
            raise ValueError(f'Unresolved admitted warning: {folder}')
    elif result['status'] == 'failed_diagnostic_gate':
        if admitted or len(result['attempts']) != 3 or (folder/'recovery.json').exists():
            raise ValueError(f'Incomplete or mislabelled failed case: {folder}')
    else:
        raise ValueError(f'Unknown terminal status: {folder}')
    return {
        'directory': folder.as_posix(), 'scenario': scenario, 'seed': seed,
        'candidate': candidate, 'status': result['status'],
        'attempts': len(result['attempts']), 'result_sha256': file_hash(folder/'result.json'),
        'data_sha256': run['data_sha256'], 'truth_sha256': run['truth_sha256'],
        'original_truth_sha256': run['truth_original_coordinates_sha256'],
        'source_control': run['source_control'],
        'compute_thread_policy': run.get('compute_thread_policy', 'original library defaults'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, default=Path('artifacts/stage7/recovery-v6'))
    parser.add_argument('--sensitivity', type=Path, default=Path('artifacts/stage7/sensitivity-v2'))
    parser.add_argument('--severity-repair', type=Path,
                        default=Path('artifacts/stage7/severity-conditional-v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    activation = verify_activation(Path.cwd(), CURRENT_PRIOR_ID)
    records, primary = [], {}
    for scenario in SCENARIOS:
        for seed in [7101, 7102, 7103]:
            if scenario == 'representative' and seed != 7101:
                continue
            record = checked_run(args.primary/f'{scenario}-{seed}', 'primary', scenario,
                                 seed, activation)
            record['campaign'] = 'primary'
            records.append(record)
            primary[scenario, seed] = record
    for candidate in CANDIDATES:
        if candidate == 'primary':
            continue
        for scenario in ['factors_sparse', 'representative', 'fallback_zero']:
            record = checked_run(args.sensitivity/candidate/f'{scenario}-7101', candidate,
                                 scenario, 7101, activation)
            record['campaign'] = 'sensitivity'
            records.append(record)
    # The predeclared repair continues beyond seed 7101 only if that fit passes.
    first = checked_run(args.severity_repair/'severity_missing-7101', 'primary',
                        'severity_missing', 7101, activation)
    first['campaign'] = 'severity_coordinates'
    records.append(first)
    if first['status'] == 'accepted':
        for seed in [7102, 7103]:
            record = checked_run(args.severity_repair/f'severity_missing-{seed}', 'primary',
                                 'severity_missing', seed, activation)
            record['campaign'] = 'severity_coordinates'
            records.append(record)
    for record in records:
        reference = primary[record['scenario'], record['seed']]
        for key in ['data_sha256', 'truth_sha256', 'original_truth_sha256']:
            if record[key] != reference[key]:
                raise ValueError(f'Datasets differ across repeated fits: {record["directory"]}')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'audit.json', {
        'prior_id': CURRENT_PRIOR_ID, 'activation': activation,
        'complete': True, 'unique_primary_datasets': len(primary),
        'counts': {name: dict(Counter(r['status'] for r in records if r['campaign'] == name))
                   for name in ['primary', 'sensitivity', 'severity_coordinates']},
        'runs': records, 'held_out_scoring_enabled': False,
        'interpretation': 'Completion includes explicit diagnostic failures. Sensitivity and '
                          'coordinate refits are not independent generated datasets.',
        'source_sha256': file_hash(Path(__file__)),
    })
    print(json.dumps({'complete': True, 'fits': len(records), 'datasets': len(primary)}))


if __name__ == '__main__':
    main()
