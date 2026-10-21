"""Assemble immutable synthetic fit/recovery audits, with explicit failure denominators."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic import SCENARIOS
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls-from', type=Path)
    parser.add_argument('--include', nargs='*', choices=SCENARIOS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=[7101, 7102, 7103],
                        default=[7101, 7102, 7103])
    args = parser.parse_args()
    expected = [(name, seed) for name in SCENARIOS for seed in args.seeds
                if (name != 'representative' or seed == 7101)
                and (not args.include or name in args.include)]
    diagnostics, recoveries, parameters, audit = [], [], [], []
    for name, seed in expected:
        source = args.controls_from if args.controls_from and name in (
            'independent', 'latent_ar'
        ) else args.runs
        folder = source / f'{name}-{seed}'
        path = folder / 'result.json'
        record = json.loads(path.read_text())
        for relative, digest in record['artifacts'].items():
            if file_hash(folder / relative) != digest:
                raise ValueError(f'Changed artifact: {folder / relative}')
        run = json.loads((folder / 'run.json').read_text())
        audit.append({'scenario': name, 'seed': seed, 'status': record['status'],
                      'directory': folder.as_posix(), 'result_sha256': file_hash(path),
                      'options': record['options'],
                      'source_control': run['source_control'],
                      'data_sha256': run['data_sha256'], 'truth_sha256': run['truth_sha256']})
        for attempt in record['attempts']:
            markers = list(attempt.get('chain_wall_markers', {}).values())
            measured = bool(markers) and all('warmup_perf_counter_diff_sum' in m for m in markers)
            diagnostics.append({
                'scenario': name, 'seed': seed, 'retry': attempt['retry'],
                **attempt['diagnostics'], 'accepted': attempt['accepted'],
                **{k: v for k, v in attempt.items() if k.endswith('_seconds')},
                'unresolved_warnings': json.dumps(attempt['unresolved_warnings']),
                'early_terminated': bool(attempt.get('early_failure')),
                'early_failure': json.dumps(attempt.get('early_failure')),
                'diagnostic_counts_are_lower_bounds': bool(attempt.get('early_failure')),
                'peak_working_set_bytes': attempt.get('peak_working_set_bytes'),
                'warmup_sampler_wall_seconds': sum(
                    m['warmup_perf_counter_diff_sum'] for m in markers
                ) if measured else None,
                'retained_sampler_wall_seconds': sum(
                    m.get('retained_perf_counter_diff_sum', 0) for m in markers
                ) if measured else None,
                'retained_draws_completed': sum(
                    m.get('retained_draws_completed', 0) for m in markers
                ) if markers and all('retained_draws_completed' in m for m in markers) else None,
            })
        if record['status'] != 'accepted':
            continue
        recovery = json.loads((folder / 'recovery.json').read_text())
        for quantity in ['eta_operational', 'eta_generating_interpretation',
                         'expected_observed_counts', 'training_posterior_predictive',
                         'severity_location', 'combined_relative_overdispersion']:
            if quantity in recovery:
                recoveries.append({'scenario': name, 'seed': seed, 'quantity': quantity,
                                   **recovery[quantity]})
        for h, metrics in recovery['future_predictive'].items():
            recoveries.append({'scenario': name, 'seed': seed,
                               'quantity': f'future_counts_h{h}', **metrics})
        for parameter, stats in recovery['parameters'].items():
            truth, mean, sd, lo, hi = [np.asarray(stats[key]).ravel() for key in
                                       ['truth', 'mean', 'sd', 'q05', 'q95']]
            for j in range(len(mean)):
                parameters.append({'scenario': name, 'seed': seed, 'parameter': parameter,
                                   'index': j, 'truth': truth[j], 'mean': mean[j], 'sd': sd[j],
                                   'q05': lo[j], 'q95': hi[j]})
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(diagnostics).to_csv(args.output/'diagnostics.csv', index=False)
    pd.DataFrame(recoveries).to_csv(args.output/'recovery.csv', index=False)
    pd.DataFrame(parameters).to_csv(args.output/'parameters.csv', index=False)
    write_json(args.output/'audit.json', {
        'runs': audit, 'expected_datasets': len(expected),
        'accepted': sum(r['status'] == 'accepted' for r in audit),
        'held_out_scoring_enabled': False, 'real_data_read': False,
        'interpretation': 'Fixed-truth stress/recovery cases; not SBC or real-data validation.',
        'artifacts': {p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()},
    })


if __name__ == '__main__':
    main()
