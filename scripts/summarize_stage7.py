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
    args = parser.parse_args()
    expected = [(name, seed) for name in SCENARIOS for seed in (7101, 7102, 7103)
                if name != 'representative' or seed == 7101]
    diagnostics, recoveries, parameters, audit = [], [], [], []
    for name, seed in expected:
        folder = args.runs / f'{name}-{seed}'
        path = folder / 'result.json'
        record = json.loads(path.read_text())
        for relative, digest in record['artifacts'].items():
            if file_hash(folder / relative) != digest:
                raise ValueError(f'Changed artifact: {folder / relative}')
        run = json.loads((folder / 'run.json').read_text())
        audit.append({'scenario': name, 'seed': seed, 'status': record['status'],
                      'directory': folder.as_posix(), 'result_sha256': file_hash(path),
                      'source_control': run['source_control'],
                      'data_sha256': run['data_sha256'], 'truth_sha256': run['truth_sha256']})
        for attempt in record['attempts']:
            diagnostics.append({
                'scenario': name, 'seed': seed, 'retry': attempt['retry'],
                'accepted': attempt['accepted'], **attempt['diagnostics'],
                **{k: v for k, v in attempt.items() if k.endswith('_seconds')},
                'unresolved_warnings': json.dumps(attempt['unresolved_warnings']),
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
