"""Resume completed synthetic cases by hash; preserve failures and partial runs.

This coordinates ordinary local Python processes, not agents. Each fit uses
four sequential chains so the requested worker limit bounds sampling concurrency.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cyber_threat_modelling.evaluation.synthetic import SCENARIOS
from cyber_threat_modelling.models.stage7_priors import PRIOR_ID
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--exclude', nargs='*', default=[])
    coordinates = parser.add_mutually_exclusive_group()
    coordinates.add_argument('--center-factors', action='store_true')
    coordinates.add_argument('--anchor-factors', action='store_true')
    parser.add_argument('--dense-mass', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    jobs = [(name, seed) for name in SCENARIOS for seed in (7101, 7102, 7103)
            if name not in args.exclude and (name != 'representative' or seed == 7101)]

    def execute(name, seed):
        folder = args.root / f'{name}-{seed}'
        if folder.exists():
            result = folder / 'result.json'
            if not result.exists():
                return {'scenario': name, 'seed': seed, 'status': 'partial_run_preserved'}
            record = json.loads(result.read_text())
            if record['options'].get('prior_id') != PRIOR_ID:
                raise ValueError(f'Existing run has a different prior: {folder}')
            requested = ('anchored' if args.anchor_factors else
                         'centered' if args.center_factors else 'innovations')
            if record['options'].get('factor_parameterization', 'innovations') != requested:
                raise ValueError(f'Existing run has different factor coordinates: {folder}')
            metric = 'dense' if args.dense_mass else 'diagonal'
            if record['options'].get('mass_matrix', 'diagonal') != metric:
                raise ValueError(f'Existing run has a different mass matrix: {folder}')
            for relative, digest in record['artifacts'].items():
                if file_hash(folder / relative) != digest:
                    raise ValueError(f'Changed artifact: {folder / relative}')
            return {'scenario': name, 'seed': seed, 'status': record['status'], 'resumed': True}
        log = args.root / f'{name}-{seed}.log'
        with log.open('xb') as stream:
            command = [
                sys.executable, '-u', 'scripts/run_stage7.py', '--scenario', name,
                '--seed', str(seed), '--output', str(folder),
            ]
            if args.center_factors:
                command.append('--center-factors')
            if args.anchor_factors:
                command.append('--anchor-factors')
            if args.dense_mass:
                command.append('--dense-mass')
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        record = json.loads((folder / 'result.json').read_text()) if (
            folder / 'result.json'
        ).exists() else {'status': 'execution_failed'}
        return {'scenario': name, 'seed': seed, 'status': record['status'],
                'exit_code': process.returncode}

    outcomes = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, name, seed) for name, seed in jobs]
        for future in as_completed(futures):
            record = future.result()
            outcomes.append(record)
            print(json.dumps(record), flush=True)
    # Each coordinator invocation is distinct; completed fit directories are immutable.
    print(json.dumps({'outcomes': outcomes}), flush=True)


if __name__ == '__main__':
    main()
