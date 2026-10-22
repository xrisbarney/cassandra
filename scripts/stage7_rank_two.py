"""Run the declared rank-two grid under an explicit computational admission decision."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.stage7_priors import CANDIDATES, CURRENT_PRIOR_ID
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representative', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--synthetic-grid', action='store_true',
                        help='Complete the declared synthetic grid after verified joint algebra, '
                             'independently of representative posterior admission.')
    args = parser.parse_args()
    if args.synthetic_grid:
        decision = Path('docs/decisions/STAGE7_SYNTHETIC_GRID_COMPLETION_20261022.md')
        audit_path = Path('docs/checks/stage7-full-tests-v5-20261022/audit.json')
        audit = json.loads(audit_path.read_text())
        if not audit['all_passed'] or not audit['source_unchanged']:
            raise ValueError('Joint coordinate verification did not pass.')
        for name in ['joint_coordinates.py', 'gaussian_coordinates.py', 'block_gaussian.py',
                     'synthetic_state_space.py', 'stage7_priors.py']:
            relative = f'src/cyber_threat_modelling/models/{name}'
            if file_hash(Path(relative)) != audit['source_sha256'][relative]:
                raise ValueError(f'Untested model change: {relative}')
        generator = 'src/cyber_threat_modelling/evaluation/synthetic.py'
        if file_hash(Path(generator)) != audit['source_sha256'][generator]:
            raise ValueError('Untested synthetic generator change.')
        parent_path = args.representative/'run.json'
        run = json.loads(parent_path.read_text())
        if run['real_data_read'] or run['held_out_scoring_enabled']:
            raise ValueError('The independent grid is synthetic only.')
        options = run['options']
        if (options.get('level_parameterization') != 'joint_conditional'
                or options['mass_matrix'] != 'diagonal'):
            raise ValueError('The synthetic-grid decision fixes joint/diagonal coordinates.')
        selection = {
            'selection': 'Verified joint coordinates fixed before grid results; numerical '
                         'admission remains required for every recovery result.',
            'parent_run_sha256': file_hash(parent_path),
            'decision_sha256': file_hash(decision),
            'verification_sha256': file_hash(audit_path),
        }
    else:
        result_path = args.representative/'result.json'
        while not result_path.exists():
            time.sleep(30)
        result = json.loads(result_path.read_text())
        if result['status'] != 'accepted':
            raise RuntimeError('Representative not admitted; do not start the dependent grid.')
        options = result['options']
        accepted = result['attempts'][-1]
        if not accepted['accepted'] or accepted['unresolved_warnings'] or accepted['chains'] != 4:
            raise ValueError('Inconsistent representative admission.')
        for relative, expected in result['artifacts'].items():
            if file_hash(args.representative/relative) != expected:
                raise ValueError(f'Changed representative artifact: {relative}')
        selection = {
            'result_sha256': file_hash(result_path),
            'selection': 'Predeclared diagnostic admission; recovery scores were not read.',
        }
    for key, expected in [('prior_id', CURRENT_PRIOR_ID), ('prior_candidate', 'primary'),
                          ('scenario', 'representative'), ('seed', 7101),
                          ('count_parameterization', 'conditional'), ('gaussian_solver', 'block')]:
        if options[key] != expected:
            raise ValueError(f'Unexpected representative {key}.')
    metric = options['mass_matrix']
    if metric not in {'dense', 'diagonal'}:
        raise ValueError('Unknown representative metric.')
    args.root.mkdir(parents=True, exist_ok=False)
    write_json(args.root/'selection.json', {
        'representative': args.representative.as_posix(),
        **selection, 'metric': metric,
        'synthetic_grid': args.synthetic_grid,
        'numba_optimization_env': os.environ.get('NUMBA_OPT', '3 (library default)'),
        'level_parameterization': options.get('level_parameterization'),
        'source_sha256': file_hash(Path(__file__)), 'workers': args.workers,
    })
    jobs = [('primary', 'fallback_zero', seed) for seed in [7101, 7102, 7103]]
    jobs += [(candidate, scenario, 7101) for candidate in CANDIDATES if candidate != 'primary'
             for scenario in ['representative', 'fallback_zero']]

    def execute(candidate, scenario, seed):
        log = args.root/f'{candidate}-{scenario}-{seed}.log'
        command = [sys.executable, '-u', 'scripts/stage7_batch.py', '--root',
                   str(args.root/candidate), '--include', scenario, '--seeds', str(seed),
                   '--workers', '1', '--prior-candidate', candidate, '--anchor-factors',
                   '--conditional-counts', '--block-coordinates']
        if metric == 'dense':
            command.append('--dense-mass')
        if options.get('level_parameterization') == 'joint_conditional':
            command.append('--joint-levels')
        with log.open('xb') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        path = args.root/candidate/f'{scenario}-{seed}'/'result.json'
        status = json.loads(path.read_text())['status'] if path.exists() else 'execution_failed'
        return {'candidate': candidate, 'scenario': scenario, 'seed': seed,
                'status': status, 'coordinator_exit_code': process.returncode}

    outcomes = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute, *job) for job in jobs]):
            outcome = future.result()
            outcomes.append(outcome)
            write_json(args.root/'progress.json', outcomes)
            print(json.dumps(outcome), flush=True)
    write_json(args.root/'outcomes.json', outcomes)
    if any(r['status'] not in {'accepted', 'failed_diagnostic_gate'} for r in outcomes):
        raise RuntimeError('The rank-two grid has unfinished or failed executions.')


if __name__ == '__main__':
    main()
