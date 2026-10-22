"""Separate recorded Stage 7 costs without summing duplicated graph timings."""

import argparse
import json
from pathlib import Path

import pandas as pd

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audits', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases, chains, seen = [], [], set()
    for audit_path in args.audits:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        for case in audit['runs']:
            folder = Path(case['directory'])
            if folder.resolve() in seen:
                raise ValueError(f'Duplicate timing case: {folder}')
            seen.add(folder.resolve())
            result_path = folder / 'result.json'
            if file_hash(result_path) != case['result_sha256']:
                raise ValueError(f'Changed result: {folder}')
            result = json.loads(result_path.read_text(encoding='utf-8'))
            for name, digest in result['artifacts'].items():
                if file_hash(folder / name) != digest:
                    raise ValueError(f'Changed timing source: {folder / name}')
            attempts = result['attempts']
            graph_builds = {}
            for attempt in attempts:
                build = attempt.get('graph_build_retry', 0)
                graph = attempt['graph_seconds']
                if build in graph_builds and graph_builds[build] != graph:
                    raise ValueError(f'Graph timing changed within one process: {folder}')
                graph_builds[build] = graph
            row = {'directory': folder.as_posix(), 'scenario': case['scenario'],
                   'seed': case['seed'], 'candidate': result['options']['prior_candidate'],
                   'status': case['status'], 'attempts': len(attempts),
                   'graph_seconds_once_per_process': sum(graph_builds.values()),
                   'graph_builds': len(graph_builds),
                   'result_sha256': file_hash(result_path)}
            for key in ('step_compile_seconds', 'sample_call_seconds',
                        'diagnostic_seconds', 'serialization_seconds'):
                row[key] = sum(a[key] for a in attempts)
            preflights = sorted(folder.glob('gradient_timing*.json'))
            row['preflight_compile_seconds'] = 0.
            row['preflight_first_call_seconds'] = 0.
            for preflight in preflights:
                timing = json.loads(preflight.read_text(encoding='utf-8'))
                row['preflight_compile_seconds'] += timing['function_compile_seconds']
                row['preflight_first_call_seconds'] += timing[
                    'first_call_including_deferred_jit_seconds']
                row[preflight.stem + '_median_repeat_seconds'] = timing[
                    'median_repeated_call_seconds']
            row['recorded_component_seconds'] = sum(row[key] for key in (
                'graph_seconds_once_per_process', 'step_compile_seconds', 'sample_call_seconds',
                'diagnostic_seconds', 'serialization_seconds')) + sum(
                    row.get(key, 0) for key in
                    ('preflight_compile_seconds', 'preflight_first_call_seconds'))
            cases.append(row)
            for attempt in attempts:
                for chain, markers in attempt.get('chain_wall_markers', {}).items():
                    item = {'directory': folder.as_posix(), 'retry': attempt['retry'],
                            'chain': int(chain), 'attempt_accepted': attempt['accepted'],
                            'early_terminated': bool(attempt.get('early_failure'))}
                    for phase in ('warmup', 'retained'):
                        for suffix in ('draws_completed', 'perf_counter_diff_sum',
                                       'process_time_diff_sum'):
                            key = f'{phase}_{suffix}'
                            item[key] = markers.get(key)
                    chains.append(item)
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(cases).to_csv(args.output / 'cases.csv', index=False)
    pd.DataFrame(chains).to_csv(args.output / 'chains.csv', index=False)
    write_json(args.output / 'audit.json', {
        'source_sha256': file_hash(Path(__file__)),
        'input_audits': {p.as_posix(): file_hash(p) for p in args.audits},
        'cases': len(cases),
        'recorded_component_job_seconds': sum(r['recorded_component_seconds'] for r in cases),
        'interpretation': 'Components are measured within individual concurrent processes; '
        'their sum is job-seconds, not campaign elapsed time. Graph construction is counted '
        'once per execution process, including checkpoint continuations. Sample-call time '
        'includes warmup, retained draws and internal setup, '
        'including deferred JIT; chain phase sums are breakdowns, not additional costs. '
        'Preflight repeat medians are descriptive and not added to totals. Python/import '
        'startup, post-admission forecast/recovery and outer coordinator overhead were not '
        'timed. Four chains were sequential; no four-process startup benchmark is implied. '
        'These are synthetic costs, not a production rolling-refit budget.',
        'artifacts': {p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()},
    })


if __name__ == '__main__':
    main()
