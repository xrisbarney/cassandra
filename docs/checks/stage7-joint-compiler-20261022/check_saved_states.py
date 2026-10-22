"""Check visited states saved at failed retained draws, not leapfrog failure points."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytensor
from threadpoolctl import threadpool_limits

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--difference-steps', nargs='+', type=float, default=[1e-5])
    args = parser.parse_args()
    if any(step <= 0 for step in args.difference_steps):
        raise ValueError('Difference steps must be positive.')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for folder in args.runs:
        run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        if run['real_data_read'] or run['held_out_scoring_enabled']:
            raise ValueError('Synthetic checks only.')
        with np.load(folder / 'data.npz') as loaded:
            data = dict(loaded)
        with np.load(folder / 'truth.npz') as loaded:
            truth = dict(loaded)
        model = build_model(data, run['options'], truth)
        fused = model.compile_fn([model.logp(), model.dlogp()],
                                 inputs=model.value_vars, mode='NUMBA')
        native = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars)
        for path in sorted(folder.glob('failed_state_*.npz')):
            with np.load(path) as loaded:
                point = {v.name: loaded[v.name].copy() for v in model.value_vars}
            a, b = fused(point), native(point)
            np.testing.assert_allclose(a[0], b[0], rtol=1e-8, atol=1e-6)
            np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-6)
            assert np.isfinite(a[0]) and np.isfinite(a[1]).all()
            differences, offset = [], 0
            for variable in model.value_vars:
                name = variable.name
                # Both factor coefficients, and first/last entries of each other block.
                indices = sorted({0, point[name].size - 1})
                for index in indices:
                    analytic = float(b[1][offset + index])
                    estimates = []
                    for step in args.difference_steps:
                        delta = np.zeros_like(point[name])
                        delta.flat[index] = step
                        plus, minus = dict(point), dict(point)
                        plus[name], minus[name] = point[name] + delta, point[name] - delta
                        fd = float((native(plus)[0] - native(minus)[0]) / (2 * step))
                        estimates.append({'step': step, 'finite_difference': fd,
                                          'absolute_error': abs(fd - analytic),
                                          'passed': bool(np.isclose(fd, analytic,
                                                                   rtol=5e-5, atol=5e-5))})
                    if len(estimates) == 1:
                        passed = estimates[0]['passed']
                    else:
                        passed = any(
                            left['passed'] and right['passed'] and np.isclose(
                                left['finite_difference'], right['finite_difference'],
                                rtol=5e-5, atol=5e-5)
                            for left, right in zip(estimates[:-1], estimates[1:], strict=True)
                        )
                    differences.append({'variable': name, 'index': index,
                                        'analytic': analytic, 'estimates': estimates,
                                        'passed': bool(passed)})
                offset += point[name].size
            record = {'directory': folder.as_posix(), 'checkpoint': path.name,
                      'checkpoint_sha256': file_hash(path),
                      'run_sha256': file_hash(folder / 'run.json'),
                      'passed': all(d['passed'] for d in differences),
                      'logp_abs_error': float(abs(a[0] - b[0])),
                      'gradient_max_abs_error': float(np.max(abs(a[1] - b[1]))),
                      'finite_differences': differences}
            records.append(record)
            write_json(args.output / f'checkpoint_{len(records)}.json', record)
            print(json.dumps({k: record[k] for k in ('directory', 'checkpoint', 'passed',
                                                    'gradient_max_abs_error')}), flush=True)
    if not records:
        raise ValueError('No saved failure checkpoints were checked.')
    write_json(args.output / 'audit.json', {
        'passed': all(r['passed'] for r in records),
        'source_sha256': file_hash(Path(__file__)),
        'model_source_sha256': {p.as_posix(): file_hash(p)
                                for p in Path('src/cyber_threat_modelling/models').glob('*.py')},
        'numba_opt': os.environ.get('NUMBA_OPT'),
        'numba_fastmath': bool(pytensor.config.numba__fastmath),
        'checkpoints': len(records), 'real_data_read': False,
        'difference_steps': args.difference_steps,
        'difference_rule': 'Original 5e-5 absolute/relative tolerance. If a step grid is '
        'requested, two adjacent estimates must agree with each other and the analytic '
        'gradient. All estimates, including failures, are retained.',
        'interpretation': 'Complete target and gradient at saved visited states, with selected '
        'finite differences. A rejected leapfrog proposal may differ from the saved state; '
        'passing these checks neither explains nor waives a divergence.',
        'artifacts': {p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()},
    })
    if not all(r['passed'] for r in records):
        raise RuntimeError('A saved-state numerical check remains unresolved; see audit.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
