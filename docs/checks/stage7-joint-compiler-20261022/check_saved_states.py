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
    args = parser.parse_args()
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
                    delta = np.zeros_like(point[name])
                    delta.flat[index] = 1e-5
                    plus, minus = dict(point), dict(point)
                    plus[name], minus[name] = point[name] + delta, point[name] - delta
                    fd = float((native(plus)[0] - native(minus)[0]) / 2e-5)
                    analytic = float(b[1][offset + index])
                    np.testing.assert_allclose(fd, analytic, rtol=5e-5, atol=5e-5,
                                               err_msg=f'{folder}: {name}[{index}]')
                    differences.append({'variable': name, 'index': index,
                                        'finite_difference': fd, 'analytic': analytic,
                                        'absolute_error': abs(fd - analytic)})
                offset += point[name].size
            record = {'directory': folder.as_posix(), 'checkpoint': path.name,
                      'checkpoint_sha256': file_hash(path),
                      'run_sha256': file_hash(folder / 'run.json'),
                      'passed': True, 'logp_abs_error': float(abs(a[0] - b[0])),
                      'gradient_max_abs_error': float(np.max(abs(a[1] - b[1]))),
                      'finite_differences': differences}
            records.append(record)
            write_json(args.output / f'checkpoint_{len(records)}.json', record)
            print(json.dumps({k: record[k] for k in ('directory', 'checkpoint', 'passed',
                                                    'gradient_max_abs_error')}), flush=True)
    if not records:
        raise ValueError('No saved failure checkpoints were checked.')
    write_json(args.output / 'audit.json', {
        'passed': True, 'source_sha256': file_hash(Path(__file__)),
        'model_source_sha256': {p.as_posix(): file_hash(p)
                                for p in Path('src/cyber_threat_modelling/models').glob('*.py')},
        'numba_opt': os.environ.get('NUMBA_OPT'),
        'numba_fastmath': bool(pytensor.config.numba__fastmath),
        'checkpoints': len(records), 'real_data_read': False,
        'interpretation': 'Complete target and gradient at saved visited states, with selected '
        'finite differences. A rejected leapfrog proposal may differ from the saved state; '
        'passing these checks neither explains nor waives a divergence.',
        'artifacts': {p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()},
    })


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
