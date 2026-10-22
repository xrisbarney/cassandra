"""Investigate a failed difference check without changing its tolerance."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash

with threadpool_limits(limits=1):
    folder = Path('artifacts/stage7/rank-two-joint-v1/primary/fallback_zero-7102')
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    with np.load(folder / 'data.npz') as f:
        data = dict(f)
    with np.load(folder / 'truth.npz') as f:
        truth = dict(f)
    model = build_model(data, run['options'], truth)
    with np.load(folder / 'failed_state_0.npz') as f:
        point = {v.name: f[v.name].copy() for v in model.value_vars}
    function = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars)
    base, gradient = function(point)
    records, offset = [], 0
    for variable in model.value_vars:
        name = variable.name
        if name == 'f_phi_interval__':
            for index in range(point[name].size):
                differences = []
                for step in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
                    delta = np.zeros_like(point[name])
                    delta.flat[index] = step
                    plus, minus = dict(point), dict(point)
                    plus[name], minus[name] = point[name] + delta, point[name] - delta
                    a, b = function(plus)[0], function(minus)[0]
                    fd = float((a - b) / (2 * step))
                    analytic = float(gradient[offset + index])
                    differences.append({'step': step, 'finite_difference': fd,
                                        'absolute_error': abs(fd - analytic),
                                        'passes_original_tolerance': bool(np.isclose(
                                            fd, analytic, rtol=5e-5, atol=5e-5))})
                records.append({'variable': name, 'index': index, 'analytic': analytic,
                                'persistence': float(2 * expit(point[name][index]) - 1),
                                'differences': differences})
        offset += point[name].size
    output = Path('docs/checks/stage7-saved-state-numerics-20261022/difference_steps.json')
    record = {'source_sha256': file_hash(Path(__file__)),
              'checkpoint_sha256': file_hash(folder / 'failed_state_0.npz'),
              'run_sha256': file_hash(folder / 'run.json'), 'logp': float(base),
              'records': records, 'interpretation': 'Step sensitivity diagnosis only; '
              'the original failed check is preserved. No acceptance tolerance changed.'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record), flush=True)
