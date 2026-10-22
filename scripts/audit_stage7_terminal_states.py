"""Audit preserved failure points without sampling or altering sealed sources."""

import argparse
import time
from pathlib import Path

import numpy as np
from run_stage7_supplement import classify
from stage7_terminal_oracle import dense_density

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import load_fixture, verify_fit_authority
from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash


def main():
    import warnings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['masked_fallback', 'matched_one_factor_fallback'],
                        required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_fit_authority(root, args.case)
    output = root/'artifacts/stage7/terminal-v1/audit'/args.case
    output.mkdir(parents=True, exist_ok=False)
    data, truth, reference, options = load_fixture(root, args.case)
    source = root/'artifacts/stage7/supplement-v1'/args.case/'failed_state.npz'
    with np.load(source) as saved:
        point = dict(saved)
    rng = np.random.default_rng(221022)
    points = [('saved_callback_point', point)]
    for index in range(2):
        points.append((f'perturbation_{index+1}', {
            k: v+rng.normal(0, 1e-4, v.shape) for k, v in point.items()}))
    start = time.perf_counter()
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = build_model(data, options, truth,
                            level_reference=reference if args.case == 'masked_fallback' else None)
        native = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars)
        compiled = model.compile_fn([model.logp(), model.dlogp()],
                                    inputs=model.value_vars, mode='NUMBA')
        for label, p in points:
            a, b = native(p), compiled(p)
            oracle, canonical = dense_density(p, data, options)
            np.testing.assert_allclose(a[0], b[0], rtol=1e-8, atol=1e-6)
            np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(a[0], oracle, rtol=1e-8, atol=1e-6)
            assert np.isfinite(a[0]) and np.isfinite(a[1]).all()
            derivatives, offset = [], 0
            for variable in model.value_vars:
                name = variable.name
                for index in sorted({0, p[name].size-1}):
                    delta = np.zeros_like(p[name])
                    delta.flat[index] = 1e-5
                    plus, minus = dict(p), dict(p)
                    plus[name], minus[name] = p[name]+delta, p[name]-delta
                    fd = (dense_density(plus, data, options)[0]
                          - dense_density(minus, data, options)[0])/2e-5
                    actual = float(a[1][offset+index])
                    np.testing.assert_allclose(fd, actual, rtol=5e-5, atol=5e-5)
                    derivatives.append({'variable': name, 'index': index,
                                        'analytic': actual, 'independent_fd': fd})
                offset += p[name].size
            rows.append({'point': label, 'native_logp': float(a[0]), 'dense_logp': oracle,
                         'compiled_logp': float(b[0]), 'derivatives': derivatives,
                         'phi': canonical['phi'].tolist(),
                         'constant_shift_prior_curvature':
                         canonical['shift_prior_curvature'].tolist()})
    performance, unresolved = classify(caught)
    write_json(output/'audit.json', {
        'case': args.case, 'passed': not unresolved, 'points': rows,
        'saved_state_sha256': file_hash(source), 'elapsed_seconds': time.perf_counter()-start,
        'performance_warnings': performance, 'unresolved_warnings': unresolved,
        'source_sha256': {Path(__file__).name: file_hash(Path(__file__)),
                          'stage7_terminal_oracle.py':
                          file_hash(Path(__file__).with_name('stage7_terminal_oracle.py'))},
        'sampling_performed': False,
        'limitation': 'Callback points are not full divergent trajectories. Local agreement '
                      'does not certify all integration paths or identify a unique mechanism.'})
    if unresolved:
        raise ValueError('Unresolved audit warning.')
    print(args.case, 'audit passed', flush=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
