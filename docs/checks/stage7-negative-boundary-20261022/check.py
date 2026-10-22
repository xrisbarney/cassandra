"""Check an actual retained negative-persistence boundary state, without refitting."""

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.special import logit
from threadpoolctl import threadpool_limits

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash


def main():
    folder = Path('artifacts/stage7/hosted-downloads/39754804660/attempt1/'
                  'dynamics_tight-representative-7101/rank-two-joint-v1/'
                  'dynamics_tight/representative-7101')
    output = Path('artifacts/stage7/negative-boundary-numerics-20261022')
    output.mkdir(parents=True, exist_ok=False)
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    assert not run['real_data_read'] and not run['held_out_scoring_enabled']
    with np.load(folder / 'data.npz') as loaded:
        data = dict(loaded)
    with np.load(folder / 'truth.npz') as loaded:
        truth = dict(loaded)
    posterior_path = folder / 'posterior_1.nc'
    with xr.open_dataset(posterior_path, group='posterior') as post:
        phi = post['f_phi'].values[:, :, 1]
        chain, draw = np.unravel_index(np.argmin(phi), phi.shape)
        saved = {name: value.values[chain, draw].copy() for name, value in post.items()}
    options = run['options']
    bound = 2 * options['level_settings']['amplitude_multiplier'] / np.sqrt(options['factors'])
    point = {
        'joint_level_whitened': saved['joint_level_whitened'],
        'f_whitened_states': saved['f_whitened_states'],
        'm_sd_interval__': logit(saved['m_sd'] / options['level_settings']['scale_upper']),
        'psi_log__': np.log(saved['psi']),
        'f_phi_interval__': logit((saved['f_phi'] + 1) / 2),
        'f_diag_interval__': logit(saved['f_diag'] / bound),
        'f_free_interval__': logit((saved['f_free'] + bound) / (2 * bound)),
    }
    model = build_model(data, options, truth)
    assert set(point) == set(model.initial_point())
    for name, expected in model.initial_point().items():
        assert point[name].shape == expected.shape
    names = list(saved)
    reconstructed = model.compile_fn(
        model.replace_rvs_by_values([model[name] for name in names]),
        inputs=model.value_vars, on_unused_input='ignore')(point)
    reconstruction = {}
    for name, values in zip(names, reconstructed, strict=True):
        error = float(np.max(np.abs(values - saved[name])))
        np.testing.assert_allclose(values, saved[name], rtol=1e-7, atol=1e-7)
        reconstruction[name] = error
    np.savez(output / 'visited_point.npz', **point)
    write_json(output / 'selection.json', {
        'posterior_sha256': file_hash(posterior_path), 'directory': folder.as_posix(),
        'chain_zero_based': int(chain), 'draw_zero_based': int(draw),
        'second_persistence': float(saved['f_phi'][1]),
        'selection': 'Minimum retained second-factor persistence in the failed retry 1.',
        'roundtrip_max_abs_errors': reconstruction, 'source_sha256': file_hash(Path(__file__)),
    })
    native = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars)
    fused = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars, mode='NUMBA')
    a, b = fused(point), native(point)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-6)
    assert np.isfinite(a[0]) and np.isfinite(a[1]).all()
    steps = [.003, .001, .0003, .0001, .00003, .00001]
    comparisons, offset = [], 0
    for variable in model.value_vars:
        name = variable.name
        for index in sorted({0, point[name].size - 1}):
            analytic = float(b[1][offset + index])
            estimates = []
            for step in steps:
                delta = np.zeros_like(point[name])
                delta.flat[index] = step
                plus, minus = dict(point), dict(point)
                plus[name], minus[name] = point[name] + delta, point[name] - delta
                fd = float((native(plus)[0] - native(minus)[0]) / (2 * step))
                estimates.append({'step': step, 'value': fd,
                                  'absolute_error': abs(fd - analytic),
                                  'passed': bool(np.isclose(fd, analytic, rtol=5e-5, atol=5e-5))})
            passed = any(left['passed'] and right['passed'] and np.isclose(
                left['value'], right['value'], rtol=5e-5, atol=5e-5)
                for left, right in zip(estimates[:-1], estimates[1:], strict=True))
            comparisons.append({'variable': name, 'index': index, 'analytic': analytic,
                                'estimates': estimates, 'passed': bool(passed)})
        offset += point[name].size
    record = {
        'passed': all(x['passed'] for x in comparisons),
        'selection_sha256': file_hash(output / 'selection.json'),
        'point_sha256': file_hash(output / 'visited_point.npz'),
        'source_sha256': file_hash(Path(__file__)), 'numba_opt': os.environ.get('NUMBA_OPT'),
        'logp_abs_error': float(abs(a[0] - b[0])),
        'gradient_max_abs_error': float(np.max(np.abs(a[1] - b[1]))),
        'finite_differences': comparisons,
        'rule': 'Existing tolerances: two adjacent finite differences must agree with '
                'each other and the analytic gradient at rtol=atol=5e-5.',
        'interpretation': 'Actual retained state, not a rejected leapfrog proposal. '
                          'No posterior admission or recovery eligibility change.',
    }
    write_json(output / 'audit.json', record)
    print({key: record[key] for key in ('passed', 'logp_abs_error', 'gradient_max_abs_error')},
          flush=True)
    if not record['passed']:
        raise ValueError('Inspect preserved numerical comparisons.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
