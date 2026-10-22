"""Full-size one-factor target/gradient and coordinate checks before diagnostics."""

import time
import warnings
from pathlib import Path

import numpy as np
from run_stage7_supplement import classify
from stage7_terminal_model import build_joint
from stage7_terminal_oracle import dense_density

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import load_fixture, verify_fit_authority
from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash


def main():
    root = Path(__file__).resolve().parents[1]
    verify_fit_authority(root, 'matched_one_factor_fallback')
    data, truth, _, options = load_fixture(root, 'matched_one_factor_fallback')
    output = root/'artifacts/stage7/terminal-v1/verification'
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    reports = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        old_model = build_model(data, options, truth)
        old_states = old_model.compile_fn(old_model.replace_rvs_by_values(
            [old_model['m'], old_model['f_states']]), inputs=old_model.value_vars,
            on_unused_input='ignore')
        for name, fixed in [('joint_free', None), ('joint_fixed', truth['phi'])]:
            model = build_joint(data, options, fixed_phi=fixed)
            native = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars)
            compiled = model.compile_fn([model.logp(), model.dlogp()],
                                        inputs=model.value_vars, mode='NUMBA')
            canonical = model.compile_fn(model.replace_rvs_by_values(
                [model['m'], model['f_states']]), inputs=model.value_vars,
                on_unused_input='ignore')
            point = model.initial_point()
            rng = np.random.default_rng(221023)
            nearby = {k: v+rng.normal(0, .03, v.shape) for k, v in point.items()}
            near = {k: v.copy() for k, v in point.items()}
            if fixed is None:
                near['f_phi_interval__'][:] = np.log((1+.9999)/(1-.9999))
            points = [('initial', point), ('perturbed', nearby), ('near_unit', near)]
            if fixed is not None:
                points = points[:2]
            for label, p in points:
                a, b = native(p), compiled(p)
                oracle, expected = dense_density(p, data, options, fixed_phi=fixed)
                actual = canonical(p)
                np.testing.assert_allclose(actual[0], expected['m'], rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(actual[1], expected['f_states'], rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(a[0], oracle, rtol=1e-8, atol=1e-6)
                np.testing.assert_allclose(a[0], b[0], rtol=1e-8, atol=1e-6)
                np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-6)
                assert np.isfinite(a[0]) and np.isfinite(a[1]).all()
                offset, derivatives = 0, []
                for var in model.value_vars:
                    key = var.name
                    delta = np.zeros_like(p[key])
                    delta.flat[0] = 1e-5
                    plus, minus = dict(p), dict(p)
                    plus[key], minus[key] = p[key]+delta, p[key]-delta
                    fd = (dense_density(plus, data, options, fixed_phi=fixed)[0]
                          - dense_density(minus, data, options, fixed_phi=fixed)[0])/2e-5
                    np.testing.assert_allclose(fd, a[1][offset], rtol=5e-5, atol=5e-5)
                    derivatives.append({'variable': key, 'fd': fd, 'analytic': float(a[1][offset])})
                    offset += p[key].size
                if fixed is None:
                    old = old_model.initial_point()
                    for key in old:
                        if key in p:
                            old[key] = p[key]
                    old['m_known'] = actual[0][options['level_settings']['known_indices']]
                    old['m_fallback'] = np.array(actual[0][9])
                    recovered = old_states(old)
                    np.testing.assert_allclose(recovered[0], actual[0], rtol=1e-7, atol=1e-7)
                    np.testing.assert_allclose(recovered[1], actual[1], rtol=1e-7, atol=1e-7)
                reports.append({'model': name, 'point': label,
                                'density_error': abs(float(a[0])-oracle),
                                'gradient_max_error': float(np.max(abs(a[1]-b[1]))),
                                'independent_derivatives': derivatives})
                print(name, label, 'passed', flush=True)
    performance, unresolved = classify(caught)
    sources = [Path(__file__), Path(__file__).with_name('stage7_terminal_model.py'),
               Path(__file__).with_name('stage7_terminal_oracle.py')]
    write_json(output/'audit.json', {'passed': not unresolved, 'checks': reports,
               'elapsed_seconds': time.perf_counter()-start,
               'performance_warnings': performance, 'unresolved_warnings': unresolved,
               'source_sha256': {p.name: file_hash(p) for p in sources},
               'sampling_performed': False})
    if unresolved:
        raise ValueError('Unresolved verification warning.')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
