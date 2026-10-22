"""Verify or execute one of four declared cases, exactly one attempt per case."""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
from ci_stage7 import environment
from run_stage7 import IrrecoverableSamplingGate, peak_memory_bytes, retained_failure

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import (
    ACTIVATION,
    DECLARATION,
    FIXTURES,
    declaration,
    load_fixture,
    make_fixture,
    verify_fit_authority,
)
from cyber_threat_modelling.models.compiler_warnings import is_dynamic_global_cache_warning
from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash, git_revision


def budget(record):
    expected = (4, 2000, 2000, .99, 15, 7103, 7301)
    actual = tuple(record[k] for k in ('chains', 'warmup', 'retained_draws', 'target_accept',
                                      'max_treedepth', 'sampler_seed', 'initialization_seed'))
    if actual != expected or record['attempts_per_case'] != 1:
        raise ValueError('The supplement budget or seeds changed.')
    return expected


def verified_model(root, case, output):
    host = environment({'python_version': '3.12.0', 'numba_opt': '3'}, fitting=True)
    saved = load_fixture(root, case)
    regenerated = make_fixture(root, case)
    for a, b in zip(saved[:3], regenerated[:3]):
        if set(a) != set(b):
            raise ValueError('Supplement regeneration has different arrays.')
        for key in a:
            if np.issubdtype(a[key].dtype, np.inexact):
                np.testing.assert_allclose(a[key], b[key], rtol=1e-14, atol=1e-14)
            else:
                np.testing.assert_array_equal(a[key], b[key])
    if saved[3] != regenerated[3]:
        raise ValueError('Supplement regenerated options differ.')
    data, truth, reference, options = saved
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = build_model(data, options, truth,
                            level_reference=reference if case == 'masked_fallback' else None)
        graph_seconds = time.perf_counter() - start
        start = time.perf_counter()
        compiled = model.compile_fn([model.logp(), model.dlogp()],
                                    inputs=model.value_vars, mode='NUMBA')
        native = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars)
        compile_seconds = time.perf_counter() - start
        point = model.initial_point()
        rng = np.random.default_rng(251022)
        perturbed = {k: v + rng.normal(0, .05, np.shape(v)) for k, v in point.items()}
        near = {k: np.array(v, copy=True) for k, v in point.items()}
        phi = np.full(options['factors'], .9999)
        near['f_phi_interval__'] = np.log((1 + phi) / (1 - phi))
        comparisons = []
        for label, p in [('initial', point), ('perturbed', perturbed), ('near_unit', near)]:
            a, b = compiled(p), native(p)
            np.testing.assert_allclose(a[0], b[0], rtol=1e-8, atol=1e-6)
            np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-6)
            if not np.isfinite(a[0]) or not np.isfinite(a[1]).all():
                raise ValueError('Supplement complete target is nonfinite.')
            comparisons.append({'point': label, 'logp_abs_error': float(abs(a[0] - b[0])),
                                'gradient_max_abs_error': float(np.max(abs(a[1] - b[1])))})
        times = []
        for _ in range(20):
            start = time.perf_counter()
            compiled(point)
            times.append(time.perf_counter() - start)
    performance, unresolved = classify(caught)
    record = {'case': case, 'passed': not unresolved, 'host': host,
              'regeneration_matches': True, 'comparisons': comparisons,
              'options': options, 'graph_seconds': graph_seconds,
              'explicit_compile_seconds': compile_seconds,
              'median_full_target_seconds': float(np.median(times)),
              'performance_warnings': performance, 'unresolved_warnings': unresolved,
              'fixture_manifest_sha256': file_hash(root / FIXTURES / 'manifest.json'),
              'declaration_sha256': file_hash(root / DECLARATION),
              'held_out_scoring_enabled': False}
    write_json(output / 'preflight.json', record)
    if unresolved:
        raise ValueError('Preflight has unresolved warnings; fitting is prohibited.')
    return model, options, record


def classify(caught):
    performance, unresolved = [], []
    for warning in caught:
        message = str(warning.message)
        known = (message.startswith('PyTensor could not link to a BLAS installation.')
                 or message.startswith("Numba will use object mode to run AdvancedIncSubtensor's")
                 or 'np.dot() is faster on contiguous arrays' in message
                 or is_dynamic_global_cache_warning(message, warning.category.__name__))
        entry = {'message': message, 'category': warning.category.__name__,
                 'filename': warning.filename, 'lineno': warning.lineno}
        (performance if known else unresolved).append(entry)
    return performance, unresolved


def fit(root, case, output):
    import pymc as pm

    from cyber_threat_modelling.models.bayesian_controls import diagnostic_report

    if os.environ.get('GITHUB_RUN_ATTEMPT', '1') != '1':
        raise ValueError('Hosted reruns are outside the one-attempt supplement budget.')
    verify_fit_authority(root, case)
    record = declaration(root)
    chains, warmup, draws, accept, depth, seed, initial_seed = budget(record)
    if output.resolve() != (root / 'artifacts/stage7/supplement-v1' / case).resolve():
        raise ValueError('Fit must use its single declared case directory.')
    output.mkdir(parents=True, exist_ok=False)
    run = {'case': case, 'source_control': git_revision(root), 'real_data_read': False,
           'held_out_scoring_enabled': False, 'activation_sha256': file_hash(root / ACTIVATION),
           'declaration_sha256': file_hash(root / DECLARATION),
           'fixture_manifest_sha256': file_hash(root / FIXTURES / 'manifest.json'),
           'budget': {'chains': chains, 'warmup': warmup, 'draws': draws,
                      'target_accept': accept, 'max_treedepth': depth,
                      'sampler_seed': seed, 'initialization_seed': initial_seed},
           'source_sha256': {p.relative_to(root).as_posix(): file_hash(p) for p in
                            [Path(__file__).resolve(), *list((root/'src').rglob('*.py'))]}}
    write_json(output / 'run.json', run)
    model, options, preflight = verified_model(root, case, output)
    timing, failure = {}, {}
    with model, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        step = pm.NUTS(target_accept=accept, max_treedepth=depth, compile_kwargs={'mode': 'NUMBA'})
        compile_seconds = time.perf_counter() - start
        initial = model.initial_point()
        rng = np.random.default_rng(initial_seed)
        initvals = [{k: v + rng.normal(0, .1, size=v.shape) for k, v in initial.items()}
                    for _ in range(chains)]
        start = time.perf_counter()

        def callback(trace, draw):
            elapsed = time.perf_counter() - start
            row = timing.setdefault(str(draw.chain), {'first_draw_seconds': elapsed})
            phase = 'warmup' if draw.tuning else 'retained'
            if not draw.tuning and 'first_retained_seconds' not in row:
                row['first_retained_seconds'] = elapsed
            row['last_draw_seconds'] = elapsed
            row[phase + '_draws_completed'] = row.get(phase + '_draws_completed', 0) + 1
            for clock in ('perf_counter_diff', 'process_time_diff'):
                key = phase + '_' + clock + '_sum'
                row[key] = row.get(key, 0.) + sum(float(s[clock]) for s in draw.stats)
            if (draw.draw_idx + 1) % 50 == 0:
                write_json(output/'heartbeat.json', {
                    'chain': int(draw.chain), 'draw_index': int(draw.draw_idx), 'phase': phase,
                    'sample_call_elapsed_seconds': elapsed, 'chain_wall_markers': timing})
            if (draw.draw_idx + 1) % 500 == 0:
                print(case, draw.chain, draw.draw_idx + 1, flush=True)
            reason = retained_failure(draw.stats, draw.tuning, depth)
            if reason:
                failure.update(reason, chain=int(draw.chain), draw_index=int(draw.draw_idx))
                np.savez_compressed(output/'failed_state.npz', **draw.point)
                write_json(output/'early_failure.json', failure)
                raise IrrecoverableSamplingGate('Retained zero-tolerance diagnostic failed.')

        try:
            idata = pm.sample(draws=draws, tune=warmup, chains=chains, cores=1, step=step,
                              random_seed=seed, initvals=initvals, callback=callback,
                              progressbar=False)
        except IrrecoverableSamplingGate:
            idata = None
        sample_seconds = time.perf_counter() - start
    performance, unresolved = classify(caught)
    start = time.perf_counter()
    if idata is None:
        diagnostic = {'accepted': False, 'all_diagnostics_finite': False,
                      'max_rhat': None, 'min_bulk_ess': None, 'min_tail_ess': None,
                      'min_bfmi': None, 'divergences': failure['divergences'],
                      'tree_depth_saturations': failure['tree_depth_saturations']}
    else:
        diagnostic, table = diagnostic_report(idata, depth)
        table.to_csv(output/'diagnostics.csv')
    diagnostic_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if idata is not None:
        idata.to_netcdf(output/'posterior.nc')
    serialization_seconds = time.perf_counter() - start
    accepted = diagnostic['accepted'] and not unresolved
    diagnostic = {k: None if isinstance(v, float) and not np.isfinite(v) else v
                  for k, v in diagnostic.items()}
    write_json(output/'result.json', {
        'case': case, 'status': 'accepted' if accepted else 'failed_diagnostic_gate',
        'options': options, 'attempt_count': 1, 'budget': run['budget'],
        'diagnostics': diagnostic, 'early_failure': failure or None,
        'performance_warnings': performance, 'unresolved_warnings': unresolved,
        'graph_seconds': preflight['graph_seconds'], 'step_compile_seconds': compile_seconds,
        'sample_call_seconds': sample_seconds, 'chain_wall_markers': timing,
        'diagnostic_seconds': diagnostic_seconds, 'serialization_seconds': serialization_seconds,
        'peak_working_set_bytes': peak_memory_bytes(),
        'artifacts': {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'case': case, 'accepted': bool(accepted), 'diagnostics': diagnostic}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'fit'])
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    budget(declaration(root))
    # Validate the case before creating any run directory.
    load_fixture(root, args.case)
    if args.action == 'verify':
        args.output.mkdir(parents=True, exist_ok=False)
        verified_model(root, args.case, args.output)
    else:
        fit(root, args.case, args.output)


if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'
    import pymc  # noqa: F401
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
