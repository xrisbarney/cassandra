"""Verify or run only the eight declared hosted synthetic cases."""

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.stage7_priors import (
    CURRENT_PRIOR_ID,
    level_settings,
    stationary_truth,
    verify_activation,
)
from cyber_threat_modelling.provenance import file_hash

PLAN = Path('docs/decisions/STAGE7_HOSTED_PENDING_20261022.json')


def checked_plan(root, fitting=False):
    plan = json.loads((root / PLAN).read_text(encoding='utf-8'))
    if plan['real_data_read'] or plan['held_out_scoring_enabled']:
        raise ValueError('Hosted work must remain synthetic.')
    if fitting and plan['status'] != 'active':
        raise ValueError('Hosted fitting has not been activated.')
    for name, expected in plan['files_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f'Changed or escaping declared source: {name}')
    for case in plan['cases']:
        if not (root / case['directory']).resolve().is_relative_to(root.resolve()):
            raise ValueError('Case output escapes the project.')
    return plan


def fixture(case, comparisons=None):
    data, original, options = generate(case['scenario'], case['seed'])
    truth = stationary_truth(original, options)
    canonical = {}
    for key, values in [('data_sha256', data), ('truth_sha256', truth),
                        ('truth_original_coordinates_sha256', original)]:
        path = Path(case['fixture_directory']) / (key.removesuffix('_sha256') + '.npz')
        if file_hash(path) != case[key]:
            raise ValueError(f'Hosted synthetic fixture differs: {case["key"]} {key}')
        with np.load(path) as loaded:
            saved = dict(loaded)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **values)
        if comparisons is not None:
            comparisons.append({
                'case': case['key'], 'fixture': key, 'canonical_sha256': case[key],
                'regenerated_sha256': hashlib.sha256(buffer.getvalue()).hexdigest(),
                'maximum_array_absolute_errors': {
                    name: float(np.max(np.abs(values[name].astype(float)
                                               - saved[name].astype(float)), initial=0))
                    for name in values if name in saved},
            })
        if set(saved) != set(values):
            raise ValueError('Canonical fixture array names differ from generator.')
        for name in values:
            if np.issubdtype(values[name].dtype, np.inexact):
                np.testing.assert_allclose(values[name], saved[name], rtol=1e-14, atol=1e-14)
            else:
                np.testing.assert_array_equal(values[name], saved[name])
        canonical[key] = saved
    options.update(prior_id=CURRENT_PRIOR_ID, prior_candidate=case['candidate'],
                   count_parameterization='conditional', gaussian_solver='block',
                   level_parameterization='joint_conditional',
                   factor_parameterization='anchored', mass_matrix='diagonal')
    options['level_settings'] = level_settings(canonical['data_sha256'], options)
    return canonical['data_sha256'], canonical['truth_sha256'], options


def environment(plan, fitting=False):
    if platform.python_version() != plan['python_version'] or sys.platform != 'win32':
        raise ValueError('Hosted work requires the declared Windows/Python environment.')
    allowed = ([plan['numba_opt']] if fitting else
               plan.get('verification_numba_profiles', [plan['numba_opt']]))
    if os.environ.get('NUMBA_OPT') not in allowed:
        raise ValueError('Unexpected Numba optimization profile.')
    import pytensor

    installed = json.loads(Path('.tools/compiler/installation.json').read_text(encoding='utf-8'))
    spec = json.loads(Path('configs/windows_compiler.json').read_text(encoding='utf-8'))
    if installed['sha256'] != spec['sha256'] or not pytensor.config.cxx:
        raise ValueError('Pinned compiler is not configured.')
    if Path(pytensor.config.cxx).resolve() != Path(installed['compiler_path']).resolve():
        raise ValueError('PyTensor is using a different compiler.')
    return {'platform': platform.platform(), 'python': sys.version,
            'processor': platform.processor(), 'compiler': installed,
            'numba_opt': os.environ['NUMBA_OPT'],
            'numba_fastmath': bool(pytensor.config.numba__fastmath),
            'numba_cache': bool(pytensor.config.numba__cache),
            'github_run_id': os.environ.get('GITHUB_RUN_ID'),
            'github_sha': os.environ.get('GITHUB_SHA')}


def verification(plan, output, host):
    from cyber_threat_modelling.models.synthetic_state_space import build_model

    fixtures, regeneration = [], []
    for case in plan['cases']:
        try:
            fixture(case, regeneration)
        finally:
            write_json(output / 'regeneration.json', regeneration)
        fixtures.append({'key': case['key'], 'matched': True,
                         **{k: v for k, v in case.items() if k.endswith('_sha256')}})
    # Full-size fixed points supplement the independent dense-coordinate tests in CI.
    comparisons = []
    for scenario in ('representative', 'fallback_zero'):
        case = next(c for c in plan['cases'] if c['candidate'] == 'level_tight'
                    and c['scenario'] == scenario)
        data, truth, options = fixture(case)
        model = build_model(data, options, truth)
        compiled = model.compile_fn([model.logp(), model.dlogp()],
                                    inputs=model.value_vars, mode='NUMBA')
        native = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars)
        point = model.initial_point()
        rng = np.random.default_rng(251022)
        perturbed = {k: v + rng.normal(0, .05, np.shape(v)) for k, v in point.items()}
        near = {k: np.array(v, copy=True) for k, v in point.items()}
        near['f_phi_interval__'] = np.log((1 + np.array([.35, .9999]))
                                         / (1 - np.array([.35, .9999])))
        errors = []
        for p in (point, perturbed, near):
            a, b = compiled(p), native(p)
            np.testing.assert_allclose(a[0], b[0], rtol=1e-8, atol=1e-6)
            np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-6)
            if not np.isfinite(a[0]) or not np.isfinite(a[1]).all():
                raise ValueError('Hosted complete target is nonfinite.')
            errors.append({'logp_abs_error': float(abs(a[0] - b[0])),
                           'gradient_max_abs_error': float(np.max(abs(a[1] - b[1])))})
        durations = []
        for _ in range(30):
            start = time.perf_counter()
            compiled(point)
            durations.append(time.perf_counter() - start)
        comparisons.append({'scenario': scenario, 'candidate': 'level_tight',
                            'errors': errors, 'median_fused_seconds': float(np.median(durations))})
    write_json(output / 'verification.json', {
        'passed': True, 'plan_sha256': file_hash(PLAN), 'source_sha256': file_hash(Path(__file__)),
        'environment': host, 'fixtures': fixtures, 'complete_targets': comparisons,
        'regeneration': regeneration,
        'real_data_read': False, 'held_out_scoring_enabled': False,
        'interpretation': 'Environment/fixture/numerical verification only; '
        'no posterior admission.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['verify', 'fit'])
    parser.add_argument('--case')
    parser.add_argument('--attempt', type=int, choices=[0, 1, 2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    plan = checked_plan(root, fitting=args.mode == 'fit')
    activation = verify_activation(root, CURRENT_PRIOR_ID)
    host = environment(plan, fitting=args.mode == 'fit')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'host.json', host)
    if args.mode == 'verify':
        verification(plan, args.output, host)
        return
    matches = [case for case in plan['cases'] if case['key'] == args.case]
    if len(matches) != 1:
        raise ValueError('Case is outside the declared hosted grid.')
    case = matches[0]
    fixture(case)
    if args.attempt is None:
        raise ValueError('Hosted fitting requires its prescribed attempt index.')
    folder = Path(case['directory'])
    from audit_stage7_campaign import checked_run

    from cyber_threat_modelling.evaluation.synthetic_checkpoint import restore

    if (folder / 'result.json').exists():
        record = checked_run(folder, case['candidate'], case['scenario'], case['seed'],
                             activation, recompute=True)
        write_json(args.output / 'completion.json', {
            'case': record, 'plan_sha256': file_hash(PLAN), 'environment': host,
            'case_is_terminal': True, 'already_completed_no_refit': True})
        return
    if args.attempt:
        _, attempts, _ = restore(folder, root)
        if len(attempts) != args.attempt:
            raise ValueError('Hosted attempt does not follow its sealed predecessor.')
    elif folder.exists():
        raise ValueError('Initial hosted case directory already exists.')
    command = [sys.executable, '-u', 'scripts/run_stage7.py', '--scenario', case['scenario'],
               '--seed', str(case['seed']), '--prior-candidate', case['candidate'],
               '--output', case['directory'], *plan['fit_flags'], '--pause-after-attempt',
               '--fixture-directory', case['fixture_directory']]
    if args.attempt:
        command.append('--resume-attempts')
    process = subprocess.run(command, check=False)
    if process.returncode:
        raise RuntimeError('Hosted fit did not finish; preserve partial files.')
    if not (folder / 'result.json').exists():
        _, attempts, digest = restore(folder, root)
        if len(attempts) != args.attempt + 1:
            raise ValueError('Hosted pause did not complete exactly one prescribed attempt.')
        write_json(args.output / 'completion.json', {
            'plan_sha256': file_hash(PLAN), 'environment': host, 'case_is_terminal': False,
            'completed_attempt': args.attempt, 'checkpoint_sha256': digest,
            'interpretation': 'Completed failed attempt; the next prescribed attempt is pending.'})
        return
    for name, key in [('data.npz', 'data_sha256'), ('truth.npz', 'truth_sha256'),
                      ('truth_original_coordinates.npz', 'truth_original_coordinates_sha256')]:
        if file_hash(folder / name) != case[key]:
            raise ValueError('Completed hosted fit used a different synthetic fixture.')
    # Recompute diagnostics independently from saved posterior files before transfer.
    record = checked_run(folder, case['candidate'], case['scenario'], case['seed'], activation,
                         recompute=True)
    write_json(args.output / 'completion.json', {
        'case': record, 'plan_sha256': file_hash(PLAN),
        'source_sha256': file_hash(Path(__file__)), 'environment': host,
        'case_is_terminal': True, 'statistical_failure_is_not_execution_failure': True})


if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
