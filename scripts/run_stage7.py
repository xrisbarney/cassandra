"""Run one auditable fixed-truth Stage 7 experiment; never reads real data."""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic import SCENARIOS, generate
from cyber_threat_modelling.models.components import fitted_effort
from cyber_threat_modelling.models.stage7_priors import (
    CANDIDATES,
    CURRENT_PRIOR_ID,
    level_settings,
    stationary_truth,
    verify_activation,
)
from cyber_threat_modelling.provenance import file_hash, git_revision


class IrrecoverableSamplingGate(RuntimeError):
    """A retained divergence/depth hit makes this attempt ineligible permanently."""


def retained_failure(stats, tuning, depth):
    if tuning:
        return None
    divergences = sum(bool(s.get('diverging', False)) for s in stats)
    saturations = sum(int(s.get('depth', 0)) >= depth for s in stats)
    return {'divergences': divergences, 'tree_depth_saturations': saturations} if (
        divergences or saturations
    ) else None


def peak_memory_bytes():
    """Process peak working set on Windows; unavailable platforms are explicit."""
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                'PagefileUsage', 'PeakPagefileUsage',
            )
        ]

    kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                         wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                     counters.cb):
        raise OSError('Could not obtain process peak memory.')
    return int(counters.PeakWorkingSetSize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--seed', type=int, choices=[7101, 7102, 7103], required=True)
    parser.add_argument('--output', type=Path, required=True)
    coordinates = parser.add_mutually_exclusive_group()
    coordinates.add_argument('--center-factors', action='store_true')
    coordinates.add_argument('--anchor-factors', action='store_true')
    parser.add_argument('--dense-mass', action='store_true')
    parser.add_argument('--prior-candidate', choices=CANDIDATES, default='primary')
    parser.add_argument('--conditional-severity', action='store_true')
    parser.add_argument('--conditional-counts', action='store_true')
    parser.add_argument('--conditional-iid', action='store_true')
    parser.add_argument('--sign-anchor', type=int, choices=[3])
    parser.add_argument('--joint-levels', action='store_true')
    parser.add_argument('--block-coordinates', action='store_true')
    args = parser.parse_args()
    if args.joint_levels and (args.scenario not in {'representative', 'fallback_zero'}
                              or not args.conditional_counts or not args.block_coordinates
                              or args.dense_mass):
        parser.error('Joint levels require the declared rank-two block/diagonal configuration.')
    if args.sign_anchor is not None and (args.scenario != 'effort_proxy'
                                         or not args.conditional_counts):
        parser.error('The declared working sign chart applies to conditional proxy fits only.')
    if args.conditional_severity and args.scenario != 'severity_missing':
        parser.error('Conditional severity coordinates apply only to the severity fixture.')
    if args.conditional_iid and args.scenario not in {'dispersion_dense', 'dispersion_sparse'}:
        parser.error('Conditional iid coordinates apply only to the noise-separation fixtures.')
    root = Path(__file__).resolve().parents[1]
    activation = verify_activation(root, CURRENT_PRIOR_ID)
    audit_path = root/'docs/checks/stage7-levels-20261021/audit.json'
    prior_audit = json.loads(audit_path.read_text())
    if not prior_audit.get('complete') or not prior_audit.get('primary_passed'):
        raise ValueError('The primary prior-predictive gate has not passed.')
    if prior_audit['prior_activation']['sha256'] != activation['sha256']:
        raise ValueError('Prior audit belongs to a different activation.')
    # The candidate grid is predeclared; wide cases may fail plausibility and
    # still run as explicitly labelled sensitivity stress comparisons.
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    data, truth, options = generate(args.scenario, args.seed)
    np.savez_compressed(folder / 'truth_original_coordinates.npz', **truth)
    truth = stationary_truth(truth, options)
    options['prior_id'] = CURRENT_PRIOR_ID
    options['prior_candidate'] = args.prior_candidate
    if args.conditional_severity:
        options['severity_parameterization'] = 'conditional'
    if args.conditional_counts:
        options['count_parameterization'] = 'conditional'
    if args.sign_anchor is not None:
        options['working_sign_anchor'] = args.sign_anchor
    if args.joint_levels:
        options['level_parameterization'] = 'joint_conditional'
    if args.conditional_iid:
        options['residual_parameterization'] = 'conditional'
    if args.block_coordinates:
        options['gaussian_solver'] = 'block'
    options['factor_parameterization'] = ('anchored' if args.anchor_factors else
                                          'centered' if args.center_factors else 'innovations')
    options['mass_matrix'] = 'dense' if args.dense_mass else 'diagonal'
    effort = None
    if args.scenario == 'effort_proxy':
        data['effort'], effort = fitted_effort(data['N'], data['mask_N'].all(axis=1))
    options['level_settings'] = level_settings(data, options)
    np.savez_compressed(folder / 'data.npz', **data)
    np.savez_compressed(folder / 'truth.npz', **truth)
    files = list((root / 'src').rglob('*.py')) + [Path(__file__), root / 'uv.lock',
             root / 'docs/STAGE7_DESIGN.md', root / 'docs/STAGE2_FREEZE.json']
    if args.anchor_factors:
        files.append(root/'docs/decisions/STAGE7_AMPLITUDE_COORDINATES_20261021.md')
    if args.conditional_severity:
        files.append(root/'docs/decisions/STAGE7_SEVERITY_COORDINATES_20261021.md')
    if args.conditional_iid:
        files.append(root/'docs/decisions/STAGE7_IID_COORDINATES_20261022.md')
    if args.sign_anchor is not None:
        files.append(root/'docs/decisions/STAGE7_SIGN_GAUGE_20261022.md')
    if args.joint_levels:
        files.append(root/'docs/decisions/STAGE7_JOINT_LEVEL_COORDINATES_20261022.md')
    if args.conditional_counts:
        files.append(root/'docs/decisions/STAGE7_COUNT_COORDINATES_20261021.md')
        if args.scenario in {'effort_oracle', 'effort_none', 'factors_sparse'}:
            files.append(root/'docs/decisions/STAGE7_MATCHED_COUNT_REPAIRS_20261022.md')
        if not args.dense_mass:
            files.append(root/'docs/decisions/STAGE7_WHITENED_METRIC_20261022.md')
    write_json(folder / 'run.json', {
        'source_control': git_revision(root), 'options': options, 'effort_fit': effort,
        'pid': os.getpid(), 'source_sha256': {
            p.relative_to(root).as_posix(): file_hash(p) for p in files
        }, 'data_sha256': file_hash(folder / 'data.npz'),
        'truth_sha256': file_hash(folder / 'truth.npz'), 'real_data_read': False,
        'held_out_scoring_enabled': False,
        'prior_amendment': activation,
        'prior_predictive_audit_sha256': file_hash(audit_path),
        'compute_threadpools': threadpool_info(),
        'numba_optimization_env': os.environ.get('NUMBA_OPT', '3 (library default)'),
        'compute_thread_policy': 'One native numerical thread per sequential-chain fit process.',
        'truth_original_coordinates_sha256': file_hash(folder/'truth_original_coordinates.npz'),
    })
    import pymc as pm
    import pytensor

    from cyber_threat_modelling.evaluation.synthetic_recovery import summarize
    from cyber_threat_modelling.models.bayesian_controls import diagnostic_report
    from cyber_threat_modelling.models.synthetic_state_space import build_model

    if not pytensor.config.cxx:
        raise RuntimeError('Activate the verified compiler before fitting.')
    start = time.perf_counter()
    model = build_model(data, options, truth)
    graph_seconds = time.perf_counter() - start
    if args.scenario == 'representative':
        start = time.perf_counter()
        gradient = model.compile_dlogp(mode='NUMBA')
        compile_seconds = time.perf_counter() - start
        point = model.initial_point()
        start = time.perf_counter()
        evaluated = gradient(point)
        first_call = time.perf_counter() - start
        if not np.isfinite(evaluated).all():
            raise ValueError('Representative gradient is nonfinite.')
        repeats = []
        for _ in range(30):
            start = time.perf_counter()
            gradient(point)
            repeats.append(time.perf_counter() - start)
        write_json(folder / 'gradient_timing.json', {
            'graph_seconds': graph_seconds, 'function_compile_seconds': compile_seconds,
            'first_call_including_deferred_jit_seconds': first_call,
            'median_repeated_call_seconds': float(np.median(repeats)),
            'peak_working_set_bytes': peak_memory_bytes(),
            'cache_scope': 'First call in this process, not an empty on-disk cache experiment.',
            'fit_follows_this_preflight': True,
        })
    budgets = [(1000, 1000, .95, 12), (2000, 1000, .99, 15), (2000, 2000, .99, 15)]
    attempts = []
    for retry, (warmup, draws, accept, depth) in enumerate(budgets):
        timing, failure = {}, {}
        with model, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warning_contexts = []
            warning_phase = {'phase': 'setup', 'chain': None, 'draw_index': None}

            def capture_warning(message, category, filename, lineno, file=None, line=None):
                # Preserve the ordinary warning gate while locating numerical
                # warnings in setup, warmup or retained sampling for diagnosis.
                import traceback

                caught.append(warnings.WarningMessage(message, category, filename, lineno,
                                                       file, line))
                warning_contexts.append({
                    'message': str(message), 'filename': filename, 'lineno': lineno,
                    'category': category.__name__, **warning_phase,
                    'stack': [{'filename': frame.filename, 'lineno': frame.lineno,
                               'function': frame.name}
                              for frame in traceback.extract_stack(limit=12)[:-1]],
                })

            warnings.showwarning = capture_warning
            start = time.perf_counter()
            initial = model.initial_point()
            potential_args = {}
            if args.dense_mass:
                from pymc.blocking import DictToArrayBijection
                from pymc.step_methods.hmc.quadpotential import QuadPotentialFullAdapt

                initial_vector = DictToArrayBijection.map(initial).data
                dimension = len(initial_vector)
                potential_args['potential'] = QuadPotentialFullAdapt(
                    dimension, initial_vector, np.eye(dimension), initial_weight=10,
                    adaptation_window=max(101, 2 * dimension), update_window=10,
                )
            step = pm.NUTS(target_accept=accept, max_treedepth=depth,
                           compile_kwargs={'mode': 'NUMBA'}, **potential_args)
            compile_seconds = time.perf_counter() - start
            rng = np.random.default_rng(args.seed + retry * 100)
            initvals = [{k: v + rng.normal(0, .1, size=v.shape) for k, v in initial.items()}
                        for _ in range(4)]
            start = time.perf_counter()

            def callback(trace, draw):
                elapsed = time.perf_counter() - start
                record = timing.setdefault(str(draw.chain), {'first_draw_seconds': elapsed})
                if not draw.tuning and 'first_retained_seconds' not in record:
                    record['first_retained_seconds'] = elapsed
                # The installed callback's is_last flag compares i == draws
                # inside range(draws), so record the latest time explicitly.
                record['last_draw_seconds'] = elapsed
                phase = 'warmup' if draw.tuning else 'retained'
                record[phase + '_draws_completed'] = record.get(phase + '_draws_completed', 0) + 1
                for clock in ('perf_counter_diff', 'process_time_diff'):
                    key = f'{phase}_{clock}_sum'
                    record[key] = record.get(key, 0.) + sum(
                        float(stat[clock]) for stat in draw.stats
                    )
                if (draw.draw_idx + 1) % 50 == 0:
                    write_json(folder/'heartbeat.json', {
                        'retry': retry, 'chain': int(draw.chain),
                        'draw_index': int(draw.draw_idx), 'phase': phase,
                        'sample_call_elapsed_seconds': elapsed,
                        'chain_wall_markers': timing,
                        'interpretation': 'Progress only; not diagnostic admission.',
                    })
                if (draw.draw_idx + 1) % 500 == 0:
                    print(args.scenario, args.seed, retry, draw.chain, draw.draw_idx + 1,
                          flush=True)
                next_index = int(draw.draw_idx) + 1
                next_chain = int(draw.chain)
                if next_index == warmup + draws:
                    next_index, next_chain = 0, next_chain + 1
                next_phase = ('post_sampling' if next_chain >= 4 else
                              'warmup' if next_index < warmup else 'retained')
                warning_phase.update(chain=next_chain, draw_index=next_index, phase=next_phase)
                reason = retained_failure(draw.stats, draw.tuning, depth)
                if reason:
                    failure.update(reason, chain=int(draw.chain), draw_index=int(draw.draw_idx),
                                   reason='Retained draw failed a zero-tolerance gate.')
                    # This is a diagnostic checkpoint, never an admitted posterior.
                    np.savez_compressed(folder/f'failed_state_{retry}.npz', **draw.point)
                    write_json(folder/f'early_failure_{retry}.json', failure)
                    raise IrrecoverableSamplingGate(failure['reason'])

            try:
                warning_phase.update(phase='warmup', chain=0, draw_index=0)
                idata = pm.sample(draws=draws, tune=warmup, chains=4, cores=1, step=step,
                                  random_seed=args.seed + retry, initvals=initvals,
                                  callback=callback, progressbar=False)
            except IrrecoverableSamplingGate:
                idata = None
            sample_seconds = time.perf_counter() - start
        start = time.perf_counter()
        if idata is None:
            diagnostics = {
                'accepted': False, 'all_diagnostics_finite': False,
                'max_rhat': None, 'min_bulk_ess': None, 'min_tail_ess': None, 'min_bfmi': None,
                'divergences': failure['divergences'],
                'tree_depth_saturations': failure['tree_depth_saturations'],
            }
            table = None
        else:
            diagnostics, table = diagnostic_report(idata, depth)
        diagnostic_seconds = time.perf_counter() - start
        messages = list(dict.fromkeys(str(w.message) for w in caught))
        # These narrowly identified messages describe execution speed, not a
        # numerical failure. Native/NUMBA value and gradient agreement is tested.
        performance = [m for m in messages if (
            m.startswith('PyTensor could not link to a BLAS installation.')
            or m.startswith("Numba will use object mode to run AdvancedIncSubtensor's")
            or 'np.dot() is faster on contiguous arrays' in m
        )]
        configuration_warnings = [m for m in messages if
                                  m == 'QuadPotentialFullAdapt is an experimental feature']
        unresolved = [m for m in messages if m not in performance + configuration_warnings]
        accepted = diagnostics['accepted'] and not unresolved
        if table is not None:
            table.to_csv(folder / f'diagnostics_{retry}.csv')
        start = time.perf_counter()
        if idata is not None:
            idata.to_netcdf(folder / f'posterior_{retry}.nc')
        serialization = time.perf_counter() - start
        attempt = {
            'retry': retry, 'warmup': warmup, 'draws': draws, 'chains': 4, 'cores': 1,
            'target_accept': accept, 'max_treedepth': depth, 'accepted': accepted,
            'diagnostics': {k: v if not isinstance(v, float) or np.isfinite(v) else None
                            for k, v in diagnostics.items()},
            'graph_seconds': graph_seconds, 'step_compile_seconds': compile_seconds,
            'sample_call_seconds': sample_seconds, 'chain_wall_markers': timing,
            'diagnostic_seconds': diagnostic_seconds, 'serialization_seconds': serialization,
            'peak_working_set_bytes': peak_memory_bytes(),
            'warnings': messages, 'performance_warnings': performance,
            'warning_locations': [{'message': str(w.message), 'filename': w.filename,
                                   'lineno': w.lineno, 'category': w.category.__name__}
                                  for w in caught],
            'warning_contexts': warning_contexts,
            'sampler_configuration_warnings': configuration_warnings,
            'unresolved_warnings': unresolved,
            'early_failure': failure or None,
        }
        write_json(folder / f'attempt_{retry}.json', attempt)
        attempts.append(attempt)
        print(json.dumps(attempt['diagnostics']), flush=True)
        if accepted:
            report, paths = summarize(idata.posterior, data, truth, options, args.seed + 10000)
            np.savez_compressed(folder / 'future.npz', counts=paths)
            write_json(folder / 'recovery.json', report)
            break
    write_json(folder / 'result.json', {
        'status': 'accepted' if accepted else 'failed_diagnostic_gate',
        'options': options, 'attempts': attempts,
        'artifacts': {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()},
    })


if __name__ == '__main__':
    # Three outer fit workers should not each claim all eight logical CPUs.
    # Set defaults for subsequently loaded libraries, then limit loaded BLAS.
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'
    import pymc  # noqa: F401
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
