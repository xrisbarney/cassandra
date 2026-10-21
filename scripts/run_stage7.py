"""Run one auditable fixed-truth Stage 7 experiment; never reads real data."""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic import SCENARIOS, generate
from cyber_threat_modelling.models.components import fitted_effort
from cyber_threat_modelling.provenance import file_hash, git_revision


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
    parser.add_argument('--center-factors', action='store_true')
    parser.add_argument('--dense-mass', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    data, truth, options = generate(args.scenario, args.seed)
    options['factor_parameterization'] = 'centered' if args.center_factors else 'innovations'
    options['mass_matrix'] = 'dense' if args.dense_mass else 'diagonal'
    effort = None
    if args.scenario == 'effort_proxy':
        data['effort'], effort = fitted_effort(data['N'], data['mask_N'].all(axis=1))
    np.savez_compressed(folder / 'data.npz', **data)
    np.savez_compressed(folder / 'truth.npz', **truth)
    files = list((root / 'src').rglob('*.py')) + [Path(__file__), root / 'uv.lock',
             root / 'docs/STAGE7_DESIGN.md', root / 'docs/STAGE2_FREEZE.json']
    write_json(folder / 'run.json', {
        'source_control': git_revision(root), 'options': options, 'effort_fit': effort,
        'pid': os.getpid(), 'source_sha256': {
            p.relative_to(root).as_posix(): file_hash(p) for p in files
        }, 'data_sha256': file_hash(folder / 'data.npz'),
        'truth_sha256': file_hash(folder / 'truth.npz'), 'real_data_read': False,
        'held_out_scoring_enabled': False,
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
        timing = {}
        with model, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
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
                for clock in ('perf_counter_diff', 'process_time_diff'):
                    key = f'{phase}_{clock}_sum'
                    record[key] = record.get(key, 0.) + sum(
                        float(stat[clock]) for stat in draw.stats
                    )
                if (draw.draw_idx + 1) % 500 == 0:
                    print(args.scenario, args.seed, retry, draw.chain, draw.draw_idx + 1,
                          flush=True)

            idata = pm.sample(draws=draws, tune=warmup, chains=4, cores=1, step=step,
                              random_seed=args.seed + retry, initvals=initvals,
                              callback=callback, progressbar=False)
            sample_seconds = time.perf_counter() - start
        start = time.perf_counter()
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
        table.to_csv(folder / f'diagnostics_{retry}.csv')
        start = time.perf_counter()
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
            'sampler_configuration_warnings': configuration_warnings,
            'unresolved_warnings': unresolved,
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
    main()
