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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--seed', type=int, choices=[7101, 7102, 7103], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    data, truth, options = generate(args.scenario, args.seed)
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
    budgets = [(1000, 1000, .95, 12), (2000, 1000, .99, 15), (2000, 2000, .99, 15)]
    attempts = []
    for retry, (warmup, draws, accept, depth) in enumerate(budgets):
        timing = {}
        with model, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            start = time.perf_counter()
            step = pm.NUTS(target_accept=accept, max_treedepth=depth,
                           compile_kwargs={'mode': 'NUMBA'})
            compile_seconds = time.perf_counter() - start
            initial = model.initial_point()
            rng = np.random.default_rng(args.seed + retry * 100)
            initvals = [{k: v + rng.normal(0, .1, size=v.shape) for k, v in initial.items()}
                        for _ in range(4)]
            start = time.perf_counter()

            def callback(trace, draw):
                elapsed = time.perf_counter() - start
                record = timing.setdefault(str(draw.chain), {'first_draw_seconds': elapsed})
                if not draw.tuning and 'first_retained_seconds' not in record:
                    record['first_retained_seconds'] = elapsed
                if draw.is_last:
                    record['last_draw_seconds'] = elapsed
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
        unresolved = [m for m in messages if not m.startswith(
            'PyTensor could not link to a BLAS installation.')]
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
            'warnings': messages, 'unresolved_warnings': unresolved,
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
