"""One fresh attempt per reauthorized diagnostic; preserve every completed chain."""

import argparse
import json
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pymc as pm
from run_stage7 import IrrecoverableSamplingGate, retained_failure
from run_stage7_supplement import classify
from stage7_chain_checkpoint import save_complete_chain
from stage7_terminal_model import build_joint

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import load_fixture, verify_fit_authority
from cyber_threat_modelling.models.bayesian_controls import diagnostic_report
from cyber_threat_modelling.provenance import file_hash, git_revision

RECORD = Path('docs/decisions/STAGE7_DIAGNOSTIC_REAUTHORIZATION_20261023.json')
CLOSURE = Path('docs/decisions/STAGE7_REAUTHORIZED_OUTCOME_20261023.json')
BASE = Path('artifacts/stage7/reauthorized-20261023')
CASES = ('joint_free', 'joint_fixed')


def authority(root):
    if (root/CLOSURE).exists():
        raise ValueError('The reauthorized diagnostic round has already closed.')
    record = json.loads((root/RECORD).read_text())
    if record['status'] != 'active' or record['cases'] != list(CASES):
        raise ValueError('No active authority for exactly these two diagnostics.')
    expected = (2, 1, 5400, 4, 2000, 2000, .99, 15, 7103, 7301)
    actual = tuple(record[k] for k in ('attempt_limit', 'attempts_per_case',
        'per_attempt_wall_seconds', 'chains', 'warmup', 'draws', 'target_accept',
        'max_treedepth', 'sampler_seed', 'initialization_seed'))
    if actual != expected or record['stage8_enabled'] or record['held_out_scoring_enabled']:
        raise ValueError('Declared diagnostic budget or scope changed.')
    if datetime.now(UTC) >= datetime.fromisoformat(record['global_deadline_utc']):
        raise TimeoutError('Global sampling deadline reached.')
    verify_fit_authority(root, 'matched_one_factor_fallback')
    parent = json.loads((root/'docs/decisions/STAGE7_TERMINAL_DIAGNOSTICS_20261022.json')
                        .read_text())
    for name, digest in {**parent['files_sha256'], **record['files_sha256']}.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError('Sealed source or evidence changed: '+name)
    return record


def fit(root, case):
    if case not in CASES:
        raise ValueError('Undeclared diagnostic.')
    record = authority(root)
    deadline = datetime.fromisoformat(record['global_deadline_utc'])
    output = root/BASE/'fits'/case
    output.mkdir(parents=True, exist_ok=False)
    data, truth, _, options = load_fixture(root, 'matched_one_factor_fallback')
    fixed = truth['phi'] if case == 'joint_fixed' else None
    write_json(output/'run.json', {
        'case': case, 'attempt_count': 1, 'chains': 4, 'warmup': 2000, 'draws': 2000,
        'target_accept': .99, 'max_treedepth': 15, 'sampler_seed': 7103,
        'initialization_seed': 7301, 'fixed_phi': None if fixed is None else fixed.tolist(),
        'activation_sha256': file_hash(root/RECORD), 'source_control': git_revision(root),
        'held_out_scoring_enabled': False, 'stage8_enabled': False,
        'fresh_attempt_not_continuation': True, 'numba_opt': '3', 'numerical_threads': 1})
    start = time.perf_counter()
    timing, failure, checkpoints = {}, {}, []
    checkpoint_seconds = 0.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = build_joint(data, options, fixed_phi=fixed)
        graph_seconds = time.perf_counter()-start
        with model:
            stamp = time.perf_counter()
            step = pm.NUTS(target_accept=.99, max_treedepth=15, compile_kwargs={'mode': 'NUMBA'})
            compile_seconds = time.perf_counter()-stamp
            point = model.initial_point()
            rng = np.random.default_rng(7301)
            init = [{k: v+rng.normal(0, .1, v.shape) for k, v in point.items()} for _ in range(4)]

            def callback(trace, draw):
                nonlocal checkpoint_seconds
                elapsed = time.perf_counter()-start
                row = timing.setdefault(str(draw.chain), {})
                phase = 'warmup' if draw.tuning else 'retained'
                row[phase] = row.get(phase, 0)+1
                if (draw.draw_idx+1) % 50 == 0:
                    write_json(output/'heartbeat.json', {'chain': int(draw.chain),
                               'draw_index': int(draw.draw_idx), 'elapsed_seconds': elapsed,
                               'completed_transitions': timing, 'checkpoints': checkpoints})
                if (draw.draw_idx+1) % 500 == 0:
                    print(case, draw.chain, draw.draw_idx+1, round(elapsed, 1), flush=True)
                reason = retained_failure(draw.stats, draw.tuning, 15)
                if draw.draw_idx == 3999:
                    checkpoint_start = time.perf_counter()
                    checkpoints.append(save_complete_chain(
                        trace, model, output, int(draw.chain), 2000, 2000))
                    np.savez_compressed(output/f'chain_{draw.chain}_last_point.npz', **draw.point)
                    performance, unresolved = classify(caught)
                    write_json(output/f'chain_{draw.chain}_warnings.json', {
                        'performance_warnings': performance, 'unresolved_warnings': unresolved})
                    checkpoint_seconds += time.perf_counter()-checkpoint_start
                    print(case, 'checkpointed chain', draw.chain, flush=True)
                if reason or time.perf_counter()-start >= 5400 or datetime.now(UTC) >= deadline:
                    failure.update(reason or {'time_budget_exhausted': True},
                                   chain=int(draw.chain), draw_index=int(draw.draw_idx))
                    np.savez_compressed(output/'failed_state.npz', **draw.point)
                    write_json(output/'early_failure.json', failure)
                    raise IrrecoverableSamplingGate('Attempt stopped at its declared gate.')

            stamp = time.perf_counter()
            try:
                idata = pm.sample(draws=2000, tune=2000, chains=4, cores=1, step=step,
                                  random_seed=7103, initvals=init, callback=callback,
                                  progressbar=False)
            except IrrecoverableSamplingGate:
                idata = None
            sampling_seconds = time.perf_counter()-stamp
    performance, unresolved = classify(caught)
    if idata is not None:
        diagnostics, table = diagnostic_report(idata, 15)
        idata.to_netcdf(output/'posterior.nc')
        table.to_csv(output/'diagnostics.csv')
        status = 'accepted' if diagnostics['accepted'] and not unresolved else 'failed_diagnostics'
    else:
        diagnostics = None
        status = ('time_budget_exhausted' if failure.get('time_budget_exhausted')
                  else 'early_failure')
    if diagnostics:
        diagnostics = {k: None if isinstance(v, float) and not np.isfinite(v) else v
                       for k, v in diagnostics.items()}
    write_json(output/'result.json', {
        'case': case, 'status': status, 'attempt_count': 1, 'diagnostics': diagnostics,
        'early_failure': failure or None, 'graph_seconds': graph_seconds,
        'step_compile_seconds': compile_seconds, 'sample_call_seconds': sampling_seconds,
        'checkpoint_seconds_within_sample_call': checkpoint_seconds,
        'elapsed_seconds': time.perf_counter()-start, 'completed_transitions': timing,
        'performance_warnings': performance, 'unresolved_warnings': unresolved,
        'fixed_persistence_is_conditional_only': fixed is not None, 'checkpoints': checkpoints,
        'artifacts': {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    print(case, status, flush=True)


def main():
    import os

    if os.environ.get('NUMBA_OPT') != '3':
        raise ValueError('Reuse the verified NUMBA_OPT=3 execution profile.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    args = parser.parse_args()
    fit(Path(__file__).resolve().parents[1], args.case)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
