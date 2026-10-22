"""Verify the complete v2 campaign without treating failed fits as recovery evidence."""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.recovery_eligibility import recovery_eligibility
from cyber_threat_modelling.evaluation.synthetic import SCENARIOS
from cyber_threat_modelling.models.stage7_priors import (
    CANDIDATES,
    CURRENT_PRIOR_ID,
    verify_activation,
)
from cyber_threat_modelling.provenance import file_hash


def checked_run(folder, candidate, scenario, seed, activation, recompute=False):
    result = json.loads((folder/'result.json').read_text())
    run = json.loads((folder/'run.json').read_text())
    options = result['options']
    for key, expected in [('prior_id', CURRENT_PRIOR_ID), ('prior_candidate', candidate),
                          ('scenario', scenario), ('seed', seed)]:
        if options[key] != expected or run['options'][key] != expected:
            raise ValueError(f'Wrong {key}: {folder}')
    if run['real_data_read'] or run['held_out_scoring_enabled']:
        raise ValueError(f'Unexpected real-data access: {folder}')
    if run['prior_amendment']['sha256'] != activation['sha256']:
        raise ValueError(f'Wrong prior activation: {folder}')
    for relative, expected in result['artifacts'].items():
        if file_hash(folder/relative) != expected:
            raise ValueError(f'Changed artifact: {folder/relative}')
    for filename, key in [('data.npz', 'data_sha256'), ('truth.npz', 'truth_sha256'),
                          ('truth_original_coordinates.npz', 'truth_original_coordinates_sha256')]:
        if file_hash(folder/filename) != run[key]:
            raise ValueError(f'Changed generating record: {folder/filename}')
    admitted = [attempt for attempt in result['attempts'] if attempt['accepted']]
    budgets = [(1000, 1000, .95, 12), (2000, 1000, .99, 15), (2000, 2000, .99, 15)]
    for index, attempt in enumerate(result['attempts']):
        if index >= len(budgets) or attempt['retry'] != index:
            raise ValueError(f'Unexpected retry schedule: {folder}')
        observed = tuple(attempt[key] for key in
                         ['warmup', 'draws', 'target_accept', 'max_treedepth'])
        if observed != budgets[index] or attempt['chains'] != 4:
            raise ValueError(f'Changed sampling budget: {folder}')
        diagnostic = attempt['diagnostics']
        numerical_pass = bool(
            diagnostic.get('all_diagnostics_finite', False)
            and diagnostic['max_rhat'] < 1.01
            and diagnostic['min_bulk_ess'] > 400 and diagnostic['min_tail_ess'] > 400
            and diagnostic['min_bfmi'] > .3 and diagnostic['divergences'] == 0
            and diagnostic['tree_depth_saturations'] == 0
        )
        if (diagnostic['accepted'] != numerical_pass or
                attempt['accepted'] != (numerical_pass and not attempt['unresolved_warnings'])):
            raise ValueError(f'Admission does not follow diagnostic gates: {folder}')
    if result['status'] == 'accepted':
        if len(admitted) != 1 or not admitted[0]['diagnostics']['accepted']:
            raise ValueError(f'Inconsistent admission: {folder}')
        if admitted[0]['unresolved_warnings']:
            raise ValueError(f'Unresolved admitted warning: {folder}')
        if admitted[0] is not result['attempts'][-1]:
            raise ValueError(f'Continued after admission: {folder}')
        import xarray as xr

        retry = admitted[0]['retry']
        with xr.open_dataset(folder/f'posterior_{retry}.nc', group='posterior') as posterior:
            if posterior.sizes['chain'] != 4 or posterior.sizes['draw'] != budgets[retry][1]:
                raise ValueError(f'Incomplete accepted posterior: {folder}')
        if recompute:
            import arviz as az

            from cyber_threat_modelling.models.bayesian_controls import diagnostic_report

            idata = az.from_netcdf(folder/f'posterior_{retry}.nc')
            actual, _ = diagnostic_report(idata, budgets[retry][3])
            idata.close()
            for key, expected in admitted[0]['diagnostics'].items():
                if not np.isclose(actual[key], expected, rtol=1e-10, atol=1e-10):
                    raise ValueError(f'Recomputed diagnostic differs ({key}): {folder}')
    elif result['status'] == 'failed_diagnostic_gate':
        if admitted or len(result['attempts']) != 3 or (folder/'recovery.json').exists():
            raise ValueError(f'Incomplete or mislabelled failed case: {folder}')
    else:
        raise ValueError(f'Unknown terminal status: {folder}')
    return {
        'directory': folder.as_posix(), 'scenario': scenario, 'seed': seed,
        'candidate': candidate, 'status': result['status'],
        'attempts': len(result['attempts']), 'result_sha256': file_hash(folder/'result.json'),
        'data_sha256': run['data_sha256'], 'truth_sha256': run['truth_sha256'],
        'original_truth_sha256': run['truth_original_coordinates_sha256'],
        'source_control': run['source_control'],
        'options': options,
        'accepted_diagnostics_recomputed': recompute and result['status'] == 'accepted',
        'compute_thread_policy': run.get('compute_thread_policy', 'original library defaults'),
    }


def routed_folder(routing, campaign, key, default, root):
    """Use only an explicit, project-contained route; never select by outcome."""
    chosen = Path(routing.get(campaign, {}).get(key, default))
    resolved = chosen.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'Route escapes the project: {chosen}')
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, default=Path('artifacts/stage7/recovery-v6'))
    parser.add_argument('--sensitivity', type=Path, default=Path('artifacts/stage7/sensitivity-v2'))
    parser.add_argument('--severity-repair', type=Path,
                        default=Path('artifacts/stage7/severity-conditional-v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--routing', type=Path,
                        help='Explicit replacement routes and preserved earlier attempts.')
    parser.add_argument('--recompute-diagnostics', action='store_true')
    args = parser.parse_args()
    routing = json.loads(args.routing.read_text()) if args.routing else {}
    primary_keys = {f'{name}/{seed}' for name in SCENARIOS for seed in [7101, 7102, 7103]
                    if name != 'representative' or seed == 7101}
    sensitivity_keys = {f'{candidate}/{name}/7101' for candidate in CANDIDATES
                        if candidate != 'primary'
                        for name in ['factors_sparse', 'representative', 'fallback_zero']}
    severity_keys = {f'severity_missing/{seed}' for seed in [7101, 7102, 7103]}
    for campaign, expected in [('primary', primary_keys), ('sensitivity', sensitivity_keys),
                               ('severity_coordinates', severity_keys)]:
        if set(routing.get(campaign, {})) - expected:
            raise ValueError(f'Unknown {campaign} routing keys.')
    activation = verify_activation(Path.cwd(), CURRENT_PRIOR_ID)
    records, primary = [], {}
    for scenario in SCENARIOS:
        for seed in [7101, 7102, 7103]:
            if scenario == 'representative' and seed != 7101:
                continue
            folder = routed_folder(routing, 'primary', f'{scenario}/{seed}',
                                   args.primary/f'{scenario}-{seed}', Path.cwd())
            record = checked_run(folder, 'primary', scenario, seed, activation,
                                 args.recompute_diagnostics)
            record['campaign'] = 'primary'
            records.append(record)
            primary[scenario, seed] = record
    count_panels = {}
    for record in primary.values():
        with np.load(Path(record['directory'])/'data.npz') as observed:
            mask = observed['mask_N'].astype(bool)
            counts = np.where(mask, observed['N'], 0).astype('<i8')
            digest = sha256(np.asarray(counts.shape, dtype='<i8').tobytes()
                            + mask.astype('u1').tobytes() + counts.tobytes()).hexdigest()
        record['observed_count_fingerprint'] = digest
        count_panels.setdefault(digest, []).append({
            'scenario': record['scenario'], 'seed': record['seed'],
        })
    for candidate in CANDIDATES:
        if candidate == 'primary':
            continue
        for scenario in ['factors_sparse', 'representative', 'fallback_zero']:
            folder = routed_folder(routing, 'sensitivity', f'{candidate}/{scenario}/7101',
                                   args.sensitivity/candidate/f'{scenario}-7101', Path.cwd())
            record = checked_run(folder, candidate, scenario, 7101, activation,
                                 args.recompute_diagnostics)
            record['campaign'] = 'sensitivity'
            records.append(record)
    # The predeclared repair continues beyond seed 7101 only if that fit passes.
    first_folder = routed_folder(routing, 'severity_coordinates', 'severity_missing/7101',
                                 args.severity_repair/'severity_missing-7101', Path.cwd())
    first = checked_run(first_folder, 'primary', 'severity_missing', 7101, activation,
                        args.recompute_diagnostics)
    first['campaign'] = 'severity_coordinates'
    records.append(first)
    if first['status'] == 'accepted':
        for seed in [7102, 7103]:
            folder = routed_folder(routing, 'severity_coordinates', f'severity_missing/{seed}',
                                   args.severity_repair/f'severity_missing-{seed}', Path.cwd())
            record = checked_run(folder, 'primary', 'severity_missing', seed, activation,
                                 args.recompute_diagnostics)
            record['campaign'] = 'severity_coordinates'
            records.append(record)
    for record in records:
        record.update(recovery_eligibility(record['directory'], record['status'],
                                           routing, Path.cwd()))
        reference = primary[record['scenario'], record['seed']]
        for key in ['data_sha256', 'truth_sha256', 'original_truth_sha256']:
            if record[key] != reference[key]:
                raise ValueError(f'Datasets differ across repeated fits: {record["directory"]}')
    counts = {name: dict(Counter(r['status'] for r in records if r['campaign'] == name))
              for name in ['primary', 'sensitivity', 'severity_coordinates']}
    unique = {}
    for record in records:
        key = Path(record['directory']).resolve()
        if key in unique:
            if unique[key]['result_sha256'] != record['result_sha256']:
                raise ValueError(f'Conflicting repeated verification: {key}')
            unique[key].setdefault('also_verified_as', []).append(record['campaign'])
        else:
            unique[key] = record
    overlapping_groups = len(unique) != len(records)
    records = list(unique.values())
    earlier = []
    for item in routing.get('earlier_attempts', []):
        folder = routed_folder({}, '', '', item['directory'], Path.cwd())
        run = json.loads((folder/'run.json').read_text())
        reference = primary[run['options']['scenario'], run['options']['seed']]
        for key, filename in [('data_sha256', 'data.npz'), ('truth_sha256', 'truth.npz'),
                              ('original_truth_sha256', 'truth_original_coordinates.npz')]:
            if file_hash(folder/filename) != reference[key]:
                raise ValueError(f'Changed data in earlier attempt: {folder}')
        for relative, digest in item['artifacts'].items():
            path = (folder/relative).resolve()
            if not path.is_relative_to(folder.resolve()) or file_hash(path) != digest:
                raise ValueError(f'Changed earlier attempt artifact: {path}')
        if (folder/'result.json').exists():
            old = checked_run(folder, run['options']['prior_candidate'],
                              run['options']['scenario'], run['options']['seed'], activation)
            if item['status'] != old['status']:
                raise ValueError(f'Wrong earlier attempt status: {folder}')
        elif item['status'] not in ['interrupted', 'execution_failed']:
            raise ValueError(f'Unfinished attempt needs explicit status: {folder}')
        earlier.append(item)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'audit.json', {
        'prior_id': CURRENT_PRIOR_ID, 'activation': activation,
        'complete': True, 'primary_cases': len(primary),
        'distinct_observed_training_count_panels': len(count_panels),
        'observed_training_count_panel_groups': count_panels,
        'counts': counts, 'verification_groups_overlap': overlapping_groups,
        'runs': records, 'held_out_scoring_enabled': False,
        'routing_sha256': file_hash(args.routing) if args.routing else None,
        'recompute_diagnostics_requested': args.recompute_diagnostics,
        'earlier_attempts': earlier,
        'interpretation': 'Completion includes explicit diagnostic failures. Sensitivity and '
                          'coordinate refits are not independent generated datasets. '
                          'Severity-coordinate verification may reference the same files as '
                          'routed primary cases; runs contains each directory only once. '
                          'Effort variants share observations. Count-panel fingerprints show '
                          'reuse; distinct fingerprints are not a claim of independent '
                          'replication or simulation-based calibration.',
        'source_sha256': file_hash(Path(__file__)),
    })
    print(json.dumps({'complete': True, 'fits': len(records), 'primary_cases': len(primary),
                      'distinct_count_panels': len(count_panels)}))


if __name__ == '__main__':
    main()
