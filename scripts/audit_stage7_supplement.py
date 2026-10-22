"""Verify/import supplement evidence; summarize only numerically admitted fits."""

import argparse
import json
import shutil
from contextlib import closing
from pathlib import Path

import arviz as az
import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import (
    ACTIVATION,
    FIXTURES,
    declaration,
    load_fixture,
)
from cyber_threat_modelling.evaluation.synthetic_recovery import (
    future_counts,
    interval_metrics,
    posterior_array,
    reconstructed_loading,
)
from cyber_threat_modelling.models.bayesian_controls import diagnostic_report
from cyber_threat_modelling.provenance import file_hash


def check(root, folder, case):
    declared = declaration(root)
    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    run = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    preflight = json.loads((folder/'preflight.json').read_text(encoding='utf-8'))
    if result['case'] != case or run['case'] != case or not preflight['passed']:
        raise ValueError('Case or preflight mismatch.')
    if run['real_data_read'] or run['held_out_scoring_enabled']:
        raise ValueError('Supplement accessed real data.')
    if run['activation_sha256'] != file_hash(root / ACTIVATION):
        raise ValueError('Supplement activation differs.')
    if run['fixture_manifest_sha256'] != file_hash(root / FIXTURES / 'manifest.json'):
        raise ValueError('Fixture manifest differs.')
    for path, digest in run['source_sha256'].items():
        resolved = (root/path).resolve()
        if not resolved.is_relative_to(root.resolve()) or file_hash(resolved) != digest:
            raise ValueError('Execution source differs.')
    if set(p.name for p in folder.iterdir()) != set(result['artifacts']) | {'result.json'}:
        raise ValueError('Incomplete or unexpected artifact inventory.')
    for name, digest in result['artifacts'].items():
        resolved = (folder/name).resolve()
        if not resolved.is_relative_to(folder.resolve()) or file_hash(resolved) != digest:
            raise ValueError('Changed or escaping result artifact.')
    expected = {'chains': 4, 'warmup': 2000, 'draws': 2000, 'target_accept': .99,
                'max_treedepth': 15, 'sampler_seed': declared['sampler_seed'],
                'initialization_seed': declared['initialization_seed']}
    if run['budget'] != expected or result['budget'] != expected or result['attempt_count'] != 1:
        raise ValueError('Sampling budget changed.')
    if result['options'] != load_fixture(root, case)[3]:
        raise ValueError('Fitted options differ from declared fixture.')
    diagnostics = result['diagnostics']
    complete = not result['early_failure']
    if complete:
        with closing(az.from_netcdf(folder/'posterior.nc')) as idata:
            if idata.posterior.sizes['chain'] != 4 or idata.posterior.sizes['draw'] != 2000:
                raise ValueError('Incomplete posterior dimensions.')
            actual, _ = diagnostic_report(idata, 15)
        for key, value in diagnostics.items():
            valid = not np.isfinite(actual[key]) if value is None else np.isclose(
                actual[key], value, rtol=1e-10, atol=1e-10)
            if not valid:
                raise ValueError('Independently recomputed diagnostic differs: ' + key)
    elif (folder/'posterior.nc').exists() or diagnostics['accepted'] or not any(
            result['early_failure'][key] > 0 for key in ('divergences', 'tree_depth_saturations')):
        raise ValueError('Early stop cannot have a complete admitted posterior.')
    accepted = diagnostics['accepted'] and not result['unresolved_warnings']
    if result['status'] != ('accepted' if accepted else 'failed_diagnostic_gate'):
        raise ValueError('Result does not follow unchanged admission rules.')
    return {'case': case, 'status': result['status'], 'diagnostics': diagnostics,
            'complete_posterior_recomputed': complete,
            'result_sha256': file_hash(folder/'result.json'),
            'github_sha': preflight['host']['github_sha'],
            'github_run_id': preflight['host']['github_run_id'],
            'artifact_hashes_verified': True, 'held_out_scoring_enabled': False}


def import_case(root, args):
    target = root / 'artifacts/stage7/supplement-v1' / args.case
    if target.exists():
        raise FileExistsError('Preserve the existing supplement case; never overwrite it.')
    review = check(root, args.download, args.case)
    if review['github_sha'] != args.head or str(review['github_run_id']) != args.run_id:
        raise ValueError('Downloaded result comes from a different hosted execution.')
    target.mkdir(parents=True)
    # Publish the verified terminal marker last, preserving the original bytes.
    for path in args.download.iterdir():
        if path.name != 'result.json':
            shutil.copy2(path, target/path.name)
            if file_hash(path) != file_hash(target/path.name):
                raise ValueError('Imported bytes differ.')
    shutil.copy2(args.download/'result.json', target/'result.json')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'import.json', {
        **review, 'directory': target.relative_to(root).as_posix(),
        'download': args.download.as_posix(), 'source_sha256': file_hash(Path(__file__))})
    print(args.case, review['status'], 'independently verified and imported', flush=True)


def observable_review(root, folder, case, output):
    review = check(root, folder, case)
    if review['status'] != 'accepted':
        raise ValueError('No recovery summaries for a failed posterior.')
    data, truth, _, options = load_fixture(root, case)
    with closing(az.from_netcdf(folder/'posterior.nc')) as idata:
        posterior = idata.posterior
        eta = posterior_array(posterior, 'eta_by_regime')[:, :, 0]
        indices = np.arange(len(eta))
        known = options['level_settings']['known_indices']
        metrics = {label: {'log_intensity': interval_metrics(
                               eta[:, :, cols], truth['eta'][:, cols]),
                           'expected_counts': interval_metrics(
                               np.exp(eta[:, :, cols]), truth['mu'][:, cols])}
                   for label, cols in [('all', list(range(10))), ('known', known)]}
        if options.get('fallback_index') is not None:
            metrics['fallback'] = {
                'log_intensity': interval_metrics(eta[:, :, [9]], truth['eta'][:, [9]]),
                'expected_counts': interval_metrics(np.exp(eta[:, :, [9]]), truth['mu'][:, [9]])}
        loading = reconstructed_loading(posterior, 'f', 10, options['factors'], indices)
        phi = posterior_array(posterior, 'f_phi')
        covariance = {}
        for lag in (0, 1, 6, 12):
            draws = np.einsum('nkr,nr,nlr->nkl', loading, phi**lag, loading)
            target = (truth['loading'] * truth['phi']**lag) @ truth['loading'].T
            covariance[str(lag)] = interval_metrics(draws, target)
        # Preserve chains for a separate exploration review, even if numerical gates pass.
        summaries = {}
        for name in ('m_sd', 'm', 'f_phi', 'f_diag', 'f_free'):
            values = posterior[name].values
            summaries[name] = {'chain_means': values.mean(axis=1).tolist(),
                               'chain_quantiles_05_50_95':
                               np.quantile(values, [.05, .5, .95], axis=1).tolist()}
        paths = future_counts(posterior, data, truth, options, indices,
                              np.zeros(len(indices), dtype=int), np.random.default_rng(17101))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'future.npz', counts=paths)
    write_json(output/'review.json', {
        **review, 'observable_recovery': metrics, 'covariance': covariance,
        'chain_summaries': summaries, 'future_sha256': file_hash(output/'future.npz'),
        'interpretation': 'Numerical admission only; exploration review remains separate. '
                          'Reduced cases approximate a two-factor truth. Masked fallback '
                          'predictions do not validate the original all-zero count target. '
                          'These fixed-truth cases are not general calibration or SBC.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['import', 'review'])
    parser.add_argument('--case', required=True)
    parser.add_argument('--download', type=Path)
    parser.add_argument('--head')
    parser.add_argument('--run-id')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_fixture(root, args.case)
    if args.action == 'import':
        if not all((args.download, args.head, args.run_id)):
            parser.error('Import requires download, head and run ID.')
        import_case(root, args)
    else:
        observable_review(root, root/'artifacts/stage7/supplement-v1'/args.case,
                          args.case, args.output)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
