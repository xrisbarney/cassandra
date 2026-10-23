"""Apply existing recovery summaries only after complete free-fit admission/review."""

import json
from contextlib import closing
from pathlib import Path

import arviz as az
import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import load_fixture
from cyber_threat_modelling.evaluation.synthetic_recovery import (
    future_counts,
    interval_metrics,
    posterior_array,
    reconstructed_loading,
)
from cyber_threat_modelling.provenance import file_hash


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'docs/simulations/stage7-reauthorized-20261023'
    audit = json.loads((output/'diagnostics.json').read_text())
    review_path = output/'review-decision.json'
    review = json.loads(review_path.read_text())
    selected = next(row for row in audit['cases'] if row['case'] == 'joint_free')
    if selected['status'] != 'accepted' or not review['free_persistence_exploration_clear']:
        raise ValueError('Recovery requires a complete admitted and reviewed free posterior.')
    base = root/'artifacts/stage7/reauthorized-20261023'
    posterior_path = base/'fits/joint_free/posterior.nc'
    assert file_hash(posterior_path) == selected['result']['artifacts']['posterior.nc']
    data, truth, _, options = load_fixture(root, 'matched_one_factor_fallback')
    with closing(az.from_netcdf(posterior_path)) as idata:
        posterior = idata.posterior
        eta = posterior_array(posterior, 'eta_by_regime')[:, :, 0]
        indices = np.arange(len(eta))
        known = options['level_settings']['known_indices']
        groups = [('all', list(range(10))), ('known', known), ('fallback', [9])]
        metrics = {
            label: {
                'log_intensity': interval_metrics(eta[:, :, cols], truth['eta'][:, cols]),
                'expected_counts': interval_metrics(np.exp(eta[:, :, cols]),
                                                    truth['mu'][:, cols]),
            }
            for label, cols in groups
        }
        loading = reconstructed_loading(posterior, 'f', 10, 1, indices)
        phi = posterior_array(posterior, 'f_phi')
        covariance = {}
        for lag in (0, 1, 6, 12):
            draws = np.einsum('nkr,nr,nlr->nkl', loading, phi**lag, loading)
            target = (truth['loading']*truth['phi']**lag) @ truth['loading'].T
            covariance[str(lag)] = interval_metrics(draws, target)
        paths = future_counts(posterior, data, truth, options, indices,
                              np.zeros(len(indices), dtype=int), np.random.default_rng(17101))
        future = {str(h): {label: interval_metrics(paths[:, h-1, cols],
                                                  truth['future_counts'][h-1, cols])
                           for label, cols in groups} for h in (1, 3, 6, 12)}
        parameters = {}
        for name, target in [('m', 'm'), ('psi', 'psi'), ('f_phi', 'phi')]:
            x = posterior_array(posterior, name)
            parameters[name] = {
                'truth': truth[target].tolist(), 'mean': x.mean(axis=0).tolist(),
                'q05': np.quantile(x, .05, axis=0).tolist(),
                'q95': np.quantile(x, .95, axis=0).tolist(),
                **interval_metrics(x, truth[target]),
            }
    artifact = base/'recovery'
    artifact.mkdir(exist_ok=False)
    np.savez_compressed(artifact/'future.npz', counts=paths)
    if (output/'recovery.json').exists():
        raise FileExistsError('Preserve the existing recovery summary.')
    write_json(output/'recovery.json', {
        'case': 'joint_free', 'source_sha256': file_hash(Path(__file__)),
        'posterior_sha256': file_hash(posterior_path),
        'review_decision_sha256': file_hash(review_path),
        'observable_recovery': metrics, 'covariance': covariance, 'parameters': parameters,
        'synthetic_future_predictive': future,
        'future_paths_sha256': file_hash(artifact/'future.npz'), 'future_rng_seed': 17101,
        'fallback_observed_total': int(data['N'][:, 9].sum()),
        'fallback_future_zero_fraction': float((paths[:, :, 9] == 0).mean()),
        'interpretation': 'One fixed-truth matched fixture, not SBC or general calibration. '
                          'Recovery is descriptive; no new acceptance cutoff. '
                          'All-zero observations need not identify the exact generating rate. '
                          'Future draws concern generated data only; no real held-out scoring.',
        'stage8_enabled': False, 'held_out_scoring_enabled': False,
    })


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
